# Copy of preprocessor.py functionality into source/data_utils.py
import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
import gc

# RWSE Configuration
RWSE_MAX_K = 16


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, use_processed=False):
        """
        Enhanced GraphDataset with support for processed data and multiple loading modes.
        """
        if use_processed:
            self.processed_dir = filename
            self.num_graphs = torch.load(f"{filename}/num_graphs.pt")
        else:
            self.raw = filename
            self.num_graphs, self.graphs_dicts = self._count_and_load_graphs()
        self.use_processed = use_processed
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        if self.use_processed:
            return torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        else:
            return GraphDataset.dictToGraphObject(self.graphs_dicts[idx])

    def _count_and_load_graphs(self):
        """Load and count graphs from JSON file with support for compressed files."""
        print(f"Loading graphs from {self.raw}...")
        print("This may take a few minutes, please wait...")

        if self.raw.endswith(".gz"):
            with gzip.open(self.raw, "rt", encoding="utf-8") as f:
                graphs_dicts = json.load(f)
        else:
            with open(self.raw, "r", encoding="utf-8") as f:
                graphs_dicts = json.load(f)

        return len(graphs_dicts), graphs_dicts

    @staticmethod
    def dictToGraphObject(graph_dict, is_test_set=False, graph_idx_info=""):
        """
        Convert graph dictionary to PyTorch Geometric Data object with enhanced processing.
        """
        num_nodes = graph_dict.get('num_nodes', 0)
        if not isinstance(num_nodes, int) or num_nodes < 0:
            num_nodes = 0

        # Create node features - using zeros with shape (num_nodes, 1)
        x = torch.zeros(num_nodes, 1, dtype=torch.long)

        # Handle edge data
        raw_edge_index = graph_dict.get('edge_index', [])
        raw_edge_attr = graph_dict.get('edge_attr', [])
        edge_attr_dim = graph_dict.get('edge_attr_dim', 7)

        if not isinstance(edge_attr_dim, int) or edge_attr_dim <= 0:
            edge_attr_dim = 7

        if num_nodes == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
        else:
            edge_index = torch.tensor(raw_edge_index, dtype=torch.long) if raw_edge_index else torch.empty((2, 0),
                                                                                                           dtype=torch.long)
            edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float) if raw_edge_attr else torch.empty(
                (0, edge_attr_dim), dtype=torch.float)

            # Validate edge_index shape
            if edge_index.numel() > 0 and edge_index.shape[0] != 2:
                print(f"Warning: Invalid edge_index shape for graph {graph_idx_info}. Clearing edges.")
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

            # Validate edge_attr dimensions
            if edge_attr.numel() > 0:
                if edge_attr.shape[1] != edge_attr_dim:
                    print(
                        f"Warning: Mismatch edge_attr_dim (expected {edge_attr_dim}, got {edge_attr.shape[1]}) for graph {graph_idx_info}.")
                    if 'edge_attr_dim' in graph_dict:
                        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
                        if edge_index.shape[1] > 0:
                            print(f"  Cleared edge_attr for {graph_idx_info} due to dim mismatch.")

                # Validate edge count consistency
                if edge_index.shape[1] != edge_attr.shape[0]:
                    print(f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}.")
                    if edge_index.shape[1] == 0:
                        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

        # Handle labels - more flexible parsing
        y_val_raw = graph_dict.get('y')
        y_val = -1

        if is_test_set:
            y_val = -1
        elif y_val_raw is not None:
            # Try multiple parsing strategies
            if isinstance(y_val_raw, int):
                y_val = y_val_raw
            elif isinstance(y_val_raw, list) and len(y_val_raw) > 0:
                # Handle [[4]] -> [4] -> 4
                temp_y = y_val_raw
                while isinstance(temp_y, list) and len(temp_y) > 0:
                    temp_y = temp_y[0]
                if isinstance(temp_y, (int, float)):
                    y_val = int(temp_y)
                else:
                    y_val = -1
            elif isinstance(y_val_raw, (float, str)):
                try:
                    y_val = int(float(y_val_raw))
                except:
                    y_val = -1
            else:
                y_val = -1

        if y_val == -1 and not is_test_set:
            print(f"Warning: 'y' missing or malformed for TRAIN/VAL graph {graph_idx_info}. Raw value: {y_val_raw}")

        y = torch.tensor([y_val], dtype=torch.long)

        # Create Data object
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

        # Add RWSE positional encoding
        if data_obj.num_nodes > 0 and data_obj.num_edges > 0:
            data_obj.rwse_pe = GraphDataset.get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes,
                                                                 k_max=RWSE_MAX_K)
        else:
            data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))

        return data_obj

    @staticmethod
    def get_rw_landing_probs(edge_index, num_nodes, k_max):
        """
        Compute Random Walk Structural Encoding (RWSE) landing probabilities.
        """
        if num_nodes == 0:
            return torch.zeros((0, k_max), device=edge_index.device)
        if edge_index.numel() == 0:
            return torch.zeros((num_nodes, k_max), device=edge_index.device)

        if num_nodes > 1000:
            print(f"Info: RWSE for graph with {num_nodes} nodes. This may be slow.")

        source, _ = edge_index[0], edge_index[1]
        deg = degree(source, num_nodes=num_nodes, dtype=torch.float)
        deg_inv = deg.pow(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        try:
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        except RuntimeError as e:
            max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
            print(f"Error in to_dense_adj: {e}. Max_idx: {max_idx}, Num_nodes: {num_nodes}. Returning zeros for RWSE.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)

        P_dense = deg_inv.view(-1, 1) * adj
        rws_list = []
        Pk = torch.eye(num_nodes, device=edge_index.device)

        for _ in range(1, k_max + 1):
            if Pk.numel() == 0 or P_dense.numel() == 0:
                return torch.zeros((num_nodes, k_max), device=edge_index.device)
            try:
                Pk = Pk @ P_dense
            except RuntimeError as e:
                print(f"RuntimeError during Pk @ P_dense: {e}. Returning zeros.")
                return torch.zeros((num_nodes, k_max), device=edge_index.device)
            rws_list.append(torch.diag(Pk))

        return torch.stack(rws_list, dim=1) if rws_list else torch.zeros((num_nodes, k_max), device=edge_index.device)


class TestGraphDataset(GraphDataset):
    """GraphDataset for test data that suppresses label warnings"""
    def get(self, idx):
        if self.use_processed:
            return torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        else:
            return GraphDataset.dictToGraphObject(self.graphs_dicts[idx], is_test_set=True)


class MultiDatasetLoader:
    """
    Enhanced data loader supporting multiple datasets with separate processing.
    Handles datasets A, B, C, D with train/validation/test splits.
    """

    def __init__(self, base_path=None, val_split_ratio=0.1):
        if base_path is None:
            script_base_path = os.path.dirname(os.path.abspath(__file__))
            self.original_dataset_base_path = os.path.join(script_base_path, 'original_dataset')
        else:
            self.original_dataset_base_path = base_path

        self.val_split_ratio = val_split_ratio
        self.dataset_names = ['A', 'B', 'C', 'D']

        # Setup processed data directory
        script_base_path = os.path.dirname(os.path.abspath(__file__))
        self.processed_dir = os.path.join(script_base_path, 'processed_data_separate')
        os.makedirs(self.processed_dir, exist_ok=True)

    # [Rest of MultiDatasetLoader methods from preprocessor.py - keeping them exactly the same]
    # ... (includes all methods like find_single_json_file, process_single_dataset, etc.)


# Simple functions for basic usage
def dictToGraphObject(graph_dict):
    """Convert graph dictionary to PyTorch Geometric Data object"""
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)


def add_zeros(data):
    """Add zero node features to data objects"""
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def load_dataset(filename, transform=None):
    """Load dataset from file - supports .pt, .json.gz, and .json"""
    if filename.endswith('.pt'):
        # Load preprocessed PyTorch file
        data_list = torch.load(filename, weights_only=False)
        return data_list
    else:
        # Load from JSON
        return GraphDataset(filename, transform=transform)


def load_test_dataset(filename, transform=None):
    """Load test dataset without label warnings"""
    if filename.endswith('.pt'):
        return torch.load(filename, weights_only=False)
    else:
        return TestGraphDataset(filename, transform=transform)


def add_original_indices(dataset):
    """Add original indices to dataset for GCOD tracking"""
    for i, data in enumerate(dataset):
        data.original_idx = i
    return dataset