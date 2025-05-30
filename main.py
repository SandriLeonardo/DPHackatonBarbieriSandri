import argparse
import torch
import os
import pandas as pd
import signal
import sys
from torch_geometric.loader import DataLoader
from source.models import GNN
from source.losses import get_loss_function
from source.training import train_model, evaluate_model
from source.data_utils import load_dataset, add_zeros, add_original_indices
from source.utils import set_seed, setup_logging
from sklearn.model_selection import train_test_split


def cleanup_cuda():
    """Clean up CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA memory cleared")


def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt"""
    print('\nKeyboard interrupt detected. Cleaning up...')
    cleanup_cuda()
    sys.exit(0)


def get_model_defaults(model_type):
    """Get default parameters for specific model types"""
    if model_type == "gce_model":
        return {
            'num_layer': 2,
            'emb_dim': 128,
            'drop_ratio': 0.3,
            'virtual_node': True,
            'residual': True,
            'JK': 'last',
            'graph_pooling': 'mean',
            'edge_drop_ratio': 0.15,
            'batch_norm': True,
            'batch_size': 64,
            'epochs': 200,
            'learning_rate': 5e-3,
            'patience': 25,
            'scheduler': 'plateau',
            'gce_q': 0.9,
            'loss_type': 'gce',
            'best_metric': 'f1'
        }
    elif model_type == "gcod_model":
        return {
            'num_layer': 3,
            'emb_dim': 218,
            'drop_ratio': 0.7,
            'virtual_node': True,
            'residual': True,
            'JK': 'last',
            'graph_pooling': 'mean',
            'edge_drop_ratio': 0.1,
            'batch_norm': True,
            'batch_size': 64,
            'epochs': 250,
            'learning_rate': 5e-3,
            'patience': 25,
            'scheduler': 'plateau',
            'loss_type': 'gcod',
            'gcod_lambda_p': 2.0,
            'gcod_lambda_r': 0.1,
            'gcod_T_u': 15,
            'gcod_lr_u': 0.1,
            'best_metric': 'accuracy'
        }
    else:
        return get_model_defaults("gce_model")


def apply_model_defaults(args):
    """Apply model-specific defaults, but don't override user-specified values"""
    defaults = get_model_defaults(args.model_type)

    for key, default_value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, default_value)

    return args


def split_dataset_indices(dataset, val_split=0.2, seed=777):
    """Split dataset indices for train/validation split"""
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=None
    )
    return train_indices, val_indices


def create_subset_loader(dataset, indices, batch_size, shuffle=False):
    """Create DataLoader for a subset of the dataset"""
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def main(args):
    # Register signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # Apply model-specific defaults
    args = apply_model_defaults(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup device with detailed info
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(args.device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(args.device).total_memory / 1e9:.1f}GB")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Loss type: {args.loss_type}")
    print(f"Best metric: {args.best_metric}")

    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submission", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Setup logging
    setup_logging(args)

    # Load test dataset (for final predictions only)
    test_dataset = load_dataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine dataset folder name for output
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]

    # Model configuration from arguments
    model_config = {
        'num_class': 6,
        'num_layer': args.num_layer,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'virtual_node': args.virtual_node,
        'residual': args.residual,
        'JK': args.JK,
        'graph_pooling': args.graph_pooling,
        'edge_drop_ratio': args.edge_drop_ratio,
        'batch_norm': args.batch_norm
    }

    # Initialize model
    model = GNN(**model_config).to(device)

    # Get loss function with parameters
    loss_kwargs = {'num_classes': 6}
    if args.loss_type == "gce":
        loss_kwargs['q'] = args.gce_q
    elif args.loss_type == "noisy":
        loss_kwargs['p_noisy'] = getattr(args, 'noise_prob', 0.2)
    elif args.loss_type == "gcod":
        loss_kwargs['alpha_train'] = getattr(args, 'gcod_lambda_p', 2.0)
        loss_kwargs['lambda_r'] = getattr(args, 'gcod_lambda_r', 0.1)

    criterion = get_loss_function(args.loss_type, **loss_kwargs)

    try:
        # Training mode
        if args.train_path:
            print("Training mode: Training model on provided dataset")

            # Load full training dataset
            full_train_dataset = load_dataset(args.train_path, transform=add_zeros)

            # Split training data into train/validation
            train_indices, val_indices = split_dataset_indices(
                full_train_dataset,
                val_split=args.val_split,
                seed=args.seed
            )

            print(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")

            # Add original indices if using GCOD (must be done before creating subsets)
            if args.loss_type == "gcod":
                full_train_dataset = add_original_indices(full_train_dataset)

            # Create train and validation loaders
            train_loader = create_subset_loader(
                full_train_dataset, train_indices, args.batch_size, shuffle=True
            )
            val_loader = create_subset_loader(
                full_train_dataset, val_indices, args.batch_size, shuffle=False
            )

            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            # Setup scheduler if specified
            scheduler = None
            if args.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=getattr(args, 'scheduler_step_size', 50),
                    gamma=getattr(args, 'scheduler_gamma', 0.5)
                )
            elif args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=getattr(args, 'scheduler_gamma', 0.5),
                    patience=getattr(args, 'scheduler_patience', 10)
                )

            # Setup checkpoint path
            checkpoint_path = f"checkpoints/model_{test_dir_name}_{args.model_type}"

            # Prepare GCOD arguments if needed
            gcod_args = None
            if args.loss_type == "gcod":
                gcod_args = argparse.Namespace(
                    gcod_T_u=getattr(args, 'gcod_T_u', 15),
                    gcod_lr_u=getattr(args, 'gcod_lr_u', 0.1)
                )

            # Train model (now using proper val_loader instead of test_loader)
            best_model_path = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=val_loader,  # This is now validation data, not test data
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                epochs=args.epochs,
                checkpoint_path=checkpoint_path,
                patience=args.patience,
                best_metric=args.best_metric,  # This will now correctly use F1 for gce_model
                loss_type=args.loss_type,
                gcod_args=gcod_args
            )

            # Load best model for prediction
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from: {best_model_path}")

        else:
            print("Inference mode: Using pre-trained model")

            # Load pre-trained model
            checkpoint_path = f"checkpoints/model_{test_dir_name}_{args.model_type}_best.pth"
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f"Loaded pre-trained model from: {checkpoint_path}")
            else:
                print(f"Warning: No pre-trained model found at {checkpoint_path}")
                print("Using randomly initialized model")

        # Generate predictions on actual test set
        predictions = evaluate_model(test_loader, model, device)

        # Save predictions to CSV
        output_csv_path = f"submission/testset_{test_dir_name}.csv"
        test_graph_ids = list(range(len(predictions)))
        output_df = pd.DataFrame({
            "id": test_graph_ids,
            "pred": predictions
        })
        output_df.to_csv(output_csv_path, index=False)
        print(f"Test predictions saved to {output_csv_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_cuda()
        sys.exit(0)
    finally:
        # Always cleanup at the end
        cleanup_cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GNN model on graph datasets.")

    # Required arguments
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to the test dataset.")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Path to the training dataset (optional).")

    # Model selection with defaults
    parser.add_argument("--model_type", type=str, default="gce_model",
                        choices=["gce_model", "gcod_model"],
                        help="Model configuration: 'gce_model' (default) or 'gcod_model'")

    # Loss function override (optional)
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["gce", "gcod", "standard", "noisy"],
                        help="Override loss function (uses model_type default if not specified)")

    # Model architecture parameters
    parser.add_argument("--num_layer", type=int, default=None)
    parser.add_argument("--emb_dim", type=int, default=None)
    parser.add_argument("--drop_ratio", type=float, default=None)
    parser.add_argument("--virtual_node", type=bool, default=None)
    parser.add_argument("--residual", type=bool, default=None)
    parser.add_argument("--JK", type=str, default=None, choices=["last", "sum", "cat"])
    parser.add_argument("--graph_pooling", type=str, default=None,
                        choices=["sum", "mean", "max", "attention", "set2set"])
    parser.add_argument("--edge_drop_ratio", type=float, default=None)
    parser.add_argument("--batch_norm", type=bool, default=None)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of training data to use for validation (default: 0.2)")

    # Scheduler parameters
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["step", "cosine", "plateau"])
    parser.add_argument("--scheduler_step_size", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)

    # Best model criteria - FIXED: default=None and added "f1" to choices
    parser.add_argument("--best_metric", type=str, default=None,
                        choices=["accuracy", "loss", "f1"])

    # Loss function parameters
    parser.add_argument("--gce_q", type=float, default=0.9)
    parser.add_argument("--noise_prob", type=float, default=0.2)
    parser.add_argument("--gcod_lambda_p", type=float, default=2.0)
    parser.add_argument("--gcod_lambda_r", type=float, default=0.1)
    parser.add_argument("--gcod_T_u", type=int, default=15)
    parser.add_argument("--gcod_lr_u", type=float, default=0.1)

    # System parameters
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=777)

    args = parser.parse_args()
    main(args)