import argparse
import torch
import os
import pandas as pd
from torch_geometric.loader import DataLoader
from source.models import GNN
from source.losses import get_loss_function
from source.training import train_model, evaluate_model
from source.data_utils import load_dataset, add_zeros
from source.utils import set_seed, setup_logging


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
            'learning_rate':5e-3,
            'patience': 25,
            'scheduler': 'plateau',
            'gce_q': 0.9,
            'loss_type': 'gce'
        }
    elif model_type == "gcod_model":
        return {
            'num_layer': 3,
            'emb_dim': 128,
            'drop_ratio': 0.3,
            'virtual_node': True,
            'residual': True,
            'JK': 'last',
            'graph_pooling': 'mean',
            'edge_drop_ratio': 0.2,
            'batch_norm': True,
            'batch_size': 32,
            'epochs': 200,
            'learning_rate': 1e-4,
            'patience': 25,
            'scheduler': 'plateau',
            'loss_type': 'gcod'
        }
    else:  # Default to gce_model
        return get_model_defaults("gce_model")


def apply_model_defaults(args):
    """Apply model-specific defaults, but don't override user-specified values"""
    defaults = get_model_defaults(args.model_type)

    # Only set defaults for arguments that weren't explicitly provided by user
    for key, default_value in defaults.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, default_value)

    # Set loss_type if not explicitly set
    if not hasattr(args, 'loss_type') or args.loss_type is None:
        args.loss_type = defaults['loss_type']

    return args


def main(args):
    # Apply model-specific defaults
    args = apply_model_defaults(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Loss type: {args.loss_type}")

    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submission", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Setup logging
    setup_logging(args)

    # Load test dataset
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

    # Get loss function based on loss type with parameters
    loss_kwargs = {}
    if args.loss_type == "gce":
        loss_kwargs['q'] = args.gce_q
    elif args.loss_type == "noisy":
        loss_kwargs['p_noisy'] = args.noise_prob

    criterion = get_loss_function(args.loss_type, **loss_kwargs)

    # Training mode
    if args.train_path:
        print("Training mode: Training model on provided dataset")

        # Load training dataset
        train_dataset = load_dataset(args.train_path, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Setup scheduler if specified
        scheduler = None
        if args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size,
                                                        gamma=args.scheduler_gamma)
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_gamma,
                                                                   patience=args.scheduler_patience)

        # Setup checkpoint path
        checkpoint_path = f"checkpoints/model_{test_dir_name}"

        # Train model
        best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,  # Use as validation
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            checkpoint_path=checkpoint_path,
            patience=args.patience,
            best_metric=args.best_metric
        )

        # Load best model for prediction
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")

    else:
        print("Inference mode: Using pre-trained model")

        # Load pre-trained model
        checkpoint_path = f"checkpoints/model_{test_dir_name}_best.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded pre-trained model from: {checkpoint_path}")
        else:
            print(f"Warning: No pre-trained model found at {checkpoint_path}")
            print("Using randomly initialized model")

    # Generate predictions
    predictions = evaluate_model(test_loader, model, device)

    # Save predictions to CSV
    output_csv_path = f"submission/testset_{test_dir_name}.csv"
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")


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

    # Model architecture parameters (defaults set by model_type)
    parser.add_argument("--num_layer", type=int, default=None,
                        help="Number of GNN layers")
    parser.add_argument("--emb_dim", type=int, default=None,
                        help="Embedding dimension")
    parser.add_argument("--drop_ratio", type=float, default=None,
                        help="Dropout ratio")
    parser.add_argument("--virtual_node", type=bool, default=None,
                        help="Use virtual node")
    parser.add_argument("--residual", type=bool, default=None,
                        help="Use residual connections")
    parser.add_argument("--JK", type=str, default=None, choices=["last", "sum", "cat"],
                        help="Jumping Knowledge type")
    parser.add_argument("--graph_pooling", type=str, default=None,
                        choices=["sum", "mean", "max", "attention", "set2set"],
                        help="Graph pooling method")
    parser.add_argument("--edge_drop_ratio", type=float, default=None,
                        help="Edge dropout ratio")
    parser.add_argument("--batch_norm", type=bool, default=None,
                        help="Use batch normalization")

    # Training parameters (defaults set by model_type)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training and testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience")

    # Scheduler parameters
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=["step", "cosine", "plateau"],
                        help="Learning rate scheduler type")
    parser.add_argument("--scheduler_step_size", type=int, default=50,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5,
                        help="Decay factor for scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler")

    # Best model criteria
    parser.add_argument("--best_metric", type=str, default="accuracy",
                        choices=["accuracy", "loss"],
                        help="Metric to use for best model selection")

    # Loss function parameters
    parser.add_argument("--gce_q", type=float, default=0.5,
                        help="q parameter for GCE loss")
    parser.add_argument("--noise_prob", type=float, default=0.2,
                        help="Noise probability for Noisy CE loss")

    # System parameters
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device number")
    parser.add_argument("--seed", type=int, default=777,
                        help="Random seed")

    # Checkpointing
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of checkpoints to save during training")

    args = parser.parse_args()
    main(args)