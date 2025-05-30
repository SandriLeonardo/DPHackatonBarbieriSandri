import torch
import random
import numpy as np
import tarfile
import os
import logging


def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.

    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")

# Example usage
# folder_path = "./testfolder/submission"            # Path to the folder you want to compress
# output_file = "./testfolder/submission.gz"         # Output .gz file name
# gzip_folder(folder_path, output_file)

def setup_logging(args):
    """Setup logging configuration"""
    # Determine dataset name from test_path
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]

    # Create logs directory
    logs_dir = f"logs/dataset_{test_dir_name}_{args.model_type}"
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(logs_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting training with model_type: {args.model_type}")
    logging.info(f"Test path: {args.test_path}")
    logging.info(f"Train path: {args.train_path}")