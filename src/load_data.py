"""
Data loading utility for Hugging Face datasets with CLI support.

This module provides functionality to load datasets from Hugging Face Hub with
the ability to control the number of examples via command line arguments.
Supports the Salesforce/xlam-function-calling-60k dataset and other compatible datasets.

By default, datasets are automatically saved to the root/data folder with descriptive filenames.
You can specify a custom output path if needed.

Usage:
    python load_data.py --dataset Salesforce/xlam-function-calling-60k --split train --num_examples 1000
    python load_data.py --dataset Salesforce/xlam-function-calling-60k --split train --num_examples 1000 --output_file custom_path.arrow
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Union
from datasets import load_dataset, Dataset
from datasets.utils.logging import set_verbosity, WARNING


def load_data(dataset_name: str, split: str, num_examples: Optional[int] = None) -> Dataset:
    """
    Load a dataset from Hugging Face Hub with optional example limiting.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        split (str): Dataset split to load (e.g., 'train', 'validation', 'test')
        num_examples (Optional[int]): Maximum number of examples to load. If None, loads all examples.
        
    Returns:
        Dataset: Loaded dataset with specified number of examples
        
    Raises:
        ValueError: If dataset_name or split is invalid
        ConnectionError: If unable to connect to Hugging Face Hub
    """
    if not dataset_name or not split:
        raise ValueError("Both dataset_name and split must be provided")
    
    try:
        # Set logging to warning level to reduce verbosity
        set_verbosity(WARNING)
        
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Limit number of examples if specified
        if num_examples is not None and num_examples > 0:
            if len(dataset) > num_examples:
                dataset = dataset.select(range(num_examples))
                print(f"Loaded {num_examples} examples from {dataset_name} ({split} split)")
            else:
                print(f"Dataset has only {len(dataset)} examples, loaded all available")
        else:
            print(f"Loaded {len(dataset)} examples from {dataset_name} ({split} split)")
            
        return dataset
        
    except Exception as e:
        raise ConnectionError(f"Failed to load dataset '{dataset_name}': {str(e)}")


def save_dataset(dataset: Dataset, output_file: str) -> None:
    """
    Save dataset to file.
    
    Args:
        dataset (Dataset): Dataset to save
        output_file (str): Output file path
    """
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(output_file)
        print(f"Dataset saved to {output_file}")
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")


def get_default_output_path(dataset_name: str, split: str, num_examples: Optional[int] = None) -> str:
    """
    Generate default output path for saving dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        split (str): Dataset split
        num_examples (Optional[int]): Number of examples loaded
        
    Returns:
        str: Default output path
    """
    # Get the root directory (one level up from this script)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    data_dir = root_dir / "data" / "training"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on dataset name, split, and number of examples
    dataset_short = dataset_name.split("/")[-1]  # Get just the dataset name part
    if num_examples:
        filename = f"{dataset_short}_{split}_{num_examples}examples"
    else:
        filename = f"{dataset_short}_{split}_all"
    
    return str(data_dir / filename)


def main():
    """Main function to handle CLI execution."""
    parser = argparse.ArgumentParser(
        description="Load datasets from Hugging Face Hub with configurable example count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_data.py --dataset Salesforce/xlam-function-calling-60k --split train --num_examples 1000
  python load_data.py --dataset Salesforce/xlam-function-calling-60k --split train --num_examples 1000 --output_file custom_path.arrow
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="Salesforce/xlam-function-calling-60k",
        help="Name of the dataset on Hugging Face Hub (default: Salesforce/xlam-function-calling-60k)"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        choices=["train", "validation", "test", "all"],
        help="Dataset split to load (default: train)"
    )
    
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=None,
        help="Maximum number of examples to load (default: all available)"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Output file path to save the dataset (default: auto-generated path in root/data folder)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load the dataset
        dataset = load_data(args.dataset, args.split, args.num_examples)
        
        # Always save to file
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = get_default_output_path(args.dataset, args.split, args.num_examples)
        
        save_dataset(dataset, output_path)
        
        # Print dataset info
        print(f"\nDataset Info:")
        print(f"  Name: {args.dataset}")
        print(f"  Split: {args.split}")
        print(f"  Examples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")
        print(f"  Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
