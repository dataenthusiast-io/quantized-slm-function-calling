"""
Comprehensive data preparation and splitting script for MLX framework compatibility.

This script handles the complete pipeline:
1. Loads function calling datasets with conversations format
2. Transforms conversations into proper chat format for Gemma model
3. Splits data into train/validation/test sets
4. Saves in MLX-compatible JSONL format

The script expects datasets with 'conversations' field containing:
- 'from': 'system' -> system prompt with tools
- 'from': 'human' -> user query
- 'from': 'gpt' -> assistant response with function calls

Usage:
    python prep_data.py --input_file data/xlam-function-calling-60k_train_1000examples
    python prepare_and_split_data.py --input_file data/xlam-function-calling-60k_train_1000examples --output_dir data --num_examples 1000
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset, load_from_disk
from datasets.utils.logging import set_verbosity, WARNING


def transform_function_calling_data(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Transform function calling data into separate system_prompt, user_input, and assistant_output columns.
    
    Args:
        example (Dict[str, Any]): Raw example from the dataset with 'query', 'tools', 'answers' fields
        
    Returns:
        Dict[str, str]: Transformed example with separate columns
    """
    query = example.get('query', '')
    tools = example.get('tools', '')
    answers = example.get('answers', '')
    
    # Create system prompt with tools (without code block formatting)
    system_prompt = "## Instructions\nYou are a helpful assistant with access to the following tools:\n\n" + tools
    
    # User input is the query
    user_input = query
    
    # Assistant output is the function calls in the required format
    assistant_output = answers
    
    return {
        'system_prompt': system_prompt,
        'user_input': user_input,
        'assistant_output': assistant_output
    }


def format_function_call_example(example: Dict[str, Any]) -> str:
    """
    Format a single function calling example into Gemma chat format.
    
    Args:
        example (Dict[str, Any]): Example with system_prompt, user_input, assistant_output
        
    Returns:
        str: Formatted text in Gemma chat format
    """
    system_prompt = example.get('system_prompt', '')
    user_input = example.get('user_input', '')
    assistant_output = example.get('assistant_output', '')
    
    # Build the formatted conversation
    formatted_text = "<bos>"
    
    # Add system prompt and user query as a single user turn
    if system_prompt.strip() and user_input.strip():
        formatted_text += f"<start_of_turn>user\n{system_prompt}\n\n## User\n{user_input}<end_of_turn>\n"
    elif user_input.strip():
        formatted_text += f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
    
    # Add assistant response with function calls
    if assistant_output.strip():
        # Convert the answers to the required function call format
        try:
            import json
            answers = json.loads(assistant_output)
            function_calls = []
            for i, answer in enumerate(answers):
                call_id = f"call_id_{i}"
                function_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": answer["name"],
                        "arguments": json.dumps(answer["arguments"])
                    }
                }
                function_calls.append(function_call)
            
            # Format as markdown JSON (matching base model behavior)
            formatted_text += f"<start_of_turn>model\n```json\n{json.dumps(function_calls, indent=2)}\n```<end_of_turn>\n"
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to raw output if parsing fails
            formatted_text += f"<start_of_turn>model\n{assistant_output}<end_of_turn>\n"
    
    # Add end of sequence token
    formatted_text += "<eos>"
    
    return formatted_text


def prepare_data(dataset: Dataset, num_examples: Optional[int] = None) -> Dataset:
    """
    Prepare dataset for gemma-3-1b-it-bf16 model training.
    
    Args:
        dataset (Dataset): Input dataset to prepare
        num_examples (Optional[int]): Maximum number of examples to process
        
    Returns:
        Dataset: Prepared dataset with formatted text
    """
    # Limit examples if specified
    if num_examples is not None and num_examples > 0:
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
            print(f"Processing {num_examples} examples")
        else:
            print(f"Processing all {len(dataset)} examples")
    else:
        print(f"Processing all {len(dataset)} examples")
    
    # Transform function calling data into separate columns
    print("Transforming function calling data...")
    transformed_dataset = dataset.map(transform_function_calling_data)
    
    # Format each example
    formatted_examples = []
    for i, example in enumerate(transformed_dataset):
        try:
            formatted_text = format_function_call_example(example)
            # Only add non-empty examples
            if len(formatted_text.strip()) > len("<bos><eos>"):
                formatted_examples.append({"text": formatted_text})
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} examples...")
                
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue
    
    print(f"Successfully formatted {len(formatted_examples)} examples")
    
    # Create new dataset with formatted text
    return Dataset.from_list(formatted_examples)


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/validation/test splits.
    
    Args:
        dataset (Dataset): Input dataset to split
        train_ratio (float): Proportion for training set (default: 0.8)
        val_ratio (float): Proportion for validation set (default: 0.1)
        test_ratio (float): Proportion for test set (default: 0.1)
        random_seed (int): Random seed for reproducible splits (default: 42)
        
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get total number of examples
    total_examples = len(dataset)
    print(f"Total examples: {total_examples}")
    
    # Calculate split sizes
    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    test_size = total_examples - train_size - val_size  # Ensure we use all examples
    
    print(f"Split sizes: train={train_size}, validation={val_size}, test={test_size}")
    
    # Create indices and shuffle
    indices = list(range(total_examples))
    random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    return train_dataset, val_dataset, test_dataset


def save_jsonl(dataset: Dataset, output_file: str) -> None:
    """
    Save dataset to JSONL file for MLX compatibility.
    
    Args:
        dataset (Dataset): Dataset to save
        output_file (str): Output file path
    """
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(dataset)} examples to {output_file}")
        
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")


def get_default_output_dir(input_file: str) -> str:
    """
    Generate default output directory for split data.
    
    Args:
        input_file (str): Input file path
        
    Returns:
        str: Default output directory
    """
    # Get the root directory (one level up from this script)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    return str(root_dir / "data" / "training")


def main():
    """Main function to handle CLI execution."""
    parser = argparse.ArgumentParser(
        description="Prepare and split function calling datasets for MLX framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_and_split_data.py --input_file data/xlam-function-calling-60k_train_1000examples
  python prepare_and_split_data.py --input_file data/xlam-function-calling-60k_train_1000examples --num_examples 1000
  python prepare_and_split_data.py --input_file data/xlam-function-calling-60k_train_1000examples --output_dir data --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
        """
    )
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to input dataset file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for split data (default: root/data)"
    )
    
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=None,
        help="Maximum number of examples to process (default: all available)"
    )
    
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Proportion for training set (default: 0.8)"
    )
    
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.1,
        help="Proportion for validation set (default: 0.1)"
    )
    
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.1,
        help="Proportion for test set (default: 0.1)"
    )
    
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        # Set logging to warning level
        set_verbosity(WARNING)
        
        # Load the dataset
        print(f"Loading dataset from {args.input_file}...")
        dataset = load_from_disk(args.input_file)
        print(f"Loaded dataset with {len(dataset)} examples")
        
        # Prepare the data
        print("Preparing data...")
        prepared_dataset = prepare_data(dataset, args.num_examples)
        
        # Split the dataset
        print(f"Splitting dataset with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
        train_dataset, val_dataset, test_dataset = split_dataset(
            prepared_dataset, 
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio, 
            args.random_seed
        )
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = get_default_output_dir(args.input_file)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save splits as JSONL files
        train_file = Path(output_dir) / "train.jsonl"
        val_file = Path(output_dir) / "valid.jsonl"
        test_file = Path(output_dir) / "test.jsonl"
        
        print(f"Saving splits to {output_dir}...")
        save_jsonl(train_dataset, str(train_file))
        save_jsonl(val_dataset, str(val_file))
        save_jsonl(test_dataset, str(test_file))
        
        # Print summary
        print(f"\nData Preparation and Splitting Summary:")
        print(f"  Input: {args.input_file}")
        print(f"  Output directory: {output_dir}")
        print(f"  Train: {len(train_dataset)} examples -> train.jsonl")
        print(f"  Validation: {len(val_dataset)} examples -> valid.jsonl")
        print(f"  Test: {len(test_dataset)} examples -> test.jsonl")
        print(f"  Format: JSONL (MLX compatible)")
        print(f"  Random seed: {args.random_seed}")
        print(f"  Chat format: Gemma with <bos>, <start_of_turn>, <end_of_turn>, <eos> tokens")
        
        # Show a sample of the formatted data
        if len(train_dataset) > 0:
            print(f"\nSample formatted data:")
            sample_text = train_dataset[0]['text']
            # Truncate for display
            display_text = sample_text[:500] + "..." if len(sample_text) > 500 else sample_text
            print(display_text)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
