#!/usr/bin/env python3
"""
Simple test script for the fine-tuned Gemma function calling model.

This script demonstrates how to load and test the fine-tuned model on sample data.
It's designed to be easy to use for anyone wanting to test the model.

Usage:
    python test_model.py [--num_examples N] [--verbose]
"""

import argparse
import json
import sys
import os
from mlx_lm import load, generate

# Add the src directory to the path so we can import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.test_data_loader import (
    load_standardized_test_data, 
    save_standardized_results, 
    create_standardized_result
)
from utils.inference import (
    extract_tools_and_query,
    format_prompt_for_inference,
    extract_function_calls_from_response
)


def test_finetuned_model(num_examples=100, verbose=False):
    """
    Test the fine-tuned model on standardized data.
    
    Args:
        num_examples (int): Number of examples to test
        verbose (bool): Whether to show detailed output
        
    Returns:
        List[Dict]: Test results
    """
    print("🧪 Testing Fine-tuned Model")
    print("=" * 60)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model, tokenizer = load('mlx-community/gemma-3-1b-it-4bit', adapter_path='gemma-3-1b-function-calling-4bit')
    print("✓ Fine-tuned model loaded")
    
    # Load standardized test data
    test_file = "data/training/test.jsonl"
    print(f"Loading {num_examples} standardized examples from {test_file}...")
    test_examples = load_standardized_test_data(test_file, num_examples)
    print(f"✓ Loaded {len(test_examples)} standardized examples")
    
    # Test the model
    results = []
    for example in test_examples:
        print(f"Testing Example {example['example_id']}/{len(test_examples)}")
        
        # Format prompt for inference (same format as training)
        prompt = format_prompt_for_inference(example['tools_json'], example['user_query'])
        
        # Generate response
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                verbose=False,
                max_tokens=200
            )
            
            # Extract function calls from response
            extracted_calls = extract_function_calls_from_response(response)
            print(f"\nExtracted Function Calls:")
            print(f"{extracted_calls}")
            
            # Create standardized result (NO is_success - calculated in analysis)
            result = create_standardized_result(
                example=example,
                actual_response=response,
                extracted_function_calls=extracted_calls,
                prompt_used=prompt,
                model_name="finetuned_model"
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            result = create_standardized_result(
                example=example,
                actual_response=f"ERROR: {str(e)}",
                extracted_function_calls="",
                prompt_used=prompt,
                model_name="finetuned_model"
            )
            results.append(result)
    
    # Save results
    output_file = "data/results/fintuned_model_test_results.json"
    save_standardized_results(results, output_file)
    
    # Print summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Examples tested: {len(results)}")
    print(f"Model: gemma-3-1b-it-4bit (fine-tuned)")
    print(f"Results saved to: {output_file}")
    
    return results


def main():
    """Main function to handle CLI execution."""
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned Gemma function calling model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py                    # Test with 5 examples
  python test_model.py --num_examples 10  # Test with 10 examples
  python test_model.py --verbose          # Show detailed output
        """
    )
    
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=100,
        help="Number of examples to test (default: 100)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed output including extracted function calls"
    )
    
    args = parser.parse_args()
    
    try:
        test_finetuned_model(args.num_examples, args.verbose)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
