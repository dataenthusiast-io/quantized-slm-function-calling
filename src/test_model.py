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
from mlx_lm import load, generate
from inference import load_test_data, extract_tools_and_query, format_prompt_for_base_model, extract_function_calls_from_response


def test_finetuned_model(num_examples=5, verbose=False):
    """
    Test the fine-tuned model on sample data.
    
    Args:
        num_examples (int): Number of examples to test
        verbose (bool): Whether to show detailed output
    """
    print("🚀 Testing Fine-tuned Gemma Function Calling Model")
    print("=" * 60)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model, tokenizer = load('mlx-community/gemma-3-1b-it-4bit', adapter_path='gemma-3-1b-function-calling-4bit')
    print("✓ Model loaded successfully")
    
    # Load test data
    print(f"\nLoading {num_examples} test examples...")
    test_examples = load_test_data('data/training/test.jsonl', num_examples)
    print(f"✓ Loaded {len(test_examples)} examples")
    
    # Test each example
    print(f"\nTesting model on {len(test_examples)} examples...")
    print("=" * 60)
    
    results = []
    for i, example in enumerate(test_examples, 1):
        print(f"\n📝 Example {i}/{len(test_examples)}")
        print("-" * 40)
        
        # Extract tools and query from the test data
        tools_json, user_query = extract_tools_and_query(example['text'])
        
        # Format prompt for the model
        prompt = format_prompt_for_base_model(tools_json, user_query)
        
        # Generate response
        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=200)
        
        # Extract function calls from response
        extracted_calls = extract_function_calls_from_response(response)
        
        # Check if response contains valid function calls
        is_success = False
        try:
            calls = json.loads(extracted_calls)
            is_success = isinstance(calls, list) and len(calls) > 0
        except:
            pass
        
        # Store result
        result = {
            'example_id': i,
            'user_query': user_query,
            'tools_json': tools_json,
            'response': response,
            'extracted_function_calls': extracted_calls,
            'is_success': is_success
        }
        results.append(result)
        
        # Display results
        print(f"Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        if verbose:
            print(f"Extracted calls: {extracted_calls}")
            print(f"Success: {'✅' if is_success else '❌'}")
        
        print()
    
    # Calculate and display summary
    success_count = sum(1 for r in results if r['is_success'])
    success_rate = (success_count / len(results)) * 100
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Examples tested: {len(results)}")
    print(f"Successful function calls: {success_count}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Model: gemma-3-1b-it-4bit (fine-tuned)")
    
    # Save results
    output_file = "data/results/fintuned_model_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
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
