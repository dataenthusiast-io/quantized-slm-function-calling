#!/usr/bin/env python3
"""
Analyze and compare test results from base model vs fine-tuned model.

This script loads the test results from both models and provides a detailed
comparison showing the performance improvement from fine-tuning.

Usage:
    python analyze_results.py [--verbose]
"""

import argparse
import json
import os
from pathlib import Path


def load_results(file_path):
    """Load test results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON from {file_path}: {e}")
        return None


def calculate_syntactic_success(extracted_calls: str) -> bool:
    """
    Calculate syntactic success based on extracted function calls.
    
    Args:
        extracted_calls (str): Extracted function calls string
        
    Returns:
        bool: True if syntactically valid JSON function calls
    """
    if not extracted_calls or extracted_calls.strip() == "":
        return False
    
    try:
        calls = json.loads(extracted_calls)
        return isinstance(calls, list) and len(calls) > 0
    except json.JSONDecodeError:
        return False


def calculate_metrics(results):
    """Calculate performance metrics from standardized test results."""
    if not results:
        return None
    
    total_examples = len(results)
    
    # Calculate syntactic success based on extracted_function_calls
    successful_calls = sum(1 for r in results if calculate_syntactic_success(r.get('extracted_function_calls', '')))
    
    success_rate = (successful_calls / total_examples) * 100 if total_examples > 0 else 0
    
    # Count different types of responses
    valid_json_responses = 0
    markdown_json_responses = 0
    invalid_responses = 0
    
    for result in results:
        # Check actual_response field
        response = result.get('actual_response', '')
        if '```json' in response:
            markdown_json_responses += 1
        elif response.strip().startswith('[') or response.strip().startswith('{'):
            valid_json_responses += 1
        else:
            invalid_responses += 1
    
    return {
        'total_examples': total_examples,
        'successful_calls': successful_calls,
        'success_rate': success_rate,
        'valid_json_responses': valid_json_responses,
        'markdown_json_responses': markdown_json_responses,
        'invalid_responses': invalid_responses
    }


def print_metrics(title, metrics):
    """Print formatted metrics."""
    if not metrics:
        print(f"❌ {title}: No data available")
        return
    
    print(f"\n📊 {title}")
    print("=" * 50)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Successful function calls: {metrics['successful_calls']}")
    print(f"Success rate: {metrics['success_rate']:.1f}%")
    print(f"Valid JSON responses: {metrics['valid_json_responses']}")
    print(f"Markdown JSON responses: {metrics['markdown_json_responses']}")
    print(f"Invalid responses: {metrics['invalid_responses']}")


def print_comparison(base_metrics, finetuned_metrics):
    """Print comparison between base and fine-tuned models."""
    if not base_metrics or not finetuned_metrics:
        print("❌ Cannot compare - missing data")
        return
    
    print(f"\n🔄 COMPARISON")
    print("=" * 50)
    
    # Success rate comparison
    base_rate = base_metrics['success_rate']
    finetuned_rate = finetuned_metrics['success_rate']
    improvement = finetuned_rate - base_rate
    improvement_pct = (improvement / base_rate) * 100 if base_rate > 0 else 0
    
    print(f"Success Rate:")
    print(f"  Base model:      {base_rate:.1f}%")
    print(f"  Fine-tuned:      {finetuned_rate:.1f}%")
    print(f"  Improvement:     +{improvement:.1f} percentage points ({improvement_pct:+.1f}%)")
    
    # Response quality comparison
    base_valid = base_metrics['valid_json_responses'] + base_metrics['markdown_json_responses']
    finetuned_valid = finetuned_metrics['valid_json_responses'] + finetuned_metrics['markdown_json_responses']
    
    print(f"\nValid JSON Responses:")
    print(f"  Base model:      {base_valid}/{base_metrics['total_examples']} ({base_valid/base_metrics['total_examples']*100:.1f}%)")
    print(f"  Fine-tuned:      {finetuned_valid}/{finetuned_metrics['total_examples']} ({finetuned_valid/finetuned_metrics['total_examples']*100:.1f}%)")


def show_failure_examples(results, model_name, max_examples=3):
    """Show examples of failed cases."""
    failures = [r for r in results if not r.get('is_success', False)]
    
    if not failures:
        print(f"\n✅ {model_name}: No failures to show!")
        return
    
    print(f"\n❌ {model_name} - Failure Examples (showing {min(len(failures), max_examples)}):")
    print("-" * 60)
    
    for i, failure in enumerate(failures[:max_examples], 1):
        print(f"\nExample {i}:")
        print(f"Query: {failure.get('user_query', 'N/A')[:100]}...")
        print(f"Response: {failure.get('response', 'N/A')[:200]}...")
        print(f"Extracted: {failure.get('extracted_function_calls', 'N/A')[:100]}...")


def main():
    """Main function to analyze and compare results."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare test results from base model vs fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py              # Basic analysis
  python analyze_results.py --verbose    # Show failure examples
        """
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed failure examples"
    )
    
    args = parser.parse_args()
    
    print("🔍 Analyzing Test Results")
    print("=" * 60)
    
    # Define file paths
    base_results_file = "data/results/base_model_test_results.json"
    finetuned_results_file = "data/results/fintuned_model_test_results.json"
    
    # Load results
    print("Loading test results...")
    base_results = load_results(base_results_file)
    finetuned_results = load_results(finetuned_results_file)
    
    if not base_results and not finetuned_results:
        print("❌ No results files found. Please run the test scripts first:")
        print("   python src/inference.py  # For base model")
        print("   python test_model.py     # For fine-tuned model")
        return 1
    
    # Calculate metrics
    base_metrics = calculate_metrics(base_results)
    finetuned_metrics = calculate_metrics(finetuned_results)
    
    # Print individual metrics
    if base_metrics:
        print_metrics("Base Model Results", base_metrics)
    
    if finetuned_metrics:
        print_metrics("Fine-tuned Model Results", finetuned_metrics)
    
    # Print comparison
    if base_metrics and finetuned_metrics:
        print_comparison(base_metrics, finetuned_metrics)
    
    # Show failure examples if verbose
    if args.verbose:
        if base_results:
            show_failure_examples(base_results, "Base Model")
        if finetuned_results:
            show_failure_examples(finetuned_results, "Fine-tuned Model")
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    if base_metrics and finetuned_metrics:
        improvement = finetuned_metrics['success_rate'] - base_metrics['success_rate']
        print(f"Fine-tuning improved function calling success rate by {improvement:.1f} percentage points")
        print(f"From {base_metrics['success_rate']:.1f}% to {finetuned_metrics['success_rate']:.1f}%")
        
        if improvement > 0:
            print("🎉 Fine-tuning was successful!")
        else:
            print("⚠️  Fine-tuning may need adjustment")
    else:
        print("Run both test scripts to see the full comparison")
    
    return 0


if __name__ == "__main__":
    exit(main())
