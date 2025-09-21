#!/usr/bin/env python3
"""
Semantic correctness analysis for function calling models.

This script compares the actual generated function calls against the expected
responses to measure semantic accuracy, not just syntactic validity.
"""

import json
import re
from typing import Dict, List, Any, Tuple


def normalize_function_calls(calls_str: str) -> List[Dict[str, Any]]:
    """
    Normalize function calls by extracting and parsing JSON from markdown or plain text.
    """
    if not calls_str or calls_str.strip() == "":
        return []
    
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```json\s*', '', calls_str)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    
    try:
        calls = json.loads(cleaned)
        if isinstance(calls, list):
            return calls
        else:
            return [calls] if isinstance(calls, dict) else []
    except json.JSONDecodeError:
        return []


def extract_function_name_and_args(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Extract function name and arguments from a function call object.
    """
    if 'function' in call and isinstance(call['function'], dict):
        func_name = call['function'].get('name', '')
        args_str = call['function'].get('arguments', '')
        
        # Parse arguments if they're a string
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except:
                args = {}
        else:
            args = args_str if isinstance(args_str, dict) else {}
            
        return func_name, args
    else:
        # Handle direct format: {"name": "func", "arguments": {...}}
        func_name = call.get('name', '')
        args = call.get('arguments', {})
        return func_name, args


def compare_function_calls(expected: List[Dict], actual: List[Dict]) -> Dict[str, Any]:
    """
    Compare expected vs actual function calls and return detailed metrics.
    """
    if not expected and not actual:
        return {
            'exact_match': True,
            'function_name_matches': 0,
            'argument_matches': 0,
            'total_expected': 0,
            'total_actual': 0,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0
        }
    
    if not expected:
        return {
            'exact_match': False,
            'function_name_matches': 0,
            'argument_matches': 0,
            'total_expected': 0,
            'total_actual': len(actual),
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0
        }
    
    if not actual:
        return {
            'exact_match': False,
            'function_name_matches': 0,
            'argument_matches': 0,
            'total_expected': len(expected),
            'total_actual': 0,
            'precision': 1.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Extract function names and arguments
    expected_funcs = []
    for call in expected:
        name, args = extract_function_name_and_args(call)
        expected_funcs.append((name, args))
    
    actual_funcs = []
    for call in actual:
        name, args = extract_function_name_and_args(call)
        actual_funcs.append((name, args))
    
    # Count matches
    function_name_matches = 0
    argument_matches = 0
    exact_matches = 0
    
    # Check each expected function
    for exp_name, exp_args in expected_funcs:
        found_match = False
        for act_name, act_args in actual_funcs:
            if exp_name == act_name:
                function_name_matches += 1
                found_match = True
                
                # Check argument similarity
                if exp_args == act_args:
                    argument_matches += 1
                    exact_matches += 1
                break
    
    # Calculate metrics
    total_expected = len(expected_funcs)
    total_actual = len(actual_funcs)
    
    precision = function_name_matches / total_actual if total_actual > 0 else 0.0
    recall = function_name_matches / total_expected if total_expected > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    exact_match = (exact_matches == total_expected == total_actual)
    
    return {
        'exact_match': exact_match,
        'function_name_matches': function_name_matches,
        'argument_matches': argument_matches,
        'exact_matches': exact_matches,
        'total_expected': total_expected,
        'total_actual': total_actual,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def extract_expected_from_user_query(user_query: str) -> str:
    """
    Extract expected response from user_query field for fine-tuned model results.
    """
    # Look for the expected response pattern in the user_query
    # Pattern: <start_of_turn>model\n```json\n[...]\n```<end_of_turn>
    pattern = r'<start_of_turn>model\n```json\n(.*?)\n```<end_of_turn>'
    match = re.search(pattern, user_query, re.DOTALL)
    if match:
        return f"```json\n{match.group(1)}\n```"
    return ""


def analyze_semantic_correctness(results_file: str, model_name: str) -> Dict[str, Any]:
    """
    Analyze semantic correctness for a model's results.
    """
    print(f"🔍 Analyzing semantic correctness for {model_name}...")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_examples = len(results)
    semantic_metrics = []
    
    for result in results:
        # Get expected response
        expected_response = result.get('expected_response', '')
        
        # If no expected_response field, try to extract from user_query (for fine-tuned model)
        if not expected_response and 'user_query' in result:
            expected_response = extract_expected_from_user_query(result['user_query'])
        
        # Get actual response - use extracted_function_calls if available (it's already parsed)
        if 'extracted_function_calls' in result:
            actual_calls_str = result['extracted_function_calls']
            actual_calls = normalize_function_calls(actual_calls_str)
        elif 'base_model_response' in result:
            actual_response = result['base_model_response']
            actual_calls = normalize_function_calls(actual_response)
        elif 'response' in result:
            actual_response = result['response']
            actual_calls = normalize_function_calls(actual_response)
        else:
            continue
        
        # Normalize expected function calls
        expected_calls = normalize_function_calls(expected_response)
        
        # Compare
        comparison = compare_function_calls(expected_calls, actual_calls)
        semantic_metrics.append(comparison)
    
    # Calculate aggregate metrics
    exact_matches = sum(1 for m in semantic_metrics if m['exact_match'])
    avg_precision = sum(m['precision'] for m in semantic_metrics) / len(semantic_metrics)
    avg_recall = sum(m['recall'] for m in semantic_metrics) / len(semantic_metrics)
    avg_f1 = sum(m['f1_score'] for m in semantic_metrics) / len(semantic_metrics)
    
    # Function name accuracy
    total_expected_functions = sum(m['total_expected'] for m in semantic_metrics)
    total_actual_functions = sum(m['total_actual'] for m in semantic_metrics)
    total_name_matches = sum(m['function_name_matches'] for m in semantic_metrics)
    total_arg_matches = sum(m['argument_matches'] for m in semantic_metrics)
    
    function_name_accuracy = total_name_matches / total_expected_functions if total_expected_functions > 0 else 0.0
    argument_accuracy = total_arg_matches / total_expected_functions if total_expected_functions > 0 else 0.0
    
    return {
        'model_name': model_name,
        'total_examples': total_examples,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / total_examples,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1,
        'function_name_accuracy': function_name_accuracy,
        'argument_accuracy': argument_accuracy,
        'total_expected_functions': total_expected_functions,
        'total_actual_functions': total_actual_functions,
        'total_name_matches': total_name_matches,
        'total_argument_matches': total_arg_matches
    }


def main():
    """Main function to run semantic analysis."""
    print("🧠 Semantic Correctness Analysis")
    print("=" * 60)
    
    # Analyze base model
    base_metrics = analyze_semantic_correctness(
        "data/results/base_model_test_results.json", 
        "Base Model (4-bit)"
    )
    
    # Analyze fine-tuned model
    finetuned_metrics = analyze_semantic_correctness(
        "data/results/fintuned_model_test_results.json", 
        "Fine-tuned Model"
    )
    
    # Display results
    print("\n📊 SEMANTIC CORRECTNESS RESULTS")
    print("=" * 60)
    
    for metrics in [base_metrics, finetuned_metrics]:
        print(f"\n🔹 {metrics['model_name']}")
        print(f"   Total Examples: {metrics['total_examples']}")
        print(f"   Exact Matches: {metrics['exact_matches']} ({metrics['exact_match_rate']:.1%})")
        print(f"   Function Name Accuracy: {metrics['function_name_accuracy']:.1%}")
        print(f"   Argument Accuracy: {metrics['argument_accuracy']:.1%}")
        print(f"   Average Precision: {metrics['avg_precision']:.3f}")
        print(f"   Average Recall: {metrics['avg_recall']:.3f}")
        print(f"   Average F1 Score: {metrics['avg_f1_score']:.3f}")
        print(f"   Expected Functions: {metrics['total_expected_functions']}")
        print(f"   Generated Functions: {metrics['total_actual_functions']}")
    
    # Calculate improvement
    exact_improvement = finetuned_metrics['exact_match_rate'] - base_metrics['exact_match_rate']
    f1_improvement = finetuned_metrics['avg_f1_score'] - base_metrics['avg_f1_score']
    name_acc_improvement = finetuned_metrics['function_name_accuracy'] - base_metrics['function_name_accuracy']
    arg_acc_improvement = finetuned_metrics['argument_accuracy'] - base_metrics['argument_accuracy']
    
    print(f"\n🔄 IMPROVEMENT FROM FINE-TUNING")
    print("=" * 60)
    print(f"Exact Match Rate: {exact_improvement:+.1%}")
    print(f"F1 Score: {f1_improvement:+.3f}")
    print(f"Function Name Accuracy: {name_acc_improvement:+.1%}")
    print(f"Argument Accuracy: {arg_acc_improvement:+.1%}")
    
    # Save detailed results
    output = {
        'base_model': base_metrics,
        'finetuned_model': finetuned_metrics,
        'improvements': {
            'exact_match_rate': exact_improvement,
            'f1_score': f1_improvement,
            'function_name_accuracy': name_acc_improvement,
            'argument_accuracy': arg_acc_improvement
        }
    }
    
    with open('data/results/semantic_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: data/results/semantic_analysis.json")


if __name__ == "__main__":
    main()
