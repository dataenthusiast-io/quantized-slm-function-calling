#!/usr/bin/env python3
"""
Standardized test data loader for consistent evaluation across models.

This module ensures both base and fine-tuned models are tested on identical
examples with consistent data structure and evaluation criteria.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_standardized_test_data(test_file: str, num_examples: int = 100) -> List[Dict[str, Any]]:
    """
    Load standardized test data ensuring both models use identical examples.
    
    Args:
        test_file (str): Path to test.jsonl file
        num_examples (int): Number of examples to load (default: 100)
        
    Returns:
        List[Dict[str, Any]]: List of standardized test examples
    """
    examples = []
    
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
                
            try:
                data = json.loads(line.strip())
                
                # Extract the standardized fields
                text = data.get('text', '')
                
                # Parse tools and user query from the text
                tools_json, user_query = extract_tools_and_query(text)
                
                # Extract expected response from the text
                expected_response = extract_expected_response(text)
                
                # Create standardized example
                example = {
                    'example_id': i + 1,
                    'tools_json': tools_json,
                    'user_query': user_query,
                    'expected_response': expected_response,
                    'original_text': text
                }
                
                examples.append(example)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {i+1}: {e}")
                continue
    
    return examples


def extract_tools_and_query(text: str) -> tuple[str, str]:
    """
    Extract tools JSON and user query from the formatted text.
    
    Args:
        text (str): The formatted text from the dataset
        
    Returns:
        tuple[str, str]: (tools_json, user_query)
    """
    # Extract tools from the system prompt
    tools_start = text.find('```json\n') + 8
    tools_end = text.find('\n```', tools_start)
    
    if tools_start > 7 and tools_end > tools_start:
        tools_json = text[tools_start:tools_end]
    else:
        tools_json = ""
    
    # Extract user query
    user_start = text.find('## User\n') + 8
    user_end = text.find('<end_of_turn>', user_start)
    
    if user_start > 7 and user_end > user_start:
        user_query = text[user_start:user_end].strip()
    else:
        user_query = ""
    
    return tools_json, user_query


def extract_expected_response(text: str) -> str:
    """
    Extract expected response from the formatted text.
    
    Args:
        text (str): The formatted text from the dataset
        
    Returns:
        str: Expected response (function calls)
    """
    # Look for the expected response pattern
    pattern_start = text.find('<start_of_turn>model\n```json\n') + len('<start_of_turn>model\n```json\n')
    pattern_end = text.find('\n```<end_of_turn>', pattern_start)
    
    if pattern_start > len('<start_of_turn>model\n```json\n') - 1 and pattern_end > pattern_start:
        expected = text[pattern_start:pattern_end]
        return f"```json\n{expected}\n```"
    
    return ""


def save_standardized_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results in standardized format.
    
    Args:
        results (List[Dict[str, Any]]): Results to save
        output_file (str): Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")


def create_standardized_result(
    example: Dict[str, Any],
    actual_response: str,
    extracted_function_calls: str,
    prompt_used: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Create a standardized result entry.
    
    Args:
        example (Dict[str, Any]): Original test example
        actual_response (str): Model's actual response
        extracted_function_calls (str): Extracted function calls
        prompt_used (str): Prompt used for generation
        model_name (str): Name of the model (for identification)
        
    Returns:
        Dict[str, Any]: Standardized result entry
    """
    return {
        'example_id': example['example_id'],
        'model_name': model_name,
        'tools_json': example['tools_json'],
        'user_query': example['user_query'],
        'expected_response': example['expected_response'],
        'actual_response': actual_response,
        'extracted_function_calls': extracted_function_calls,
        'prompt_used': prompt_used
    }
