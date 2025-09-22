#!/usr/bin/env python3
"""
Inference utility functions for function calling models.

This module contains shared utility functions used by both base and fine-tuned
model inference scripts, including prompt formatting and response parsing.
"""

import json
from typing import Dict, List, Any, Tuple


def extract_tools_and_query(text: str) -> Tuple[str, str]:
    """
    Extract tools and user query from the formatted text.
    
    Args:
        text (str): Formatted text from test data
        
    Returns:
        Tuple[str, str]: (tools_json, user_query)
    """
    # Find the tools section (between "## Instructions" and "## User")
    instructions_start = text.find("## Instructions")
    user_start = text.find("## User")
    
    if instructions_start == -1 or user_start == -1:
        return "", ""
    
    # Extract tools (everything between "## Instructions" and "## User")
    tools_section = text[instructions_start:user_start]
    
    # Find the JSON array in the tools section
    json_start = tools_section.find('[')
    json_end = tools_section.rfind(']') + 1
    
    if json_start != -1 and json_end != -1:
        tools_json = tools_section[json_start:json_end]
    else:
        tools_json = ""
    
    # Extract user query (everything after "## User")
    user_query = text[user_start + len("## User"):].strip()
    
    return tools_json, user_query


def format_prompt_for_inference(tools_json: str, user_query: str) -> str:
    """
    Format the prompt for inference (used by both base and fine-tuned models).
    
    Args:
        tools_json (str): JSON string of available tools
        user_query (str): User's question
        
    Returns:
        str: Formatted prompt for inference
    """
    if tools_json:
        system_prompt = f"""<bos><start_of_turn>user
## Instructions
You are a helpful assistant with access to the following tools:

{tools_json}

## User
{user_query}<end_of_turn>
<start_of_turn>model
"""
    else:
        system_prompt = f"""<bos><start_of_turn>user
## Instructions
You are a helpful assistant.

## User
{user_query}<end_of_turn>
<start_of_turn>model
"""
    
    return system_prompt


def extract_function_calls_from_response(response: str) -> str:
    """
    Extract function calls from model response, handling both plain JSON and markdown JSON.
    
    Args:
        response (str): Model response
        
    Returns:
        str: Extracted function calls in JSON format
    """
    # Try to find JSON in markdown code blocks first
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    # Try to find plain JSON array
    if response.strip().startswith('['):
        # Find the end of the JSON array
        bracket_count = 0
        for i, char in enumerate(response):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return response[:i+1].strip()
    
    # Return original response if no JSON found
    return response.strip()


def load_test_data(test_file: str, num_examples: int = 5) -> List[Dict[str, Any]]:
    """
    Load test data from JSONL file.
    
    Args:
        test_file (str): Path to test.jsonl file
        num_examples (int): Number of examples to load
        
    Returns:
        List[Dict[str, Any]]: List of test examples
    """
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            example = json.loads(line.strip())
            examples.append(example)
    return examples


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: List of results
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
