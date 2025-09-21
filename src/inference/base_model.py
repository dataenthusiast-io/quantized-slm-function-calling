
from mlx_lm import load, generate
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add the src directory to the path so we can import from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.test_data_loader import (
    load_standardized_test_data, 
    save_standardized_results, 
    create_standardized_result
)


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


def extract_tools_and_query(text: str) -> tuple[str, str]:
    """
    Extract tools and user query from the formatted text.
    
    Args:
        text (str): Formatted text from test data
        
    Returns:
        tuple[str, str]: (tools_json, user_query)
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


def format_prompt_for_base_model(tools_json: str, user_query: str) -> str:
    """
    Format the prompt for the base model (without fine-tuning).
    
    Args:
        tools_json (str): JSON string of available tools
        user_query (str): User's question
        
    Returns:
        str: Formatted prompt for base model
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


def test_base_model(model, tokenizer, test_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Test the base model on standardized examples.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        test_examples: List of standardized test examples
        
    Returns:
        List[Dict[str, Any]]: Results with predictions
    """
    results = []
    
    for example in test_examples:
        print(f"Testing Example {example['example_id']}/{len(test_examples)}")
        
        # Format prompt for base model
        prompt = format_prompt_for_base_model(example['tools_json'], example['user_query'])
        
        try:
            # Generate response
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
                model_name="base_model"
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            result = create_standardized_result(
                example=example,
                actual_response=f"ERROR: {str(e)}",
                extracted_function_calls="",
                prompt_used=prompt,
                model_name="base_model"
            )
            results.append(result)
    
    return results


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


def main():
    """Main function to test base model."""
    print("🧪 Testing Base Model on Function Calling Data")
    print("=" * 60)
    
    # Load test data
    test_file = "data/training/test.jsonl"
    num_examples = 100  # Test all examples in the test set
    
    print(f"Loading {num_examples} standardized examples from {test_file}...")
    test_examples = load_standardized_test_data(test_file, num_examples)
    print(f"✓ Loaded {len(test_examples)} standardized examples")
    
    # Load base model (without fine-tuning)
    print(f"\nLoading base model...")
    model, tokenizer = load("mlx-community/gemma-3-1b-it-4bit")
    print(f"✓ Base model loaded")
    
    # Test the model
    print(f"\nTesting base model on {len(test_examples)} examples...")
    results = test_base_model(model, tokenizer, test_examples)
    
    # Save results
    output_file = "data/results/base_model_test_results.json"
    save_standardized_results(results, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Examples tested: {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"\nReview the results to see how the base model performs on function calling tasks.")


if __name__ == "__main__":
    main() 