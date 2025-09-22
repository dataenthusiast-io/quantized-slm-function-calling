
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
from utils.inference import (
    format_prompt_for_inference,
    extract_function_calls_from_response,
)




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
        
        # Format prompt for inference
        prompt = format_prompt_for_inference(example['tools_json'], example['user_query'])
        
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