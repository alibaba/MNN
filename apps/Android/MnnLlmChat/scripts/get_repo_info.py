import json
import argparse
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

def get_repo_size_in_bytes(repo_id: str) -> int:
    """
    Calculates the total size of all files in a Hugging Face model repository.

    Args:
        repo_id (str): The ID of the model repository, e.g., 'google-bert/bert-base-uncased'.

    Returns:
        int: The total size of the files in bytes. Returns -1 if the repository
             cannot be found or an error occurs.
    """
    # Initialize the HfApi client
    api = HfApi()
    total_size = 0
    
    try:
        # Fetch model information, including metadata for each file
        print(f"Fetching metadata for repository: '{repo_id}'...")
        model_info = api.model_info(repo_id=repo_id, files_metadata=True)
        
        # Sum the size of each file in the repository
        for file in model_info.siblings:
            if file.size is not None:
                total_size += file.size
        
        print(f"Successfully calculated size for '{repo_id}': {total_size} bytes")
        return total_size
        
    except HfHubHTTPError as e:
        # Handle cases where the repository is not found or other HTTP errors
        print(f"Error: Could not retrieve info for repository '{repo_id}'. It might not exist or be private.")
        print(f"Details: {e}")
        return -1
    except Exception as e:
        # Handle other potential exceptions
        print(f"An unexpected error occurred while processing '{repo_id}': {e}")
        return -1

def process_model_list(models: list):
    """
    Iterates through a list of model objects and adds the 'file_size' field.

    Args:
        models (list): A list of dictionaries, where each dictionary represents a model.
    """
    if not models:
        return # Do nothing if the list is empty

    for model in models:
        # Check if the model has a HuggingFace source
        if 'sources' in model and 'HuggingFace' in model['sources'] and 'file_size' not in model:
            repo_id = model['sources']['HuggingFace']
            if repo_id:
                # Get the repository size and add it to the model object
                file_size = get_repo_size_in_bytes(repo_id)
                model['file_size'] = file_size
            else:
                # Handle cases where the HuggingFace repo_id is empty
                model['file_size'] = -1
                print(f"Warning: Empty HuggingFace repo_id for model '{model.get('modelName', 'N/A')}'.")
        else:
            # If no HuggingFace source, you can decide what to do.
            # Here we skip adding the field, or you could add 'file_size': 0 or -1
            print(f"Skipping model '{model.get('modelName', 'N/A')}' as it has no HuggingFace source.")


def main():
    """
    Main function to parse arguments, read the input JSON, process it,
    and write to the output JSON file.
    """
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description="Process a market config JSON file to add Hugging Face model sizes."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output JSON file."
    )
    
    args = parser.parse_args()

    # Read the input JSON file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the input file '{args.input}'.")
        return

    # Define the keys that contain lists of models to process
    model_list_keys = ['models', 'tts_models', 'asr_models']

    # Process each list of models
    for key in model_list_keys:
        if key in data and isinstance(data[key], list):
            print(f"\n--- Processing models in '{key}' ---")
            process_model_list(data[key])
        else:
            print(f"\n--- No models found in '{key}', skipping ---")

    # Write the updated data to the output JSON file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nProcessing complete. Output successfully written to '{args.output}'")
    except IOError as e:
        print(f"Error: Could not write to output file '{args.output}'. Details: {e}")


if __name__ == '__main__':
    main()
