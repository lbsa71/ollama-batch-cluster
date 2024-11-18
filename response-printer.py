import os
import sys
import json

def extract_prompts_and_responses(directory):
    # Check if the provided directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Only process JSON files
        if filename.lower().endswith(".json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    # Extract "prompt" and "response"
                    prompt = data.get("prompt", "No prompt found")
                    response = data.get("response", "No response found")

                    # Print with a separator
                    print("#" * 40)
                    print(f"# File: {filename}")
                    print("#" * 40)
                    print(f"Prompt:\n{prompt}\n")
                    print(f"Response:\n{response}\n")

            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON in file '{filename}'.")
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

if __name__ == "__main__":
    # Ensure the directory path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python extract_json.py <directory_path>")
        sys.exit(1)

    # Get the directory path from the command-line argument
    directory_path = sys.argv[1]
    extract_prompts_and_responses(directory_path)

