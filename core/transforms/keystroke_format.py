# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

import json
import os
import re

def convert_button_format(button_str):
    """
    Convert button string from '<Key.shift_r: <60>>' format to 'Key.shift_r' format
    or from '<Key.space: ' '>' format to 'Key.space' format
    """
    # Match pattern like '<Key.something: whatever>'
    match = re.match(r'<Key\.([^:]+):.*>', button_str)
    if match:
        return f"Key.{match.group(1)}"
    return button_str

def convert_json_file(input_dir):
    """
    Convert actions_screenshots_standardtime.json to actions_keystroke_standard.json
    in the specified directory
    """
    input_file = os.path.join(input_dir, 'actions_screenshots_standardtime.json')
    output_file = os.path.join(input_dir, 'actions_keystroke_standard.json')
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Convert each entry
        for entry in data:
            if 'button' in entry:
                entry['button'] = convert_button_format(entry['button'])
        
        # Write output file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

def main():
    while True:
        # Get directory path from user
        input_dir = input("Please enter the input directory path (or 'q' to quit): ").strip()
        
        # Check if user wants to quit
        if input_dir.lower() == 'q':
            print("Exiting program...")
            break
            
        # Validate directory exists
        if not os.path.isdir(input_dir):
            print("Error: Invalid directory path. Please try again.")
            continue
            
        # Process the file
        convert_json_file(input_dir)
        
        # Ask if user wants to process another directory
        another = input("Would you like to process another directory? (y/n): ").strip().lower()
        if another != 'y':
            print("Exiting program...")
            break

if __name__ == "__main__":
    main()