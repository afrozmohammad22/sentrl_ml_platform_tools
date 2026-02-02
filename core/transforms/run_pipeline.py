# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Data Transform Pipeline for Sentrl AI Agent Platform
Runs all data standardization steps in sequence
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Import transform modules
from core.transforms.screenshot_match import EventProcessor
from core.transforms.datetime_format import process_json_file as convert_to_standard_time
from core.transforms.keystroke_format import convert_json_file as convert_keystroke

# Note: convert_screenshot2isoformat functionality is included in screenshot_match

def backup_actions_json(directory):
    """Create a backup of the original actions.json file."""
    original_file = os.path.join(directory, 'actions.json')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(directory, f'actions_backup_{timestamp}.json')
    
    try:
        shutil.copy2(original_file, backup_file)
        print(f"Backup created: {backup_file}")
        return True
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return False

def rename_final_output_to_standardized(directory):
    """Rename actions_keystroke_standard.json to actions_standardized.json."""
    source_file = os.path.join(directory, 'actions_keystroke_standard.json')

    # Determine output location
    directory_path = Path(directory)
    if directory_path.name == 'raw':
        # Output to processed/ directory
        processed_dir = directory_path.parent / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        target_file = processed_dir / 'actions_standardized.json'
    else:
        # Output to same directory
        target_file = Path(directory) / 'actions_standardized.json'

    try:
        # Copy (not move) the final output
        shutil.copy2(source_file, target_file)
        print(f"Final output saved to: {target_file}")
        return True
    except Exception as e:
        print(f"Error saving final output: {str(e)}")
        return False

def validate_directory(directory):
    """Validate that the directory exists and contains required files."""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist")
    
    # Check for actions.json
    if not os.path.exists(os.path.join(directory, 'actions.json')):
        raise ValueError(f"actions.json not found in {directory}")
    
    # Check for screenshots
    screenshot_files = [f for f in os.listdir(directory) if f.startswith('screen_') and f.endswith('.png')]
    if not screenshot_files:
        raise ValueError(f"No screenshot files found in {directory}")

def run_conversion_sequence(directory):
    """Run all conversion scripts in sequence."""
    try:
        print("\n1. Creating backup of original actions.json...")
        if not backup_actions_json(directory):
            raise Exception("Backup creation failed, aborting conversion sequence")
        
        print("\n2. Running screenshot matching...")
        processor = EventProcessor()
        processor.process_events(directory)
        
        print("\n3. Converting datetime format...")
        convert_to_standard_time(directory)
        
        print("\n4. Standardizing keystroke format...")
        convert_keystroke(directory)
        
        print("\n5. Renaming final output...")
        # Rename to actions_standardized.json for processed/ directory
        if not rename_final_output_to_standardized(directory):
            raise Exception("Final renaming failed")
        
        print("\nAll conversions completed successfully!")
        
    except Exception as e:
        print(f"\nError during conversion sequence: {str(e)}")
        import traceback
        print("\nFull error trace:")
        print(traceback.format_exc())
        return False
    
    return True

def main():
    # Get directory path from command line or user input
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Please enter the directory path containing actions.json and screenshots: ").strip()
        directory = directory.strip("'\"")  # Remove quotes if path was dragged and dropped
    
    try:
        # Validate directory and required files
        validate_directory(directory)
        
        # Run the conversion sequence
        success = run_conversion_sequence(directory)
        
        if success:
            print("\nFinal output files:")
            print("- processed/actions_standardized.json (standardized data)")
            print("- raw/actions_backup_[timestamp].json (original backup)")
            print("- Intermediate files in raw/ directory")
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        print("\nFull error trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()