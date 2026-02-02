# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

import json
import os
from datetime import datetime
import copy

def convert_datetime_format(timestamp_str):
    # Parse the datetime string (handle both ISO 8601 and legacy formats)
    timestamp_str = timestamp_str.rstrip('Z')

    # Try ISO 8601 format first (2024-01-31T12:02:22.584)
    if 'T' in timestamp_str:
        dt = datetime.fromisoformat(timestamp_str)
    else:
        # Fall back to space-separated format (2024-01-31 12:02:22.584000)
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')

    # Format to the new format with 3 decimal places
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

def convert_pressed_to_boolean(pressed_value):
    # If it's already a boolean, return it as is
    if isinstance(pressed_value, bool):
        return pressed_value
    # If it's a string, convert it
    if isinstance(pressed_value, str):
        return pressed_value.lower() == 'true'
    # For any other case, return False
    return False

def update_screenshot_filename(filename):
    # Extract timestamp from filename
    timestamp_start = filename.find('screen_') + 7
    old_timestamp = filename[timestamp_start:-4]
    new_timestamp = convert_datetime_format(old_timestamp)
    return f"screen_{new_timestamp}.png"

def process_json_file(input_dir):
    input_file = os.path.join(input_dir, 'actions_with_screenshots.json')
    output_file = os.path.join(input_dir, 'actions_screenshots_standardtime.json')
    
    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"actions_with_screenshots.json not found in {input_dir}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Store previous mouse position
    prev_x = None
    prev_y = None
    
    # Create a new list for processed events
    processed_events = []
    
    for event in data:
        new_event = copy.deepcopy(event)
        
        # Convert timestamp format
        new_event['ts'] = convert_datetime_format(event['ts'])
        
        # Convert screenshot filename
        if 'screenshot' in new_event:
            new_event['screenshot'] = update_screenshot_filename(event['screenshot'])
        
        # Convert pressed to boolean if it exists
        if 'pressed' in new_event:
            new_event['pressed'] = convert_pressed_to_boolean(event['pressed'])
        
        # Store position from current event if it has X and Y
        if 'X' in event and 'Y' in event:
            prev_x = event['X']
            prev_y = event['Y']
        
        # Add position to mouse_click events using previous position
        if event['event'] == 'mouse_click' and prev_x is not None and prev_y is not None:
            new_event['position'] = {
                'X': prev_x,
                'Y': prev_y
            }
        
        processed_events.append(new_event)
    
    # Write the processed events to the output file
    with open(output_file, 'w') as f:
        json.dump(processed_events, f, indent=2)
    
    print(f"\nConversion completed! Output file created: {output_file}")

def main():
    while True:
        # Get directory path from user
        print("\nPlease enter the directory path containing actions_with_screenshots.json")
        print("(You can drag and drop the folder here)")
        input_dir = input("> ").strip()
        
        # Remove quotes if the path was dragged and dropped
        input_dir = input_dir.strip("'\"")
        
        if not input_dir:
            print("Directory path cannot be empty. Please try again.")
            continue
            
        if not os.path.exists(input_dir):
            print(f"Error: Directory '{input_dir}' does not exist. Please try again.")
            continue
        
        try:
            process_json_file(input_dir)
            break  # Exit the loop if processing was successful
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            continue  # Ask for the directory again
        except Exception as e:
            print(f"\nError during conversion: {str(e)}")
            print("Would you like to try again with a different directory? (y/n)")
            if input("> ").lower() != 'y':
                break
            continue

if __name__ == "__main__":
    main()