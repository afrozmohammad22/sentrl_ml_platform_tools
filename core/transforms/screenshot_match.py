# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

import json
import os
from datetime import datetime
import glob
from collections import defaultdict

class EventProcessor:
    def __init__(self):
        self.eligible_buttons = {'Key.tab', 'Key.enter'}
        
    def is_pressed(self, event):
        """Check if the event is in pressed state, handling both bool and string cases."""
        pressed = event.get('pressed')
        if isinstance(pressed, bool):
            return pressed
        if isinstance(pressed, str):
            return pressed.lower() == 'true'
        return False

    def parse_timestamp(self, ts_str):
        """Convert timestamp string to datetime object."""
        try:
            ts_str = ts_str.rstrip('Z')
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError as e:
            print(f"Error parsing timestamp {ts_str}: {e}")
            return None

    def is_eligible_event(self, event):
        """
        Check if event should have a screenshot.
        Eligible events are:
        1. Mouse clicks (any button) - both pressed and released states
        2. Keystrokes for Key.enter and Key.tab with pressed=true/"True" only
        """
        if not isinstance(event, dict):
            return False
            
        event_type = event.get('event')
        
        # Handle mouse clicks - accept both pressed and released states
        if event_type == 'mouse_click':
            return True
            
        # Handle keystrokes - only accept pressed state
        if event_type == 'keystroke':
            button = str(event.get('button', '')).replace('<', '').replace('>', '')
            is_eligible_button = any(key in button for key in self.eligible_buttons)
            return is_eligible_button and self.is_pressed(event)
            
        return False

    def find_closest_event(self, screenshot_ts, events, used_event_indices):
        """Find closest eligible event to screenshot timestamp."""
        screenshot_dt = self.parse_timestamp(screenshot_ts)
        if not screenshot_dt:
            return None
            
        min_diff = float('inf')
        closest_event_idx = None
        
        for idx, event in enumerate(events):
            if idx in used_event_indices:
                continue
                
            if not self.is_eligible_event(event):
                continue
                
            event_dt = self.parse_timestamp(event.get('ts'))
            if not event_dt:
                continue
                
            time_diff = abs((screenshot_dt - event_dt).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_event_idx = idx
        
        return closest_event_idx

    def process_events(self, folder_path):
        """Process events and match with screenshots from the same directory."""
        json_path = os.path.join(folder_path, 'actions.json')
        with open(json_path, 'r') as f:
            events = json.load(f)
        
        events = events if isinstance(events, list) else [events]
        
        screenshot_pattern = os.path.join(folder_path, 'screen_*.png')
        screenshot_files = sorted(glob.glob(screenshot_pattern))
        screenshot_files = [os.path.basename(f) for f in screenshot_files]
        
        # Count eligible events
        event_counts = defaultdict(int)
        for event in events:
            if self.is_eligible_event(event):
                event_type = event['event']
                if event_type == 'keystroke':
                    button = str(event.get('button', '')).replace('<', '').replace('>', '')
                    event_counts[f"keystroke_{button} (pressed)"] += 1
                else:
                    pressed_state = "pressed" if self.is_pressed(event) else "released"
                    event_counts[f"{event_type} ({pressed_state})"] += 1
        
        # Match screenshots to events
        used_event_indices = set()
        matched_events = defaultdict(int)
        
        for screenshot in screenshot_files:
            try:
                screenshot_ts = screenshot[7:-4]  # Remove 'screen_' and '.png'
                
                closest_idx = self.find_closest_event(
                    screenshot_ts, events, used_event_indices
                )
                
                if closest_idx is not None:
                    event = events[closest_idx]
                    event['screenshot'] = screenshot
                    used_event_indices.add(closest_idx)
                    
                    event_type = event['event']
                    if event_type == 'keystroke':
                        button = str(event.get('button', '')).replace('<', '').replace('>', '')
                        matched_events[f"keystroke_{button} (pressed)"] += 1
                    else:
                        pressed_state = "pressed" if self.is_pressed(event) else "released"
                        matched_events[f"{event_type} ({pressed_state})"] += 1
            except Exception as e:
                print(f"Error processing screenshot {screenshot}: {e}")
                continue
        
        # Print statistics
        print("\nEvent Processing Statistics:")
        print(f"Total events: {len(events)}")
        print(f"Screenshots available: {len(screenshot_files)}")
        print(f"Matches made: {len(used_event_indices)}")
        
        print("\nEligible events found:")
        print("- Mouse clicks: both pressed and released states")
        print("- Keystrokes (Enter/Tab): pressed state only")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")
            
        print("\nMatched events:")
        for event_type, count in matched_events.items():
            print(f"  {event_type}: {count}")
        
        # Save processed events
        output_path = os.path.join(folder_path, 'actions_with_screenshots.json')
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)
            
        print(f"\nProcessed events saved to: {output_path}")
        
        return events

def main():
    folder_path = input("Enter the folder path containing actions.json and screenshots: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
        
    try:
        processor = EventProcessor()
        processor.process_events(folder_path)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("Full error trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()