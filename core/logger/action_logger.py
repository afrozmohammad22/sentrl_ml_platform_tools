# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
User Action Logger for Sentrl AI Agent Platform
Captures keyboard, mouse, and window events on macOS
"""

import datetime
import numpy as np
import os
import cv2
import pyautogui
from pynput.keyboard import Listener as KeyboardListener, Key, KeyCode
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Key
import json
from AppKit import NSWorkspace
import threading
import time
import signal
import sys
from pathlib import Path

# Import config system
from core.config import load_config, get_session_dir

# Global variables
running = True
mouse_listener = None
keyboard_listener = None
current_window = None
event_list = []
action_logger = None
data_folder = None
log_filename = None
scroll_timer = None
SCROLL_TIMEOUT = None  # Will be loaded from config

class ActionLogger:
    def __init__(self):
        self.window_scroll_positions = {}
        self.current_window = None
        
    def update_window(self, window_title):
        if window_title != self.current_window:
            self.current_window = window_title
            if window_title not in self.window_scroll_positions:
                self.window_scroll_positions[window_title] = {
                    'vertical_position': 0,
                    'horizontal_position': 0,
                    'max_scroll_down': 0,
                    'max_scroll_right': 0,
                    'last_known_height': None,
                    'last_known_width': None
                }

    def update_scroll_position(self, dx, dy):
        if self.current_window:
            window_data = self.window_scroll_positions[self.current_window]
            
            window_data['vertical_position'] += dy
            window_data['horizontal_position'] += dx
            
            if window_data['vertical_position'] < window_data['max_scroll_down']:
                window_data['max_scroll_down'] = window_data['vertical_position']
            if window_data['horizontal_position'] > window_data['max_scroll_right']:
                window_data['max_scroll_right'] = window_data['horizontal_position']
            
            return {
                'window_title': self.current_window,
                'current_position': {
                    'vertical': window_data['vertical_position'],
                    'horizontal': window_data['horizontal_position']
                },
                'scroll_extent': {
                    'max_down': window_data['max_scroll_down'],
                    'max_right': window_data['max_scroll_right']
                }
            }
        return None

def get_data_folder():
    """Get data folder from config, create session directory"""
    config = load_config()
    session_dir = get_session_dir(config=config)
    # Return the raw subdirectory for action logging
    raw_dir = session_dir / 'raw'
    screenshots_dir = raw_dir / 'screenshots'
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    return str(raw_dir)

def get_active_window_info():
    workspace = NSWorkspace.sharedWorkspace()
    active_app = workspace.activeApplication()
    if active_app:
        return {
            "window_title": active_app['NSApplicationName'],
            "process_name": active_app['NSApplicationName'],
            "process_path": active_app['NSApplicationPath'],
            "pid": active_app['NSApplicationProcessIdentifier']
        }
    return {"error": "No active window found"}

def event_screen(ts):
    """Capture screenshot and save to screenshots subdirectory"""
    img = np.array(pyautogui.screenshot())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filename = f'screen_{ts.isoformat(timespec="milliseconds")}.png'
    # Save to screenshots subdirectory
    screenshots_dir = os.path.join(data_folder, 'screenshots')
    filepath = os.path.join(screenshots_dir, filename)
    cv2.imwrite(filepath, img)
    # Return relative path for JSON
    return f'screenshots/{filename}'

def write_logs(event):
    global event_list, log_filename
    event_list.append(event)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use 'w+' mode and flush after writing
            with open(log_filename, 'w+') as f:
                json.dump(event_list, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            break
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error writing to log file (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(0.1)  # Small delay between retries

def normalize_key(key):
    if isinstance(key, KeyCode):
        if key.char is not None:
            return key.char
        return f"Key.{key.vk}"
    elif isinstance(key, Key):
        return str(key)
    return str(key)

def cleanup():
    global running, mouse_listener, keyboard_listener, event_list, log_filename, scroll_timer
    running = False
    
    # Cancel any pending scroll timer
    if scroll_timer is not None:
        scroll_timer.cancel()
    
    if mouse_listener:
        mouse_listener.stop()
    if keyboard_listener:
        keyboard_listener.stop()
    
    final_event = {
        'ts': datetime.datetime.now().isoformat(timespec='milliseconds'),
        'event': 'session_end',
        'reason': 'user_initiated_shutdown'
    }
    event_list.append(final_event)
    
    try:
        with open(log_filename, 'w') as f:
            json.dump(event_list, f, indent=2)
        print("\nSession ended gracefully. Data saved successfully.")
    except Exception as e:
        print(f"\nError saving final data: {e}")
    
    sys.exit(0)

def signal_handler(signum, frame):
    if signum == signal.SIGTSTP:
        print("\nReceived CTRL+Z signal. Cleaning up...")
        cleanup()
    elif signum in (signal.SIGINT, signal.SIGTERM):
        print("\nReceived termination signal. Cleaning up...")
        cleanup()

def delayed_scroll_screenshot(x, y, scroll_position, initial_ts):
    """Capture screenshot and log final scroll position"""
    global event_list
    
    final_ts = datetime.datetime.now()
    screenshot_file = event_screen(final_ts)
    
    mousescroll_info = {
        'ts': initial_ts.isoformat(timespec='milliseconds'),
        'event': 'mouse_scroll',
        'position': {'X': x, 'Y': y},
        'scroll_context': scroll_position,
        'screenshot': screenshot_file,
        'scroll_end_ts': final_ts.isoformat(timespec='milliseconds')
    }
    write_logs(mousescroll_info)

def on_press(key):
    global event_list
    ts = datetime.datetime.now()
    normalized_key = normalize_key(key)
    
    screenshot_file = None
    if normalized_key in ['Key.enter', 'Key.tab']:
        screenshot_file = event_screen(ts)
    
    keypress_info = {
        'ts': ts.isoformat(timespec='milliseconds'),
        'event': 'keystroke',
        'button': normalized_key,
        'pressed': True
    }
    
    if screenshot_file:
        keypress_info['screenshot'] = screenshot_file
        
    write_logs(keypress_info)

def on_release(key):
    global event_list
    ts = datetime.datetime.now()
    keyrelease_info = {
        'ts': ts.isoformat(timespec='milliseconds'),
        'event': 'keystroke',
        'button': normalize_key(key),
        'pressed': False
    }
    write_logs(keyrelease_info)

def on_move(x, y):
    global event_list
    ts = datetime.datetime.now()
    mousepos_info = {
        'ts': ts.isoformat(timespec='milliseconds'),
        'event': 'mouse_position',
        'X': x,
        'Y': y
    }
    write_logs(mousepos_info)

def on_click(x, y, button, pressed):
    global event_list
    ts = datetime.datetime.now()
    screenshot_file = event_screen(ts)
    mouseclick_info = {
        'ts': ts.isoformat(timespec='milliseconds'),
        'event': 'mouse_click',
        'button': button.name,
        'pressed': pressed,
        'position': {'X': x, 'Y': y},
        'screenshot': screenshot_file
    }
    write_logs(mouseclick_info)

def on_scroll(x, y, dx, dy):
    global action_logger, event_list, scroll_timer
    
    ts = datetime.datetime.now()
    
    window_info = get_active_window_info()
    window_title = window_info["window_title"]
    
    action_logger.update_window(window_title)
    scroll_position = action_logger.update_scroll_position(dx, dy)
    
    # Cancel any existing timer
    if scroll_timer is not None:
        scroll_timer.cancel()
    
    # Set new timer for this scroll event
    scroll_timer = threading.Timer(
        SCROLL_TIMEOUT, 
        delayed_scroll_screenshot, 
        args=[x, y, scroll_position, ts]
    )
    scroll_timer.start()

def monitor_active_window():
    global current_window, running, event_list
    while running:
        try:
            window_info = get_active_window_info()
            window_title = window_info["window_title"]
            
            if current_window != window_title:
                ts = datetime.datetime.now()
                current_window = window_title
                
                screenshot_file = event_screen(ts)
                
                window_change_info = {
                    'ts': ts.isoformat(timespec='milliseconds'),
                    'event': 'window_change',
                    'window_info': window_info,
                    'screenshot': screenshot_file
                }
                write_logs(window_change_info)
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in monitor_active_window: {e}")
            if not running:
                break

if __name__ == "__main__":
    try:
        # Load config
        config = load_config()

        # Initialize global variables
        SCROLL_TIMEOUT = config['logger']['scroll_timeout']

        data_folder = get_data_folder()
        log_filename = os.path.join(data_folder, 'actions.json')
        action_logger = ActionLogger()

        print(f"Logging to: {data_folder}")
        print(f"Config loaded: scroll_timeout={SCROLL_TIMEOUT}s")
        
        # Register signal handlers
        signal.signal(signal.SIGTSTP, signal_handler)  # CTRL+Z
        signal.signal(signal.SIGINT, signal_handler)   # CTRL+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Add session start event
        start_event = {
            'ts': datetime.datetime.now().isoformat(timespec='milliseconds'),
            'event': 'session_start'
        }
        write_logs(start_event)
        
        # Start the window monitoring thread
        window_monitor = threading.Thread(target=monitor_active_window, daemon=True)
        window_monitor.start()
        
        # Set up input listeners
        with MouseListener(
            on_click=on_click,
            on_scroll=on_scroll,
            on_move=on_move
        ) as mouse_listener:
            with KeyboardListener(
                on_press=on_press,
                on_release=on_release
            ) as keyboard_listener:
                print("Logging started. Press CTRL+Z to exit gracefully...")
                keyboard_listener.join()
                
    except Exception as e:
        print(f"Error during execution: {e}")
        cleanup()