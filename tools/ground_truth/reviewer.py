# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Human Augmentation Form
1. Load directory with actions.json and screenshots. If there is a actions_human_evaluated.json, load that instead
2. Select a batch of 15 events to annotate randomly. The 15 events should be in sequential order.
3. Preview the 15 events with 2 second interval, then display first event in that batch to annotate
4. Allow user to input additional information in the human augmentation section
5. Copy all events from actions.json and append human_augmentation to actions_human_evaluated.json (create if not exists). 
6. The output should be in the json format. Each human_augmentation is appended to the corresponding event in the actions_human_evaluated.json file.
7. Use the below questions to consctruct the form questions with hints. 

{{  
                "goal": {{"What is the user trying to achieve?"}},
                "interaction_hierarchy": {{
                    "workflow": "Describe the workflow the task is part of",
                    "task": "What is the current task being performed?",
                    "action": "Identify the user action",
                    
                }},
                "state_description": {{
                    "active_window": "Identify the active window on machine",
                    "active_subwindow": "Identify the active subwindow on machine",
                    "content": "Describe the viewed content",
                    "workflow_progress": "How far in the workflow is the user?",
                    "type_of_screen": "What type of UI screen is in the active app?",
                    "cta": "What is the main CTA on the active app?",
                    "attention": "Which part of the screen is the user currently focussed to perform the action and task?"
                }},
                
                "next_steps": {{
                    "next_action": "What should be the next action to finish this task?",
                    "next_task": "What should be the next task to progress the workflow?",
                    "next_workflow": "what should be the next workflow to achieve the user goal?"
                }},

                "user_prompts": {{
                    "Prompt 1": "What might the user prompt us to achieve their goal?",
                    "Prompt 2": "What other user prompt can you think of?",
                    "Prompt 3": "What other user prompt can you think of?"
                }}
            }}

"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
import shutil
from typing import Dict, List, Optional
import requests
import time
import threading
import random

class EventLabelingApp:
    def __init__(self, root):
        """
        Initialize the main application window and setup UI components
        
        Args:
            root: The main tkinter window
        """
        self.root = root
        self.root.title("GroundTruth")
        
        # Get screen dimensions
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        
        # Calculate maximum dimensions (80% of screen size)
        self.max_width = int(self.screen_width * 0.8)
        self.max_height = int(self.screen_height * 0.8)
        
        # Initialize data structures
        self.events_data = []
        self.screenshot_events = []
        self.current_event_index = 0
        self.workspace_directory = None
        self.current_image = None  # Store the current PIL Image
        self.current_scale = 1.0   # Track current scale factor
        
        # Add dictionary to store edits for each event
        self.event_edits = {}  # Will store edits using event index as key
        
        # Add form field structure
        self.form_fields = {
            "interaction_hierarchy": {
                "action": "Identify the user action",
                "task": "What is the current task being performed?",
                "workflow": "Describe the workflow the task is part of",
            },
            "state_description": {
                "active_window": "Identify the active window on machine",
                "active_app": "Name of the active application (web or local app)",
                "content": "Describe the viewed content",
                "type_of_screen": "What type of UI screen is in the active app?",
                "cta": "What is the main CTA on the active app?"
            },
            "goal": "What is the user trying to achieve?",
            "attention": "Which part of the screen is the user currently focussed to perform the action and task?",
            "next_steps": {
                "next_action": "What should be the next action to finish this task?",
                "next_task": "What should be the next task to progress the workflow?"
            },
            "user_prompts": {
                "Prompt 1": "What might the user prompt us to achieve their goal?",
                "Prompt 2": "What other user prompt can you think of?",
                "Prompt 3": "What other user prompt can you think of?"
            }
        }
        
        # Initialize form entries dictionary
        self.form_entries = {}
        
        # Add flag for preview in progress
        self.preview_in_progress = False
        
        # Add status label for notifications
        self.status_label = None
        
        # Add new attributes for random event handling
        self.unannotated_indices = []  # Store indices of unannotated events
        self.current_random_index = 0   # Track position in random sequence
        
        # Setup main UI components
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface components"""
        # Create main frames with grid
        self.root.grid_rowconfigure(1, weight=1)  # Make content frame expandable
        self.root.grid_columnconfigure(0, weight=1)  # Make columns expandable
        
        # Control frame at top - center aligned
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.control_frame.grid_columnconfigure(0, weight=1)  # Enable center alignment
        
        # Directory label at top
        self.dir_label = ttk.Label(
            self.control_frame,
            text="No directory selected",
            font=('TkDefaultFont', 10),
            wraplength=800
        )
        self.dir_label.grid(row=0, column=0, pady=(0,5))
        
        # Container for buttons below directory label
        self.button_container = ttk.Frame(self.control_frame)
        self.button_container.grid(row=1, column=0)
        
        # Add Upload Directory button
        self.upload_btn = ttk.Button(
            self.button_container,
            text="Upload Directory",
            command=self.upload_directory
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Preview Task button to button_container
        self.preview_btn = ttk.Button(
            self.button_container,
            text="Preview Task",
            command=self.start_preview_task
        )
        self.preview_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Random Batch button
        self.random_batch_btn = ttk.Button(
            self.button_container,
            text="Random Batch",
            command=self.jump_to_random_batch
        )
        self.random_batch_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Show Next Annotated button
        self.next_annotated_btn = ttk.Button(
            self.button_container,
            text="Show Next Annotated",
            command=self.show_next_annotated_event
        )
        self.next_annotated_btn.pack(side=tk.LEFT, padx=5)
        
        # Create canvas with scrollbar for entire content
        self.main_canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        
        # Configure main canvas
        self.main_canvas.grid(row=1, column=0, sticky='nsew')
        self.scrollbar.grid(row=1, column=1, sticky='ns')
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create main container frame inside canvas
        self.main_container = ttk.Frame(self.main_canvas)
        self.canvas_frame = self.main_canvas.create_window((0, 0), window=self.main_container, anchor='nw')
        
        # Create content frame
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Configure grid weights
        self.content_frame.grid_columnconfigure(0, weight=3)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)
        
        # Image frame setup
        self.setup_image_frame()
        
        # Event info frame setup
        self.setup_event_info_frame()
        
        # Create bottom container for form
        self.bottom_container = ttk.Frame(self.content_frame)
        self.bottom_container.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(2, 0))
        self.bottom_container.grid_columnconfigure(0, weight=1)
        
        # Create form container
        self.form_container = ttk.Frame(self.bottom_container)
        self.form_container.grid(row=0, column=0, sticky='nsew')
        self.form_container.grid_columnconfigure(1, weight=1)
        
        # Initialize tab order widgets list
        self.tab_order_widgets = []
        
        # Create form fields in the form container
        self.create_form_fields(self.form_container)
        
        # Now create navigation frame
        self.navigation_frame = ttk.Frame(self.bottom_container)
        self.navigation_frame.grid(row=1, column=0, sticky='ew', pady=(10,0))
        self.navigation_frame.grid_columnconfigure(1, weight=1)
        
        # Add navigation buttons to left
        self.prev_btn = ttk.Button(
            self.navigation_frame,
            text="Previous",
            command=self.show_previous_event,
            state=tk.DISABLED
        )
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.next_btn = ttk.Button(
            self.navigation_frame,
            text="Next",
            command=self.show_next_event,
            state=tk.DISABLED
        )
        self.next_btn.grid(row=0, column=1, padx=5)
        
        # Save button at bottom right
        self.save_btn = ttk.Button(
            self.navigation_frame,
            text="Save Changes",
            command=self.save_event_changes,
            state=tk.DISABLED
        )
        self.save_btn.grid(row=0, column=2, padx=5)
        
        # Now add navigation buttons to tab order
        self.tab_order_widgets.extend([self.prev_btn, self.next_btn, self.save_btn])
        
        # Set up tab navigation
        self.setup_tab_navigation()
        
        # Bind events for scrolling
        self.main_container.bind('<Configure>', self._configure_scroll_region)
        self.main_canvas.bind('<Configure>', self._configure_canvas_window)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Button-4>", self._on_mousewheel)
        self.root.bind("<Button-5>", self._on_mousewheel)
        
        # Create status label that overlays the entire window
        self.status_label = ttk.Label(
            self.root,  # Changed to root window
            text="",
            font=('TkDefaultFont', 14, 'bold'),
            foreground='#2E8B57',
            background='white'
        )
        # Initially hide it
        self.status_label.place_forget()
        
        # Configure the control_frame to center the status label
        self.control_frame.grid_columnconfigure(0, weight=1)  # This ensures center alignment
        
        # Add counter for annotated events
        self.annotated_counter = ttk.Label(
            self.control_frame,
            text="Annotated: 0/0",
            font=('TkDefaultFont', 10)
        )
        self.annotated_counter.grid(row=0, column=1, padx=5, sticky='e')
        
        # Bind keyboard shortcuts
        self.root.bind('<Shift-Left>', lambda e: self.show_previous_event())
        self.root.bind('<Shift-Right>', lambda e: self.show_next_event())
        
    def _configure_scroll_region(self, event=None):
        """Configure the scroll region to encompass the inner frame"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _configure_canvas_window(self, event=None):
        """Configure the canvas window size"""
        width = event.width if event else self.main_canvas.winfo_width()
        self.main_canvas.itemconfig(self.canvas_frame, width=width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        # Handle different event types for cross-platform compatibility
        if event.num == 4 or event.delta > 0:
            # Scroll up
            self.main_canvas.yview_scroll(-2, "units")
        elif event.num == 5 or event.delta < 0:
            # Scroll down
            self.main_canvas.yview_scroll(2, "units")

    def upload_directory(self):
        """
        Handle directory upload and process the actions.json file
        """
        self.workspace_directory = filedialog.askdirectory(
            title="Select Directory with actions.json and screenshots"
        )
        
        if not self.workspace_directory:
            return
            
        # Update directory label
        self.dir_label.config(text=f"Current Directory: {self.workspace_directory}")
            
        # First check for actions_human_evaluated.json
        human_evaluated_path = os.path.join(self.workspace_directory, 'actions_human_evaluated.json')
        actions_path = os.path.join(self.workspace_directory, 'actions.json')
        
        try:
            # Load the appropriate file
            if os.path.exists(human_evaluated_path):
                with open(human_evaluated_path, 'r') as f:
                    self.events_data = json.load(f)
            else:
                with open(actions_path, 'r') as f:
                    self.events_data = json.load(f)
            
            # Filter events with screenshots
            self.screenshot_events = [
                event for event in self.events_data 
                if 'screenshot' in event
            ]
            
            if self.screenshot_events:
                # Get indices of unannotated events
                self.unannotated_indices = [
                    i for i, event in enumerate(self.screenshot_events)
                    if not event.get('Human Augmentation')
                ]
                
                # Randomly shuffle the unannotated indices
                random.shuffle(self.unannotated_indices)
                
                if not self.unannotated_indices:
                    messagebox.showinfo(
                        "No Unannotated Events",
                        "All events have been annotated!"
                    )
                    return
                
                # Set current index to first random unannotated event
                self.current_random_index = 0
                self.current_event_index = self.unannotated_indices[0]
                
                # Update slider range to total number of unannotated events
                self.event_slider.configure(
                    from_=0,
                    to=len(self.unannotated_indices) - 1
                )
                self.event_slider.set(self.current_event_index)
                
                self.update_event_counter()
                self.enable_navigation()
                self.display_current_event()
                
            else:
                messagebox.showwarning(
                    "No Screenshots",
                    "No events with screenshots found in the file."
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            
        # Update annotated counter after loading directory
        self.update_annotated_counter()
        
    def save_event_changes(self):
        """Save the form data for the current event"""
        if not self.screenshot_events:
            return
            
        # Save current edits
        self.save_current_edits()
        
        # Get current event
        current_event = self.screenshot_events[self.current_event_index]
        output_path = os.path.join(
            self.workspace_directory,
            'actions_human_evaluated.json'
        )
        
        try:
            # If file exists, load it; otherwise use original events_data
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                # Use original events_data to preserve order
                existing_data = self.events_data
            
            # Find and update the matching event in existing_data
            for event in existing_data:
                if (event.get('ts') == current_event['ts'] and 
                    event.get('event') == current_event['event']):
                    # Update or add Human Augmentation data
                    event['Human Augmentation'] = self.event_edits[self.current_event_index]['human_augmentation']
                    break
            
            # Write back the updated data
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            messagebox.showinfo("Success", "Changes saved successfully!")
            
            # Move to next event if available
            if self.current_event_index < len(self.screenshot_events) - 1:
                self.show_next_event()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving changes: {str(e)}")
        
        # Update annotated counter after saving
        self.update_annotated_counter()

    def show_previous_event(self):
        """Navigate to previous event"""
        if self.current_event_index > 0:
            # Save current edits before moving
            self.save_current_edits()
            
            self.current_event_index -= 1
            
            # Update the random index to match the new event
            if self.current_event_index in self.unannotated_indices:
                self.current_random_index = self.unannotated_indices.index(self.current_event_index)
            
            self.display_current_event()
            self.enable_navigation()
            
            # Load edits for the new event
            self.load_current_edits()

    def show_next_event(self):
        """Navigate to next event"""
        if self.current_event_index < len(self.screenshot_events) - 1:
            # Save current edits before moving
            self.save_current_edits()
            
            self.current_event_index += 1
            
            # Update the random index to match the new event
            if self.current_event_index in self.unannotated_indices:
                self.current_random_index = self.unannotated_indices.index(self.current_event_index)
            
            self.display_current_event()
            self.enable_navigation()
            
            # Load edits for the new event
            self.load_current_edits()

    def enable_navigation(self):
        """Enable/disable navigation buttons based on current position"""
        self.prev_btn.configure(
            state=tk.NORMAL if self.current_event_index > 0 else tk.DISABLED
        )
        self.next_btn.configure(
            state=tk.NORMAL 
            if self.current_event_index < len(self.screenshot_events) - 1 
            else tk.DISABLED
        )

    def display_current_event(self):
        """Display the current event's screenshot and information"""
        if not self.screenshot_events:
            return
            
        current_event = self.screenshot_events[self.current_event_index]
        
        # Skip first click of double click
        if self.current_event_index > 0:
            previous_event = self.screenshot_events[self.current_event_index - 1]
            if (is_double_click(current_event, previous_event) and 
                current_event.get('pressed', False)):
                self.current_event_index += 1
                current_event = self.screenshot_events[self.current_event_index]
        
        # Update event counter with actual event number
        self.update_event_counter()
        
        # Update slider event label with actual event number
        self.slider_event_label.config(text=f"Event: {self.current_event_index + 1}")
        
        # Update slider value to match current event index
        self.event_slider.set(self.current_event_index)
        
        # Display screenshot
        screenshot_path = os.path.join(
            self.workspace_directory,
            current_event['screenshot']
        )
        
        if os.path.exists(screenshot_path):
            self.current_image = Image.open(screenshot_path)
            self.display_image()
        
        # Update event info and other displays
        self.update_event_info(current_event)
        
        # Clear form fields for new event
        self.clear_form_fields()
        
        # Update annotated counter
        self.update_annotated_counter()

    def display_image(self):
        """Display the image fitted to the current screen size"""
        if not self.current_image:
            return
        
        # Get current screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate maximum dimensions (70% of screen size to ensure visibility)
        max_width = int(screen_width * 0.7)
        max_height = int(screen_height * 0.7)
        
        # Get original image dimensions
        original_width = self.current_image.width
        original_height = self.current_image.height
        
        # Calculate scaling factor to fit within max dimensions
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        self.scale_factor = min(width_ratio, height_ratio)  # Store scale factor for coordinate scaling
        
        # Calculate new dimensions
        new_width = int(original_width * self.scale_factor)
        new_height = int(original_height * self.scale_factor)
        
        # Resize image
        resized_image = self.current_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas size to match resized image
        self.image_canvas.config(
            width=new_width,
            height=new_height
        )
        
        # Clear previous content and display new image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, image=photo, anchor='nw')
        self.image_canvas.image = photo  # Keep a reference

    def update_event_info(self, current_event):
        """Update event information display"""
        # Configure state to normal temporarily to update content
        self.event_info.configure(state='normal')
        
        # Clear existing content
        self.event_info.delete(1.0, tk.END)
        
        # Get event details
        event_type = current_event.get('event', 'N/A')
        button = current_event.get('button', 'N/A')
        
        # Check for double click
        is_double = False
        if self.current_event_index > 0:
            previous_event = self.screenshot_events[self.current_event_index - 1]
            is_double = is_double_click(current_event, previous_event)
        
        # Handle pressed state based on event type
        pressed = 'N/A'
        if event_type == 'scroll':
            delta = current_event.get('delta', 0)
            pressed = 'Scroll Up' if delta > 0 else 'Scroll Down'
        elif 'pressed' in current_event:
            pressed = current_event['pressed']
            if is_double and not pressed:
                event_type = 'double_click'  # Show as double click instead of click
        
        # Handle window change events
        if event_type == 'window_change':
            prev_window = current_event.get('previous_window', 'N/A')
            next_window = current_event.get('next_window', 'N/A')
            pressed = f"From: {prev_window}\nTo: {next_window}"
        
        # Create timeline of 5 events
        start_idx = max(0, self.current_event_index - 2)
        end_idx = min(len(self.screenshot_events), start_idx + 5)
        start_idx = max(0, end_idx - 5)  # Adjust start if we're near the end
        
        timeline = []
        i = start_idx
        while i < end_idx:
            event = self.screenshot_events[i]
            
            # Check for double click
            is_double = False
            if i > 0:
                prev_evt = self.screenshot_events[i-1]
                is_double = is_double_click(event, prev_evt)
            
            # Check for click and drag
            is_drag = False
            if i < len(self.screenshot_events) - 1:
                next_evt = self.screenshot_events[i+1]
                is_drag = is_click_and_drag(event, next_evt)
            
            # Skip first click of double click in timeline
            if is_double and event.get('pressed', False):
                i += 1
                continue
            
            # Handle click and drag sequence
            if is_drag and event.get('pressed', True):
                # Find the end of drag sequence
                next_idx = i + 1
                while (next_idx < len(self.screenshot_events) and 
                       self.screenshot_events[next_idx]['event'] in ['mousemove', 'click']):
                    if (self.screenshot_events[next_idx]['event'] == 'click' and 
                        not self.screenshot_events[next_idx].get('pressed', True)):
                        break
                    next_idx += 1
                
                timeline.append("click and drag")
                i = next_idx + 1
                continue
            
            # Handle event aggregation
            if event['event'] in ['mousemove', 'scroll', 'keystroke']:
                # Count similar consecutive events
                event_type = event['event']
                count = 1
                next_idx = i + 1
                
                while (next_idx < end_idx and 
                       self.screenshot_events[next_idx]['event'] == event_type):
                    count += 1
                    next_idx += 1
                
                if count > 1:
                    # Add aggregated event description
                    if event_type == 'mousemove':
                        event_str = "many mouse movements"
                    elif event_type == 'scroll':
                        event_str = "many scroll events"
                    elif event_type == 'keystroke':
                        event_str = "many keystrokes"
                    
                    timeline.append(event_str)
                    i = next_idx
                    continue
            
            # Regular event display
            event_str = event.get('event', 'N/A')
            if is_double and not event.get('pressed', False):
                event_str = "double click"
            elif event_str == 'click':
                event_str = "click"
            
            if i == self.current_event_index:
                event_str = f"**{event_str}**"  # Bold current event
            
            timeline.append(event_str)
            i += 1
        
        # Configure text tags with larger fonts
        self.event_info.tag_configure('header', font=('TkDefaultFont', 22, 'bold'), justify='right')
        self.event_info.tag_configure('content', font=('TkDefaultFont', 20), justify='right')
        
        self.event_info.insert(tk.END, "Event Details:\n", 'header')
        self.event_info.insert(tk.END, f"Type: {event_type}\n", 'content')
        self.event_info.insert(tk.END, f"Button: {button}\n", 'content')
        self.event_info.insert(tk.END, f"Pressed: {pressed}\n\n", 'content')
        
        self.event_info.insert(tk.END, "Timeline:\n", 'header')
        for event_str in timeline:
            self.event_info.insert(tk.END, f"{event_str}\n", 'content')
        
        # Set back to disabled
        self.event_info.configure(state='disabled')
        
        # Enable save button
        self.save_btn.configure(state=tk.NORMAL)

    def save_current_edits(self):
        """Save current form entries"""
        form_data = {}
        for field_id, entry in self.form_entries.items():
            form_data[field_id] = entry.get('1.0', tk.END).strip()
        
        # Always update the event_edits with the latest form data
        self.event_edits[self.current_event_index] = {
            'human_augmentation': form_data
        }

    def load_current_edits(self):
        """Load saved form entries for current event"""
        # Clear all entries first
        for entry in self.form_entries.values():
            entry.delete('1.0', tk.END)
        
        # Check if there are saved edits in event_edits
        if self.current_event_index in self.event_edits:
            saved_data = self.event_edits[self.current_event_index]['human_augmentation']
        else:
            # If no saved edits, load existing Human Augmentation data from the event
            current_event = self.screenshot_events[self.current_event_index]
            saved_data = current_event.get('Human Augmentation', {})
        
        # Populate form fields with saved data
        for field_id, value in saved_data.items():
            if field_id in self.form_entries:
                self.form_entries[field_id].insert('1.0', value)

    def on_slider_change(self, value):
        """Handle slider value changes"""
        try:
            # Convert slider value to integer
            new_index = int(float(value))
            
            if new_index != self.current_event_index:
                # Save current edits before moving
                self.save_current_edits()
                
                # Update current index
                self.current_event_index = new_index
                
                # Update random index if needed
                if new_index in self.unannotated_indices:
                    self.current_random_index = self.unannotated_indices.index(new_index)
                
                # Update display
                self.display_current_event()
                self.enable_navigation()
                
                # Load edits for the new event
                self.load_current_edits()
        except Exception as e:
            print(f"Slider error: {str(e)}")

    def create_form_fields(self, parent):
        """Create form fields based on the form structure"""
        # Configure grid to expand horizontally
        parent.grid_columnconfigure(1, weight=1)
        
        row = 0
        for section, content in self.form_fields.items():
            # Add section header with minimal padding
            ttk.Label(
                parent,
                text=section.replace('_', ' ').title(),
                font=('TkDefaultFont', 11, 'bold')
            ).grid(row=row, column=0, columnspan=2, pady=(2,1), sticky='w')  # Reduced padding
            row += 1
            
            if isinstance(content, dict):
                for field, hint in content.items():
                    entry = self.create_field(parent, f"{section}.{field}", hint, row)
                    # Only add to tab order if entry is not None (editable field)
                    if entry is not None:
                        self.tab_order_widgets.append(entry)
                    row += 1
            else:
                entry = self.create_field(parent, section, content, row)
                # Only add to tab order if entry is not None (editable field)
                if entry is not None:
                    self.tab_order_widgets.append(entry)
                row += 1

    def create_field(self, parent, field_id, hint, row):
        """Create individual form field with label and text entry"""
        # Create label with fixed width and reduced padding
        label = ttk.Label(
            parent,
            text=field_id.split('.')[-1].replace('_', ' ').title(),
            wraplength=150,
            width=20
        )
        label.grid(row=row, column=0, padx=2, pady=1, sticky='nw')  # Reduced padding
        
        # Create text entry that fills horizontal space
        entry = tk.Text(
            parent,
            height=2,  # Slightly reduced height if needed
            wrap=tk.WORD,
            # Make specific fields non-editable
            state='disabled' if field_id in ['state_description.workflow_progress', 'next_steps.next_workflow'] else 'normal'
        )
        entry.grid(row=row, column=1, padx=2, pady=1, sticky='ew')  # Reduced padding
        
        # Add tooltip with hint
        ToolTip(entry, hint)
        
        # Store entry widget reference
        self.form_entries[field_id] = entry
        
        # Only add editable fields to tab order
        if field_id not in ['state_description.workflow_progress', 'next_steps.next_workflow']:
            return entry
        return None

    def setup_tab_navigation(self):
        """Setup tab navigation order for all widgets"""
        for idx, widget in enumerate(self.tab_order_widgets):
            widget.bind('<Tab>', lambda e, idx=idx: self.focus_next_widget(e, idx))
            widget.bind('<Shift-Tab>', lambda e, idx=idx: self.focus_previous_widget(e, idx))

    def focus_next_widget(self, event, current_idx=None):
        """Move focus to next widget in tab order with auto-scroll"""
        if current_idx is None:
            try:
                current_idx = self.tab_order_widgets.index(event.widget)
            except ValueError:
                current_idx = -1
        
        next_idx = (current_idx + 1) % len(self.tab_order_widgets)
        next_widget = self.tab_order_widgets[next_idx]
        next_widget.focus_set()
        
        # If it's a Text widget, move cursor to start and ensure visibility
        if isinstance(next_widget, tk.Text):
            next_widget.mark_set('insert', '1.0')
            self.ensure_widget_visible(next_widget)
        
        return 'break'

    def focus_previous_widget(self, event, current_idx=None):
        """Move focus to previous widget in tab order with auto-scroll"""
        if current_idx is None:
            try:
                current_idx = self.tab_order_widgets.index(event.widget)
            except ValueError:
                current_idx = 0
        
        prev_idx = (current_idx - 1) % len(self.tab_order_widgets)
        prev_widget = self.tab_order_widgets[prev_idx]
        prev_widget.focus_set()
        
        # If it's a Text widget, move cursor to start and ensure visibility
        if isinstance(prev_widget, tk.Text):
            prev_widget.mark_set('insert', '1.0')
            self.ensure_widget_visible(prev_widget)
        
        return 'break'

    def ensure_widget_visible(self, widget):
        """Ensure the widget is visible in the scrollable area with context"""
        try:
            # Get widget's position relative to the form container
            widget_y = widget.winfo_rooty() - self.main_container.winfo_rooty()
            
            # Get current scroll position
            current_scroll = self.main_canvas.yview()[0]
            
            # Calculate the visible area
            visible_top = current_scroll * self.main_canvas.winfo_height()
            visible_bottom = visible_top + self.main_canvas.winfo_height()
            
            # Increased scroll amount (50% of visible area)
            scroll_amount = self.main_canvas.winfo_height() * 0.5
            
            # Calculate target scroll position with increased movement
            if widget_y < visible_top + scroll_amount:
                # Scroll up with increased amount
                target_scroll = max(0, (widget_y - scroll_amount) / self.main_canvas.winfo_height())
                self.main_canvas.yview_moveto(target_scroll)
            else:
                # Always scroll down by 50% of the window height
                current_y = self.main_canvas.yview()[0]
                new_y = min(1.0, current_y + 0.5)  # Scroll down by 50% of window height
                self.main_canvas.yview_moveto(new_y)
                
                # If we're near the bottom, scroll to show the last few fields
                if widget_y > self.main_canvas.winfo_height() * 0.7:
                    self.main_canvas.yview_moveto(1.0)
                
        except Exception as e:
            # Fallback to basic scrolling with increased amount
            self.main_canvas.yview_scroll(2, "units")

    def setup_image_frame(self):
        """Setup the frame that will contain the screenshot image"""
        # Create frame for image
        self.image_frame = ttk.Frame(self.content_frame)
        self.image_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        # Create canvas for image
        self.image_canvas = tk.Canvas(
            self.image_frame,
            bg='white',
            width=800,
            height=600
        )
        self.image_canvas.pack(expand=True, fill='both')
        
        # Create status label overlay for image canvas
        self.status_label = ttk.Label(
            self.image_canvas,
            text="",
            font=('TkDefaultFont', 14, 'bold'),
            foreground='#2E8B57',
            background='white'  # Match canvas background
        )
        # Initially place it in center but make it invisible
        self.status_label.place(relx=0.5, rely=0.5, anchor='center')
        self.status_label.place_forget()  # Hide it initially
        
        # Create navigation container
        nav_container = ttk.Frame(self.image_frame)
        nav_container.pack(fill='x', padx=5, pady=5)
        
        # Add event counter and jump field
        self.event_counter = ttk.Label(
            nav_container,
            text="Event: 0/0"
        )
        self.event_counter.pack(side='left', padx=5)
        
        # Add slider event number label
        self.slider_event_label = ttk.Label(
            nav_container,
            text="Event: 0"
        )
        self.slider_event_label.pack(side='right', padx=5)
        
        # Add jump to event entry
        ttk.Label(nav_container, text="Jump to:").pack(side='left', padx=5)
        self.jump_entry = ttk.Entry(nav_container, width=10)
        self.jump_entry.pack(side='left', padx=5)
        
        # Add jump button
        ttk.Button(
            nav_container,
            text="Go",
            command=self.jump_to_event
        ).pack(side='left', padx=5)
        
        # Create slider for event navigation
        self.event_slider = ttk.Scale(
            nav_container,
            from_=0,
            to=len(self.screenshot_events) - 1 if hasattr(self, 'screenshot_events') else 0,
            orient='horizontal',
            command=self.on_slider_change
        )
        self.event_slider.pack(side='left', fill='x', expand=True, padx=5)

    def setup_event_info_frame(self):
        """Setup the frame that will contain event information"""
        # Create frame for event info
        self.info_frame = ttk.Frame(self.content_frame)
        self.info_frame.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)
        
        # Create text widget for event info with smaller width and right justification
        self.event_info = tk.Text(
            self.info_frame,
            wrap=tk.WORD,
            width=20,  # Reduced from 30 to 20 for narrower pane
            height=15,
            state='disabled',
            font=('TkDefaultFont', 20)
        )
        self.event_info.pack(expand=True, fill='both')
        
        # Configure tags with right justification
        self.event_info.tag_configure('header', font=('TkDefaultFont', 22, 'bold'), justify='right')
        self.event_info.tag_configure('content', font=('TkDefaultFont', 20), justify='right')

    def clear_form_fields(self):
        """Clear all form fields"""
        for entry in self.form_entries.values():
            entry.delete('1.0', tk.END)

    def update_event_counter(self):
        """Update the event counter display"""
        total_events = len(self.screenshot_events)
        current_position = self.current_event_index + 1
        self.event_counter.config(
            text=f"Event: {current_position}/{total_events}"
        )

    def jump_to_event(self):
        """Jump to a specific event number"""
        try:
            event_num = int(self.jump_entry.get())
            if 1 <= event_num <= len(self.screenshot_events):
                # Convert to 0-based index
                target_event_index = event_num - 1
                
                # Save current edits before jumping
                self.save_current_edits()
                
                # Update indices
                self.current_event_index = target_event_index
                if target_event_index in self.unannotated_indices:
                    self.current_random_index = self.unannotated_indices.index(target_event_index)
                
                # Update display
                self.display_current_event()
                self.enable_navigation()
                
                # Load edits for the new event
                self.load_current_edits()
                
                # Clear the jump entry
                self.jump_entry.delete(0, tk.END)
                
            else:
                messagebox.showwarning(
                    "Invalid Input",
                    f"Please enter a number between 1 and {len(self.screenshot_events)}"
                )
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Please enter a valid number"
            )
        except Exception as e:
            print(f"Jump error: {str(e)}")
            messagebox.showerror(
                "Error",
                "An error occurred while jumping to the event"
            )

    def load_human_augmentation_data(self):
        """Load existing human augmentation data into the form"""
        current_event = self.screenshot_events[self.current_event_index]
        if 'Human Augmentation' in current_event:
            self.event_edits[self.current_event_index] = {
                'human_augmentation': current_event['Human Augmentation']
            }
            self.load_current_edits()

    def start_preview_task(self):
        """Start the preview task in a separate thread"""
        if self.preview_in_progress:
            return
            
        if len(self.screenshot_events) < 15:
            messagebox.showwarning(
                "Preview Not Available",
                "Not enough events to preview"
            )
            return
            
        # Store current position before preview
        self.preview_start_index = self.current_event_index
        
        # Get the current batch's start and end indices
        batch_start = (self.current_event_index // 15) * 15
        batch_end = min(batch_start + 15, len(self.screenshot_events))
        
        # Use the current batch as the preview sequence
        self.random_sequence = list(range(batch_start, batch_end))
        
        # Disable navigation during preview
        self.disable_navigation_during_preview()
        
        # Start preview in separate thread
        preview_thread = threading.Thread(target=self.show_preview_events)
        preview_thread.daemon = True
        preview_thread.start()
    
    def show_preview_events(self):
        """Show the selected 15 events with 2.5 second interval"""
        try:
            self.preview_in_progress = True
            
            # Show the 15 events in the current batch
            for index in self.random_sequence:
                if not self.preview_in_progress:  # Check for cancellation
                    break
                self.current_event_index = index
                
                # Load human augmentation data if it exists
                self.root.after(0, self.load_and_display_event)
                time.sleep(2.5)  # Changed to 2.5 second interval
            
            # Only proceed if preview wasn't cancelled
            if self.preview_in_progress:
                # Return to the starting position
                self.current_event_index = self.preview_start_index
                self.root.after(0, self.load_and_display_event)
                
                # Show completion message
                self.root.after(0, lambda: self.show_status_message("✓ Preview completed"))
                self.root.after(3000, lambda: self.show_status_message(""))
                
        finally:
            self.preview_in_progress = False
            self.root.after(0, self.enable_navigation_after_preview)

    def disable_navigation_during_preview(self):
        """Disable navigation controls during preview"""
        self.prev_btn.configure(state=tk.DISABLED)
        self.next_btn.configure(state=tk.DISABLED)
        self.event_slider.configure(state=tk.DISABLED)
        self.preview_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        
    def enable_navigation_after_preview(self):
        """Re-enable navigation controls after preview"""
        try:
            self.event_slider.configure(state=tk.NORMAL)
            self.preview_btn.configure(state=tk.NORMAL)
            self.save_btn.configure(state=tk.NORMAL)
            
            # Update current indices to match the starting event
            if hasattr(self, 'preview_start_index'):
                self.current_event_index = self.preview_start_index
                try:
                    # Find the corresponding random index
                    if self.current_event_index in self.unannotated_indices:
                        self.current_random_index = self.unannotated_indices.index(self.current_event_index)
                except ValueError:
                    # If the event isn't in unannotated_indices, find the next unannotated event
                    next_unannotated = next(
                        (i for i in self.unannotated_indices if i > self.current_event_index),
                        self.unannotated_indices[0]
                    )
                    self.current_random_index = self.unannotated_indices.index(next_unannotated)
                    self.current_event_index = next_unannotated
            
            # Enable navigation without recursive calls
            self.prev_btn.configure(state=tk.NORMAL if self.current_random_index > 0 else tk.DISABLED)
            self.next_btn.configure(
                state=tk.NORMAL 
                if self.current_random_index < len(self.unannotated_indices) - 1 
                else tk.DISABLED
            )
            
            # Update display for the current event
            self.display_current_event()
            self.load_current_edits()
            
        except Exception as e:
            print(f"Navigation error: {str(e)}")
            # Ensure controls are enabled even if there's an error
            self.event_slider.configure(state=tk.NORMAL)
            self.preview_btn.configure(state=tk.NORMAL)
            self.save_btn.configure(state=tk.NORMAL)

    def jump_back_after_preview(self):
        """Jump back to the event prior to preview"""
        # Return to the starting position
        self.current_event_index = self.preview_start_index
        # Find the corresponding random index for this event
        try:
            self.current_random_index = self.unannotated_indices.index(self.current_event_index)
        except ValueError:
            # If the event isn't in unannotated_indices, find the next unannotated event
            next_unannotated = next(
                (i for i in self.unannotated_indices if i > self.current_event_index),
                self.unannotated_indices[0]
            )
            self.current_random_index = self.unannotated_indices.index(next_unannotated)
            self.current_event_index = next_unannotated
            
        self.event_slider.set(self.current_event_index)
        self.display_current_event()
        self.load_current_edits()  # Load any existing edits for this event
        self.enable_navigation()

    def show_status_message(self, message: str):
        """Show a status message in the center of the window"""
        if self.status_label:
            if message:
                # Calculate center position of the window
                window_width = self.root.winfo_width()
                window_height = self.root.winfo_height()
                
                # Show and update the message
                self.status_label.config(text=message)
                self.status_label.place(
                    relx=0.5,
                    rely=0.5,
                    anchor='center'
                )
                # Bring label to front
                self.status_label.lift()
            else:
                # Hide the label when message is empty
                self.status_label.place_forget()

    def check_batch_completion(self):
        """Check if current batch is completed and handle navigation"""
        batch_start = (self.current_event_index // 15) * 15
        batch_end = min(batch_start + 15, len(self.screenshot_events))
        batch_events = list(range(batch_start, batch_end))
        
        # Check if all events in batch are annotated
        all_annotated = all(
            idx in self.event_edits or 
            self.screenshot_events[idx].get('Human Augmentation') is not None
            for idx in batch_events
        )
        
        if all_annotated:
            # Find next batch with unannotated events
            next_batch_start = batch_end
            while next_batch_start < len(self.screenshot_events):
                next_batch_end = min(next_batch_start + 15, len(self.screenshot_events))
                next_batch = list(range(next_batch_start, next_batch_end))
                
                # Check if any event in next batch is unannotated
                for idx in next_batch:
                    if (idx not in self.event_edits and 
                        self.screenshot_events[idx].get('Human Augmentation') is None):
                        self.current_event_index = idx
                        self.current_random_index = self.unannotated_indices.index(idx)
                        self.display_current_event()
                        return True
                
                next_batch_start = next_batch_end
                
            # If no more unannotated events found
            messagebox.showinfo("Complete", "All events have been annotated!")
            return False
        
        return True

    def load_and_display_event(self):
        """Display current event and load human augmentation data if it exists"""
        # First display the event
        self.display_current_event()
        
        # Then load any existing human augmentation data
        current_event = self.screenshot_events[self.current_event_index]
        
        # Clear existing form fields
        self.clear_form_fields()
        
        # Check for human augmentation data in the event
        if 'Human Augmentation' in current_event:
            # Store the data in event_edits
            self.event_edits[self.current_event_index] = {
                'human_augmentation': current_event['Human Augmentation']
            }
            # Load the data into form fields
            self.load_current_edits()
        # Check if we have data in event_edits
        elif self.current_event_index in self.event_edits:
            self.load_current_edits()

    def jump_to_random_batch(self):
        """Jump to a random batch of 15 events"""
        if not self.screenshot_events:
            return
        
        # Calculate number of complete batches
        num_batches = len(self.screenshot_events) // 15
        
        # If there are no complete batches, show warning
        if num_batches == 0:
            messagebox.showwarning(
                "Not Enough Events",
                "Need at least 15 events for random batch navigation"
            )
            return
        
        # Pick a random batch
        random_batch = random.randint(0, num_batches - 1)
        
        # Calculate the starting index of the chosen batch
        new_index = random_batch * 15
        
        # Save current edits before moving
        self.save_current_edits()
        
        # Update current index
        self.current_event_index = new_index
        
        # Update random index if needed
        if new_index in self.unannotated_indices:
            self.current_random_index = self.unannotated_indices.index(new_index)
        
        # Update display
        self.display_current_event()
        self.enable_navigation()
        
        # Load edits for the new event
        self.load_current_edits()
        
        # Show status message
        self.show_status_message(f"✓ Jumped to batch {random_batch + 1}")
        self.root.after(3000, lambda: self.show_status_message(""))

    def show_next_annotated_event(self):
        """Jump to the next annotated event"""
        for i in range(self.current_event_index + 1, len(self.screenshot_events)):
            if 'Human Augmentation' in self.screenshot_events[i]:
                self.current_event_index = i
                self.display_current_event()
                self.enable_navigation()
                self.load_current_edits()
                return
        messagebox.showinfo("Info", "No more annotated events found.")

    def update_annotated_counter(self):
        """Update the counter showing number of annotated events"""
        if not self.screenshot_events:
            self.annotated_counter.config(text="Annotated: 0/0")
            return
            
        annotated_count = sum(
            1 for event in self.screenshot_events 
            if event.get('Human Augmentation') is not None or 
            self.screenshot_events.index(event) in self.event_edits
        )
        total_count = len(self.screenshot_events)
        self.annotated_counter.config(text=f"Annotated: {annotated_count}/{total_count}")

def is_double_click(current_event, previous_event):
    """
    Determine if current click event is part of a double click by comparing with previous event
    """
    if not previous_event:
        return False
        
    # Check if both events are clicks
    if current_event.get('event') != 'click' or previous_event.get('event') != 'click':
        return False
        
    # Check if clicks happened within 500ms of each other
    time_diff = current_event.get('ts', 0) - previous_event.get('ts', 0)
    if time_diff > 500:  # 500ms threshold
        return False
        
    # Check if clicks were at same/similar position
    current_pos = (current_event.get('x', 0), current_event.get('y', 0))
    previous_pos = (previous_event.get('x', 0), previous_event.get('y', 0))
    
    # Allow for small mouse movement (within 5 pixels)
    position_diff = (
        abs(current_pos[0] - previous_pos[0]),
        abs(current_pos[1] - previous_pos[1])
    )
    if position_diff[0] > 5 or position_diff[1] > 5:
        return False
        
    return True

def is_click_and_drag(current_event, next_event):
    """
    Determine if events represent a click-and-drag action
    """
    if not next_event:
        return False
    
    # Check if it starts with a click (pressed=True) and ends with release (pressed=False)
    if (current_event.get('event') == 'click' and next_event.get('event') == 'click' and
        current_event.get('pressed', False) == True and next_event.get('pressed', False) == False):
        
        # Check if there are mouse movements between press and release
        time_diff = next_event.get('ts', 0) - current_event.get('ts', 0)
        if time_diff > 100:  # More than 100ms between press and release
            return True
    
    return False

# Add ToolTip class for hints
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip,
            text=self.text,
            justify='left',
            background="#ffffe0",
            relief='solid',
            borderwidth=1,
            wraplength=300
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set initial window size to 80% of screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Calculate position for center of screen
    position_x = int((screen_width - window_width) / 2)
    position_y = int((screen_height - window_height) / 2)
    
    # Set window size and position
    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
    
    app = EventLabelingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
