# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Vision-based Annotation Tool for Sentrl AI Agent Platform
Uses Llama 3.2 Vision (via Ollama) to annotate screenshots with LLM analysis
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                           QHBoxLayout, QWidget, QFileDialog, QProgressBar,
                           QLabel, QTextEdit, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import requests
import base64
import datetime
import time
import io
import platform
import torch

# Import config system
from core.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
logger.info(f"Using device: {device}")

def extract_section(text: str, section_name: str) -> List[Dict]:
    """Extract a section from the LLM response text."""
    try:
        # Try to parse the entire text as JSON first
        data = json.loads(text)
        section_key = section_name.lower().replace(" ", "_")
        return data.get(section_key, [])
    except json.JSONDecodeError:
        # If JSON parsing fails, return empty list
        return []

class AnalysisWorker(QThread):
    """Worker thread for processing screenshots"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.is_running = True

    def run(self):
        try:
            actions = self.analyzer.load_actions()
            if not actions:
                self.error.emit("No actions found in actions.json")
                return

            total_screenshots = sum(1 for action in actions if "screenshot" in action)
            processed = 0

            for i, action in enumerate(actions):
                if not self.is_running:
                    break

                if "screenshot" in action:
                    screenshot_path = self.analyzer.get_screenshot_path(action["screenshot"])
                    if screenshot_path:
                        self.status.emit(f"Processing: {screenshot_path.name}")
                        
                        analysis = self.analyzer.analyze_screenshot(screenshot_path)
                        action["llm_analysis"] = analysis
                        processed += 1
                        self.progress.emit(int((processed / total_screenshots) * 100))

            if self.is_running:
                self.analyzer.save_augmented_actions(actions)
                self.status.emit("Processing complete!")
                self.finished.emit()

        except Exception as e:
            self.error.emit(f"Error during processing: {str(e)}")

    def stop(self):
        self.is_running = False

class ActionAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Action Analysis Tool')
        self.setMinimumSize(800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create top section with buttons
        button_layout = QHBoxLayout()
        
        self.select_btn = QPushButton('Select Directory')
        self.select_btn.clicked.connect(self.select_directory)
        self.select_btn.setMinimumWidth(150)
        
        self.process_new_btn = QPushButton('Start New Processing')
        self.process_new_btn.clicked.connect(lambda: self.start_processing(restart=True))
        self.process_new_btn.setEnabled(False)
        self.process_new_btn.setMinimumWidth(150)
        
        self.resume_btn = QPushButton('Resume Processing')
        self.resume_btn.clicked.connect(lambda: self.start_processing(restart=False))
        self.resume_btn.setEnabled(False)
        self.resume_btn.setMinimumWidth(150)
        
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(150)

        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.process_new_btn)
        button_layout.addWidget(self.resume_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()

        # Add status section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        status_layout = QVBoxLayout(status_frame)
        
        self.dir_label = QLabel('No directory selected')
        self.status_label = QLabel('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        status_layout.addWidget(self.dir_label)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        # Add log viewer
        log_frame = QFrame()
        log_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        log_layout = QVBoxLayout(log_frame)
        
        log_label = QLabel('Processing Log:')
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text)

        # Add all sections to main layout
        layout.addLayout(button_layout)
        layout.addWidget(status_frame)
        layout.addWidget(log_frame)

        self.show()

    def select_directory(self):
        """Handle directory selection"""
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir_path:
            self.dir_label.setText(f'Selected: {dir_path}')
            self.analyzer = ActionAnalyzer(dir_path)
            self.process_new_btn.setEnabled(True)
            
            # Check if there are existing responses
            responses_dir = Path(dir_path) / "llm_responses"
            if responses_dir.exists() and any(responses_dir.iterdir()):
                self.resume_btn.setEnabled(True)
                self.log_message("Found existing responses. You can resume processing.")
            else:
                self.resume_btn.setEnabled(False)
            
            self.log_message(f"Directory selected: {dir_path}")

    def start_processing(self, restart=True):
        """Start processing the selected directory"""
        if self.analyzer:
            self.worker = AnalysisWorker(self.analyzer)
            self.worker.progress.connect(self.update_progress)
            self.worker.status.connect(self.update_status)
            self.worker.finished.connect(self.processing_finished)
            self.worker.error.connect(self.handle_error)
            
            # Set restart flag in analyzer
            self.analyzer.restart = restart
            
            self.worker.start()
            
            self.process_new_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.select_btn.setEnabled(False)
            
            if restart:
                self.log_message("Starting new processing...")
            else:
                self.log_message("Resuming processing...")

    def stop_processing(self):
        """Stop the processing"""
        if self.worker:
            self.worker.stop()
            self.log_message("Processing stopped by user")
            self.processing_finished()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        self.log_message(message)

    def processing_finished(self):
        """Handle processing completion"""
        self.process_new_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.status_label.setText("Ready")
        self.log_message("Processing complete")

    def handle_error(self, error_message):
        """Handle error messages"""
        self.status_label.setText("Error occurred")
        self.log_message(f"ERROR: {error_message}")
        self.processing_finished()

    def log_message(self, message):
        """Add message to log viewer"""
        self.log_text.append(f"{message}")

class ActionAnalyzer:
    def __init__(self, input_dir: str, config: Dict = None):
        """Initialize the analyzer with input directory path."""
        self.input_dir = Path(input_dir)

        # Load config if not provided
        if config is None:
            config = load_config()
        self.config = config

        # Determine if we're in raw/ or processed/ directory
        # If input_dir ends with 'raw', load from raw/actions.json
        # If input_dir ends with 'processed', load from processed/actions_standardized.json
        if self.input_dir.name == 'raw':
            self.actions_file = self.input_dir / "actions.json"
        elif self.input_dir.name == 'processed':
            self.actions_file = self.input_dir / "actions_standardized.json"
        else:
            # Assume processed by default
            self.actions_file = self.input_dir / "actions_standardized.json"
            if not self.actions_file.exists():
                self.actions_file = self.input_dir / "actions.json"

        # Create annotations directory structure
        annotations_dir = self.input_dir.parent / "annotations"
        self.responses_dir = annotations_dir / "llm_responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

        # Get Ollama configuration from config
        self.model_name = config['vision']['model']
        self.ollama_endpoint = config['vision']['endpoint']
        self.temperature = config['vision']['temperature']

        logger.info(f"Initialized ActionAnalyzer with input directory: {input_dir}")
        logger.info(f"Using model: {self.model_name}")
        logger.info(f"Actions file: {self.actions_file}")
        logger.info(f"Output directory: {self.responses_dir}")

    def save_response(self, screenshot_name: str, analysis: Dict):
        """Save individual response to a JSON file."""
        try:
            # Ensure the directory exists
            self.responses_dir.mkdir(parents=True, exist_ok=True)
            
            # Create the response file path
            response_file = self.responses_dir / f"{screenshot_name}_analysis.json"
            
            # Save the analysis
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved analysis to: {response_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis for {screenshot_name}: {e}")
            raise

    def analyze_screenshot(self, screenshot_path: Path) -> Dict:
        """Analyze screenshot using Llama Vision model."""
        try:
            system_prompt = r"""You are an expert AI data labeller who can analyze user interfaces, and interpret the user actions, context and states of the system. 
            Your task is to analyze screenshots and the user interactions."""

            user_prompt = r"""Analyze this screenshot and event log carefully. 
            These screenshots are captured whenever the user clicks, scrolls, changes window, presses enter or tab. 

            Instructions:
            Provide a detailed analysis that can be used to replicate user activity and predict next steps.
            Provide a direct and comprehensive response using the template below, formatted as a JSON object:

            Example 1: 
            Input: 
            {
                "ts": "2024-01-31T12:02:22.584",
                "event": "mouse_click",
                "button": "left",
                "pressed": "true",
                "screenshot": "screen_2024-01-31T12:02:22.584.png",
                "position": {
                    "X": "639.3046875",
                    "Y": "866.8125"
                }
            }"""

            # Convert image to base64
            with open(screenshot_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            # Force garbage collection after processing
            import gc
            gc.collect()

            # Make API call to local Ollama instance
            response = requests.post(
                self.ollama_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "temperature": self.temperature
                },
                timeout=self.config['vision'].get('timeout', 60)
            )

            try:
                response_data = response.json()
                text_response = response_data.get("response", "")
                logger.debug(f"Raw response text:\n{text_response}")

                # Store as raw text without any formatting
                result = {
                    "raw_response": str(text_response),  # Force string conversion
                    "timestamp": str(datetime.datetime.now()),
                    "screenshot_name": str(screenshot_path.stem)  # Force string conversion
                }

                self.save_response(screenshot_path.stem, result)
                return result

            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
                raise

        except Exception as e:
            logger.error(f"Error analyzing screenshot {screenshot_path}: {e}")
            raise

    def load_actions(self) -> List[Dict]:
        """Load actions from JSON file."""
        try:
            with open(self.actions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading actions file: {e}")
            raise

    def process_actions(self):
        """Process all actions and save responses individually."""
        try:
            actions = self.load_actions()
            if not actions:
                raise ValueError("No actions found in actions.json")

            total_screenshots = sum(1 for action in actions if "screenshot" in action)
            processed = 0

            for action in actions:
                if "screenshot" in action:
                    screenshot_path = self.get_screenshot_path(action["screenshot"])
                    if screenshot_path:
                        logger.info(f"Processing: {screenshot_path.name}")
                        analysis = self.analyze_screenshot(screenshot_path)
                        action["llm_analysis"] = analysis
                        processed += 1
                        logger.info(f"Progress: {processed}/{total_screenshots}")

            logger.info("Processing complete!")
            return actions

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

    def get_screenshot_path(self, screenshot_name: str) -> Optional[Path]:
        """Get full path for screenshot."""
        # Handle both relative paths (screenshots/file.png) and absolute
        if '/' in screenshot_name:
            # Relative path from actions.json
            screenshot_path = self.input_dir / screenshot_name
        else:
            # Try direct file in input_dir
            screenshot_path = self.input_dir / screenshot_name

        # Also check in raw/screenshots if not found
        if not screenshot_path.exists():
            alt_path = self.input_dir.parent / 'raw' / screenshot_name
            if alt_path.exists():
                screenshot_path = alt_path

        return screenshot_path if screenshot_path.exists() else None

    def save_augmented_actions(self, actions: List[Dict]):
        """Save the augmented actions back to a JSON file."""
        try:
            # Create a backup of the original actions file
            backup_path = self.actions_file.with_suffix('.json.backup')
            if self.actions_file.exists():
                import shutil
                shutil.copy2(self.actions_file, backup_path)
                logger.info(f"Created backup of actions file: {backup_path}")

            # Save the augmented actions
            output_path = self.actions_file.with_name('actions_augmented.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(actions, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved augmented actions to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving augmented actions: {e}")
            raise

def main():
    app = QApplication(sys.argv)
    ex = ActionAnalyzerGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()