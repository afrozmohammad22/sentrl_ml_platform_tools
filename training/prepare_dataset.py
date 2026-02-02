# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Dataset Preparation for Sentrl AI Agent Platform
Unified formatter merging OAI-chat, Alpaca, and Instruction formats
Prepares HuggingFace dataset structure for training
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import jsonlines
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
import logging

from core.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetFormatter:
    """Unified dataset formatter for training"""

    def __init__(self, data_dir: Path, config: Dict = None):
        """
        Initialize dataset formatter.

        Args:
            data_dir: Session directory containing annotations or ground_truth
            config: Optional config dict, will load if not provided
        """
        self.data_dir = Path(data_dir)

        if config is None:
            config = load_config()
        self.config = config

        # Determine data source: prioritize ground_truth over annotations
        self.ground_truth_file = self.data_dir / 'ground_truth' / 'actions_labeled.json'
        self.annotations_file = self.data_dir / 'annotations' / 'actions_annotated.json'

        if self.ground_truth_file.exists():
            self.source_file = self.ground_truth_file
            self.source_type = 'ground_truth'
            logger.info(f"Using ground truth labels from: {self.source_file}")
        elif self.annotations_file.exists():
            self.source_file = self.annotations_file
            self.source_type = 'annotations'
            logger.info(f"Using LLM annotations from: {self.source_file}")
        else:
            raise FileNotFoundError(
                f"No data found. Expected either:\n"
                f"  - {self.ground_truth_file}\n"
                f"  - {self.annotations_file}"
            )

        # Determine screenshot location
        self.screenshots_dir = self.data_dir / 'raw' / 'screenshots'
        if not self.screenshots_dir.exists():
            self.screenshots_dir = self.data_dir / 'processed' / 'screenshots'

        logger.info(f"Screenshots directory: {self.screenshots_dir}")

    def load_data(self) -> List[Dict]:
        """Load action data from source file"""
        with open(self.source_file, 'r') as f:
            data = json.load(f)

        if not data:
            raise ValueError(f"No data found in {self.source_file}")

        logger.info(f"Loaded {len(data)} events from {self.source_file}")
        return data

    @staticmethod
    def convert_flat_to_nested(flat_dict: Dict) -> Dict:
        """
        Convert flat dot-notation keys to nested structure.

        Example:
            {'interaction_hierarchy.action': 'Click', 'goal': 'Open app'}
            → {'interaction_hierarchy': {'action': 'Click'}, 'goal': 'Open app'}

        Args:
            flat_dict: Dictionary with dot-notation keys

        Returns:
            Nested dictionary structure
        """
        nested = {}
        for key, value in flat_dict.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested

    def format_alpaca_style(self, event: Dict) -> Optional[Dict]:
        """
        Format event in Alpaca instruction-tuning style.

        Format:
        {
            "instruction": "Describe the current state...",
            "input": "Action: click at X,Y\nWindow: Chrome\nScreenshot: <image>",
            "output": "Task: ... Workflow: ... Goal: ..."
        }
        """
        try:
            # Extract components
            if self.source_type == 'ground_truth':
                # Check if using flat "Human Augmentation" format (legacy demo data)
                if 'Human Augmentation' in event:
                    # Convert flat format to nested
                    nested = self.convert_flat_to_nested(event['Human Augmentation'])
                    action_desc = nested.get('interaction_hierarchy', {}).get('action', '')
                    task = nested.get('interaction_hierarchy', {}).get('task', '')
                    workflow = nested.get('interaction_hierarchy', {}).get('workflow', '')
                    goal = nested.get('goal', '')
                    active_window = nested.get('state_description', {}).get('active_window', '')
                    active_app = nested.get('state_description', {}).get('active_app', '')
                    content = nested.get('state_description', {}).get('content', '')
                    attention = nested.get('attention', '')
                elif 'interaction_hierarchy' in event:
                    # Standard nested ground truth format
                    action_desc = event.get('interaction_hierarchy', {}).get('action', '')
                    task = event.get('interaction_hierarchy', {}).get('task', '')
                    workflow = event.get('interaction_hierarchy', {}).get('workflow', '')
                    goal = event.get('goal', '')
                    active_window = event.get('state_description', {}).get('active_window', '')
                    active_app = event.get('state_description', {}).get('active_app', '')
                    content = event.get('state_description', {}).get('content', '')
                    attention = event.get('attention', '')
                else:
                    # No labels found, skip this event
                    return None
            else:
                # LLM annotation format - extract from llm_analysis
                llm_data = event.get('llm_analysis', {})
                action_desc = llm_data.get('action', '')
                task = llm_data.get('task', '')
                workflow = llm_data.get('workflow', '')
                goal = llm_data.get('goal', '')
                active_window = llm_data.get('active_window', '')
                active_app = llm_data.get('active_app', '')
                content = llm_data.get('content', '')
                attention = llm_data.get('attention', '')

            # Build instruction
            instruction = (
                "Describe the current state of the system based on the user action, "
                "visual context, and system state. Identify the task, workflow, and goal."
            )

            # Build input
            event_type = event.get('event', 'unknown')
            position = event.get('position', {})
            input_text = (
                f"Action: {event_type} at position X:{position.get('X', 0)}, Y:{position.get('Y', 0)}\n"
                f"Window: {active_window}\n"
                f"Application: {active_app}\n"
                f"Timestamp: {event.get('ts', '')}"
            )

            # Build output
            output_text = (
                f"Action: {action_desc}\n"
                f"Task: {task}\n"
                f"Workflow: {workflow}\n"
                f"Goal: {goal}\n"
                f"Content: {content}\n"
                f"Attention: {attention}"
            )

            return {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "screenshot": event.get('screenshot', '')
            }

        except Exception as e:
            logger.error(f"Error formatting event in Alpaca style: {e}")
            return None

    def format_oai_chat(self, event: Dict) -> Optional[Dict]:
        """
        Format event in OpenAI chat completion style.

        Format:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        try:
            # Similar extraction as Alpaca
            if self.source_type == 'ground_truth':
                action_desc = event.get('interaction_hierarchy', {}).get('action', '')
                task = event.get('interaction_hierarchy', {}).get('task', '')
                workflow = event.get('interaction_hierarchy', {}).get('workflow', '')
                goal = event.get('goal', '')
                active_window = event.get('state_description', {}).get('active_window', '')
            else:
                llm_data = event.get('llm_analysis', {})
                action_desc = llm_data.get('action', '')
                task = llm_data.get('task', '')
                workflow = llm_data.get('workflow', '')
                goal = llm_data.get('goal', '')
                active_window = llm_data.get('active_window', '')

            # System message
            system_msg = (
                "You are an expert at analyzing user interactions and describing "
                "the state of computer systems based on screenshots and user actions."
            )

            # User message
            event_type = event.get('event', 'unknown')
            position = event.get('position', {})
            user_text = (
                f"Analyze this user action: {event_type} at position "
                f"X:{position.get('X', 0)}, Y:{position.get('Y', 0)} "
                f"in window: {active_window}. What is the user trying to accomplish?"
            )

            # Assistant response
            assistant_response = (
                f"Action: {action_desc}\n"
                f"Task: {task}\n"
                f"Workflow: {workflow}\n"
                f"Goal: {goal}"
            )

            return {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image", "image_url": {"url": event.get('screenshot', '')}}
                        ]
                    },
                    {"role": "assistant", "content": assistant_response}
                ]
            }

        except Exception as e:
            logger.error(f"Error formatting event in OAI chat style: {e}")
            return None

    def get_image_metadata(self, screenshot_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Get image dimensions"""
        try:
            with Image.open(screenshot_path) as img:
                return img.size
        except Exception as e:
            logger.warning(f"Could not read image metadata: {e}")
            return None, None

    def create_hf_dataset(
        self,
        output_dir: Path = None,
        format_type: str = 'alpaca',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, int]:
        """
        Create HuggingFace dataset structure with train/val/test splits.

        Args:
            output_dir: Where to save dataset (defaults to data_dir/training)
            format_type: 'alpaca' or 'oai-chat'
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)

        Returns:
            Dict with split sizes
        """
        if output_dir is None:
            output_dir = self.data_dir / 'training'

        output_dir = Path(output_dir)
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        test_dir = output_dir / 'test'

        # Create directories
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load and filter events
        events = self.load_data()
        events_with_screenshots = [e for e in events if 'screenshot' in e]

        logger.info(f"Filtering: {len(events_with_screenshots)} events with screenshots")

        if not events_with_screenshots:
            raise ValueError("No events with screenshots found")

        # Split data
        train_events, test_events = train_test_split(
            events_with_screenshots,
            test_size=test_size,
            random_state=42
        )

        train_events, val_events = train_test_split(
            train_events,
            test_size=val_size,
            random_state=42
        )

        logger.info(f"Splits: train={len(train_events)}, val={len(val_events)}, test={len(test_events)}")

        # Process each split
        splits = [
            ('train', train_events, train_dir),
            ('val', val_events, val_dir),
            ('test', test_events, test_dir)
        ]

        for split_name, split_events, split_dir in splits:
            logger.info(f"Processing {split_name} split...")
            jsonl_path = split_dir / 'metadata.jsonl'

            with jsonlines.open(jsonl_path, mode='w') as writer:
                for event in split_events:
                    # Format event
                    if format_type == 'alpaca':
                        formatted = self.format_alpaca_style(event)
                    elif format_type == 'oai-chat':
                        formatted = self.format_oai_chat(event)
                    else:
                        raise ValueError(f"Unknown format_type: {format_type}")

                    if formatted is None:
                        continue

                    screenshot_name = event.get('screenshot', '')
                    if not screenshot_name:
                        continue

                    # Handle screenshot path (may be relative like screenshots/file.png)
                    if '/' in screenshot_name:
                        screenshot_path = self.screenshots_dir.parent / screenshot_name
                    else:
                        screenshot_path = self.screenshots_dir / screenshot_name

                    # Copy screenshot
                    if screenshot_path.exists():
                        dest_path = split_dir / screenshot_path.name
                        shutil.copy2(screenshot_path, dest_path)

                        # Get metadata
                        width, height = self.get_image_metadata(screenshot_path)

                        # Write to JSONL
                        metadata_entry = {
                            "file_name": screenshot_path.name,
                            "formatted_data": formatted,
                            "metadata": {
                                "width": width,
                                "height": height
                            }
                        }
                        writer.write(metadata_entry)
                    else:
                        logger.warning(f"Screenshot not found: {screenshot_path}")

        logger.info(f"Dataset created successfully at: {output_dir}")

        return {
            'train_size': len(train_events),
            'val_size': len(val_events),
            'test_size': len(test_events)
        }


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--data', type=str, required=True, help='Session directory')
    parser.add_argument('--output', type=str, help='Output directory (default: data_dir/training)')
    parser.add_argument('--format', type=str, default='alpaca', choices=['alpaca', 'oai-chat'],
                        help='Output format')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set proportion')

    args = parser.parse_args()

    try:
        formatter = DatasetFormatter(args.data)
        result = formatter.create_hf_dataset(
            output_dir=args.output,
            format_type=args.format,
            test_size=args.test_size,
            val_size=args.val_size
        )

        print("\n✅ Dataset preparation complete!")
        print(f"  Train: {result['train_size']} samples")
        print(f"  Val: {result['val_size']} samples")
        print(f"  Test: {result['test_size']} samples")

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise


if __name__ == '__main__':
    main()
