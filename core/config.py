# Copyright (c) 2024-2025 Sentrl AI Inc. All rights reserved.
# This software is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""
Configuration loader for Sentrl AI Agent Platform
Loads YAML config with environment variable substitution
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve environment variables in config values.
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(value, str):
        # Replace ${VAR_NAME} with environment variable value
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for match in matches:
            env_value = os.getenv(match, '')
            value = value.replace(f'${{{match}}}', env_value)

        # Expand ~ to home directory
        if value.startswith('~'):
            value = str(Path(value).expanduser())

        return value

    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]

    else:
        return value


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict containing resolved configuration

    Example:
        >>> config = load_config()
        >>> data_dir = config['data']['root_dir']
    """
    # Get absolute path relative to project root
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    config = resolve_env_vars(config)

    return config


def get_data_dir(config: Dict[str, Any] = None) -> Path:
    """
    Get the main data directory path.
    Creates it if it doesn't exist.
    """
    if config is None:
        config = load_config()

    data_dir = Path(config['data']['root_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def get_session_dir(session_name: str = None, config: Dict[str, Any] = None) -> Path:
    """
    Get or create a session directory.

    Args:
        session_name: Optional session name, otherwise generates from format
        config: Optional config dict

    Returns:
        Path to session directory
    """
    if config is None:
        config = load_config()

    data_dir = get_data_dir(config)

    if session_name is None:
        from datetime import datetime
        session_format = config['data']['session_format']
        session_name = datetime.now().strftime(session_format)

    session_dir = data_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (session_dir / 'raw' / 'screenshots').mkdir(parents=True, exist_ok=True)
    (session_dir / 'processed').mkdir(parents=True, exist_ok=True)
    (session_dir / 'annotations' / 'llm_responses').mkdir(parents=True, exist_ok=True)
    (session_dir / 'ground_truth').mkdir(parents=True, exist_ok=True)
    (session_dir / 'training' / 'train').mkdir(parents=True, exist_ok=True)
    (session_dir / 'training' / 'test').mkdir(parents=True, exist_ok=True)

    return session_dir


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully:")
    print(f"Data root: {config['data']['root_dir']}")
    print(f"Vision model: {config['vision']['model']}")
    print(f"Training base model: {config['training']['base_model']}")
    print(f"Device: {config['training']['device']}")
