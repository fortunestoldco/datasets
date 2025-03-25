#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import logging

# Configuration file path
CONFIG_FILE = os.path.expanduser("~/.doc_downloader_config.json")

logger = logging.getLogger(__name__)

# Function to create or load configuration
def load_or_create_config() -> dict:
    default_config = {
        'github_token': '',
        'huggingface_token': '',  # Keep this for uploads
        'repositories': [],
        'output_directory': './downloaded_docs',
        'test_ratio': 0.2
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            # Ensure all keys exist in the loaded config
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]

            return config
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            return default_config
    else:
        return default_config

# Function to save configuration
def save_config(config: dict) -> None:
    """
    Save configuration to file.

    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Secure the file permissions (read/write only for the user)
        os.chmod(CONFIG_FILE, 0o600)
    except Exception as e:
        logger.error(f"Error saving configuration file: {str(e)}")

