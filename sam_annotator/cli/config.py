"""Configuration management for SAM Annotator."""

import json
import os

# Constants for config file
CONFIG_FILE = ".sam_config.json"

def load_config():
    """Load configuration from config file if it exists."""
    config = {
        "last_category_path": None,
        "last_classes_csv": None,
        "last_sam_version": "sam1",
        "last_model_type": None
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")  
    
    return config

def save_config(config):
    """Save configuration to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}") 