"""Utility functions for the phishing detection project."""

import os
import json


def get_project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    """Save a dictionary as a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load a JSON file and return as a dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)
