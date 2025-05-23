"""Unit tests for the CLI config module."""

import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock, call
from sam_annotator.cli.config import load_config, save_config

class TestConfig:
    """Tests for the config module in the CLI package."""
    
    def test_load_config_default(self, monkeypatch):
        """Test that load_config returns default values when no config file exists."""
        # Patch os.path.exists to return False (file doesn't exist)
        monkeypatch.setattr(os.path, "exists", lambda path: False)
        
        config = load_config()
        
        # Check that default values are returned
        assert config["last_category_path"] is None
        assert config["last_classes_csv"] is None
        assert config["last_sam_version"] == "sam1"
        assert config["last_model_type"] is None
    
    def test_load_config_existing(self, monkeypatch):
        """Test that load_config loads values from an existing config file."""
        # Sample config data
        sample_config = {
            "last_category_path": "/path/to/category",
            "last_classes_csv": "/path/to/classes.csv",
            "last_sam_version": "sam2",
            "last_model_type": "base_v2"
        }
        
        # Patch os.path.exists to return True (file exists)
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        
        # Patch open to return our sample config
        mocked_open = mock_open(read_data=json.dumps(sample_config))
        with patch("builtins.open", mocked_open):
            config = load_config()
        
        # Check that values from the file are returned
        assert config["last_category_path"] == "/path/to/category"
        assert config["last_classes_csv"] == "/path/to/classes.csv"
        assert config["last_sam_version"] == "sam2"
        assert config["last_model_type"] == "base_v2"
    
    def test_load_config_error(self, monkeypatch, capfd):
        """Test that load_config handles errors gracefully."""
        # Patch os.path.exists to return True (file exists)
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        
        # Patch open to raise an exception
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = load_config()
        
        # Check that default values are returned
        assert config["last_category_path"] is None
        assert config["last_classes_csv"] is None
        assert config["last_sam_version"] == "sam1"
        assert config["last_model_type"] is None
        
        # Check that a warning was printed
        out, err = capfd.readouterr()
        assert "Warning: Could not load config file: Test error" in out
    
    def test_save_config(self, monkeypatch):
        """Test that save_config writes config to a file."""
        # Sample config data
        sample_config = {
            "last_category_path": "/path/to/category",
            "last_classes_csv": "/path/to/classes.csv",
            "last_sam_version": "sam2",
            "last_model_type": "base_v2"
        }
        
        # Mock json.dump instead of testing the file write directly
        mock_dump = MagicMock()
        
        with patch("builtins.open", mock_open()) as mocked_open:
            with patch("json.dump", mock_dump) as mocked_dump:
                save_config(sample_config)
        
        # Check that the file was opened for writing
        mocked_open.assert_called_once_with(".sam_config.json", "w")
        
        # Check that json.dump was called with the right arguments
        mocked_dump.assert_called_once()
        args, kwargs = mocked_dump.call_args
        assert args[0] == sample_config  # First arg is the config dict
        assert 'indent' in kwargs and kwargs['indent'] == 4  # Check formatting
    
    def test_save_config_error(self, monkeypatch, capfd):
        """Test that save_config handles errors gracefully."""
        # Sample config data
        sample_config = {
            "last_category_path": "/path/to/category",
            "last_classes_csv": "/path/to/classes.csv",
            "last_sam_version": "sam2",
            "last_model_type": "base_v2"
        }
        
        # Patch open to raise an exception
        with patch("builtins.open", side_effect=Exception("Test error")):
            save_config(sample_config)
        
        # Check that a warning was printed
        out, err = capfd.readouterr()
        assert "Warning: Could not save config file: Test error" in out 