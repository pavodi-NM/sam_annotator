"""Unit tests for the CLI parser module."""

import pytest
from sam_annotator.cli.parser import create_parser, parse_args
from sam_annotator import __version__

class TestParser:
    """Tests for the parser module in the CLI package."""
    
    @pytest.fixture
    def config(self):
        """Return a sample configuration for testing."""
        return {
            "last_category_path": "/path/to/category",
            "last_classes_csv": "/path/to/classes.csv",
            "last_sam_version": "sam1",
            "last_model_type": "vit_h"
        }
    
    def test_create_parser(self, config):
        """Test that create_parser returns a properly configured parser."""
        parser = create_parser(config)
        
        # Check that the parser has the expected arguments
        actions = {action.dest: action for action in parser._actions}
        
        # Check version action
        assert "version" in actions
        assert actions["version"].version == f"%(prog)s {__version__}"
        
        # Check SAM version action
        assert "sam_version" in actions
        assert actions["sam_version"].default == "sam1"
        assert actions["sam_version"].choices == ["sam1", "sam2"]
        
        # Check model type action
        assert "model_type" in actions
        assert actions["model_type"].default == "vit_h"
        
        # Check data path actions
        assert "category_path" in actions
        assert actions["category_path"].default == "/path/to/category"
        
        assert "classes_csv" in actions
        assert actions["classes_csv"].default == "/path/to/classes.csv"
        
        # Check visualization actions
        assert "visualization" in actions
        assert actions["visualization"].default is False
        
        # Check validation actions
        assert "skip_validation" in actions
        assert actions["skip_validation"].default is False
        
        # Check debug action
        assert "debug" in actions
        assert actions["debug"].default is False
    
    def test_parse_args_defaults(self, config):
        """Test that parse_args returns the expected defaults when no args are provided."""
        args = parse_args(config, [])
        
        # Check that default values match the config
        assert args.sam_version == "sam1"
        assert args.model_type == "vit_h"
        assert args.category_path == "/path/to/category"
        assert args.classes_csv == "/path/to/classes.csv"
        
        # Check that boolean flags are false by default
        assert args.visualization is False
        assert args.export_stats is False
        assert args.skip_validation is False
        assert args.use_sample_csv is False
        assert args.create_sample is False
        assert args.debug is False
    
    def test_parse_args_custom(self, config):
        """Test that parse_args correctly handles custom arguments."""
        # Define custom arguments
        test_args = [
            "--sam_version", "sam2",
            "--model_type", "base_v2",
            "--category_path", "/custom/category",
            "--classes_csv", "/custom/classes.csv",
            "--visualization",
            "--export_stats",
            "--debug"
        ]
        
        args = parse_args(config, test_args)
        
        # Check that values match the custom arguments
        assert args.sam_version == "sam2"
        assert args.model_type == "base_v2"
        assert args.category_path == "/custom/category"
        assert args.classes_csv == "/custom/classes.csv"
        
        # Check that boolean flags are set correctly
        assert args.visualization is True
        assert args.export_stats is True
        assert args.skip_validation is False
        assert args.use_sample_csv is False
        assert args.create_sample is False
        assert args.debug is True
    
    def test_parse_args_sam_version_validation(self, config):
        """Test that parse_args validates the sam_version argument."""
        # Try an invalid SAM version
        with pytest.raises(SystemExit):
            parse_args(config, ["--sam_version", "invalid"])
    
    def test_parse_args_sample_csv(self, config):
        """Test that parse_args handles the sample CSV options correctly."""
        # Test create sample
        args = parse_args(config, ["--create_sample", "--sample_output", "test.csv"])
        assert args.create_sample is True
        assert args.sample_output == "test.csv"
        
        # Test use sample
        args = parse_args(config, ["--use_sample_csv"])
        assert args.use_sample_csv is True 