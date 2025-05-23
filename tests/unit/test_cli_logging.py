"""Unit tests for the CLI logging utilities module."""

import logging
import os
import pytest
from unittest.mock import patch, MagicMock
from sam_annotator.cli.logging_utils import setup_standard_logging, setup_debug_logging

class TestLoggingUtils:
    """Tests for the logging utilities module in the CLI package."""
    
    def test_setup_standard_logging(self):
        """Test that setup_standard_logging configures a logger properly."""
        # Reset root logger to avoid interference from previous tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Call the function to set up standard logging
        logger = setup_standard_logging()
        
        # Check that the logger was created
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        
        # Check that the logger's level is set to INFO
        assert logger.level == 0  # Logger doesn't inherit level directly from basicConfig
        
        # Check that the root logger has the correct level
        assert root_logger.level == logging.INFO
    
    def test_setup_debug_logging(self):
        """Test that setup_debug_logging configures a more detailed logger."""
        # Reset root logger to avoid interference from previous tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create proper mock handlers with real level attributes
        file_handler_mock = MagicMock()
        file_handler_mock.level = logging.DEBUG
        
        console_handler_mock = MagicMock()
        console_handler_mock.level = logging.INFO
        
        # Mock the handlers
        with patch('logging.FileHandler', return_value=file_handler_mock) as mock_file_handler:
            with patch('logging.StreamHandler', return_value=console_handler_mock) as mock_stream_handler:
                # Call the function to set up debug logging
                logger = setup_debug_logging()
                
                # Check that the logger was created and is the root logger
                assert logger is not None
                assert isinstance(logger, logging.Logger)
                assert logger is root_logger
                
                # Check that the file handler was created
                mock_file_handler.assert_called_once_with('sam_debug.log', mode='w')
                
                # Check that the root logger level is set to DEBUG
                assert root_logger.level == logging.DEBUG
                
                # Check handler levels using our pre-set attributes
                assert file_handler_mock.level == logging.DEBUG
                assert console_handler_mock.level == logging.INFO
    
    def test_handler_formatters(self):
        """Test that the logging handlers have properly configured formatters."""
        # Reset root logger to avoid interference from previous tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create a custom file handler to capture the formatter
        mock_handler = MagicMock()
        
        # Patch the handlers to use our mock
        with patch('logging.FileHandler', return_value=mock_handler) as mock_file_handler:
            with patch('logging.StreamHandler', return_value=mock_handler) as mock_stream_handler:
                # Call the function to set up debug logging
                logger = setup_debug_logging()
                
                # Check that the formatters were set
                assert mock_handler.setFormatter.call_count == 2
                
                # Get the formatter that was set
                formatter = mock_handler.setFormatter.call_args[0][0]
                
                # Check the formatter format string
                expected_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                assert formatter._fmt == expected_format 