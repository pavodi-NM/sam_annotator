"""Logging utilities for SAM Annotator."""

import logging

def setup_standard_logging():
    """Setup standard logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def setup_debug_logging():
    """Setup enhanced debug logging with file and console handlers."""
    # Create a file handler that logs to a debug.log file
    file_handler = logging.FileHandler('sam_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Also log to console with a higher level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Return the logger for convenience
    return root_logger 