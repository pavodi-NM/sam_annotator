"""Unit tests for the CLI CSV utilities module."""

import os
import pandas as pd
import pytest
from unittest.mock import patch, mock_open, MagicMock
from sam_annotator.cli.csv_utils import create_sample_csv, validate_csv, _offer_sample_creation

class TestCSVUtils:
    """Tests for the CSV utilities module in the CLI package."""
    
    @pytest.fixture
    def logger(self):
        """Return a mock logger for testing."""
        logger = MagicMock()
        return logger
    
    def test_create_sample_csv_success(self, tmp_path, logger):
        """Test that create_sample_csv creates a valid CSV file."""
        # Create a temporary file path
        output_path = os.path.join(tmp_path, "test_sample.csv")
        
        # Call the function
        result = create_sample_csv(output_path, logger)
        
        # Check that the function returned True (success)
        assert result is True
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Check that the file contains the expected content
        df = pd.read_csv(output_path)
        assert "class_name" in df.columns
        assert len(df) > 0
        assert "background" in df["class_name"].values
        assert "person" in df["class_name"].values
        
        # Check that logger was called with appropriate messages
        logger.info.assert_any_call(f"Created sample CSV file at: {output_path}")
        logger.info.assert_any_call(f"Added {len(df)} classes")
    
    def test_create_sample_csv_failure(self, logger):
        """Test that create_sample_csv handles errors gracefully."""
        # Mock pandas.DataFrame.to_csv to raise an exception
        with patch("pandas.DataFrame.to_csv", side_effect=Exception("Test error")):
            result = create_sample_csv("/invalid/path", logger)
        
        # Check that the function returned False (failure)
        assert result is False
        
        # Check that logger was called with appropriate error message
        logger.error.assert_called_once_with("Error creating sample CSV file: Test error")
    
    def test_validate_csv_valid(self, tmp_path, logger):
        """Test that validate_csv returns True for valid CSV files."""
        # Create a valid CSV file
        csv_path = os.path.join(tmp_path, "valid.csv")
        df = pd.DataFrame({"class_name": ["background", "person", "car"]})
        df.to_csv(csv_path, index=False)
        
        # Call the function
        result = validate_csv(csv_path, logger)
        
        # Check that the function returned True (valid)
        assert result is True
        
        # Check that logger was called with appropriate message
        logger.info.assert_any_call(f"CSV validation passed: Found 3 classes in {csv_path}")
    
    def test_validate_csv_alternative_column(self, tmp_path, logger, monkeypatch):
        """Test that validate_csv offers to fix CSV files with alternative column names."""
        # Create a CSV file with an alternative column name
        csv_path = os.path.join(tmp_path, "alternative.csv")
        df = pd.DataFrame({"className": ["background", "person", "car"]})
        df.to_csv(csv_path, index=False)
        
        # Mock input to simulate user selecting 'y' to fix the file
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        
        # Call the function
        result = validate_csv(csv_path, logger)
        
        # Check that the function returned True (valid after fix)
        assert result is True
        
        # Check that the file was fixed
        df_fixed = pd.read_csv(csv_path)
        assert "class_name" in df_fixed.columns
        assert "className" not in df_fixed.columns
        
        # Check that logger was called with appropriate messages
        logger.warning.assert_any_call("CSV contains 'className' column instead of 'class_name'")
        logger.info.assert_any_call("Would you like to automatically fix this by renaming the column to 'class_name'? (y/n)")
        logger.info.assert_any_call("CSV file has been fixed. The column 'className' was renamed to 'class_name'.")
    
    def test_validate_csv_nonexistent(self, logger):
        """Test that validate_csv handles nonexistent files."""
        # Call the function with a nonexistent file
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            with patch('os.path.exists', return_value=False):
                with patch('sam_annotator.cli.csv_utils._offer_sample_creation', return_value=False) as mock_offer:
                    result = validate_csv("/nonexistent/file.csv", logger)
        
        # Check that the function returned False (invalid)
        assert result is False
        
        # Check that logger was called with appropriate error message
        logger.error.assert_any_call("Error reading CSV file: File not found")
        
        # Check that _offer_sample_creation was called
        mock_offer.assert_called_once()
    
    @pytest.mark.parametrize("choice,option,expected_output", [
        ('y', '1', True),  # User chooses to create sample at original location
        ('y', '2', "sample_classes.csv"),  # User chooses default location
        ('n', None, False)  # User declines to create sample
    ])
    def test_offer_sample_creation(self, tmp_path, logger, monkeypatch, choice, option, expected_output):
        """Test the _offer_sample_creation function with various user choices."""
        # Mock the input function to return our test values
        input_values = [choice]
        if choice == 'y':
            input_values.append(option)
        input_mock = MagicMock(side_effect=input_values)
        monkeypatch.setattr('builtins.input', input_mock)
        
        # Mock create_sample_csv to avoid actually creating files
        csv_path = os.path.join(tmp_path, "test.csv")
        with patch('sam_annotator.cli.csv_utils.create_sample_csv', return_value=True) as mock_create:
            result = _offer_sample_creation(csv_path, logger)
        
        # Check the result based on our test parameters
        assert result == expected_output
        
        # If user chose to create sample, verify create_sample_csv was called
        if choice == 'y':
            mock_create.assert_called_once()
            if option == '1':
                # Should create at original location
                mock_create.assert_called_with(csv_path, logger)
            elif option == '2':
                # Should create at default location
                mock_create.assert_called_with("sample_classes.csv", logger) 