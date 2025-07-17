import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime as dt
import os
import sys
from dotenv import load_dotenv

# Add the data directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
sys.path.append(data_dir)
sys.path.append(current_dir)

# Load environment variables for testing
load_dotenv()

class TestAutoData(unittest.TestCase):
    """Unit tests for the AutoData class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock environment variables for testing
        with patch.dict(os.environ, {
            'APCA-API-KEY-ID': 'test_key',
            'APCA-API-SECRET-KEY': 'test_secret'
        }):
            from auto_data import AutoData
            self.auto_data = AutoData(
                today_price=100.0,
                predicted_price=105.0,
                test_mae=0.5,
                test_loss=0.3,
                val_mae=0.4,
                val_loss=0.2
            )
    
    def test_init_with_valid_data(self):
        """Test AutoData initialization with valid data"""
        self.assertIsNotNone(self.auto_data.trading_client)
        self.assertEqual(self.auto_data.test_row["Test Price"], 100.0)
        self.assertEqual(self.auto_data.test_row["Predicted Value"], 105.0)
        self.assertEqual(self.auto_data.test_row["Test MAE"], 0.5)
        self.assertEqual(self.auto_data.test_row["Test Loss"], 0.3)
    
    def test_init_with_none_values(self):
        """Test AutoData initialization with None values"""
        with patch.dict(os.environ, {
            'APCA-API-KEY-ID': 'test_key',
            'APCA-API-SECRET-KEY': 'test_secret'
        }):
            from auto_data import AutoData
            auto_data = AutoData()
            
            self.assertIsNone(auto_data.test_row["Test Price"])
            self.assertIsNone(auto_data.test_row["Predicted Value"])
            self.assertIsNone(auto_data.test_row["Test Diff"])
            self.assertIsNone(auto_data.test_row["Test Change(%)"])
    
    def test_test_row_calculations(self):
        """Test that test_row calculations are correct"""
        expected_diff = 105.0 - 100.0  # 5.0
        expected_change_percent = (5.0 / 100.0) * 100  # 5.0%
        
        self.assertEqual(self.auto_data.test_row["Test Diff"], expected_diff)
        self.assertEqual(self.auto_data.test_row["Test Change(%)"], expected_change_percent)
        self.assertEqual(self.auto_data.test_row["Test Diff Sign(+/-)"], expected_diff)
    
    def test_val_row_calculations(self):
        """Test that val_row calculations are correct"""
        expected_diff = 105.0 - 100.0  # 5.0
        expected_change_percent = (5.0 / 100.0) * 100  # 5.0%
        
        self.assertEqual(self.auto_data.val_row["Val Diff"], expected_diff)
        self.assertEqual(self.auto_data.val_row["Val Change(%)"], expected_change_percent)
        self.assertEqual(self.auto_data.val_row["Val MAE"], 0.4)
        self.assertEqual(self.auto_data.val_row["Val Loss"], 0.2)
    
    def test_date_fields(self):
        """Test that date fields are properly set"""
        today_str = str(dt.datetime.now().date())
        weekday_str = str(dt.datetime.now().weekday())
        
        self.assertEqual(self.auto_data.test_row["Test Date (YYYY-MM-DD)"], today_str)
        self.assertEqual(self.auto_data.test_row["Test Day Of Week"], weekday_str)
        self.assertEqual(self.auto_data.val_row["Val Date (YYYY-MM-DD)"], today_str)
        self.assertEqual(self.auto_data.val_row["Val Day Of Week"], weekday_str)
    
    def test_notes_row_initialization(self):
        """Test that notes_row is properly initialized"""
        expected_notes = {
            "Post-Break?(T/F)": None,
            'Weekend Break?(T/F)': None,
            'Else Break(T/F)': None,
            'Model Ver': None,
            'Notes': None
        }
        self.assertEqual(self.auto_data.notes_row, expected_notes)
    
    @patch('subprocess.run')
    def test_auto_data_method(self, mock_subprocess):
        """Test the auto_data method calls subprocess correctly"""
        # Mock the infinite loop to prevent hanging
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with patch('schedule.run_pending'):
                try:
                    self.auto_data.auto_data()
                except KeyboardInterrupt:
                    pass  # Expected to break out of the loop
        
        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_with(["python", "main.py"])
    
    def test_format_data_method(self):
        """Test the format_data method (currently a placeholder)"""
        # This method is currently just a pass statement
        result = self.auto_data.format_data()
        self.assertIsNone(result)
    
    def test_delivery_method(self):
        """Test the delivery method (currently a placeholder)"""
        # This method is currently just a pass statement
        result = self.auto_data.delivery()
        self.assertIsNone(result)
    
    def test_plot_data_method(self):
        """Test the plot_data method (currently a placeholder)"""
        # This method is currently just a pass statement
        result = self.auto_data.plot_data()
        self.assertIsNone(result)
    
    def test_trigger_run_method(self):
        """Test the trigger_run method (currently a placeholder)"""
        # This method is currently just a pass statement
        result = self.auto_data.trigger_run()
        self.assertIsNone(result)
    
    @patch('logging.info')
    def test_logging_method(self, mock_logging):
        """Test the logging method outputs correct messages"""
        self.auto_data.logging()
        
        # Verify logging.info was called 4 times with the correct message
        self.assertEqual(mock_logging.call_count, 4)
        mock_logging.assert_called_with("Running scheduled task...")
    
    def test_negative_price_difference(self):
        """Test calculations with negative price difference"""
        with patch.dict(os.environ, {
            'APCA-API-KEY-ID': 'test_key',
            'APCA-API-SECRET-KEY': 'test_secret'
        }):
            from auto_data import AutoData
            auto_data = AutoData(
                today_price=100.0,
                predicted_price=95.0,  # Lower than today's price
                test_mae=0.5,
                test_loss=0.3
            )
            
            expected_diff = 95.0 - 100.0  # -5.0
            expected_change_percent = (-5.0 / 100.0) * 100  # -5.0%
            
            self.assertEqual(auto_data.test_row["Test Diff"], expected_diff)
            self.assertEqual(auto_data.test_row["Test Change(%)"], expected_change_percent)
            self.assertEqual(auto_data.test_row["Test Diff Sign(+/-)"], expected_diff)

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAutoData)
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}") 