import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
import io

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import collect_user_info


class TestCollectUserInfo(unittest.TestCase):
    """Unit tests for collect_user_info function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Note: first two inputs are interest_rate and avg_salary but we don't test them
        self.valid_inputs_base = [
            '30',  # age
            'yes',  # government_employee
            'master',  # highest_education
            'no',  # study_status
            'permanent',  # employment_type
            '5',  # len_employment
            '1000',  # size_of_company
            '75000',  # reported_monthly_income
            '100000',  # total_existing_debt
            'yes',  # extra_net_worth
            '500000',  # investments_value
            '3000000',  # property_owned_value
            'own',  # housing_status
            'excellent'  # credit_history
        ]
        # Add interest_rate and avg_salary at the beginning for all tests
        self.valid_inputs = ['', ''] + self.valid_inputs_base
    
    @patch('builtins.print')  # Suppress print statements
    @patch('builtins.input')
    def test_valid_input_flow(self, mock_input, mock_print):
        """Test with all valid inputs"""
        mock_input.side_effect = self.valid_inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['government_employee'], 'yes')
        self.assertEqual(result['highest_education'], 'master')
        self.assertEqual(result['employment_type'], 'permanent')
        self.assertEqual(result['len_employment'], 5)
        self.assertEqual(result['size_of_company'], 1000)
        self.assertEqual(result['reported_monthly_income'], 75000.0)
        self.assertEqual(result['total_existing_debt'], 100000.0)
        self.assertEqual(result['investments_value'], 500000.0)
        self.assertEqual(result['property_owned_value'], 3000000.0)
        self.assertEqual(result['housing_status'], 'own')
        self.assertEqual(result['credit_history'], 'excellent')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_student_flow(self, mock_input, mock_print):
        """Test when user is a student"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '22',  # age
            'no',  # government_employee
            'bachelor',  # highest_education
            'yes',  # study_status (STUDENT)
            '0',  # reported_monthly_income
            '5000',  # total_existing_debt
            'no',  # extra_net_worth
            'rent',  # housing_status
            'fair'  # credit_history
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['employment_type'], 'unemployed')
        self.assertEqual(result['len_employment'], 0)
        self.assertEqual(result['size_of_company'], 0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_unemployed_flow(self, mock_input, mock_print):
        """Test when user is unemployed"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '35',  # age
            'no',  # government_employee
            'high_school',  # highest_education
            'no',  # study_status
            'unemployed',  # employment_type (UNEMPLOYED)
            '0',  # reported_monthly_income
            '50000',  # total_existing_debt
            'no',  # extra_net_worth
            'rent',  # housing_status
            'bad'  # credit_history
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['employment_type'], 'unemployed')
        self.assertEqual(result['len_employment'], 0)
        self.assertEqual(result['size_of_company'], 0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_no_investments_flow(self, mock_input, mock_print):
        """Test when user has no investments"""
        inputs = self.valid_inputs.copy()
        inputs[11] = 'no'  # extra_net_worth = no
        # Remove investment value inputs
        inputs = inputs[:12] + inputs[14:]
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['investments_value'], 0)
        self.assertEqual(result['property_owned_value'], 0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('builtins.exit')
    def test_age_validation_retry(self, mock_exit, mock_input, mock_print):
        """Test age validation with retries"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '-5',  # Invalid age
            '150',  # Invalid age
            '30',  # Valid age
            *self.valid_inputs[3:]  # Rest of inputs
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['age'], 30)
        mock_exit.assert_not_called()
    
    @patch('builtins.print')
    @patch('builtins.input') 
    @patch('builtins.exit')
    def test_age_validation_max_retries(self, mock_exit, mock_input, mock_print):
        """Test age validation exceeding max retries"""
        # Mock exit to raise SystemExit instead of actually exiting
        mock_exit.side_effect = SystemExit(1)
        
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            'abc',  # Invalid age
            '-10',  # Invalid age
            '200',  # Invalid age
            '0'  # Invalid age (4th attempt)
        ]
        mock_input.side_effect = inputs
        
        with self.assertRaises(SystemExit):
            collect_user_info()
        
        # Should exit after 3 failed attempts
        mock_exit.assert_called_with(1)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_government_employee_validation(self, mock_input, mock_print):
        """Test government employee input validation"""
        inputs = self.valid_inputs.copy()
        # Add invalid inputs
        inputs[3:3] = ['maybe', 'YES']  # Insert before 'yes'
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['government_employee'], 'yes')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_education_validation(self, mock_input, mock_print):
        """Test education input validation"""
        inputs = self.valid_inputs.copy()
        inputs[4] = '  MASTER  '  # With spaces and uppercase
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['highest_education'], 'master')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_employment_type_validation(self, mock_input, mock_print):
        """Test employment type validation"""
        inputs = self.valid_inputs.copy()
        # Insert invalid employment type
        inputs[6:6] = ['full-time']
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['employment_type'], 'permanent')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_numeric_input_validation(self, mock_input, mock_print):
        """Test numeric input validation for income"""
        inputs = self.valid_inputs.copy()
        # Add invalid income
        inputs[9:9] = ['not-a-number', '-5000']
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['reported_monthly_income'], 75000.0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_len_employment_boundary(self, mock_input, mock_print):
        """Test length of employment boundary validation"""
        inputs = self.valid_inputs.copy()
        # Try invalid values
        inputs[7:7] = ['190', '-1']
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['len_employment'], 5)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_credit_history_validation(self, mock_input, mock_print):
        """Test credit history validation"""
        inputs = self.valid_inputs.copy()
        # Add invalid credit history
        inputs[15:15] = ['amazing', 'poor']
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['credit_history'], 'excellent')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_housing_status_validation(self, mock_input, mock_print):
        """Test housing status validation"""
        inputs = self.valid_inputs.copy()
        # Add invalid housing status
        inputs[14:14] = ['living-with-parents']
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['housing_status'], 'own')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_edge_case_zero_income(self, mock_input, mock_print):
        """Test edge case with zero income"""
        inputs = self.valid_inputs.copy()
        inputs[9] = '0'  # Zero income
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['reported_monthly_income'], 0.0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_whitespace_handling(self, mock_input, mock_print):
        """Test that whitespace is properly handled"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '30',  # age
            '  yes  ',  # government_employee with spaces
            ' master ',  # highest_education with spaces
            'no',  # study_status
            'permanent  ',  # employment_type with spaces
            '5',  # len_employment
            '1000',  # size_of_company
            '75000',  # reported_monthly_income
            '100000',  # total_existing_debt
            'yes',  # extra_net_worth
            '500000',  # investments_value
            '3000000',  # property_owned_value
            '  own',  # housing_status with spaces
            'excellent  '  # credit_history with spaces
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['government_employee'], 'yes')
        self.assertEqual(result['highest_education'], 'master')
        self.assertEqual(result['employment_type'], 'permanent')
        self.assertEqual(result['housing_status'], 'own')
        self.assertEqual(result['credit_history'], 'excellent')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_case_insensitive_inputs(self, mock_input, mock_print):
        """Test that text inputs are case insensitive"""
        inputs = [
            '0.05',  # interest_rate
            '50000',  # avg_salary
            '30',  # age
            'YES',  # government_employee uppercase
            'MASTER',  # highest_education uppercase
            'NO',  # study_status uppercase
            'PERMANENT',  # employment_type uppercase
            '5',  # len_employment
            '1000',  # size_of_company
            '75000',  # reported_monthly_income
            '100000',  # total_existing_debt
            'NO',  # extra_net_worth uppercase
            'OWN',  # housing_status uppercase
            'EXCELLENT'  # credit_history uppercase
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['government_employee'], 'yes')
        self.assertEqual(result['highest_education'], 'master')
        self.assertEqual(result['employment_type'], 'permanent')
        self.assertEqual(result['housing_status'], 'own')
        self.assertEqual(result['credit_history'], 'excellent')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_float_interest_rate(self, mock_input, mock_print):
        """Test that interest rate accepts float values"""
        inputs = self.valid_inputs.copy()
        inputs[0] = '0.05'  # Float interest rate
        
        mock_input.side_effect = inputs
        
        # Should not raise ValueError
        result = collect_user_info()
        self.assertIsNotNone(result)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_special_characters_in_numeric_fields(self, mock_input, mock_print):
        """Test handling of special characters in numeric fields"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '30!@#',  # age with special chars
            '30',  # valid age
            'yes',  # government_employee
            'master',  # highest_education
            'no',  # study_status
            'permanent',  # employment_type
            '5$%^',  # len_employment with special chars
            '5',  # valid len_employment
            '1000',  # size_of_company
            '75000€',  # income with currency symbol
            '75000',  # valid income
            '100000',  # total_existing_debt
            'yes',  # extra_net_worth
            '500000',  # investments_value
            '3000000',  # property_owned_value
            'own',  # housing_status
            'excellent'  # credit_history
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['len_employment'], 5)
        self.assertEqual(result['reported_monthly_income'], 75000.0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_sql_injection_attempt(self, mock_input, mock_print):
        """Test that SQL injection attempts don't break the program"""
        inputs = self.valid_inputs.copy()
        # Insert SQL injection attempt and then valid input before the 3rd retry
        inputs[3:3] = ["yes'; DROP TABLE users; --", "invalid"]  # 2 attempts then valid
        
        mock_input.side_effect = inputs
        
        # Should handle as invalid input and continue
        result = collect_user_info()
        
        # Should eventually get valid input
        self.assertEqual(result['government_employee'], 'yes')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_very_large_numbers(self, mock_input, mock_print):
        """Test handling of very large numbers"""
        inputs = self.valid_inputs.copy()
        # Insert very large number and then valid input
        inputs[9:9] = ['999999999999999999999999999999999999', '99999999999999']  # 2 attempts then valid
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        # Should reject as too large (> 1e30)
        self.assertLessEqual(result['reported_monthly_income'], 1e30)
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_empty_string_for_required_fields(self, mock_input, mock_print):
        """Test empty strings for required fields"""
        inputs = [
            '',  # interest_rate (OK - has default)
            '',  # avg_salary (OK - has default)
            '',  # age (NOT OK - required)
            '30',  # valid age
            '',  # government_employee (NOT OK)
            'yes',  # valid government_employee
            'master',  # highest_education
            'no',  # study_status
            'permanent',  # employment_type
            '5',  # len_employment
            '1000',  # size_of_company
            '75000',  # reported_monthly_income
            '100000',  # total_existing_debt
            'yes',  # extra_net_worth
            '500000',  # investments_value
            '3000000',  # property_owned_value
            'own',  # housing_status
            'excellent'  # credit_history
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['government_employee'], 'yes')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_unicode_characters(self, mock_input, mock_print):
        """Test handling of Unicode characters"""
        inputs = self.valid_inputs.copy()
        inputs[3] = 'yěš'  # Unicode characters
        inputs.insert(4, 'yes')  # Add valid input after
        
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        self.assertEqual(result['government_employee'], 'yes')
    
    @patch('builtins.print')
    @patch('builtins.input')
    def test_negative_values_validation(self, mock_input, mock_print):
        """Test that negative values are properly rejected"""
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '30',  # age
            'yes',  # government_employee
            'master',  # highest_education
            'no',  # study_status
            'permanent',  # employment_type
            '-5',  # negative len_employment
            '5',  # valid len_employment
            '-1000',  # negative size_of_company
            '1000',  # valid size_of_company
            '-75000',  # negative income
            '75000',  # valid income
            '-100000',  # negative debt
            '100000',  # valid debt
            'yes',  # extra_net_worth
            '-500000',  # negative investments
            '500000',  # valid investments
            '-3000000',  # negative property value
            '3000000',  # valid property value
            'own',  # housing_status
            'excellent'  # credit_history
        ]
        mock_input.side_effect = inputs
        
        result = collect_user_info()
        
        # All numeric values should be positive
        self.assertGreaterEqual(result['len_employment'], 0)
        self.assertGreaterEqual(result['size_of_company'], 0)
        self.assertGreaterEqual(result['reported_monthly_income'], 0)
        self.assertGreaterEqual(result['total_existing_debt'], 0)
        self.assertGreaterEqual(result['investments_value'], 0)
        self.assertGreaterEqual(result['property_owned_value'], 0)
    
    @patch('builtins.print')
    @patch('builtins.input')
    @patch('builtins.exit')
    def test_multiple_fields_max_retries(self, mock_exit, mock_input, mock_print):
        """Test that exit is called when any field exceeds max retries"""
        mock_exit.side_effect = SystemExit(1)
        
        inputs = [
            '',  # interest_rate
            '',  # avg_salary
            '30',  # valid age
            'invalid1',  # invalid government_employee
            'invalid2',  # invalid government_employee
            'invalid3',  # invalid government_employee
            'invalid4'   # 4th attempt - should exit
        ]
        mock_input.side_effect = inputs
        
        with self.assertRaises(SystemExit):
            collect_user_info()
        
        mock_exit.assert_called_with(1)


if __name__ == '__main__':
    unittest.main()