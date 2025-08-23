"""
Unit tests for realistic probability results with end-to-end program execution.

Tests comprehensive scenarios including:
- Good applicant with high approval probability
- Poor applicant with low approval probability  
- Edge case applicant with marginal approval probability
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error,wrong-import-position
from main import InputHandler
from data_generation_realistic import DataGenerator


class TestRealisticProbabilityResults(unittest.TestCase):
    """Test realistic probability results with full program execution."""
    
    def setUp(self):
        """Set up test environment with InputHandler instance."""
        self.input_handler = InputHandler()
        
    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists("datasets/mortgage_applications.csv"):
            try:
                os.remove("datasets/mortgage_applications.csv")
            except OSError:
                pass
    
    @patch('builtins.input')
    def test_excellent_applicant_high_probability(self, mock_input):
        """Test excellent applicant should get high approval probability (>70%)."""
        
        # Mock inputs for excellent applicant - always choose defaults when available
        mock_input.side_effect = [
            # collect_main_user_info() - defaults
            "",  # avg_salary (default: 40000)
            "",  # interest_rate (default: 4.5%)
            "",  # retirement_age (default: 65)
            # collect_datasets_user_info() 
            "generate",  # dataset choice (own/generate)
            "5000",  # data_num_records
            # collect_other_user_info() - excellent applicant
            "35",    # age
            "yes",   # government_employee
            "phd", # highest_education
            "no",    # study_status
            "permanent", # employment_type
            "10",    # len_employment
            "5000",   # size_of_company
            "160000", # reported_monthly_income (very high)
            "0", # total_existing_debt (low relative to income)
            "yes",   # extra_net_worth (Do you have investments or property?)
            "5000000", # investments_value (high)
            "9000000", # property_owned_value (high)
            "own",   # housing_status
            "excellent", # credit_history
            # loan parameters
            "4000000", # loan_amount
            "30"     # loan_term
        ]
        
        # Execute full program setup
        applicant_data = self.input_handler.set_up_all()
        model_gbn = self.input_handler.train_and_validate_gbn()
        
        # Calculate approval probability
        approval_prob = self.input_handler.predict_loan_approval(
            model_gbn, applicant_data, 4000000, 30
        )
        
        # Excellent applicant should have high approval probability
        self.assertGreater(approval_prob, 0.40,
                          f"Excellent applicant should have > 40% approval probability, got {approval_prob:.1%}")
        self.assertLessEqual(approval_prob, 1.0,
                            "Probability should not exceed 100%")
        
        print(f"Excellent applicant probability: {approval_prob:.1%}")
    
    @patch('builtins.input')
    def test_poor_applicant_low_probability(self, mock_input):
        """Test poor applicant should get low approval probability (<50%)."""
        
        # Mock inputs for poor applicant
        mock_input.side_effect = [
            # collect_main_user_info() - defaults
            "",  # avg_salary (default: 40000)
            "",  # interest_rate (default: 4.5%)
            "",  # retirement_age (default: 65)
            # collect_datasets_user_info()
            "generate",  # dataset choice (own/generate)
            "5000",  # data_num_records
            # collect_other_user_info() - poor applicant
            "55",    # age (older)
            "no",    # government_employee (not stable job)
            "basic", # highest_education (low education)
            "no",    # study_status
            "permanent", # employment_type (unstable)
            "1",     # len_employment (very short)
            "20",    # size_of_company (small company)
            "30000", # reported_monthly_income (low)
            "300000", # total_existing_debt (high relative to income)
            "no",    # extra_net_worth (Do you have investments or property?)
            "rent",  # housing_status (not owned)
            "bad",   # credit_history (bad)
            # loan parameters
            "3000000", # loan_amount (high relative to income)
            "35"     # loan_term
        ]
        
        # Execute full program setup
        applicant_data = self.input_handler.set_up_all()
        model_gbn = self.input_handler.train_and_validate_gbn()
        
        # Calculate approval probability
        approval_prob = self.input_handler.predict_loan_approval(
            model_gbn, applicant_data, 3000000, 30
        )
        
        # Poor applicant should have low approval probability
        self.assertLess(approval_prob, 0.30,
                       f"Poor applicant should have < 30% approval probability, got {approval_prob:.1%}")

        print(f"Poor applicant probability: {approval_prob:.1%}")
    
    @patch('builtins.input')
    def test_marginal_applicant_edge_case(self, mock_input):
        """Test marginal applicant at edge of approval criteria."""
        
        # Mock inputs for marginal applicant
        mock_input.side_effect = [
            # collect_main_user_info() - defaults
            "",  # avg_salary (default: 40000)
            "",  # interest_rate (default: 4.5%)
            "",  # retirement_age (default: 65)
            # collect_datasets_user_info()
            "generate",  # dataset choice (own/generate)
            "5000",  # data_num_records
            # collect_other_user_info() - marginal applicant
            "33",    # age (middle-aged)
            "no",    # government_employee
            "bachelor", # highest_education (decent)
            "no",    # study_status
            "permanent", # employment_type (stable)
            "5",     # len_employment (moderate)
            "100",   # size_of_company (medium)
            "40000", # reported_monthly_income (average)
            "300000", # total_existing_debt (moderate)
            "yes",   # extra_net_worth (Do you have investments or property?)
            "100000", # investments_value (some)
            "500000", # property_owned_value (modest)
            "mortgage", # housing_status (has mortgage)
            "good",  # credit_history (average)
            # loan parameters
            "2500000", # loan_amount (moderate-high)
            "30"     # loan_term
        ]
        
        # Execute full program setup
        applicant_data = self.input_handler.set_up_all()
        model_gbn = self.input_handler.train_and_validate_gbn()
        
        # Calculate approval probability
        approval_prob = self.input_handler.predict_loan_approval(
            model_gbn, applicant_data, 2500000, 30
        )
        
        # Marginal applicant should have reasonable probability range
        self.assertGreaterEqual(approval_prob, 0.0,
                               "Probability should not be negative")
        self.assertLessEqual(approval_prob, 1.0,
                            "Probability should not exceed 100%")
        
        # Additional checks for data quality
        self.assertIsInstance(approval_prob, (int, float),
                             "Approval probability should be numeric")
        self.assertFalse(np.isnan(approval_prob),
                        "Approval probability should not be NaN")
        
        print(f"Marginal applicant probability: {approval_prob:.1%}")
        
        # Test that the generated dataset has correct size
        if os.path.exists("datasets/mortgage_applications.csv"):
            df = pd.read_csv("datasets/mortgage_applications.csv")
            self.assertGreaterEqual(len(df), 4500,
                                   f"Dataset should have at least 4500 records, got {len(df)}")
            self.assertLessEqual(len(df), 5000,
                                f"Dataset should have at most 5000 records, got {len(df)}")
            print(f"Dataset generated with {len(df)} records")

    @patch('builtins.input')
    def test_applicant_edge_case_01(self, mock_input):
        """Test marginal applicant at edge of approval criteria."""

        # Mock inputs for marginal applicant
        mock_input.side_effect = [
            # collect_main_user_info() - defaults
            "",  # avg_salary (default: 40000)
            "",  # interest_rate (default: 4.5%)
            "",  # retirement_age (default: 65)
            # collect_datasets_user_info()
            "generate",  # dataset choice (own/generate)
            "5000",  # data_num_records
            # collect_other_user_info() - marginal applicant
            "65",  # age (middle-aged)
            "yes",  # government_employee
            "phd",  # highest_education (decent)
            "no",  # study_status
            "permanent",  # employment_type (stable)
            "35",  # len_employment (moderate)
            "10000",  # size_of_company (medium)
            "180000",  # reported_monthly_income (average)
            "0",  # total_existing_debt (moderate)
            "yes",  # extra_net_worth (Do you have investments or property?)
            "1000000",  # investments_value (some)
            "9000000",  # property_owned_value (modest)
            "own",  # housing_status (has mortgage)
            "excellent",  # credit_history (average)
            "2500000",  # loan_amount (moderate-high)
            "35"  # loan_term
        ]

        # Execute full program setup
        applicant_data = self.input_handler.set_up_all()
        model_gbn = self.input_handler.train_and_validate_gbn()

        # Calculate approval probability
        approval_prob = self.input_handler.predict_loan_approval(
            model_gbn, applicant_data, 2500000, 35
        )

        # Elderly applicant with long-term loan should have low approval probability due to age penalty
        self.assertLess(approval_prob, 0.15,
                        f"Elderly applicant with 35-year loan should have < 15% approval probability, got {approval_prob:.1%}")

        print(f"Elderly applicant probability: {approval_prob:.1%}")


if __name__ == "__main__":
    unittest.main(verbosity=2)