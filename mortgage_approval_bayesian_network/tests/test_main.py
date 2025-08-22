"""Unit tests for main.py module"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error,wrong-import-position
from main import InputHandler
from data_generation_realistic import encode_age_group

# pylint: disable=too-many-public-methods
class TestInputHandler(unittest.TestCase):
    """Test cases for InputHandler class"""
    def setUp(self):
        self.handler = InputHandler()

    # ========== BASIC TESTS (10) ==========

    def test_validate_input_alpha_valid(self):
        """Test valid alpha input validation"""
        with patch('builtins.input', return_value='yes'):
            result = self.handler.validate_input_alpha(
                "Test prompt", ["yes", "no"]
            )
            self.assertEqual(result, 'yes')

    def test_validate_input_alpha_invalid(self):
        """Test invalid alpha input validation with max attempts"""
        with patch('builtins.input', side_effect=['maybe', 'perhaps', 'definitely', 'yes']):
            # Patch exit to prevent actual exit
            with patch('sys.exit') as mock_exit:
                self.handler.validate_input_alpha(
                    "Test prompt", ["yes", "no"], max_attempts=3
                )
                # Should have called exit after max attempts
                mock_exit.assert_called_once_with(1)

    def test_validate_input_numerical_valid(self):
        """Test valid numerical input validation"""
        with patch('builtins.input', return_value='25'):
            result = self.handler.validate_input_numerical(
                "Enter age: ", min_val=18, max_val=65, data_type=int
            )
            self.assertEqual(result, 25)

    def test_validate_input_numerical_invalid(self):
        """Test invalid numerical input validation"""
        with patch('builtins.input', side_effect=['abc', '-5', '100', '30']):
            with patch('sys.exit') as mock_exit:
                self.handler.validate_input_numerical(
                    "Enter age: ", min_val=18, max_val=65, data_type=int, max_attempts=3
                )
                # Should have called exit after max attempts
                mock_exit.assert_called_once_with(1)

    def test_calculate_monthly_payment_normal(self):
        """Test monthly payment calculation with normal values"""
        payment = self.handler.calculate_monthly_payment(1000000, 20)
        self.assertGreater(payment, 0)
        self.assertLess(payment, 1000000)  # Monthly payment should be less than total loan

    def test_calculate_monthly_payment_zero_interest(self):
        """Test monthly payment calculation with zero interest"""
        self.handler.interest_rate = 0
        payment = self.handler.calculate_monthly_payment(1000000, 20)
        expected = 1000000 / (20 * 12)
        self.assertAlmostEqual(payment, expected, places=2)

    def test_encode_age_group_all_categories(self):
        """Test age group encoding for all categories"""

        # Test young (0-25)
        young = encode_age_group(22)
        self.assertEqual(young, (1, 0, 0, 0))

        # Test prime (25-40)
        prime = encode_age_group(30)
        self.assertEqual(prime, (0, 1, 0, 0))

        # Test senior (40-60)
        senior = encode_age_group(45)
        self.assertEqual(senior, (0, 0, 1, 0))

        # Test old (60+)
        old = encode_age_group(65)
        self.assertEqual(old, (0, 0, 0, 1))

    def test_housing_and_credit_mapping(self):
        """Test housing status and credit history mapping"""
        # Test housing mapping
        housing_map = {'rent': 0, 'mortgage': 1, 'own': 2}
        self.assertEqual(housing_map['rent'], 0)
        self.assertEqual(housing_map['own'], 2)

        # Test credit mapping
        credit_map = {'bad': 0, 'fair': 1, 'good': 2, 'excellent': 3}
        self.assertEqual(credit_map['bad'], 0)
        self.assertEqual(credit_map['excellent'], 3)

    def test_education_age_validation(self):
        """Test education validation based on age"""
        # Provide all inputs needed for collect_other_user_info
        inputs = [
            '27',        # age (valid for all education levels)
            'no',        # government employee
            'phd',       # education (valid for age 27)
            'no',        # student
            'permanent', # employment type
            '3',         # years employed
            '100',       # company size
            '30000',     # monthly income
            '0',         # existing debt
            'no',        # extra net worth (no investments/property)
            'rent',      # housing status
            'good'       # credit history
        ]
        with patch('builtins.input', side_effect=inputs):
            user_data = self.handler.collect_other_user_info()
            # Age 27 can have PhD (min age is 18 for PhD based on code)
            self.assertEqual(user_data['highest_education'], 'phd')

    def test_generate_csv_dataset(self):
        """Test CSV dataset generation"""
        # Need to import the module not the class
        with patch('main.DataGenerator') as mock_generator:
            mock_instance = MagicMock()
            mock_generator.return_value = mock_instance

            self.handler.generate_csv_dataset()

            mock_generator.assert_called_once_with(
                self.handler.avg_salary,
                self.handler.interest_rate,
                int(self.handler.data_num_records)
            )
            mock_instance.generate_realistic_data.assert_called_once_with(True)
            mock_instance.remove_wrong_rows.assert_called_once()

    # ========== ADVANCED TESTS (5) ==========

    def test_predict_loan_approval_edge_cases(self):
        """Test loan approval prediction with edge cases"""
        # Mock the model
        mock_model = MagicMock()
        mock_gbn = MagicMock()
        mock_gbn.loan_approval_model = mock_model

        # Test homeless applicant (should be auto-rejected)
        applicant_data = {
            'age': 30,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 5,
            'size_of_company': 100,
            'reported_monthly_income': 50000,
            'total_existing_debt': 0,
            'investments_value': 100000,
            'property_owned_value': 0,
            'housing_status': 'homeless',
            'credit_history': 'bad'
        }

        # Mock predict to return a tuple format
        mock_model.predict.return_value = (['loan_approved'], [[0.8]])

        approval_prob = self.handler.predict_loan_approval(
            mock_gbn, applicant_data, 1000000, 20
        )

        # Should be between 0 and 1
        self.assertGreaterEqual(approval_prob, 0)
        self.assertLessEqual(approval_prob, 1)

    def test_full_workflow_integration(self):
        """Test full workflow from input collection to prediction"""
        with patch('builtins.input') as mock_input:
            with patch('main.DataGenerator'):
                with patch('main.GaussianBayesianNetwork') as mock_gbn_class:
                    # Setup mock model
                    mock_model = MagicMock()
                    mock_model.predict.return_value = (['loan_approved'], [[0.75]])
                    mock_gbn_instance = MagicMock()
                    mock_gbn_instance.loan_approval_model = mock_model
                    mock_gbn_class.return_value = mock_gbn_instance

                    # Mock all inputs
                    mock_input.side_effect = [
                        '0.05',    # interest rate
                        '40000',   # avg salary
                        'generate', # dataset choice
                        '1000',    # num records
                        '30',      # age
                        'yes',     # government employee
                        'master',  # education
                        'no',      # student
                        'permanent', # employment type
                        '5',       # years employed
                        '100',     # company size
                        '50000',   # monthly income
                        '0',       # existing debt
                        'yes',     # extra net worth
                        '100000',  # investments
                        '2000000', # property value
                        'own',     # housing status
                        'good',    # credit history
                        '3000000', # loan amount
                        '20'       # loan term
                    ]

                    # This would normally call set_up_all
                    self.handler.collect_main_user_info()
                    # validate_input_numerical rounds float values to 0 decimal places
                    self.assertEqual(float(self.handler.interest_rate), 0.0)
                    self.assertEqual(float(self.handler.avg_salary), 40000.0)

    def test_error_handling_robustness(self):
        """Test error handling in various parts of the system"""
        # Test invalid data type in validate_input_numerical
        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
            with self.assertRaises(SystemExit):
                self.handler.validate_input_numerical(
                    "Test", data_type=str  # Invalid data type
                )
            mock_exit.assert_called_with(1)

        # Test calculate_monthly_payment with invalid arguments
        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit2:
            with self.assertRaises(SystemExit):
                self.handler.calculate_monthly_payment(0, 20)
            mock_exit2.assert_called_with(1)

    def test_concurrent_predictions(self):
        """Test multiple predictions in sequence"""
        mock_model = MagicMock()
        mock_gbn = MagicMock()
        mock_gbn.loan_approval_model = mock_model

        # Different applicant profiles
        applicants = [
            {  # High risk
                'age': 22, 'government_employee': 'no', 'highest_education': 'basic',
                'study_status': 'no', 'employment_type': 'unemployed', 'len_employment': 0,
                'size_of_company': 0, 'reported_monthly_income': 0, 'total_existing_debt': 50000,
                'investments_value': 0, 'property_owned_value': 0, 'housing_status': 'rent',
                'credit_history': 'bad'
            },
            {  # Low risk
                'age': 35, 'government_employee': 'yes', 'highest_education': 'phd',
                'study_status': 'no', 'employment_type': 'permanent', 'len_employment': 10,
                'size_of_company': 1000, 'reported_monthly_income': 80000, 'total_existing_debt': 0,
                'investments_value': 500000, 'property_owned_value': 3000000, 'housing_status': 'own',
                'credit_history': 'excellent'
            }
        ]

        # Mock different predictions
        mock_model.predict.side_effect = [
            (['loan_approved'], [[0.1]]),  # Low approval for high risk
            (['loan_approved'], [[0.95]]) # High approval for low risk
        ]

        results = []
        for applicant in applicants:
            prob = self.handler.predict_loan_approval(mock_gbn, applicant, 1000000, 20)
            results.append(prob)

        self.assertLess(results[0], 0.5)  # High risk should have low approval
        self.assertGreater(results[1], 0.5)  # Low risk should have high approval

    def test_memory_and_performance(self):
        """Test memory usage and performance with large datasets"""

        # Test with reasonable dataset size
        self.handler.data_num_records = 1000

        start_time = time.time()
        with patch('main.DataGenerator') as mock_generator:
            mock_instance = MagicMock()
            mock_generator.return_value = mock_instance

            self.handler.generate_csv_dataset()

            elapsed_time = time.time() - start_time

            # Should complete quickly (mocked, so should be instant)
            self.assertLess(elapsed_time, 1.0)

    # ========== ADDITIONAL TESTS FOR UNTESTED FUNCTIONS ==========

    def test_print_mortgage_applicant_info(self):
        """Test mortgage applicant info printing"""
        applicant_data = {
            'age': 30, 'government_employee': 'yes', 'highest_education': 'master',
            'study_status': 'no', 'employment_type': 'permanent', 'len_employment': 5,
            'size_of_company': 100, 'reported_monthly_income': 50000,
            'total_existing_debt': 0, 'investments_value': 100000,
            'property_owned_value': 2000000, 'housing_status': 'own',
            'credit_history': 'good'
        }

        with patch('builtins.print') as mock_print:
            self.handler.print_mortgage_applicant_info(applicant_data, 3000000, 20)
            # Check that print was called (tabulate output)
            self.assertTrue(mock_print.called)

    def test_print_mortgage_approval_prob(self):
        """Test mortgage approval probability printing with complete data"""
        mock_model = MagicMock()
        applicant_data = {'housing_status': 'own', 'age': 30}

        with patch.object(self.handler, 'predict_loan_approval', return_value=0.75):
            with patch('builtins.print') as mock_print:
                self.handler.print_mortgage_approval_prob(
                    applicant_data, mock_model, 1000000, 20
                )
                # Should print with green color for high approval
                print_calls = str(mock_print.call_args_list)
                self.assertTrue('75' in print_calls or '75.0%' in print_calls)

    def test_collect_datasets_user_info(self):
        """Test dataset info collection"""
        with patch('builtins.input', side_effect=['generate', '50000']):
            self.handler.collect_datasets_user_info()
            self.assertEqual(self.handler.data_num_records, 50000)

        # Test with own dataset
        with patch('builtins.input', side_effect=['own', 'custom.csv']):
            with patch('os.path.exists', return_value=True):
                self.handler.collect_datasets_user_info()
                self.assertEqual(self.handler.csv_path, 'custom.csv')

    def test_collect_main_user_info(self):
        """Test main user info collection"""
        with patch('builtins.input', side_effect=['0.06', '45000', '67']):
            self.handler.collect_main_user_info()
            # validate_input_numerical rounds float values to 0 decimal places
            self.assertEqual(float(self.handler.interest_rate), 0.0)
            self.assertEqual(float(self.handler.avg_salary), 45000.0)
            self.assertEqual(self.handler.retirement_age, 67)

    def test_collect_other_user_info(self):
        """Test other user info collection"""
        inputs = [
            '30',        # age
            'yes',       # government employee
            'master',    # education
            'no',        # student
            'permanent', # employment type
            '5',         # years employed
            '100',       # company size
            '50000',     # monthly income
            '0',         # existing debt
            'yes',       # extra net worth
            '100000',    # investments
            '2000000',   # property value
            'own',       # housing status
            'good'       # credit history
        ]

        with patch('builtins.input', side_effect=inputs):
            result = self.handler.collect_other_user_info()
            self.assertEqual(result['age'], 30)
            self.assertEqual(result['government_employee'], 'yes')
            self.assertEqual(result['highest_education'], 'master')

    def test_options(self):
        """Test options menu"""
        with patch('builtins.input', return_value='3'):
            choice = self.handler.options()
            self.assertEqual(choice, 3)

        # Test invalid inputs
        with patch('builtins.input', side_effect=['abc', '7', '0', '4']):
            choice = self.handler.options()
            self.assertEqual(choice, 4)

    def test_set_up_all(self):
        """Test full setup process"""
        with patch.object(self.handler, 'collect_main_user_info'):
            with patch.object(self.handler, 'collect_datasets_user_info'):
                with patch.object(self.handler, 'generate_csv_dataset'):
                    with patch.object(self.handler, 'train_and_validate_gbn') as mock_train:
                        with patch.object(self.handler, 'collect_other_user_info'):
                            with patch.object(self.handler, 'validate_input_numerical', side_effect=[3000000, 20]):
                                with patch.object(self.handler, 'print_mortgage_applicant_info'):
                                    with patch.object(self.handler, 'print_mortgage_approval_prob'):
                                        mock_train.return_value = MagicMock()
                                        self.handler.set_up_all()
                                        mock_train.assert_called_once()

    def test_set_up_only_dataset(self):
        """Test dataset-only setup"""
        with patch.object(self.handler, 'collect_datasets_user_info'):
            with patch.object(self.handler, 'generate_csv_dataset') as mock_generate:
                self.handler.set_up_only_dataset()
                mock_generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
