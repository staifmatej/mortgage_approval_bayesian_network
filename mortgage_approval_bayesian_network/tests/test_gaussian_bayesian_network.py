"""Unit tests for gaussian_bayesian_network.py module"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error,wrong-import-position
from pgmpy.models import LinearGaussianBayesianNetwork
from gaussian_bayesian_network import GaussianBayesianNetwork


class TestGaussianBayesianNetwork(unittest.TestCase):
    """Test cases for GaussianBayesianNetwork class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create better sample data that avoids singular matrix issues
        np.random.seed(42)
        n_samples = 100  # More samples for better matrix conditioning

        # Generate more realistic and varied data
        ages = np.random.randint(25, 60, n_samples)  # Random ages between 25-60
        loan_terms = np.random.randint(10, 25, n_samples)
        self.sample_data = pd.DataFrame({
            'government_employee': np.random.choice([0, 1], n_samples),
            'age': ages,
            'age_young': np.zeros(n_samples),
            'age_prime': np.zeros(n_samples),
            'age_senior': np.zeros(n_samples),
            'age_old': np.zeros(n_samples),
            'highest_education': np.random.randint(0, 5, n_samples),
            'employment_type': np.random.randint(0, 3, n_samples),
            'len_employment': np.random.randint(0, 20, n_samples),
            'size_of_company': np.random.randint(10, 1000, n_samples),
            'reported_monthly_income': np.random.randint(20000, 80000, n_samples),
            'total_existing_debt': np.random.randint(0, 50000, n_samples),
            'investments_value': np.random.randint(0, 200000, n_samples),
            'property_owned_value': np.random.randint(0, 3000000, n_samples),
            'housing_status': np.random.randint(0, 3, n_samples),
            'credit_history': np.random.randint(0, 4, n_samples),
            'loan_amount': np.random.randint(100000, 2000000, n_samples),
            'loan_term': loan_terms,
            'stability_income': np.random.uniform(30, 90, n_samples),
            'total_stable_income_monthly': np.random.uniform(25000, 75000, n_samples),
            'ratio_income_debt': np.random.uniform(1.0, 10.0, n_samples),
            'core_net_worth': np.random.uniform(200000, 3000000, n_samples),
            'ratio_debt_net_worth': np.random.uniform(0.01, 0.3, n_samples),
            'avg_salary': np.full(n_samples, 35000),
            'ratio_income_to_avg_salary': np.random.uniform(0.7, 2.0, n_samples),
            'monthly_payment': np.random.uniform(3000, 15000, n_samples),
            'ratio_payment_to_income': np.random.uniform(0.1, 0.4, n_samples),
            'mortgage_end_age': ages + loan_terms,
            'years_of_mortgage_after_retirement': (ages + loan_terms) - 65,
            'defaulted': np.random.uniform(5, 50, n_samples),
            'loan_approved': np.random.uniform(0.2, 0.9, n_samples)
        })

        # Set age groups properly (one hot encoding)
        for i in range(n_samples):
            age_cat = np.random.choice([0, 1, 2, 3])
            if age_cat == 0:
                self.sample_data.loc[i, 'age_young'] = 1
            elif age_cat == 1:
                self.sample_data.loc[i, 'age_prime'] = 1
            elif age_cat == 2:
                self.sample_data.loc[i, 'age_senior'] = 1
            else:
                self.sample_data.loc[i, 'age_old'] = 1

    def test_network_structure(self):
        """Test that the network has the correct structure"""
        with patch('gaussian_bayesian_network.LoanDataLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_data.return_value = self.sample_data
            mock_loader.return_value = mock_loader_instance

            gbn = GaussianBayesianNetwork()

        # Check that model is LinearGaussianBayesianNetwork
        self.assertIsInstance(gbn.loan_approval_model, LinearGaussianBayesianNetwork)

        # Check number of nodes
        expected_nodes = [
            'government_employee', 'age', 'highest_education', 'employment_type', 
            'len_employment', 'size_of_company', 'stability_income', 
            'reported_monthly_income', 'total_stable_income_monthly',
            'total_existing_debt', 'ratio_income_debt', 'investments_value',
            'property_owned_value', 'core_net_worth', 'ratio_debt_net_worth',
            'avg_salary', 'ratio_income_to_avg_salary', 'housing_status',
            'credit_history', 'defaulted', 'loan_amount', 'loan_term',
            'monthly_payment', 'ratio_payment_to_income', 'mortgage_end_age',
            'years_of_mortgage_after_retirement', 'loan_approved'
        ]

        model_nodes = list(gbn.loan_approval_model.nodes())
        self.assertEqual(len(model_nodes), len(expected_nodes))
        for node in expected_nodes:
            self.assertIn(node, model_nodes)

    def test_network_edges(self):
        """Test that the network has the correct edges"""
        with patch('gaussian_bayesian_network.LoanDataLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_data.return_value = self.sample_data
            mock_loader.return_value = mock_loader_instance

            gbn = GaussianBayesianNetwork()

        # Check some key edges
        edges = list(gbn.loan_approval_model.edges())

        # Test edges to stability_income
        self.assertIn(('government_employee', 'stability_income'), edges)
        self.assertIn(('age', 'stability_income'), edges)
        self.assertIn(('highest_education', 'stability_income'), edges)

        # Test edges to loan_approved
        self.assertIn(('ratio_payment_to_income', 'loan_approved'), edges)
        self.assertIn(('defaulted', 'loan_approved'), edges)
        self.assertIn(('credit_history', 'loan_approved'), edges)

    def test_cpd_parameters(self):
        """Test that CPDs are properly fit after training"""
        with patch('gaussian_bayesian_network.LoanDataLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_data.return_value = self.sample_data
            mock_loader_instance.get_all_data_numeric.return_value = self.sample_data
            mock_loader.return_value = mock_loader_instance

            gbn = GaussianBayesianNetwork()
            gbn.train_model()

            # Check that CPDs exist for all nodes
            for node in gbn.loan_approval_model.nodes():
                cpd = gbn.loan_approval_model.get_cpds(node)
                self.assertIsNotNone(cpd, f"CPD for node {node} should not be None")




    def test_model_validation(self):
        """Test model validation using check_model"""
        with patch('gaussian_bayesian_network.LoanDataLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_data.return_value = self.sample_data
            mock_loader_instance.get_all_data_numeric.return_value = self.sample_data
            mock_loader.return_value = mock_loader_instance

            gbn = GaussianBayesianNetwork()

            # check_model_gbn should not raise any exceptions
            try:
                gbn.check_model_gbn()
                model_valid = True
            except (ValueError, KeyError, AssertionError):
                model_valid = False

            self.assertTrue(model_valid, "Model validation should pass")


    def test_data_quality_check(self):
        """Test data quality checking functionality"""
        # Create data with extreme values
        extreme_data = self.sample_data.copy()
        extreme_data.loc[0, 'reported_monthly_income'] = 1e10  # Extreme value

        with patch('gaussian_bayesian_network.LoanDataLoader') as mock_loader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_data.return_value = extreme_data
            mock_loader_instance.get_all_data_numeric.return_value = extreme_data
            mock_loader.return_value = mock_loader_instance

            with patch('builtins.print') as mock_print:
                gbn = GaussianBayesianNetwork()
                gbn.check_csv_data_quality()

                # Check that warning was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                warning_found = any('Numerical instability' in call for call in print_calls)
                self.assertTrue(warning_found, "Should warn about numerical instability")








if __name__ == '__main__':
    unittest.main()
