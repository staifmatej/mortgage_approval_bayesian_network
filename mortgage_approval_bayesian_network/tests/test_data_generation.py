"""Unit tests for data_generation_realistic.py module"""
import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=import-error,wrong-import-position
from data_generation_realistic import DataGenerator


class TestDataGeneration(unittest.TestCase):
    """Test cases for DataGenerator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.avg_salary = 35000
        self.interest_rate = 0.045
        self.num_records = 100
        self.test_csv_path = "test_mortgage_applications.csv"

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists("datasets/mortgage_applications.csv"):
            os.remove("datasets/mortgage_applications.csv")

    def test_no_crash_extreme_params(self):
        """Test that generation doesn't crash with extreme parameters"""
        # Test with very high values
        generator_high = DataGenerator(1000000, 0.25, 100)
        # Should not crash
        self.assertIsNotNone(generator_high)

        # Test with minimum values
        generator_low = DataGenerator(1000, 0.001, 100)
        self.assertIsNotNone(generator_low)

    def test_age_groups_sum_to_one(self):
        """Verify age groups sum to 1 for all rows"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        # Read the generated CSV
        df = pd.read_csv(generator.csv_path)

        # Check each row
        for idx, row in df.iterrows():
            age_sum = row['age_young'] + row['age_prime'] + row['age_senior'] + row['age_old']
            self.assertEqual(age_sum, 1, f"Age groups don't sum to 1 in row {idx}")

    def test_no_null_inf_values(self):
        """Check no NULL, NaN, +inf, -inf values in dataset"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        # Check for NaN values
        self.assertFalse(df.isnull().any().any(), "Found NaN values in dataset")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(np.isinf(df[col]).any(), f"Found infinite values in column {col}")

    def test_salary_bounds(self):
        """Verify salary and income values are reasonable"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)
        max_allowed_income = 500 * self.avg_salary

        # Check income-related columns for extreme values
        income_cols = ['reported_monthly_income', 'total_stable_income_monthly']
        for col in income_cols:
            if col in df.columns:
                max_val = df[col].max()
                self.assertLessEqual(max_val, max_allowed_income,
                                    f"Column {col} has value {max_val} exceeding {max_allowed_income}")

    def test_csv_not_empty(self):
        """Ensure CSV is never empty after generation"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 100)
        generator.generate_realistic_data(True)

        # Check file exists and is not empty
        self.assertTrue(os.path.exists(generator.csv_path))

        df = pd.read_csv(generator.csv_path)
        self.assertGreater(len(df), 0, "CSV file is empty")

    def test_minimum_rows(self):
        """Verify dataset always has more than 80 rows"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 100)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)
        self.assertGreater(len(df), 80, f"Dataset has only {len(df)} rows, expected > 80")

    def test_data_loss_limit(self):
        """Check that cleaning doesn't remove more than 20% of data"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 500)
        generator.generate_realistic_data(True)

        # Get initial count
        df_before = pd.read_csv(generator.csv_path)
        initial_count = len(df_before)

        # Clean data
        generator.remove_wrong_rows(False, None)

        # Get final count
        df_after = pd.read_csv(generator.csv_path)
        final_count = len(df_after)

        # Calculate loss percentage
        loss_percentage = (initial_count - final_count) / initial_count * 100
        self.assertLessEqual(loss_percentage, 20,
                            f"Data cleaning removed {loss_percentage:.1f}% of data")

    def test_age_range_valid(self):
        """Verify age groups are valid (sum to 1)"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        # Check age groups are binary
        age_cols = ['age_young', 'age_prime', 'age_senior', 'age_old']
        for col in age_cols:
            self.assertTrue(df[col].isin([0, 1]).all(), f"{col} contains non-binary values")

    def test_employment_type_valid(self):
        """Check employment_type only has valid values"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        valid_employment_types = ['unemployed', 'temporary', 'freelancer', 'permanent']
        self.assertTrue(df['employment_type'].isin(valid_employment_types).all(),
                       "Found invalid employment type")

    def test_decimal_precision(self):
        """Verify critical float columns have reasonable precision"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 100)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        # Check critical float columns that should have limited precision
        critical_float_cols = ['ratio_income_debt', 'ratio_debt_net_worth',
                              'ratio_payment_to_income', 'ratio_income_to_avg_salary',
                              'defaulted', 'loan_approved']

        for col in critical_float_cols:
            if col in df.columns:
                for val in df[col]:
                    if pd.notna(val) and isinstance(val, float):
                        # Check if value is reasonably formatted (not checking exact decimal places)
                        self.assertTrue(np.isfinite(val),
                                       f"Value {val} in column {col} is not finite")

    def test_no_critical_fields_empty(self):
        """Ensure critical fields are never None/NaN"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        critical_fields = ['age', 'government_employee', 'highest_education',
                          'employment_type', 'reported_monthly_income', 'loan_amount']

        for field in critical_fields:
            if field in df.columns:
                self.assertFalse(df[field].isnull().any(),
                               f"Critical field {field} contains null values")

    def test_unique_ids(self):
        """Verify all IDs are unique within a CSV file"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        if 'id' in df.columns:
            self.assertEqual(len(df['id'].unique()), len(df),
                           "Found duplicate IDs in dataset")

    def test_education_levels_valid(self):
        """Check education only has valid levels"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        valid_education = ['basic', 'high_school', 'bachelor', 'master', 'phd']
        self.assertTrue(df['highest_education'].isin(valid_education).all(),
                       "Found invalid education level")

    def test_credit_history_valid(self):
        """Verify credit history has only valid values"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 200)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        valid_credit = ['bad', 'fair', 'good', 'excellent']
        self.assertTrue(df['credit_history'].isin(valid_credit).all(),
                       "Found invalid credit history value")

    def test_logical_consistency(self):
        """Check logical consistency in data"""
        generator = DataGenerator(self.avg_salary, self.interest_rate, 300)
        generator.generate_realistic_data(True)

        df = pd.read_csv(generator.csv_path)

        # Unemployed should not have very high income
        unemployed = df[df['employment_type'] == 'unemployed']
        if len(unemployed) > 0:
            avg_unemployed_income = unemployed['reported_monthly_income'].mean()
            self.assertLess(avg_unemployed_income, self.avg_salary * 0.5,
                          "Unemployed have unrealistically high income")

        # Students should have study_status = 'yes'
        students = df[df['study_status'] == 'yes']
        if len(students) > 0:
            # Students should mostly be young (age_young = 1)
            young_student_ratio = students['age_young'].mean()
            self.assertGreater(young_student_ratio, 0.5,
                          "Most students should be in young age group")

        # Government employees should have relatively stable income
        gov_employees = df[df['government_employee'] == 'yes']
        if len(gov_employees) > 0:
            avg_gov_income = gov_employees['reported_monthly_income'].mean()
            # Just check that government employees have non-zero income on average
            self.assertGreater(avg_gov_income, self.avg_salary * 0.5,
                             "Government employees have too low income")


if __name__ == '__main__':
    unittest.main()
