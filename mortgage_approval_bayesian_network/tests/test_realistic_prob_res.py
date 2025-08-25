"""Test realistic probability results for mortgage approval scenarios."""
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import InputHandler  # pylint: disable=wrong-import-position,import-error


class TestRealisticProbResults(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Test class for realistic mortgage approval probability scenarios."""

    def setUp(self):
        self.handler = InputHandler()
        self.handler.csv_path = "datasets/mortgage_applications.csv"
        self.handler.avg_salary = 35000
        self.handler.interest_rate = 0.05
        self.handler.retirement_age = 65
        self.handler.data_num_records = 100000

        if not os.path.exists(self.handler.csv_path):
            self.handler.generate_csv_dataset()

    def test_low_probability_applicant_01(self):
        """Test low probability mortgage applicant scenario 01."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 65,
            'government_employee': 'no',
            'highest_education': 'basic',
            'study_status': 'no',
            'employment_type': 'temporary',
            'len_employment': 1,
            'size_of_company': 10,
            'reported_monthly_income': 30000,
            'total_existing_debt': 500000,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'bad'
        }

        loan_amount = 3000000
        loan_term = 25
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLess(probability, 0.3, f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0,f"Probability must be >= 0 but got: {probability:.3f}")

    def test_high_probability_applicant_01(self):
        """Test high probability mortgage applicant scenario 01."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 1000,
            'reported_monthly_income': 80000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 1500000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.9,f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,f"Probability must be <= 1 but got: {probability:.3f}")

    def test_high_probability_applicant_02(self):
        """Test high probability mortgage applicant scenario 02."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 1000,
            'reported_monthly_income': 80000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 3000000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.7,f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0, f"Probability must be <= 1 but got: {probability:.3f}")


    def test_high_probability_applicant_03(self):
        """Test high probability mortgage applicant scenario 03."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 1000,
            'reported_monthly_income': 80000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 6000000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.7,f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,f"Probability must be <= 1 but got: {probability:.3f}")


    def test_low_probability_applicant_02(self):
        """Test low probability mortgage applicant scenario 02."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 1000,
            'reported_monthly_income': 80000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 15000000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.35,f"Expected low probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,f"Probability must be <= 1 but got: {probability:.3f}")


    def test_low_probability_applicant_03(self):
        """Test low probability mortgage applicant scenario 03."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 65,
            'government_employee': 'yes',
            'highest_education': 'phd',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 34,
            'size_of_company': 10000,
            'reported_monthly_income': 200000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 8000000,
            'housing_status': 'own',
            'credit_history': 'excellent'
        }

        loan_amount = 1500000
        loan_term = 35
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.15,f"Expected low probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,f"Probability must be <= 1 but got: {probability:.3f}")

    def test_low_probability_applicant_04(self):
        """Test low probability mortgage applicant scenario 04."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 18,
            'government_employee': 'no',
            'highest_education': 'basic',
            'study_status': 'yes',
            'employment_type': 'temporary',
            'len_employment': 0,
            'size_of_company': 100,
            'reported_monthly_income': 2400,
            'total_existing_debt': 0,
            'investments_value': 500,
            'property_owned_value': 0,
            'housing_status': 'own',
            'credit_history': 'fair'
        }

        loan_amount = 300000
        loan_term = 25
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLess(probability, 0.3,f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0,f"Probability must be >= 0 but got: {probability:.3f}")


    # Teacher in High School.
    def test_high_probability_applicant_04(self):
        """Test high probability mortgage applicant scenario 04."""
        model_gbn = self.handler.train_and_validate_gbn()
        high_prob_applicant = {
            'age': 50,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 19,
            'reported_monthly_income': 39000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 2000000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, high_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.7,f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,f"Probability must be <= 1 but got: {probability:.3f}")


    def test_low_probability_applicant_05(self):
        """Test low probability mortgage applicant scenario 05."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 50,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 19,
            'reported_monthly_income': 39000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 5000000
        loan_term = 15
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLess(probability, 0.1,f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0, f"Probability must be >= 0 but got: {probability:.3f}")

    def test_low_probability_applicant_06(self):
        """Test low probability mortgage applicant scenario 06."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 50,
            'government_employee': 'yes',
            'highest_education': 'master',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 19,
            'reported_monthly_income': 39000,
            'total_existing_debt': 0,
            'investments_value': 500000,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 500000
        loan_term = 35
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLess(probability, 0.1,f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0,f"Probability must be >= 0 but got: {probability:.3f}")

    def test_high_probability_applicant_05(self):
        """Test high probability mortgage applicant scenario 05."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 30,
            'government_employee': 'yes',
            'highest_education': 'high_school',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 100,
            'reported_monthly_income': 20000,
            'total_existing_debt': 100000,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 100000
        loan_term = 35
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.9, f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0, f"Probability must be <= 1 but got: {probability:.3f}")


    def test_high_probability_applicant_06(self):
        """Test high probability mortgage applicant scenario 06."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 30,
            'government_employee': 'yes',
            'highest_education': 'high_school',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 100,
            'reported_monthly_income': 20000,
            'total_existing_debt': 100000,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 100000
        loan_term = 35
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.6, f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0, f"Probability must be <= 1 but got: {probability:.3f}")


    # Extremely high debt - risky mortgage applicant even with good credit_score.
    def test_low_probability_applicant_07(self):
        """Test low probability mortgage applicant scenario 07."""
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 100,
            'reported_monthly_income': 35000,
            'total_existing_debt': 1000000000,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 100000
        loan_term = 30
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.3, f"Expected low probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0, f"Probability must be <= 1 but got: {probability:.3f}")


    # Test payment ratio boundaries.
    def test_payment_ratio_29_percent_good_credit(self):
        """Test boundary: 29% payment ratio with good credit."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 33,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 500,
            'reported_monthly_income': 50000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 2500000
        loan_term = 20
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.6, f"Expected good probability but got: {probability:.3f}")

    def test_payment_ratio_50_percent_critical(self):
        """Test mortgage approval with critical 50% payment ratio."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 500,
            'reported_monthly_income': 40000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 3500000
        loan_term = 25
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertGreaterEqual(probability, 0.6, f"Expected moderate probability but got: {probability:.3f}")

    def test_payment_ratio_51_percent_should_fail(self):
        """Test mortgage rejection with excessive 51% payment ratio."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 8,
            'size_of_company': 500,
            'reported_monthly_income': 40000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'bad'
        }

        loan_amount = 3600000
        loan_term = 25
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.3, f"Expected low probability but got: {probability:.3f}")

    def test_bad_credit_minimal_risk(self):
        """Test mortgage approval despite bad credit with minimal risk factors."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 30,
            'government_employee': 'yes',
            'highest_education': 'high_school',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 10,
            'size_of_company': 1000,
            'reported_monthly_income': 40000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'bad'
        }

        loan_amount = 400000
        loan_term = 20
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.35, f"Bad credit should limit probability: {probability:.3f}")

    def test_CLAUDE_government_vs_freelancer_same_profile(self):
        """Test government employee vs freelancer with identical profiles."""
        model_gbn = self.handler.train_and_validate_gbn()
        base_profile = {
            'age': 35,
            'highest_education': 'master',
            'study_status': 'no',
            'len_employment': 8,
            'size_of_company': 100,
            'reported_monthly_income': 45000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 4000000
        loan_term = 15

        # Government employee
        gov_applicant = base_profile.copy()
        gov_applicant['government_employee'] = 'yes'
        gov_applicant['employment_type'] = 'permanent'

        gov_probability = self.handler.predict_loan_approval(
            model_gbn, gov_applicant, loan_amount, loan_term
        )

        # Freelancer
        freelancer_applicant = base_profile.copy()
        freelancer_applicant['government_employee'] = 'no'
        freelancer_applicant['employment_type'] = 'freelancer'
        freelancer_applicant['size_of_company'] = 1

        freelancer_probability = self.handler.predict_loan_approval(
            model_gbn, freelancer_applicant, loan_amount, loan_term
        )

        self.assertGreater(gov_probability, freelancer_probability, f"Government employee ({gov_probability:.3f}) should have higher probability than freelancer ({freelancer_probability:.3f})")

    # Income boundary tests
    def test_average_salary_boundary(self):
        """Test mortgage approval at average salary boundary."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 5,
            'size_of_company': 200,
            'reported_monthly_income': 35000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 1500000
        loan_term = 20
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.65, f"Average salary should yield decent probability: {probability:.3f}")

    def test_just_below_average_salary(self):
        """Test: Just below average salary (34,999)"""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 35,
            'government_employee': 'yes',
            'highest_education': 'bachelor',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 5,
            'size_of_company': 200,
            'reported_monthly_income': 34999,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 1500000
        loan_term = 20
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.65, f"Just below average should still be reasonable: {probability:.3f}")

    def test_minimum_living_wage(self):
        """Test: Minimum living wage (15,000) - very low income boundary"""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 27,
            'government_employee': 'yes',
            'highest_education': 'basic',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 50,
            'reported_monthly_income': 14000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 2000000
        loan_term = 30
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.2, f"Very low income should have low probability: {probability:.3f}")

    def test_minimum_living_wage_02(self):
        """Test: Minimum living wage (15,000) - very low income boundary"""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 27,
            'government_employee': 'yes',
            'highest_education': 'basic',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 50,
            'reported_monthly_income': 14000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 1400000
        loan_term = 20
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertLessEqual(probability, 0.6, f"Probability was expected lower than 60%, current probability is: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.4, f"Probability was expected higher than 40%, current probability is: {probability:.3f}")


    def test_minimum_living_wage_03(self):
        """Test mortgage approval at minimum living wage scenario 03."""
        model_gbn = self.handler.train_and_validate_gbn()
        applicant = {
            'age': 27,
            'government_employee': 'yes',
            'highest_education': 'basic',
            'study_status': 'no',
            'employment_type': 'permanent',
            'len_employment': 2,
            'size_of_company': 50,
            'reported_monthly_income': 14000,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'rent',
            'credit_history': 'excellent'
        }

        loan_amount = 400000
        loan_term = 35
        probability = self.handler.predict_loan_approval(
            model_gbn, applicant, loan_amount, loan_term
        )

        self.assertGreater(probability, 0.9, f"Very low income should have high probability wih very low monthly payment: {probability:.3f}")


if __name__ == '__main__':
    unittest.main()
