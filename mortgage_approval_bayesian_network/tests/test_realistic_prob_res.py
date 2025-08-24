import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import InputHandler


class TestRealisticProbResults(unittest.TestCase):

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
        
        self.assertLess(probability, 0.3, 
            f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0,
            f"Probability must be >= 0 but got: {probability:.3f}")

    def test_high_probability_applicant_01(self):
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
        
        self.assertGreater(probability, 0.9,
            f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,
            f"Probability must be <= 1 but got: {probability:.3f}")

    def test_high_probability_applicant_02(self):
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

        self.assertGreater(probability, 0.7,
                           f"Expected high probability but got: {probability:.3f}")
        self.assertLessEqual(probability, 1.0,
                             f"Probability must be <= 1 but got: {probability:.3f}")


    def test_high_probability_applicant_03(self):
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
        model_gbn = self.handler.train_and_validate_gbn()
        low_prob_applicant = {
            'age': 17,
            'government_employee': 'no',
            'highest_education': 'basic',
            'study_status': 'no',
            'employment_type': 'temporary',
            'len_employment': 0,
            'size_of_company': 10,
            'reported_monthly_income': 0,
            'total_existing_debt': 0,
            'investments_value': 0,
            'property_owned_value': 0,
            'housing_status': 'own',
            'credit_history': 'fair'
        }

        loan_amount = 300000
        loan_term = 25
        probability = self.handler.predict_loan_approval(
            model_gbn, low_prob_applicant, loan_amount, loan_term
        )

        self.assertLess(probability, 0.3,
                        f"Expected low probability but got: {probability:.3f}")
        self.assertGreaterEqual(probability, 0.0,
                                f"Probability must be >= 0 but got: {probability:.3f}")


if __name__ == '__main__':
    unittest.main()