import pandas as pd
from utils.error_print import *

class LoanDataLoader:
    def __init__(self):
        self._instance = None
        self._data = None

    def load_data(self, csv_path):
        if self._data is None:
            try:
                self._data = pd.read_csv(csv_path)
            except Exception as e:
                print_error_handling(f"{e}")
        return self._data

    def get_data_for_cpd(self, cpd_name):
        if self._data is None:
            print_error_handling(f"Data are not available.")
        if cpd_name == "stability_income":
            return self._data[['government_employee',
                                'age',
                                'highest_education',
                                'employment_type',
                                'len_employment',
                                'size_of_company']]

        if cpd_name == "total_stable_income_monthly":
            return self._data[['reported_monthly_income']]

        print_error_handling(f"Unknown CPD name: {cpd_name}")

    def get_all_data(self):
        return self._data
    
    def get_all_data_numeric(self):
        """Convert all text columns to numeric for fit()"""
        data = self._data.copy()
        
        # Convert text columns to numeric
        data['government_employee'] = data['government_employee'].map({'yes': 1, 'no': 0})
        data['investments'] = data['investments'].map({'yes': 1, 'no': 0})
        data['housing_status'] = data['housing_status'].map({'own': 2, 'mortgage': 1, 'rent': 0})
        data['credit_history'] = data['credit_history'].map({'excellent': 3, 'good': 2, 'fair': 1, 'bad': 0, 'poor': 0})
        data['highest_education'] = data['highest_education'].map({'phd': 4, 'master': 3, 'bachelor': 2, 'high_school': 1, 'basic': 0})
        data['employment_type'] = data['employment_type'].map({'permanent': 2, 'temporary': 1, 'freelancer': 1, 'unemployed': 0})
        
        return data
