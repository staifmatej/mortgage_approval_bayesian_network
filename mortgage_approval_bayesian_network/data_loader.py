"""Module for loading and managing loan application dataset from CSV files."""
import pandas as pd

from utils.error_print import print_error_handling

class LoanDataLoader:
    """Handles loading and accessing loan application data from CSV files."""
    def __init__(self):
        """Initialize LoanDataLoader with empty data cache."""
        self._instance = None
        self._data = None

    def load_data(self, csv_path):
        """Load data from CSV file with error handling and caching."""
        if self._data is None:
            try:
                self._data = pd.read_csv(csv_path)
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, IOError, OSError) as e:
                print_error_handling(f"{e}")
        return self._data

    def get_all_data(self):
        """Return the complete loaded dataset."""
        return self._data

    def get_all_data_numeric(self):
        """Get all numeric columns from the loaded dataset for model training."""
        data = self._data.copy()

        # Convert text columns to numeric
        data['government_employee'] = data['government_employee'].map({'yes': 1, 'no': 0})
        data['investments'] = data['investments'].map({'yes': 1, 'no': 0})
        data['housing_status'] = data['housing_status'].map({'own': 2, 'mortgage': 1, 'rent': 0})
        data['credit_history'] = data['credit_history'].map({'excellent': 3, 'good': 2, 'fair': 1, 'bad': 0, 'poor': 0})
        data['highest_education'] = data['highest_education'].map({'phd': 4, 'master': 3, 'bachelor': 2, 'high_school': 1, 'basic': 0})
        data['employment_type'] = data['employment_type'].map({'permanent': 2, 'temporary': 1, 'freelancer': 1, 'unemployed': 0})

        return data
