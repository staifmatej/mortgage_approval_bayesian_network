import networkx as nx
import logging
import warnings

logging.getLogger('numexpr').setLevel(logging.WARNING) # INFO:numexpr.utils:NumExpr
from pgmpy.models import LinearGaussianBayesianNetwork

from data_loader import LoanDataLoader
from utils.error_print import *
from utils.constants import *

warnings.filterwarnings('ignore',category=RuntimeWarning, module='sklearn')


class GaussianBayesianNetwork():
    def __init__(self, save_diagram_to_png=False, csv_path="datasets/mortgage_applications.csv", avg_salary=35000):
        self.save_diagram = save_diagram_to_png
        self.csv_path = csv_path
        self.avg_salary = avg_salary
        self.loader = self.handle_datasets_from_training()
        self.loan_approval_model = (LinearGaussianBayesianNetwork(
            [
            ("government_employee", "stability_income"),
            ("age_young", "stability_income"),
            ("age_prime", "stability_income"),
            ("age_senior", "stability_income"),
            ("age_old", "stability_income"),
            ("len_employment", "stability_income"),
            ("size_of_company", "stability_income"),
            ("highest_education", "stability_income"),
            ("employment_type", "stability_income"),

            ("reported_monthly_income","total_stable_income_monthly"),
            ("stability_income", "total_stable_income_monthly"),

            ("total_stable_income_monthly", "ratio_income_debt"),
            ("total_existing_debt", "ratio_income_debt"),

            ("core_net_worth", "ratio_debt_net_worth"),
            ("total_existing_debt", "ratio_debt_net_worth"),

            ("investments_value", "core_net_worth"),
            ("property_owned_value", "core_net_worth"),

            ("avg_salary", "ratio_income_to_avg_salary"),
            ("total_stable_income_monthly", "ratio_income_to_avg_salary"),

            ("housing_status", "defaulted"),
            ("credit_history", "defaulted"),
            ("ratio_debt_net_worth", "defaulted"),
            ("ratio_income_debt", "defaulted"),
            ("ratio_payment_to_income", "defaulted"),
            ("ratio_income_to_avg_salary", "defaulted"),

            ("monthly_payment", "ratio_payment_to_income"),
            ("total_stable_income_monthly", "ratio_payment_to_income"),

            ("loan_amount", "monthly_payment"),
            ("loan_term", "monthly_payment"),

            ("ratio_payment_to_income", "loan_approved"),
            ("ratio_debt_net_worth", "loan_approved"),
            ("credit_history", "loan_approved"),
            ("defaulted", "loan_approved")
            ]))


    def data_miss_handler(self, data):
        if data is None:
            print_error_handling(f"Dataset not found. Is {self.csv_path} correct?")

    def save_diagram_of_gbn(self):
        if self.save_diagram:
            viz = self.loan_approval_model.to_graphviz()
            viz.draw('diagram_photos/random.png', prog='dot')

    def handle_datasets_from_training(self):
        loader = LoanDataLoader()
        data = loader.load_data(self.csv_path)
        self.data_miss_handler(data)
        return loader

    def train_model(self):
        self.check_csv_data_quality()
        self.save_diagram_of_gbn()
        data = self.loader.get_all_data_numeric()

        self.loan_approval_model.fit(data)

    def check_csv_data_quality(self):
        data = self.loader.get_all_data_numeric()

        self.data_miss_handler(data)
        
        issues_found = False
        magic_constant = 2.8e4
        max_extreme_values = self.avg_salary * magic_constant
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                max_val = abs(data[col].max())
                min_val = abs(data[col].min())
                
                if max_val > max_extreme_values or min_val > max_extreme_values:
                    if not issues_found:
                        print_wrong_handling("Numerical instability detected in the data!")
                        print(f"\n{S_BOLD}Regression matrix has extreme condition number, which may leads to:{E_BOLD}")
                        print("  - Division by zero during matrix inversion")
                        print("  - Overflow during multiplication operations")
                        print("  - NaN values in coefficients")
                        print(f"\n{S_BOLD}Problematic columns:{E_BOLD}")
                        issues_found = True
                    
                    print(f"  - {col}: range [{data[col].min():.2e}, {data[col].max():.2e}]")
        
        if issues_found:
            print(f"\n{S_BOLD}Recommendation{E_BOLD}: Fix data in CSV file.\n")
        else:
            print(f"{S_GREEN}☑ Data quality check passed: no extreme values found{E_GREEN}")

    def check_model_gbn(self):
        self.train_model()
        try:
            self.loan_approval_model.check_model()
        except Exception as e:
            print_error_handling(f"Model validation error: {e}")
        print(f"{S_GREEN}☑ Model structure is valid!{E_GREEN}")

if __name__ == "__main__":
    model_gbn = GaussianBayesianNetwork(True, "datasets/mortgage_applications.csv", 35000)
    model_gbn.check_model_gbn()
