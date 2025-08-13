import networkx as nx
import logging

logging.getLogger('numexpr').setLevel(logging.WARNING) # INFO:numexpr.utils:NumExpr
from pgmpy.models import LinearGaussianBayesianNetwork

from data_loader import LoanDataLoader
from constants import *


loan_approval_model = LinearGaussianBayesianNetwork(
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
        ("ratio_payment_to_income", "loan_approved"),
        ("ratio_debt_net_worth", "loan_approved"),
        ("credit_history", "loan_approved"),

        ("loan_amount","monthly_payment"),
        ("loan_term","monthly_payment"),

        ("loan_amount","loan_approved"),
        ("loan_term","loan_approved"),

        ("defaulted","loan_approved"),

    ]
)

viz = loan_approval_model.to_graphviz()
viz.draw('diagram_photos/random.png', prog='dot')

csv_path = "datasets/mortgage_applications.csv"
loader = LoanDataLoader()
data = loader.load_data(csv_path)

if data is None:
    print(f"{S_RED}ERROR{E_RED}: Dataset not found. Is {csv_path} correct?")
    exit(1)


all_data = loader.get_all_data_numeric()
loan_approval_model.fit(all_data)

try:
    loan_approval_model.check_model()
    print(f"{S_GREEN}Model structure is valid!{E_GREEN}")
except Exception as e:
    print(f"{S_RED}ERROR{E_RED}: Model validation error: {e}")
    exit(1)