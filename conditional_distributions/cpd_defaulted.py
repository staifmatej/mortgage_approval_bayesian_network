from pgmpy.factors.continuous import LinearGaussianCPD
from constants import *

# Encoding functions for categorical variables
def encode_housing_status(status):
    """Encode housing status to risk score"""
    housing_map = {
        "homeless": 1.0,     # Highest risk
        "rent": 0.6,         # Medium-high risk
        "mortgage": 0.3,     # Low-medium risk
        "own": 0.0           # Lowest risk
    }
    return housing_map.get(status, 0.5)

def encode_credit_history(history):
    """Encode credit history to risk score"""
    credit_map = {
        "very_bad": 1.0,     # Highest risk
        "bad": 0.8,
        "fair": 0.5,
        "good": 0.2,
        "excellent": 0.0     # Lowest risk
    }
    return credit_map.get(history, 0.5)

# Default risk CPD - higher value = higher risk of defaulting
cpd_defaulted = LinearGaussianCPD(
    variable="defaulted",
    beta=[10,          # intercept - base default risk
          0.0001,      # total_existing_debt - more debt = higher risk
          -0.0002,     # total_income - higher income = lower risk
          -0.00001,    # core_net_worth - more assets = lower risk
          15,          # housing_status - encoded risk
          20,          # credit_history - encoded risk
          10,          # ratio_debt_net_worth - higher ratio = higher risk
          25],         # ratio_payment_to_income - critical factor
    std=5,
    evidence=["total_existing_debt", "total_stable_income_monthly", "core_net_worth", 
              "housing_status", "credit_history", "ratio_debt_net_worth", 
              "ratio_payment_to_income"]
)