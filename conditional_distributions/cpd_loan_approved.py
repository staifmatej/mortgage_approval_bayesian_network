from pgmpy.factors.continuous import LinearGaussianCPD

# Loan approval CPD - higher value = higher probability of approval
# This is the final decision node
cpd_loan_approved = LinearGaussianCPD(
    variable="loan_approved",
    beta=[0.8,         # intercept - base approval probability
          -0.01,       # defaulted - higher default risk = lower approval
          -20,         # ratio_payment_to_income - critical factor
          -15,         # ratio_income_debt - existing debt burden
          -10,         # ratio_debt_net_worth - leverage ratio
          0.3],        # credit_history - positive history helps
    std=0.1,
    evidence=["defaulted", "ratio_payment_to_income", "ratio_income_debt", 
              "ratio_debt_net_worth", "credit_history"]
)