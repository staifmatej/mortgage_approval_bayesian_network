from pgmpy.factors.continuous import LinearGaussianCPD

# Payment to income ratio = monthly_payment / total_income
cpd_ratio_payment_to_income = LinearGaussianCPD(
    variable="ratio_payment_to_income",
    beta=[0,          # intercept
          0.00002,    # monthly_payment coefficient
          -0.000001], # total_income coefficient (negative)
    std=0.05,         # standard deviation
    evidence=["monthly_payment", "total_stable_income_monthly"]
)