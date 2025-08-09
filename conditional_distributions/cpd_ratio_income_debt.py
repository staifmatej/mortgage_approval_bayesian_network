from pgmpy.factors.continuous import LinearGaussianCPD

# Ratio income to debt = total_existing_debt / total_income
# Linearized approximation
cpd_ratio_income_debt = LinearGaussianCPD(
    variable="ratio_income_debt",
    beta=[0,           # intercept
          0.00002,     # total_existing_debt coefficient (positive)
          -0.000001],  # total_income coefficient (negative)
    std=0.1,           # standard deviation
    evidence=["total_existing_debt", "total_stable_income_monthly"]
)