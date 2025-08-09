from pgmpy.factors.continuous import LinearGaussianCPD

# Ratio debt to net worth = total_debt / (investments + property)
cpd_ratio_debt_net_worth = LinearGaussianCPD(
    variable="ratio_debt_net_worth",
    beta=[0.1,         # intercept
          0.000001,    # total_existing_debt coefficient
          -0.0000005,  # investments_value coefficient (negative)
          -0.0000005], # property_owned_value coefficient (negative)
    std=0.2,           # standard deviation
    evidence=["total_existing_debt", "investments_value", "property_owned_value"]
)