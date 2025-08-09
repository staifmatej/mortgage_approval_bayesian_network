from pgmpy.factors.continuous import LinearGaussianCPD


def preprocess_data(raw_data):
    """ Core net worth = investments_value + property_owned_value """
    return {
        "investments_value": raw_data["investments_value"],
        "property_owned_value": raw_data["property_owned_value"]
    }

cpd_core_net_worth = LinearGaussianCPD(
    variable="core_net_worth",
    beta=[0,          # intercept
          1.0,        # investments_value coefficient (1:1)
          1.0],       # property_owned_value coefficient (1:1)
    std=50000,        # standard deviation
    evidence=["investments_value", "property_owned_value"]
)