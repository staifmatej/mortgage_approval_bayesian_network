from pgmpy.factors.continuous import LinearGaussianCPD
from constants import *


def preprocess_data(raw_data):
    return {"stability_income": raw_data["stability_income"]}

def process_data(loader):
    data = loader.get_data_for_cpd("stability_income")
    if data is None:
        print(f"{S_RED}ERROR{E_RED}: Data are not available.")
        exit(1)

    processed_data = []
    for index, row in data.iterrows():
        processed = preprocess_data(row.to_dict())
        processed_data.append(processed)
    return processed_data

def create_cpd():
    return cpd_total_stable_income_monthly

cpd_total_stable_income_monthly = LinearGaussianCPD(
    variable="total_stable_income_monthly",
    beta=[0,   # intercept
          0.2, # weight of stability of the income.
          0.8  # weight of the monthly reported income.
          ],
    std=5000,
    evidence=["stability_income", "reported_monthly_income"]
)