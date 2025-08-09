from pgmpy.factors.continuous import LinearGaussianCPD
from constants import *

EDUCATION_MAP = {
    "basic": 0.0,
    "high_school": 0.1,
    "bachelor": 0.5,
    "master": 0.75,
    "phd": 1.0
}

EMPLOYMENT_MAP = {
    "unemployed": 0.0,
    "temporary": 0.3,
    "freelancer": 0.7,
    "permanent": 1.0
}

def encode_age_groups(age: int):
    if 18 <= age < 25:   return {"young": 1, "prime": 0, "senior": 0, "old": 0}
    elif 25 <= age < 40: return {"young": 0, "prime": 1, "senior": 0, "old": 0}
    elif 40 <= age < 55: return {"young": 0, "prime": 0, "senior": 1, "old": 0}
    else:                return {"young": 0, "prime": 0, "senior": 0, "old": 1}

def encode_government_employee(government_employee: str):
    return 1 if government_employee == "yes" else 0

def encode_highest_education(highest_education: str):
    return EDUCATION_MAP.get(highest_education, 0)

def encode_employment_type(employment_type: str):
    return EMPLOYMENT_MAP.get(employment_type, 0)

def encode_employment_length(len_employment: int):
    return min(len_employment / 40.0, 1.0)

def encode_company_size(size: int):
    if size < 20: return -0.5
    if size < 100: return 0.25
    if size < 1000: return 0.5
    return 0.75

def preprocess_data(raw_data):
    age_groups = encode_age_groups(raw_data["age"])
    return {
        "government_employee": encode_government_employee(raw_data["government_employee"]),
        "age_young": age_groups["young"],
        "age_prime": age_groups["prime"],
        "age_senior": age_groups["senior"],
        "age_old": age_groups["old"],
        "highest_education": encode_highest_education(raw_data["highest_education"]),
        "employment_type": encode_employment_type(raw_data["employment_type"]), 
        "len_employment": encode_employment_length(raw_data["len_employment"]),
        "size_of_company": encode_company_size(raw_data["size_of_company"])
    }

def create_cpd():
    return cpd_stability_income

cpd_stability_income = LinearGaussianCPD(
    variable="stability_income",
    beta=[25,        # intercept (baseline)
          40,        # +40 points in "government_employee" for yes => in Czechia is this type of job very stable.
          -10,       # -10 age_young
          +25,       # +25 age_prime
          -5,        # -2.5 age_senior
          -10,       # -10 age_old
          40,        # highest_education
          20,        # employment_type
          40,        # len_employment
          10],       # size_of_company
    std=8,
    evidence=["government_employee", "age_young", "age_prime", "age_senior", "age_old", "highest_education", "employment_type", "len_employment", "size_of_company"]
)
