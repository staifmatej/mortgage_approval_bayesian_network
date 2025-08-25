"""Module for generating realistic synthetic mortgage application data with various financial and demographic features."""
# pylint: disable=too-many-lines
import csv
import os
import random
import time
import math
import warnings
import pandas as pd
import numpy as np

from utils.constants import S_RED, E_RED  # pylint: disable=import-error
from utils.error_print import print_error_handling, print_warning_handling  # pylint: disable=import-error

warnings.filterwarnings("ignore", message=".*rich is experimental.*")

def encode_age_group(age):
    """Convert age to age group indicators - matching the dataset encoding"""
    age_young = 1 if 0 <= age < 25 else 0
    age_prime = 1 if 25 <= age < 40 else 0
    age_senior = 1 if 40 <= age < 60 else 0
    age_old = 1 if age >= 60 else 0
    return age_young, age_prime, age_senior, age_old

def calculate_monthly_payment(interest_rate, loan_term, loan_amount):
    """Calculate monthly mortgage payment based on interest rate, loan term and amount."""
    monthly_interest_rate_float = interest_rate / 100 / 12
    num_of_monthly_payments = loan_term * 12

    # Loan with 0% annual interest.
    if monthly_interest_rate_float == 0:
        return loan_amount / num_of_monthly_payments

    # Calculation of monthly payment with monthly interest rate 'r' as a decimal.
    return loan_amount * (monthly_interest_rate_float * (
            1 + monthly_interest_rate_float) ** num_of_monthly_payments) / (
            (1 + monthly_interest_rate_float) ** num_of_monthly_payments - 1)

class DataGenerator:
    """Generates synthetic mortgage application data with realistic financial distributions and relationships."""
    def __init__(self, avg_salary, interest_rate, num_records, retirement_age=65):
        self.avg_salary = avg_salary
        self.interest_rate = interest_rate
        self.num_records = num_records
        self.retirement_age = retirement_age
        self.csv_path = "datasets/mortgage_applications.csv"
        self.epsilon = 1e-8
        max_num_records = 1e15
        warning_num_records = 1e7

        if self.retirement_age < 18 or self.retirement_age > 100:
            print_error_handling("Retirement_age must be between 18 and 100.")
        if self.num_records < 100:
            print_error_handling("Number of records should be at least 100.")
        if self.interest_rate > 0.27:
            print_error_handling("Interest rate is unrealistically high. Set interest rate between 0 and 27.")
        if self.num_records == 0:
            print_error_handling("CSV dataset does not contain any records.")
        if self.num_records >= warning_num_records:
            print_warning_handling("number of records is very large number. Data generation may take a while!")
        if self.num_records > max_num_records:
            print_error_handling(f"number of records exceeded the maximum permissible value {max_num_records:.0}")
        if self.epsilon < 0 or self.epsilon == 1:
            print_warning_handling("number of records is very large number. Data generation may take a while!")
        if self.avg_salary < 0 or self.avg_salary == 0:
            print_error_handling("avg_salary must be greater than or equal to zero.")
        if self.interest_rate < 0:
            print_error_handling("interest rate must be greater than zero.")

    def initialize_header(self):
        """Initialize CSV file with header row containing all feature column names."""
        self.create_folder()
        self.seed_random()
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["id",
                                       "age",
                                       "avg_salary",
                                       "government_employee",
                                       "highest_education",
                                       "employment_type",
                                       "len_employment",
                                       "study_status",
                                       "size_of_company",
                                       "investments",
                                       "investments_value",
                                       "property_owned_value",
                                       "reported_monthly_income",
                                       "total_existing_debt",
                                       "housing_status",
                                       "credit_history",
                                       "loan_amount",
                                       "loan_term",
                                       "monthly_payment",
                                       "stability_income",
                                       "total_stable_income_monthly",
                                       "core_net_worth",
                                       "ratio_income_debt",
                                       "ratio_debt_net_worth",
                                       "ratio_payment_to_income",
                                       "age_young", "age_prime", "age_senior", "age_old",
                                       "ratio_income_to_avg_salary",
                                       "mortgage_end_age",
                                       "years_of_mortgage_after_retirement",
                                       "defaulted",
                                       "loan_approved"])

    def generate_stability_len_employment(self, len_employment):
        """Calculate employment stability score based on years of employment."""
        if len_employment < 1:
            return 0
        if len_employment < 5:
            return 10
        if len_employment < 10:
            return 20
        if len_employment < 20:
            return 30
        if len_employment < 30:
            return 35
        return 0

    def create_folder(self):
        """Create datasets directory if it doesn't exist."""
        os.makedirs("datasets", exist_ok=True)

    def seed_random(self):
        """Seed random number generator with current time for reproducibility."""
        random.seed(time.time())

    def generate_random_float_from_0_to_1(self):
        """Generate random float between 0 and 1."""
        return random.random()

    def generate_base_value_property(self, reported_monthly_income):
        """Generate property value based on monthly income with realistic multiplier."""
        base_value_property = reported_monthly_income * self.generate_random_int_from_x_to_y(100, 270)
        base_value_property = math.ceil(base_value_property / 1000) * 1000
        return base_value_property

    def generate_random_investments(self, age, reported_monthly_income):
        """
        Investment simulation where a person saves 'realistic_investment_value' percent of
        their monthly salary each month for 'investment_interval' years with 8% annual return.
        """
        if age <= 18:
            return 0
        investment_interval = age - 18
        realistic_investment_value = self.generate_random_int_from_x_to_y(10,70) / 100 # Person can save up to 70% per month.
        savings_for_investments_per_year = reported_monthly_income * realistic_investment_value * 12
        return savings_for_investments_per_year * (((1.08)**investment_interval-1)/0.08)*1.08 # SPY has historically 7-9% annually return.

    def generate_random_int_from_x_to_y(self, x_from, y_to):
        """Generate random integer between x_from and y_to inclusive."""
        return random.randint(x_from, y_to)

    def remove_wrong_rows(self, debug_print=False, pbar=None):
        """Remove rows with invalid data and ensure decimal precision constraints."""
        # pylint: disable=too-many-locals
        df = pd.read_csv(self.csv_path)

        if pbar is not None:
            pbar.update(1)

        def has_max_5_decimals(x):
            str_x = str(x)
            if '.' in str_x:
                decimal_part = str_x.split('.')[1]
                return len(decimal_part) <= 5
            return True

        mask_ratio = df['ratio_income_to_avg_salary'].apply(has_max_5_decimals)
        mask_stability = df['stability_income'].apply(has_max_5_decimals)
        mask_1 = df['property_owned_value'].apply(has_max_5_decimals)
        mask_2 = df['reported_monthly_income'].apply(has_max_5_decimals)
        mask_3 = df['total_existing_debt'].apply(has_max_5_decimals)
        mask_4 = df['loan_amount'].apply(has_max_5_decimals)
        mask_5 = df['monthly_payment'].apply(has_max_5_decimals)
        mask_6 = df['core_net_worth'].apply(has_max_5_decimals)
        mask_7 = df['ratio_payment_to_income'].apply(has_max_5_decimals)
        mask_8 = df['ratio_income_debt'].apply(has_max_5_decimals)
        mask_9 = df['ratio_debt_net_worth'].apply(has_max_5_decimals)


        if pbar is not None:
            pbar.update(1)
        mask_defaulted = df['defaulted'] != 0

        # Keep only rows where all conditions are met
        mask_combined = mask_ratio & mask_stability & mask_1 & mask_2 & mask_3 & mask_4 & mask_5 & mask_6 & mask_7 & mask_8 & mask_9 & mask_defaulted
        df_clean = df[mask_combined]

        if debug_print:
            removed_rows = len(df) - len(df_clean)
            print(f"Removed {removed_rows} rows (unrounded values or defaulted=0)")

        if pbar is not None:
            pbar.update(1)

        df_clean.to_csv(self.csv_path, index=False)

    def find_thresholds_for_col(self, column_name: str) -> list:
        """Calculate percentile thresholds for a given column."""
        df = pd.read_csv(self.csv_path)
        values = df[column_name]
        thresholds = []

        for i in range(101):
            percentile = values.quantile(i / 100)
            thresholds.append(percentile)

        return thresholds

    def delete_invalid_rows(self, df):
        """Remove rows where age group indicators don't sum to exactly 1."""
        df["age_sum"] = df["age_young"] + df["age_prime"] + df["age_senior"] + df["age_old"]
        df = df[df['age_sum'] == 1]
        df = df.drop('age_sum', axis=1)
        return df

    def round_values(self, df):
        """Round numeric columns to appropriate decimal places based on their type."""
        integer_columns = ['id', 'len_employment',
                           'size_of_company', 'loan_term',
                           'age_young', 'age_prime',
                           'age_senior', 'age_old']
        money_columns = ['investments_value',
                         'property_owned_value',
                         'reported_monthly_income',
                         'total_existing_debt',
                         'loan_amount', 'monthly_payment',
                         'total_stable_income_monthly',
                         'core_net_worth']
        ratio_columns = ['ratio_payment_to_income',
                         'ratio_income_debt',
                         'ratio_debt_net_worth',
                         'ratio_income_to_avg_salary',
                         'stability_income', 'defaulted',
                         'loan_approved']

        for col in integer_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

        for col in money_columns:
            if col in df.columns:
                df[col] = np.round(df[col], 0).astype(int)

        for col in ratio_columns:
            if col in df.columns:
                df[col] = np.round(df[col], 5).astype(float)

        return df

    def delete_negative_payments(self, df):
        """Remove rows with negative monthly payment values."""
        df = df[df["monthly_payment"] > 0]
        return df

    def run_defaulted_generation(self, pbar):
        """Generate default risk scores based on multiple financial indicators."""
        df = pd.read_csv(self.csv_path)
        if pbar is not None:
            pbar.update(1)
        df = self.delete_negative_payments(df)
        df = self.delete_invalid_rows(df)
        df = self.round_values(df)

        # BASE LINE
        df["defaulted"] = 0.15

        df["defaulted"] = np.where(df["ratio_income_to_avg_salary"] == 0, df["defaulted"] * 100, df["defaulted"])
        df["defaulted"] = np.where(df["ratio_payment_to_income"] == 0, df["defaulted"] * 100, df["defaulted"])

        payment_risk = df["ratio_payment_to_income"].clip(0, 1)
        df["defaulted"] += payment_risk * 0.25

        debt_income_risk = df["ratio_income_debt"].clip(0, 1)
        df["defaulted"] += debt_income_risk * 0.15

        debt_worth_risk = df["ratio_debt_net_worth"].clip(0, 1)
        df["defaulted"] += np.where(df["ratio_debt_net_worth"] != 0, debt_worth_risk * 0.10, 0)

        if pbar is not None:
            pbar.update(1)

        income_risk = 1 - df["ratio_income_to_avg_salary"].clip(0, 1)
        df["defaulted"] += income_risk * 0.15

        df.loc[df["housing_status"] == "own", "defaulted"] *= 0.85      # 15% risk reduction
        df.loc[df["housing_status"] == "mortgage", "defaulted"] *= 1.05  # 5% risk increase
        df.loc[df["housing_status"] == "rent", "defaulted"] *= 1.15      # 15% risk increase

        df.loc[df["credit_history"] == "excellent", "defaulted"] *= 0.5  # 50% risk reduction
        df.loc[df["credit_history"] == "fair", "defaulted"] *= 1.25      # 25% risk increase
        df.loc[df["credit_history"] == "bad", "defaulted"] *= 2.0        # 100% risk increase

        df["defaulted"] = df["defaulted"].clip(0, 100)
        df["defaulted"] = np.round(df["defaulted"], 4)

        if pbar is not None:
            pbar.update(1)
        df.to_csv(self.csv_path, index=False)


    def generate_realistic_data(self, progress_bar=True):
        """Generate complete synthetic mortgage application dataset with all features."""
        self.initialize_header()
        print()

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            progress_bar_extra_value = 11
            if progress_bar:
                try:
                    from tqdm.rich import tqdm  # pylint: disable=import-outside-toplevel
                    with tqdm(total=self.num_records + progress_bar_extra_value, smoothing=1) as pbar:
                        self.run_data_generation(writer, pbar)
                        self.run_ratios_standardization(pbar)
                        self.run_defaulted_generation(pbar)
                        self.calculate_loan_approved(pbar)
                except ImportError:
                    try:
                        from tqdm import tqdm  # pylint: disable=import-outside-toplevel
                        with tqdm(total=self.num_records + progress_bar_extra_value, smoothing=1) as pbar:
                            self.run_data_generation(writer, pbar)
                            self.run_ratios_standardization(pbar)
                            self.run_defaulted_generation(pbar)
                            self.calculate_loan_approved(pbar)
                    except ImportError:
                        print(f"{S_RED}ERROR{E_RED}: Progress bar does not work.")

            if not progress_bar:
                try:
                    self.run_data_generation(writer, None)
                    self.run_ratios_standardization(None)
                    self.run_defaulted_generation(None)
                    self.calculate_loan_approved(None)
                except (ImportError, AttributeError, TypeError, ValueError) as e:
                    print(f"{S_RED}ERROR{E_RED}: Progress bar does not work. {e}")

        print("")

    def analyze_distribution(self, column_name="loan_approved", bins=10):
        """Analyze and visualize the distribution of a specified column."""
        df = pd.read_csv(self.csv_path)

        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in dataset")
            return

        max_value = df[column_name].max()
        min_value = df[column_name].min()

        if max_value <= 1.1:
            bin_edges = np.linspace(0, 1, bins + 1)
            labels = [f"{int(i*10)}-{int((i+1)*10)}" for i in range(bins)]
        else:
            bin_edges = np.linspace(0, 100, bins + 1)
            labels = [f"{int(i*10)}-{int((i+1)*10)}" for i in range(bins)]

        counts, _ = np.histogram(df[column_name], bins=bin_edges)
        percentages = (counts / len(df)) * 100

        print(f"\nDistribution analysis for '{column_name}':")
        print(f"Min: {min_value:.4f}, Max: {max_value:.4f}, Mean: {df[column_name].mean():.4f}")
        print("-" * 30)

        for i, (label, pct) in enumerate(zip(labels, percentages)):
            bar_chart = "█" * int(pct / 2)
            if label == "0-10":
                print(f"{label}%:   {pct:5.1f}% |{bar_chart}")
            elif label == "90-100":
                print(f"{label}%: {pct:5.1f}% |{bar_chart}")
            else:
                print(f"{label}%:  {pct:5.1f}% |{bar_chart}")

        print("-" * 30)
        print(f"Total records: {len(df)}")

    def calculate_loan_approved(self, pbar=None):
        """Calculate loan approval probabilities based on default scores and financial ratios."""
        df = pd.read_csv(self.csv_path)

        if pbar is not None:
            pbar.update(1)

        df["defaulted"] = df["defaulted"].clip(0, 2)

        max_defaulted = max(df["defaulted"])
        df["loan_approved"] = max(1, max_defaulted * 1000)
        df["loan_approved"] = df["loan_approved"] - df["defaulted"] * 50

        if pbar is not None:
            pbar.update(1)

        df["loan_approved"] -= df["ratio_payment_to_income"] * 90
        df["loan_approved"] -= df["ratio_debt_net_worth"] * 50
        df.loc[df["ratio_debt_net_worth"] == 0, "loan_approved"] += 25

        df.loc[df["credit_history"] == "excellent", "loan_approved"] *= 2
        df.loc[df["credit_history"] == "fair", "loan_approved"] *= 1
        df.loc[df["credit_history"] == "bad", "loan_approved"] *= 0.75

        for years in range(0, int(1e6), 5):
            mask = df["years_of_mortgage_after_retirement"] > years
            if not mask.any():
                break
            if years == 0:
                df.loc[mask, "loan_approved"] *= 0.75
            if years != 0:
                df.loc[mask, "loan_approved"] *= 0.5

        def adjust_values_to_spread_more_towards_edges02():
            df.loc[
                (df["credit_history"] != "bad") &
                (df["government_employee"] == 1) &
                (df["ratio_payment_to_income"] <= 0.55),
                "loan_approved"] *= 1.5

            df.loc[
                (df["credit_history"] != "bad") &
                (df["ratio_payment_to_income"] <= 0.55),
                "loan_approved"] *= 1.2

            df.loc[
                df["housing_status"] == "homeless",
                "loan_approved"
            ] *= 0.0001

            df.loc[
                df["study_status"] == "yes",
                "loan_approved"
            ] *= 0.00001

            # Špatné profily - od nejhorších k mírnějším
            df.loc[
                df["credit_history"] == "bad",
                "loan_approved"
            ] *= 0.02

            df.loc[
                df["ratio_payment_to_income"] > 1.0,
                "loan_approved"
            ] *= 0.001

            df.loc[
                df["ratio_payment_to_income"] > 0.7,
                "loan_approved"
            ] *= 0.01

            df.loc[
                df["ratio_payment_to_income"] > 0.5,
                "loan_approved"
            ] *= 0.1

        adjust_values_to_spread_more_towards_edges02()

        # Finální normalizace na 0-1 rozsah (zachovává interpretovatelnost)
        df["loan_approved"] = (df["loan_approved"] - df["loan_approved"].min()) / (df["loan_approved"].max() - df["loan_approved"].min())

        df.loc[
            (df["credit_history"] == "bad") |
            (df["ratio_payment_to_income"] > 0.5),
            "loan_approved"
        ] *= 0.05

        df.loc[
            (df["study_status"] == "yes") & (df["age"] < 26),
            "loan_approved"
        ] *= 0.00001

        df["loan_approved"] = np.round(df["loan_approved"], 5)

        # Check for values outside 0-1 range and print error message.
        if (df["loan_approved"] < 0).any() or (df["loan_approved"] > 1).any():
            min_val = df["loan_approved"].min()
            max_val = df["loan_approved"].max()
            count_below = (df["loan_approved"] < 0).sum()
            count_above = (df["loan_approved"] > 1).sum()
            print_error_handling(f"Invalid loan_approved values found: {count_below} below 0, {count_above} above 1. Range: [{min_val:.5f}, {max_val:.5f}]")

        if pbar is not None:
            pbar.update(1)

        df.to_csv(self.csv_path, index=False)


    def run_ratios_standardization(self, pbar):
        """Standardize financial ratios to prevent extreme values and improve model stability."""
        df = pd.read_csv(self.csv_path)
        if pbar is not None:
            pbar.update(1)
        for ratio in ["ratio_payment_to_income", "ratio_income_debt", "ratio_debt_net_worth", "ratio_income_to_avg_salary"]:
            thresholds_ratio = self.find_thresholds_for_col(ratio)
            values = df[ratio].values
            thresholds = np.array(thresholds_ratio)
            positions = np.searchsorted(thresholds, values) / 100
            df[ratio] = np.round(positions, 5)
            if pbar is not None:
                pbar.update(1)
        df.to_csv(self.csv_path, index=False)

    def run_data_generation(self, writer, pbar=None):
        """Generate individual records with all features using realistic distributions."""
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches,cell-var-from-loop,chained-comparison,pointless-string-statement
        for i in range(self.num_records):

            defaulted = 0
            loan_approved = 0

            def generate_id():
                """Generate deterministically id."""
                return i + 1

            def generate_avg_salary():
                return self.avg_salary

            avg_salary = generate_avg_salary()

            def generate_government_employee():
                """ 20% government_employee & 80% prob that person is not government_employee."""
                government_employee = "yes" if self.generate_random_float_from_0_to_1() < 0.2 else "no"
                return government_employee

            def generate_age():
                """Generate random age."""
                return self.generate_random_int_from_x_to_y(18, 65)

            def generate_highest_education():
                """
                highest_education:
                Basic: probability 20%
                High_school: probability 45%
                Bachelor: probability 20%
                Master: probability 10%
                Phd: probability 5%
                """
                r = self.generate_random_float_from_0_to_1()
                if r >= 0 and r < 0.2 and age > 6:
                    return "basic"
                if r >= 0.2 and r < 0.65 and age > 13:
                    return "high_school"
                if r >= 0.65 and r < 0.85 and age > 17:
                    return "bachelor"
                if r >= 0.85 and r < 0.95 and age > 20:
                    return "master"
                if r >= 0.95 and r <= 1 and age > 22:
                    return "phd"
                return "basic"

            record_id = generate_id()
            age = generate_age()
            government_employee = generate_government_employee()
            highest_education = generate_highest_education()

            def generate_employment_type():
                """
                government_employee == "yes" == > "permanent"
                otherwise:
                probability of unemployed is 5%.
                probability of temporary is 20%.
                probability of freelancer is 20%.
                probability of freelancer or temporary is 55%.
                """
                if government_employee == "yes":
                    return "permanent"
                if government_employee == "no":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.05:  # 5%
                        return "unemployed"
                    if r >= 0.05 and r < 0.25:  # 20%
                        return "temporary"
                    if r >= 0.25 and r < 0.45:  # 20%
                        return "freelancer"
                    if r >= 0.45 and r <= 1:  # 55%
                        return "permanent"
                return "permanent"  # default case

            employment_type = generate_employment_type()

            def generate_len_employment():
                """
                !!!! v Czechia you need to study Master & Bachelor and only than you can study Phd.
                employment_type is unemployed ==> 0 years of working for the same company.
                otherwise:
                random generated from 0 to maximum possible years of working for the same company.
                """
                if employment_type == "unemployed":
                    return 0

                penality = 0
                adult_treshold = 18  # in Czechia is person adult from age of 18.
                if highest_education == "basic":
                    penality = -2
                if highest_education == "high_school":
                    penality += 1
                if highest_education == "bachelor":
                    penality += 3
                if highest_education == "master":
                    penality += 2
                if highest_education == "phd":
                    penality += 5
                max_possible_of_len_employment = age - adult_treshold - penality

                max_possible_of_len_employment = max(max_possible_of_len_employment, 0)

                len_employment = max_possible_of_len_employment - self.generate_random_int_from_x_to_y(0,
                                                                                                       max_possible_of_len_employment)

                if len_employment < 0:
                    """ For people how study PhD. in U.S.A. for example and they are younger than in Czechia is normal. """
                    len_employment = 0

                if government_employee == "yes":
                    """Bonus years in data simulation for more stable jobs."""
                    if len_employment + len_employment / 2 < max_possible_of_len_employment:
                        len_employment += int(max_possible_of_len_employment / 2)

                return len_employment

            len_employment = generate_len_employment()

            def generate_size_of_company():
                if employment_type == "unemployed":
                    return 0
                if employment_type == "temporary":
                    """Smaller number, because "temporary" employment are the most likely as seasonal temp worker, construction work, etc."""
                    return self.generate_random_int_from_x_to_y(0, 100)
                if employment_type == "freelancer":
                    """Freelancer is working just for himself."""
                    return 1
                if employment_type == "permanent":
                    """From 0 up to 40 thousands of possible workers in companies."""
                    return self.generate_random_int_from_x_to_y(0, 40000)
                return 0  # default case

            size_of_company = generate_size_of_company()
            age_young, age_prime, age_senior, age_old = encode_age_group(age)

            def generate_study_status():
                """Study status is for people who are actively still studying."""
                # pylint: disable=too-many-branches,too-many-return-statements
                if age > 26 and highest_education != "master":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.99:
                        return "no"
                    return "yes"
                if age <= 18 and highest_education == "basic":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.05:
                        return "no"
                    if r >= 0.05 and r <= 1:
                        return "yes"
                if age <= 23 and highest_education == "high_school":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.7:
                        return "no"
                    if r >= 0.7 and r <= 1:
                        return "yes"
                if age <= 25 and highest_education == "bachelor":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.5:
                        return "no"
                    if r >= 0.5 and r <= 1:
                        return "yes"
                if age <= 30 and highest_education == "master":
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.97:
                        return "no"
                    return "yes"
                return "no"

            study_status = generate_study_status()

            def generate_reported_monthly_income():
                """Generate monthly income report with Gaussian curve."""
                # pylint: disable=too-many-branches,too-many-return-statements
                min_salary = int(self.avg_salary / 2)
                max_salary = self.avg_salary * 8
                base_salary = self.avg_salary

                if age < 18:
                    return 0

                if employment_type == "unemployed":
                    # unemployment benefit
                    benefit = self.generate_random_float_from_0_to_1() > 0.7
                    if not benefit:
                        return 0
                    if benefit:
                        return self.generate_random_int_from_x_to_y(0, int(min_salary / 2.22))
                    return 0

                if study_status == "yes" and highest_education != "master" and highest_education != "bachelor":
                    return min_salary / 2
                if study_status == "yes" and highest_education == "bachelor":
                    return min_salary * 1.1
                if study_status == "yes" and highest_education == "master":
                    return min_salary * 1.25

                # Education Bonus.
                if highest_education == "basic":
                    mean = base_salary * 0.6
                    std_dev = base_salary * 0.15
                elif highest_education == "high_school":
                    mean = base_salary * 0.8
                    std_dev = base_salary * 0.2
                elif highest_education == "bachelor":
                    mean = base_salary * 1.8
                    std_dev = base_salary * 3
                elif highest_education == "master":
                    mean = base_salary * 6
                    std_dev = base_salary * 3
                elif highest_education == "phd":
                    mean = base_salary * 10
                    std_dev = base_salary * 3

                # Age Bonus.
                if age < 25:
                    mean *= 0.7
                elif age >= 25 and age < 35:
                    mean *= 0.9
                elif age >= 35 and age < 45:
                    mean *= 1.1
                elif age >= 45 and age < 60:
                    mean *= 1.15
                else:  # 60+
                    mean *= 1.0

                if government_employee == "yes":
                    mean *= 0.85
                    std_dev *= 0.7

                if employment_type == "freelancer":
                    std_dev *= 1.8

                if employment_type == "permanent" and highest_education in ("master", "phd") and len_employment > 20:
                    std_dev *= 0.5

                # Generate salaries with Gaussian distribution
                salary = random.gauss(mean, std_dev)

                salary = max(min_salary, min(max_salary, salary))
                return math.ceil(salary / 100) * 100

            reported_monthly_income = generate_reported_monthly_income()

            def generate_housing_status():
                """Generate housing status based on income and financial situation."""
                # pylint: disable=too-many-branches,too-many-return-statements,inconsistent-return-statements

                if reported_monthly_income > self.avg_salary * 4:
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.70:
                        return "own"
                    if r >= 0.7 and r < 0.95:
                        return "mortgage"
                    if r >= 0.95 and r <= 1.0:
                        return "rent"

                if reported_monthly_income > self.avg_salary * 2:
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.40:
                        return "own"
                    if r >= 0.4 and r < 0.90:
                        return "mortgage"
                    if r >= 0.90 and r <= 1.0:
                        return "rent"

                if reported_monthly_income > self.avg_salary * 1.5:
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.10:
                        return "own"
                    if r >= 0.1 and r < 0.60:
                        return "mortgage"
                    if r >= 0.60 and r <= 1.0:
                        return "rent"

                if reported_monthly_income < self.avg_salary * 0.25:
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.0001:
                        return "own"
                    if r >= 0.0001 and r < 0.0005:
                        return "mortgage"
                    if r >= 0.0005 and r <= 1.0:
                        return "rent"

                if reported_monthly_income < self.avg_salary * 0.5:
                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.01:
                        return "own"
                    if r >= 0.01 and r < 0.09:
                        return "mortgage"
                    if r >= 0.09 and r <= 1.0:
                        return "rent"

                r = self.generate_random_float_from_0_to_1()
                if r >= 0 and r < 0.02:
                    return "own"
                if r >= 0.02 and r < 0.20:
                    return "mortgage"
                if r >= 0.20 and r <= 1.0:
                    return "rent"
                return "rent"  # default case

            housing_status = generate_housing_status()

            def generate_property_owned_value():
                value = 0
                base_value_property = self.generate_base_value_property(reported_monthly_income)

                if housing_status == "own":
                    value += base_value_property

                if housing_status == "mortgage":
                    ownership = 0.3 + self.generate_random_float_from_0_to_1() * 0.4
                    value += int(base_value_property * ownership)

                if reported_monthly_income > self.avg_salary * 4:
                    if self.generate_random_float_from_0_to_1() > 0.85:
                        value += self.generate_base_value_property(reported_monthly_income)

                if reported_monthly_income > self.avg_salary * 2 and reported_monthly_income < self.avg_salary * 4:
                    if self.generate_random_float_from_0_to_1() > 0.98:
                        value += self.generate_base_value_property(reported_monthly_income)

                if age > 55 and reported_monthly_income <= self.avg_salary * 2:
                    if self.generate_random_float_from_0_to_1() > 0.98:
                        value += self.generate_base_value_property(reported_monthly_income)
                return value

            property_owned_value = generate_property_owned_value()

            def generate_investments_value():
                if highest_education in ("master", "phd"):
                    if self.generate_random_float_from_0_to_1() > 0.50:
                        return self.generate_random_investments(age, reported_monthly_income)

                if highest_education == "bachelor":
                    if self.generate_random_float_from_0_to_1() > 0.90:
                        return self.generate_random_investments(age, reported_monthly_income)

                if reported_monthly_income > self.avg_salary * 1.25:
                    if self.generate_random_float_from_0_to_1() > 0.90:
                        return self.generate_random_investments(age, reported_monthly_income)

                if self.generate_random_float_from_0_to_1() > 0.98:
                    return self.generate_random_investments(age, reported_monthly_income)

                return 0

            investments_value = generate_investments_value()

            def generate_investments():
                if investments_value > 0 or property_owned_value > 0:
                    return "yes"
                return "no"

            investments = generate_investments()

            def generate_total_existing_debt():
                if housing_status == "mortgage":
                    return property_owned_value * self.generate_random_float_from_0_to_1()
                if self.generate_random_float_from_0_to_1() > 0.50:
                    return property_owned_value * 0.8 * self.generate_random_float_from_0_to_1()
                return 0

            total_existing_debt = generate_total_existing_debt()

            def generate_core_net_worth():
                net_worth = 0
                net_worth += property_owned_value + reported_monthly_income * 3 + investments_value - total_existing_debt
                return net_worth

            core_net_worth = generate_core_net_worth()

            def generate_loan_amount():
                min_loan_amount = 100000 # limited as an input metric in main.py
                max_monthly_payment = reported_monthly_income * 0.45

                monthly_interest_rate = self.interest_rate / 12
                loan_term_months = loan_term * 12

                if monthly_interest_rate == 0:
                    max_loan_from_income = max_monthly_payment * loan_term_months
                else:
                    max_loan_from_income = max_monthly_payment * ((1 + monthly_interest_rate) ** loan_term_months - 1) / (monthly_interest_rate * (1 + monthly_interest_rate) ** loan_term_months)

                randomness = self.generate_random_float_from_0_to_1() * 1.5 + 0.5  # 0.5 - 2.0
                loan_amount = max_loan_from_income * randomness

                return max(min_loan_amount, int(loan_amount))

            def generate_loan_term():
                return self.generate_random_int_from_x_to_y(1, 35)

            loan_term = generate_loan_term()
            loan_amount = generate_loan_amount()

            def generate_monthly_payment():
                return calculate_monthly_payment(self.interest_rate, loan_term, loan_amount)

            monthly_payment = generate_monthly_payment()

            def generate_credit_history():
                """Generate credit history based on income, education, debt, assets, employment, and age."""
                # pylint: disable=too-many-branches,too-many-statements
                score = 50  # Base score

                # Income factor (major impact)
                income_ratio = reported_monthly_income / self.avg_salary
                if income_ratio > 4:
                    score += 30
                elif income_ratio > 3:
                    score += 25
                elif income_ratio > 2:
                    score += 20
                elif income_ratio > 1.5:
                    score += 15
                elif income_ratio > 1:
                    score += 10
                elif income_ratio > 0.7:
                    score += 5
                else:
                    score -= 10

                # Education factor.
                if highest_education == "phd":
                    score += 35
                elif highest_education == "master":
                    score += 28
                elif highest_education == "bachelor":
                    score += 18
                elif highest_education == "high_school":
                    score += 1
                else:  # basic
                    score -= 5

                # Employment stability and type
                if employment_type == "permanent":
                    score += 12
                    # Length of employment bonus
                    if len_employment > 10:
                        score += 15
                    elif len_employment > 5:
                        score += 10
                    elif len_employment > 2:
                        score += 5
                elif employment_type == "freelancer":
                    score += 6
                    if len_employment > 5:
                        score += 5
                elif employment_type == "temporary":
                    score += 3
                else:  # unemployed
                    score -= 20

                # Government employee Bonus for Stable Job.
                if government_employee == "yes":
                    score += 2

                # Asset factor - property and investments combined
                total_assets = property_owned_value + investments_value
                asset_to_income_ratio = total_assets / (
                        reported_monthly_income * 12) if reported_monthly_income > 0 else 0

                if asset_to_income_ratio > 10:
                    score += 25
                elif asset_to_income_ratio > 5:
                    score += 20
                elif asset_to_income_ratio > 3:
                    score += 15
                elif asset_to_income_ratio > 1:
                    score += 10
                elif asset_to_income_ratio > 0.5:
                    score += 5
                elif total_assets == 0:
                    score -= 15  # Penalty for no assets

                if housing_status == "mortgage":
                    score -= 5  # Small penalty for existing mortgage

                # Age factor (stability indicator)
                if age >= 30 and age < 50:
                    score += 8  # Prime working age
                elif age >= 25 and age < 30:
                    score += 5
                elif age >= 50 and age < 60:
                    score += 3
                elif age < 25:
                    score -= 5  # Young, less credit history
                else:  # 60+
                    score -= 2  # Near retirement

                # Company size factor (job security)
                if size_of_company > 1000:
                    score += 5
                elif size_of_company > 100 and size_of_company <= 1000:
                    score += 3
                elif size_of_company == 1 and employment_type == "freelancer":
                    score += 1

                # Housing status bonus
                if housing_status == "own":
                    score += 10
                elif housing_status == "mortgage":
                    score += 5

                # Add small randomness
                score += self.generate_random_int_from_x_to_y(-3, 3)

                # Convert score to credit history category
                if score >= 95:
                    return "excellent"
                if score >= 75:
                    return "fair"
                return "bad"

            credit_history = generate_credit_history()

            def generate_stability_income():
                """Calculate income stability score based on multiple factors."""
                # pylint: disable=too-many-branches
                value_stability = 200  # baseline
                hgh_points_poss = value_stability + 165

                if government_employee == "yes":
                    value_stability += 50

                value_stability += self.generate_stability_len_employment(len_employment)

                if employment_type == "unemployed":
                    value_stability += -50
                if employment_type == "freelancer":
                    value_stability += -5
                if employment_type == "permanent":
                    value_stability += 5

                if highest_education == "bachelor":
                    value_stability += 10
                if highest_education == "master":
                    value_stability += 20
                if highest_education == "phd":
                    value_stability += 30
                if highest_education == "basic":
                    value_stability += -50

                if size_of_company < 10:
                    value_stability -= 25
                elif size_of_company < 100:
                    value_stability += 5
                elif size_of_company < 1000:
                    value_stability += 10
                elif size_of_company < 3000:
                    value_stability += 15
                elif size_of_company < 10000:
                    value_stability += 20
                else:
                    value_stability += 25

                if age_young == 1:
                    value_stability -= 10
                if age_old == 1:
                    value_stability -= 25
                if age_prime == 1:
                    value_stability += 25
                if age_senior == 1:
                    value_stability += 5

                # stability_income will have values between 0 and 1.
                return round(value_stability / hgh_points_poss, 5)

            stability_income = generate_stability_income()

            def generate_total_stable_income_monthly():
                """Calculate stable income as 80-100% of reported income based on stability factor."""
                return round(reported_monthly_income * (0.8 + 0.2 * stability_income), 5)

            total_stable_income_monthly = generate_total_stable_income_monthly()

            def generate_ratio_income_debt():
                if total_stable_income_monthly == 0:
                    return round(total_existing_debt / self.epsilon, 5)
                return round(total_existing_debt / total_stable_income_monthly, 5)

            def generate_ratio_debt_net_worth():
                if core_net_worth == 0:
                    return round(total_existing_debt / self.epsilon, 5)
                return round(total_existing_debt / core_net_worth, 5)

            def generate_ratio_payment_to_income():
                if total_stable_income_monthly == 0:
                    return round(monthly_payment / self.epsilon, 5)
                return round(monthly_payment / total_stable_income_monthly, 5)

            ratio_income_debt = generate_ratio_income_debt()
            ratio_debt_net_worth = generate_ratio_debt_net_worth()
            ratio_payment_to_income = generate_ratio_payment_to_income()

            def generate_ratio_income_to_avg_salary():
                if self.avg_salary == 0:
                    return round(total_stable_income_monthly / self.epsilon, 5)
                return round(total_stable_income_monthly / self.avg_salary, 5)

            ratio_income_to_avg_salary = generate_ratio_income_to_avg_salary()

            # Calculate mortgage end age and years after retirement
            mortgage_end_age = age + loan_term
            years_of_mortgage_after_retirement = mortgage_end_age - self.retirement_age

            writer.writerow([
                record_id,
                age,
                avg_salary,
                government_employee,
                highest_education,
                employment_type,
                len_employment,
                study_status,
                size_of_company,
                investments,
                investments_value,
                property_owned_value,
                reported_monthly_income,
                total_existing_debt,
                housing_status,
                credit_history,
                loan_amount,
                loan_term,
                monthly_payment,
                stability_income,
                total_stable_income_monthly,
                core_net_worth,
                ratio_income_debt,
                ratio_debt_net_worth,
                ratio_payment_to_income,
                age_young, age_prime, age_senior, age_old,
                ratio_income_to_avg_salary,
                mortgage_end_age,
                years_of_mortgage_after_retirement,
                defaulted,
                loan_approved
            ])
            if pbar is not None:
                pbar.update(1)


if __name__ == "__main__":
    dataCreate = DataGenerator(40000, 0.05, int(1e5), 65)
    dataCreate.generate_realistic_data(True)
    dataCreate.remove_wrong_rows(True, None)
    dataCreate.analyze_distribution()
