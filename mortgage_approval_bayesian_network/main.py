"""Main module for mortgage approval prediction system with user interaction interface."""
import os
import sys
import warnings
import traceback
import pandas as pd
import numpy as np
import tabulate
from gaussian_bayesian_network import GaussianBayesianNetwork

from data_generation_realistic import encode_age_group
from data_generation_realistic import DataGenerator
from utils.error_print import print_error_handling, print_invalid_input
from utils.constants import S_RED, E_RED, S_GREEN, E_GREEN, S_YELLOW, E_YELLOW, S_CYAN, E_CYAN, S_BOLD, E_BOLD
from utils.print_press_enter_to_continue import print_press_enter_to_continue
from visualize_network import create_bayesian_network_visualization

warnings.filterwarnings('ignore', category=RuntimeWarning)

TRILLION = int(1e12)

class InputHandler():
    """Handles user input collection and mortgage approval prediction workflow."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        """Initialize InputHandler with default parameters for mortgage calculations."""
        self.csv_path = "datasets/mortgage_applications.csv"
        self.avg_salary = 35000     # Average salary in Czechia in 2025.
        self.interest_rate = 0.045  # 4.5% p.a.
        self.data_num_records = 1e4 # Perfect balance between speed and data amount.
        self.retirement_age = 65
        self.mortgage_applicant_data = None
        self.model_gbn = None
        self.loan_amount = None
        self.loan_term = None

    def validate_input_alpha(self, prompt, valid_options, max_attempts=3, error_msg=None):
        """Validate alphabetic user input against list of valid options with retry mechanism."""

        for i in range(max_attempts + 1):
            if i >= max_attempts:
                print_error_handling("Error loading input")
            try:
                user_input = input(prompt).strip().lower()
            except EOFError:
                print_invalid_input(f"{e}")
            except (KeyboardInterrupt, ValueError) as e:
                print_invalid_input(f"{e}. [{i+1}/{max_attempts}]")
                continue

            if user_input in valid_options:
                return user_input

            if error_msg is None:
                options_str = '", "'.join(valid_options)
                error_msg = f'Please enter "{options_str}"'

            print_invalid_input(f"{error_msg}. [{i+1}/{max_attempts}]")

        return None  # This line should never be reached due to print_error_handling above

    def validate_input_numerical(self, prompt, min_val=None, max_val=None, default_val=None, max_attempts=3, data_type=float):
        """Validate numerical user input within specified range with type conversion."""
        # pylint: disable=too-many-arguments,R0917
        if data_type not in (float, int):
            print_error_handling("Function 'validate_input_numerical' except only data_type as float or int.")

        for i in range(max_attempts + 1):
            if i >= max_attempts:
                print_error_handling("Error loading numerical input")
                return None

            try:
                user_input = input(prompt).strip()

                if user_input == "" and default_val is not None:
                    return default_val

                value = data_type(user_input)
                if data_type == float:
                    value = np.round(value, 0)

                if min_val is not None and value < min_val:
                    print_invalid_input(f"Value must be at least {min_val}. [{i+1}/{max_attempts}]")
                    continue

                if max_val is not None and value > max_val:
                    print_invalid_input(f"Value must be at most {max_val}. [{i+1}/{max_attempts}]")
                    continue

                return value

            except ValueError:
                print_invalid_input(f"Please enter a valid number. [{i+1}/{max_attempts}]")
                continue
            except (KeyboardInterrupt, EOFError) as e:
                print_invalid_input(f"{e}. [{i+1}/{max_attempts}]")
                continue

        return None

    def generate_csv_dataset(self):
        """Generate synthetic mortgage application dataset based on configured parameters."""
        dataCreate = DataGenerator(self.avg_salary, self.interest_rate, int(self.data_num_records))
        print(f"\n{S_CYAN}Generating realistic synthetic dataset with {int(self.data_num_records):,} records...{E_CYAN}")
        dataCreate.generate_realistic_data(True)
        dataCreate.remove_wrong_rows(False, None)

    def train_and_validate_gbn(self):
        """Train and validate Gaussian Bayesian Network model on mortgage data."""
        print(f"{S_CYAN}Training & Validate Gaussain Bayesain Network...{E_CYAN}")
        model_gbn = GaussianBayesianNetwork(True, self.csv_path, self.avg_salary)
        model_gbn.check_model_gbn()
        return model_gbn

    def collect_datasets_user_info(self):
        """Collect user input for dataset generation or selection of existing dataset."""
        dataset_choice = self.validate_input_alpha(
            "\nWould you like to use your own dataset or generate a new one? (own/generate): ",
            ["own", "generate", "g", "o", "ow", "ge", "gen", "gene", "gener", "genera", "generat"],
            error_msg="Please enter 'own' to use existing dataset or 'generate' to autogenerate new realistic data")

        if "g" in dataset_choice:
            self.data_num_records = self.validate_input_numerical(
                "\nHow many records should we generate? (default 100000): ",
                min_val=100,
                max_val=TRILLION,
                default_val=100000,
                data_type=float
            )
        elif "o" in dataset_choice:
            custom_path = input(f"\nEnter path to your CSV file (default: {self.csv_path}): ").strip()
            if custom_path == "":
                custom_path = self.csv_path
            if not os.path.exists(custom_path):
                print_invalid_input("File does not exist. [1/2]")
                custom_path = input(f"\nEnter path to your CSV file (default: {self.csv_path}): ").strip()
                if custom_path == "":
                    custom_path = self.csv_path
                if not os.path.exists(custom_path):
                    print_error_handling("File does not exist. [2/2]")
            if custom_path:
                self.csv_path = custom_path

    def collect_main_user_info(self):
        """Collect main financial parameters from user for mortgage calculations."""
        print(f"\n{S_CYAN} ══════ Welcome to the BayesianHill & Co. Bank - Mortgage Approval System ══════ {E_CYAN}")

        self.interest_rate = self.validate_input_numerical("\nEnter the interest rate at which we will lend the mortgage (from 0 to 0.27; default 0.05): ", min_val=0, max_val=0.27, default_val=0.05, max_attempts=3, data_type=float)
        print(f"{S_CYAN}Note{E_CYAN}: At Bayesianhill Bank, we always provide a fixed interest rate for the entire duration of mortgage.")

        self.avg_salary = self.validate_input_numerical("\nEnter the average net salary in the Czechia (default 40000 CZK): ", min_val=1000, max_val=TRILLION, default_val=40000, max_attempts=3, data_type=float)
        print(f"{S_CYAN}Note{E_CYAN}: Unless stated otherwise, the salary is assumed to be in Czech crowns.")

        self.retirement_age = self.validate_input_numerical( "\nEnter the age at which mortgage applicant will retire (default: 65 years old) ", min_val=18, max_val=100, default_val=65, max_attempts=3, data_type=int)

    def collect_mortgage_user_info(self):
        """Collect only mortgage parameters (loan amount and term) for existing applicant."""
        print(f"\n{S_BOLD}Please enter new mortgage parameters:\n{E_BOLD}")

        self.loan_amount = self.validate_input_numerical(
            "Enter loan amount (CZK): ",
            min_val=100000,
            max_val=TRILLION,
            data_type=int
        )
        self.loan_term = self.validate_input_numerical(
            "Enter loan term (years): ",
            min_val=1,
            max_val=35,
            data_type=int
        )

    def collect_other_user_info(self):
        """Collect detailed applicant information including demographics and financial status."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        print("\nPlease enter mortgage applicant information:\n")

        age = self.validate_input_numerical(
            "Age: ",
            min_val=18,
            max_val=65,
            data_type=int
        )

        government_employee = self.validate_input_alpha(
            "Government employee? (yes/no): ",
            ["yes", "no"]
        )

        min_basic_len = 10
        min_high_school_age = min_basic_len + 4
        min_bachelor_age = min_high_school_age + 1
        min_master_age = min_bachelor_age + 1
        min_phd_age = min_master_age + 2

        education_age_map = {
            "phd": (min_phd_age, "PhD"),
            "master": (min_master_age, "master's"),
            "bachelor": (min_bachelor_age, "bachelor's"),
            "high_school": (min_high_school_age, "high school")
        }

        for attempt in range(3):
            highest_education = self.validate_input_alpha(
                "Highest education (basic/high_school/bachelor/master/phd): ",
                ["basic", "high_school", "bachelor", "master", "phd"]
            )

            # Check if education is valid for the given age
            if highest_education in education_age_map:
                min_age, degree_name = education_age_map[highest_education]
                if age < min_age:
                    if attempt < 2:
                        print_invalid_input(f"You cannot have a {degree_name} degree before age {min_age}! [{attempt+1}/3]")
                    else:
                        print_error_handling(f"You cannot have a {degree_name} degree before age {min_age}!")
                        highest_education = "basic"
                        break
                else:
                    break
            else:
                break

        study_status = self.validate_input_alpha(
            "Is mortgage applicant student? (yes/no): ",
            ["yes", "no"]
        )

        if study_status == "yes":
            employment_type = "unemployed"
            len_employment = 0
            size_of_company = 0

        else:
            employment_type = self.validate_input_alpha(
                "Employment type (unemployed/temporary/freelancer/permanent): ",
                ["unemployed", "temporary", "freelancer", "permanent"]
            )

            max_study_time = 0
            if highest_education == "basic":
                max_study_time = min_basic_len
            elif highest_education == "high_school":
                max_study_time = min_high_school_age
            elif highest_education == "bachelor":
                max_study_time = min_bachelor_age
            elif highest_education == "master":
                max_study_time = min_master_age
            elif highest_education == "phd":
                max_study_time = min_phd_age

            if employment_type != "unemployed":
                len_employment = self.validate_input_numerical(
                    "How many years is applicant in same company employee? ",
                    min_val=0,
                    max_val=age-max_study_time,
                    data_type=int
                )

                size_of_company = self.validate_input_numerical(
                    "Number of company employees, where applicant works: ",
                    min_val=0,
                    max_val=TRILLION/1000,
                    data_type=int
                )
            else:
                len_employment = 0
                size_of_company = 0

        reported_monthly_income = self.validate_input_numerical(
            "Monthly income (in Czech Crowns): ",
            min_val=0,
            max_val=TRILLION,
            data_type=float
        )

        total_existing_debt = self.validate_input_numerical(
            "Total existing debt (in Czech Crowns): ",
            min_val=0,
            max_val=TRILLION,
            data_type=float
        )

        extra_net_worth = self.validate_input_alpha(
            "Do you have investments or do you owned property? (yes/no): ",
            ["yes", "no"]
        )

        if extra_net_worth == "yes":
            investments_value = self.validate_input_numerical(
                "Investments value (in Czech Crowns): ",
                min_val=0,
                max_val=TRILLION,
                data_type=float
            )

            property_owned_value = self.validate_input_numerical(
                "Property owned value (in Czech Crowns): ",
                min_val=0,
                max_val=TRILLION,
                data_type=float
            )
        else:
            investments_value = 0
            property_owned_value = 0

        housing_status = self.validate_input_alpha(
            "Housing status (rent/mortgage/own/homeless): ",
            ["rent", "mortgage", "own", "homeless"]
        )

        if housing_status == "homeless":
            credit_history = "bad"
        else:
            credit_history = self.validate_input_alpha(
                "Mortgage applicant credit score (bad/fair/excellent): ",
                ["bad", "fair", "excellent"]
            )

        return {
            'age': age,
            'government_employee': government_employee,
            'highest_education': highest_education,
            'study_status': study_status,
            'employment_type': employment_type,
            'len_employment': len_employment,
            'size_of_company': size_of_company,
            'reported_monthly_income': reported_monthly_income,
            'total_existing_debt': total_existing_debt,
            'investments_value': investments_value,
            'property_owned_value': property_owned_value,
            'housing_status': housing_status,
            'credit_history': credit_history
        }

    def calculate_monthly_payment(self, loan_amount, loan_term):
        """Calculate monthly payment using standard amortization formula."""
        la = loan_amount
        annual_rate = self.interest_rate
        years = loan_term

        r = annual_rate / 12             # monthly interest rate (already decimal)
        loan_term_months = years * 12    # number of payments

        if la <= 0 or loan_term <= 0:
            print_error_handling("Invalid arguments in 'calculate_monthly_payment'.")

        if r == 0:  # Handle zero interest rate
            return la / loan_term_months

        return la * r / (1 - (1 + r) ** (-loan_term_months))

    def predict_loan_approval(self, model_gbn, mortgage_applicant_data, loan_amount, loan_term):
        """Predict loan approval probability using trained Bayesian Network model."""
        # pylint: disable=too-many-locals
        age_young, age_prime, age_senior, age_old = encode_age_group(mortgage_applicant_data['age'])

        housing_map = {'rent': 0, 'mortgage': 1, 'own': 2}
        credit_map = {'bad': 0, 'fair': 1, 'excellent': 2}
        education_map = {'basic': 0, 'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        employment_map = {'unemployed': 0, 'temporary': 1, 'freelancer': 1, 'permanent': 2}

        monthly_payment = self.calculate_monthly_payment(loan_amount, loan_term)

        evidence = {
            'government_employee': 1 if mortgage_applicant_data['government_employee'] == 'yes' else 0,
            'age_young': age_young,
            'age_prime': age_prime,
            'age_senior': age_senior,
            'age_old': age_old,
            'highest_education': education_map.get(mortgage_applicant_data['highest_education'], 0),
            'employment_type': employment_map.get(mortgage_applicant_data['employment_type'], 0),
            'len_employment': mortgage_applicant_data['len_employment'],
            'size_of_company': mortgage_applicant_data['size_of_company'],
            'reported_monthly_income': mortgage_applicant_data['reported_monthly_income'],
            'total_existing_debt': mortgage_applicant_data['total_existing_debt'],
            'investments_value': mortgage_applicant_data['investments_value'],
            'property_owned_value': mortgage_applicant_data['property_owned_value'],
            'housing_status': housing_map.get(mortgage_applicant_data['housing_status'], 0),
            'credit_history': credit_map.get(mortgage_applicant_data['credit_history'], 0),
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'monthly_payment': monthly_payment
        }

        try:
            evidence_df = pd.DataFrame([evidence])  # Convert dict to DataFrame
            result = model_gbn.loan_approval_model.predict(evidence_df)

            if isinstance(result, tuple) and len(result) >= 2:
                variables_list = result[0]
                mean_values = result[1]

                if 'loan_approved' in variables_list:
                    idx = variables_list.index('loan_approved')
                    approval_prob = mean_values[0][idx]  # First row, idx column
                    approval_prob = float(approval_prob)
                    approval_prob = max(0, min(1, approval_prob))

                    if mortgage_applicant_data['reported_monthly_income'] != 0:
                        ratio_payment = monthly_payment / mortgage_applicant_data['reported_monthly_income']
                    else:
                        ratio_payment = monthly_payment / 1e8


                    # Apply adjustments based on key factors
                    adjustment_factor = self._calculate_adjustment_factor(
                        mortgage_applicant_data, ratio_payment, loan_term
                    )

                    # Aplikovat adjustment
                    approval_prob *= adjustment_factor

                    # Normalizovat na 0-1
                    approval_prob = min(max(approval_prob, 0.0), 1.0)

                    return approval_prob
                print_error_handling("loan_approved not found in predictions.")
                return 0.0
            print_error_handling("Unexpected result format from predict().")
            return 0.0

        except (ValueError, KeyError, AttributeError, IndexError, TypeError) as e:
            print_error_handling(f"Prediction failed: {e}")
            traceback.print_exc()
            return 0.0

    def _calculate_adjustment_factor(self, mortgage_applicant_data, ratio_payment, loan_term):
        """Calculate adjustment factor based on applicant profile and payment ratio."""
        adjustment_factor = 1.0

        # Bonuses for excellent profiles
        adjustment_factor *= self._get_excellent_credit_bonus(
            mortgage_applicant_data, ratio_payment
        )

        # Bonuses for fair profiles
        adjustment_factor *= self._get_fair_credit_bonus(
            mortgage_applicant_data, ratio_payment
        )

        # Penalties for high payment ratios or bad credit
        adjustment_factor *= self._get_risk_penalty(
            mortgage_applicant_data, ratio_payment
        )

        # Penalty for retirement age
        adjustment_factor *= self._get_retirement_penalty(
            mortgage_applicant_data, loan_term
        )

        return adjustment_factor

    def _get_excellent_credit_bonus(self, mortgage_applicant_data, ratio_payment):
        """Calculate bonus multiplier for excellent credit profiles."""
        if mortgage_applicant_data['credit_history'] != 'excellent':
            return 1.0

        if (mortgage_applicant_data['government_employee'] == 'yes' and
            ratio_payment <= 0.6):
            return 8.0
        if ratio_payment <= 0.7:
            return 6.0
        return 1.0

    def _get_fair_credit_bonus(self, mortgage_applicant_data, ratio_payment):
        """Calculate bonus multiplier for fair credit profiles."""
        if mortgage_applicant_data['credit_history'] != 'fair':
            return 1.0

        if (mortgage_applicant_data['government_employee'] == 'yes' and
            ratio_payment <= 0.3):
            return 4.0
        if ratio_payment <= 0.2:
            return 3.0
        return 1.0

    def _get_risk_penalty(self, mortgage_applicant_data, ratio_payment):
        """Calculate penalty multiplier for high-risk factors."""
        if ratio_payment > 1.0:
            return 0.1
        if ratio_payment > 0.6:
            return 0.4
        if mortgage_applicant_data['credit_history'] == 'bad':
            return 0.2
        return 1.0

    def _get_retirement_penalty(self, mortgage_applicant_data, loan_term):
        """Calculate penalty multiplier for loans extending past retirement."""
        mortgage_end_age = mortgage_applicant_data['age'] + loan_term
        if mortgage_end_age > self.retirement_age:
            years_after_retirement = mortgage_end_age - self.retirement_age
            return 0.8 ** years_after_retirement
        return 1.0

    def print_mortgage_applicant_info(self, mortgage_applicant_data, loan_amount, loan_term):
        """Display formatted mortgage application details in tabular format."""
        applicant_info = [
            ["Age", f"{mortgage_applicant_data['age']} years"],
            ["Government Employee", mortgage_applicant_data['government_employee'].capitalize()],
            ["Education", mortgage_applicant_data['highest_education'].replace('_', ' ').title()],
            ["Student Status", mortgage_applicant_data['study_status'].capitalize()],
            ["Employment Type", mortgage_applicant_data['employment_type'].capitalize()],
            ["Years in Company", f"{mortgage_applicant_data['len_employment']} years"],
            ["Company Size", f"{mortgage_applicant_data['size_of_company']:,} employees"],
            ["Monthly Income", f"{mortgage_applicant_data['reported_monthly_income']:,.0f} CZK"],
            ["Existing Debt", f"{mortgage_applicant_data['total_existing_debt']:,.0f} CZK"],
            ["Investments", f"{mortgage_applicant_data['investments_value']:,.0f} CZK"],
            ["Property Value", f"{mortgage_applicant_data['property_owned_value']:,.0f} CZK"],
            ["Housing Status", mortgage_applicant_data['housing_status'].capitalize()],
            ["Credit Score", mortgage_applicant_data['credit_history'].capitalize()],
            ["Loan Amount", f"{loan_amount:,.0f} CZK"],
            ["Loan Term", f"{loan_term} years"],
            ["Interest Rate", f"{self.interest_rate * 100:.1f}% p.a."],
            ["Monthly Payment", f"{self.calculate_monthly_payment(loan_amount, loan_term):,.0f} CZK"]
        ]

        print(f"\n{S_CYAN}══════ Mortgage Applicant Summary ══════{E_CYAN}")
        print(tabulate.tabulate(applicant_info, tablefmt="rounded_grid", stralign="left"))

    def options(self):
        """Display menu options and get user choice for program actions."""
        print("\n══════ Options ══════")
        print("1. Generate new dataset")
        print("2. Process new mortgage applicant")
        print("3. Full reset (new salary, rate & dataset)")
        print("4. Restart entire program")
        print("5. Generate updated diagram photos")
        print("6. New mortgage parameters (loan amount & term)")
        print("7. Exit program")
        print("═════════════════════\n")

        for i in range(5):
            try:
                choice = int(input("Choose option (1-7): "))
                if 1 <= choice <= 7:
                    return choice
                if i == 3:
                    print_error_handling("Too many invalid choices.")
                print_invalid_input(f"Please enter number between 1-7. [{i}/3]")
            except ValueError:
                print_invalid_input(f"Please enter a valid number. [{i}/3]")

        return None

    def print_mortgage_approval_prob(self, mortgage_applicant_data, model_gbn, loan_amount, loan_term):
        """Display loan approval probability with color-coded output based on approval status."""
        if mortgage_applicant_data["housing_status"] == "homeless":
            approval_prob = 0.0
            print(f"\n{S_RED}Loan automatically rejected: no permanent residence{E_RED}")
        else:
            approval_prob = self.predict_loan_approval(model_gbn, mortgage_applicant_data, loan_amount, loan_term)

        # try:
        #     mortgage_age_after_end = loan_term + mortgage_applicant_data["age"]
        #     for i in range(int(1e6)):
        #         if i % 5 == 0:
        #             if mortgage_age_after_end > (self.retirement_age + i):
        #                 approval_prob = approval_prob * 0.75
        #             else:
        #                 break
        # except KeyError:
        #     print_error_handling("Missing + required 'age' field in applicant data")

        # try:
        #     monthly_payment = calculate_monthly_payment(self.interest_rate, loan_term, loan_amount)
        #     total_stable_income = mortgage_applicant_data["reported_monthly_income"]
        #     for i in range(int(1e6)):
        #         if i % 5 == 0:
        #             if 2 > (total_stable_income - self.avg_salary * 0.1 + i * (self.avg_salary / 8) ):
        #                 approval_prob = approval_prob * 0.5
        #             else:
        #                 break
        # except KeyError as e:
        #     print_error_handling(f"{e}")

        if approval_prob > 0.65:
            print(f"\nMortgage approval probability: {S_GREEN}{approval_prob:.1%}{E_GREEN}")
        elif approval_prob < 0.4:
            print(f"\nMortgage approval probability: {S_RED}{approval_prob:.1%}{E_RED}")
        else:
            print(f"\nMortgage approval probability: {S_YELLOW}{approval_prob:.1%}{E_YELLOW}")

    def main(self):
        """Main program loop handling menu navigation and user interactions."""
        # pylint: disable=too-many-statements

        # handle first-ever program running.
        applicant_data = self.set_up_all()
        self.mortgage_applicant_data = applicant_data
        model_gbn = self.train_and_validate_gbn()
        self.model_gbn = model_gbn
        print_press_enter_to_continue()

        # handle other program runs.
        while 1:
            option = self.options()

            if option == 1: # Generate new dataset only
                self.set_up_only_dataset()
                model_gbn = self.train_and_validate_gbn()
                self.model_gbn = model_gbn
                loan_amount = self.validate_input_numerical(
                    "\nEnter loan amount (CZK): ",
                    min_val=100000,
                    max_val=TRILLION,
                    data_type=int
                )
                loan_term = self.validate_input_numerical(
                    "Enter loan term (years): ",
                    min_val=1,
                    max_val=35,
                    data_type=int
                )
                # Store loan parameters in instance variables
                self.loan_amount = loan_amount
                self.loan_term = loan_term
                self.print_mortgage_applicant_info(applicant_data, loan_amount, loan_term)
                self.print_mortgage_approval_prob(applicant_data, model_gbn, loan_amount, loan_term)

            elif option == 2: # Process new mortgage applicant
                applicant_data = self.collect_other_user_info()
                self.mortgage_applicant_data = applicant_data
                if model_gbn is None:
                    model_gbn = self.train_and_validate_gbn()
                self.model_gbn = model_gbn
                loan_amount = self.validate_input_numerical(
                    "\nEnter loan amount (CZK): ",
                    min_val=100000,
                    max_val=TRILLION,
                    data_type=int
                )
                loan_term = self.validate_input_numerical(
                    "Enter loan term (years): ",
                    min_val=1,
                    max_val=35,
                    data_type=int
                )
                # Store mortgage parameters in instance variables
                self.loan_amount = loan_amount
                self.loan_term = loan_term
                self.print_mortgage_applicant_info(applicant_data, loan_amount, loan_term)
                self.print_mortgage_approval_prob(applicant_data, model_gbn, loan_amount, loan_term)

            elif option == 3: # Full reset (new salary, rate & dataset)
                self.collect_main_user_info()
                self.collect_datasets_user_info()
                self.generate_csv_dataset()
                applicant_data = self.collect_other_user_info()
                # Store data in instance variables
                self.mortgage_applicant_data = applicant_data
                model_gbn = self.train_and_validate_gbn()
                self.model_gbn = model_gbn
                loan_amount = self.validate_input_numerical(
                    "\nEnter loan amount (CZK): ",
                    min_val=100000,
                    max_val=TRILLION,
                    data_type=int
                )
                loan_term = self.validate_input_numerical(
                    "Enter loan term (years): ",
                    min_val=1,
                    max_val=35,
                    data_type=int
                )
                # Store loan parameters in instance variables
                self.loan_amount = loan_amount
                self.loan_term = loan_term
                self.print_mortgage_applicant_info(applicant_data, loan_amount, loan_term)
                self.print_mortgage_approval_prob(applicant_data, model_gbn, loan_amount, loan_term)

            elif option == 4: # Restart entire program
                applicant_data = self.set_up_all()
                self.mortgage_applicant_data = applicant_data
                model_gbn = None
                self.model_gbn = None

            elif option == 5:
                model = GaussianBayesianNetwork(save_diagram_to_png=True)
                model.save_diagram_of_gbn(print_info=True)
                create_bayesian_network_visualization()

            elif option == 6: # New mortgage parameters (loan amount & term) only
                if hasattr(self, 'mortgage_applicant_data') and self.mortgage_applicant_data and (model_gbn is not None or (hasattr(self, 'model_gbn') and self.model_gbn is not None)):
                    self.collect_mortgage_user_info()
                    # Use local model_gbn if available, otherwise use instance variable
                    active_model = model_gbn if model_gbn is not None else self.model_gbn
                    self.print_mortgage_applicant_info(self.mortgage_applicant_data, self.loan_amount, self.loan_term)
                    self.print_mortgage_approval_prob(self.mortgage_applicant_data, active_model, self.loan_amount, self.loan_term)
                else:
                    print_error_handling("No applicant data or model available. Please use option 2 first.")

            elif option == 7:
                print(f"\n{S_CYAN}Thank you for using BayesianHill & Co. Bank Mortgage Software, Goodbye!{E_CYAN}\n")
                sys.exit(0)

            print_press_enter_to_continue()


    def set_up_only_dataset(self):
        """Set up only dataset generation without model training or prediction."""
        self.collect_datasets_user_info()
        self.generate_csv_dataset()


    def set_up_all(self):
        """Complete setup including dataset generation, model training, and prediction workflow."""
        self.collect_main_user_info()
        self.collect_datasets_user_info()
        self.generate_csv_dataset()
        model_gbn = self.train_and_validate_gbn()
        mortgage_applicant_data = self.collect_other_user_info()

        loan_amount = self.validate_input_numerical(
            "\nEnter loan amount (CZK): ",
            min_val=100000,
            max_val=TRILLION,
            data_type=int
        )

        loan_term = self.validate_input_numerical(
            "Enter loan term (years): ",
            min_val=1,
            max_val=35,
            data_type=int
        )

        self.print_mortgage_applicant_info(mortgage_applicant_data, loan_amount, loan_term)
        self.print_mortgage_approval_prob(mortgage_applicant_data, model_gbn, loan_amount, loan_term)
        return mortgage_applicant_data

if __name__ == "__main__":
    handler = InputHandler()
    handler.main()
