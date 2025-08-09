import numpy as np
import logging
import pandas as pd

logging.getLogger('numexpr').setLevel(logging.WARNING) # INFO:numexpr.utils:NumExpr
from pgmpy.inference import VariableElimination

from mortgage_approval_bayesian_network.gaussian_bayesian_network import loan_approval_model
from mortgage_approval_bayesian_network.constants import *


def calculate_monthly_payment(loan_amount, loan_term_years, annual_rate=0.05):
    """Calculate monthly payment using standard amortization formula"""
    if loan_amount <= 0 or loan_term_years <= 0:
        return 0
    
    monthly_rate = annual_rate / 12
    n_payments = loan_term_years * 12
    
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / \
                      ((1 + monthly_rate)**n_payments - 1)
    
    return monthly_payment

def encode_age_group(age):
    """Convert age to age group indicators"""
    age_young = 1 if 18 <= age < 25 else 0
    age_prime = 1 if 25 <= age < 40 else 0
    age_senior = 1 if 40 <= age < 55 else 0
    age_old = 1 if age >= 55 else 0
    return age_young, age_prime, age_senior, age_old

def print_wrong_input_message(string: str):
    print(f"{S_YELLOW}Wrong Input{E_YELLOW}: {string}")

def collect_user_info():
    """Collect applicant information from banker"""
    print(f"\n{S_CYAN}=== BayesianHill Bank - Mortgage Approval System ==={E_CYAN}")

    interest_rate_str = input(f"\nEnter the interest rate at which we will lend the mortgage: (from 0 to 0.27; default 0.05) ")
    print(f"{S_CYAN}Note{E_CYAN}: At Bayesianhill Bank, we always provide a fixed interest rate for the entire duration of mortgage.")

    if interest_rate_str == "":
        interest_rate = 0.05
    else:
        interest_rate = float(interest_rate_str)

    avg_salary_str = input(f"Enter the average salary in the Czechia: (default 50000 CZK) ")
    if avg_salary_str == "":
        avg_salary = 50000
    else:
        avg_salary = float(avg_salary_str)

    # TODO: vygenerovat synteticka data na zaklade avg_salary a spocitat hypoteku pomoci interest_rate.

    print(f"\nPlease enter mortgage applicant information:\n")
    
    # Personal info
    try:
        # age
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    age = int(input("Age: ").strip())
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if age <= 0 or age >= 100:
                    print_wrong_input_message(f"Please enter age between 0 and 100. [{i+1}/3]")
                    continue
                if age > 0 and age < 100:
                    break

        # government_employee
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    government_employee = input("Government employee? (yes/no): ").strip().lower()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if government_employee != "yes" and government_employee != "no":
                    print_wrong_input_message(f"Please enter \"yes\" or \"no\". [{i+1}/3]")
                    continue
                if government_employee == "yes" or government_employee == "no":
                    break

        # highest_education
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    highest_education = input("Highest education (basic / high_school / bachelor / master / phd): ").lower().strip()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i + 1}/3]")
                    continue
                valid_options = ["basic", "high_school", "bachelor", "master", "phd"]
                if highest_education not in valid_options:
                    print_wrong_input_message(f"Please enter \"basic\", \"high_school\", \"bachelor\", \"master\" or \"phd\". [{i+1}/3]")
                    continue
                break

        # study_status
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    study_status = input("Is mortgage applicant student? (yes / no): ").strip().lower()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i + 1}/3]")
                    continue
                if study_status not in ["yes", "no"]:
                    print_wrong_input_message(f"Please enter \"yes\" or \"no\". [{i+1}/3]")
                    continue
                break

        # Person is in school or university.
        if study_status == "yes":
            employment_type = "unemployed"
            len_employment = 0
            size_of_company = 0

        # Person is not anymore in school or university.
        if study_status == "no":

            # employment_type
            for i in range(4):
                if i > 2:
                    exit(1)
                else:
                    try:
                        employment_type = input("Employment type (unemployed/temporary/freelancer/permanent): ").strip().lower()
                    except Exception as e:
                        print_wrong_input_message(f"{e}. [{i+1}/3]")
                        continue
                    valid_employment_types = ["unemployed", "temporary", "freelancer", "permanent"]
                    if employment_type not in valid_employment_types:
                        print_wrong_input_message(f"Please enter \"unemployed\", \"temporary\", \"freelancer\" or \"permanent\". [{i+1}/3]")
                        continue
                    break

            # Person is not unemployed.
            if employment_type != "unemployed":

                # len_employment
                for i in range(4):
                    if i > 2:
                        exit(1)
                    else:
                        try:
                            len_employment = int(input("How many years is applicant in same company employee? ").strip())
                        except Exception as e:
                            print_wrong_input_message(f"{e}. [{i+1}/3]")
                            continue
                        if len_employment < 0 or len_employment > 189:
                            print_wrong_input_message(f"Please enter number between 0 and 189. [{i+1}/3]")
                            continue
                        break

                # size_of_company
                for i in range(4):
                    if i > 2:
                        exit(1)
                    else:
                        try:
                            size_of_company = int(input("Number of company employees, where applicant works: ").strip())
                        except Exception as e:
                            print_wrong_input_message(f"{e}. [{i+1}/3]")
                            continue
                        if size_of_company < 0 or size_of_company > 1e30:
                            print_wrong_input_message(f"Please enter number between 0 and 1e30. [{i+1}/3]")
                            continue
                        break

            # Person is  unemployed.
            if employment_type == "unemployed":
                len_employment = 0
                size_of_company = 0


        # reported_monthly_income
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    reported_monthly_income = float(input("Monthly income (in Czech Crowns): ").strip())
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if reported_monthly_income < 0 or reported_monthly_income > 1e30:
                    print_wrong_input_message(f"Please enter number between 0 and 1e30. [{i+1}/3]")
                    continue
                break

        # total_existing_debt
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    total_existing_debt = float(input("Total existing debt (in Czech Crowns): ").strip())
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if total_existing_debt < 0 or total_existing_debt > 1e30:
                    print_wrong_input_message(f"Please enter positive number between 0 and 1e30. [{i+1}/3]")
                    continue
                break

        # extra_net_worth
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    extra_net_worth = input("Do you have investments or do you owned property? (yes/no): ").lower().strip()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if extra_net_worth not in ["yes", "no"]:
                    print_wrong_input_message(f"Please enter \"yes\" or \"no\". [{i+1}/3]")
                    continue
                break

        if extra_net_worth == "yes":
            # investments_value
            for i in range(4):
                if i > 2:
                    exit(1)
                else:
                    try:
                        investments_value = float(input("Investments value (in Czech Crowns): ").strip())
                    except Exception as e:
                        print_wrong_input_message(f"{e}. [{i+1}/3]")
                        continue
                    if investments_value < 0 or investments_value > 1e100:
                        print_wrong_input_message(f"Please enter number between 0 and 1e100. [{i+1}/3]")
                        continue
                    break

            # property_owned_value
            for i in range(4):
                if i > 2:
                    exit(1)
                else:
                    try:
                        property_owned_value = float(input("Property owned value (in Czech Crowns): ").strip())
                    except Exception as e:
                        print_wrong_input_message(f"{e}. [{i+1}/3]")
                        continue
                    if property_owned_value < 0 or property_owned_value > 1e100:
                        print_wrong_input_message(f"Please enter number between 0 and 1e100. [{i+1}/3]")
                        continue
                    break
        else:
            investments_value = 0
            property_owned_value = 0


        # housing_status
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    housing_status = input("Housing status (rent/mortgage/own): ").lower().strip()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if housing_status not in ["rent", "mortgage", "own"]:
                    print_wrong_input_message(f"Please enter \"rent\", \"mortgage\" or \"own\". [{i+1}/3]")
                    continue
                break

        # credit_history
        for i in range(4):
            if i > 2:
                exit(1)
            else:
                try:
                    credit_history = input("Mortgage applicant credit score (bad/fair/good/excellent): ").lower().strip()
                except Exception as e:
                    print_wrong_input_message(f"{e}. [{i+1}/3]")
                    continue
                if credit_history not in ["bad", "fair", "good", "excellent"]:
                    print_wrong_input_message(f"Please enter \"bad\", \"fair\", \"good\" or \"excellent\". [{i+1}/3]")
                    continue
                break

    except Exception as e:
        print_wrong_input_message(str(e))

    return {
        'age': age,
        'government_employee': government_employee,
        'highest_education': highest_education,
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

def prepare_evidence(user_info, loan_amount, loan_term):
    """Prepare evidence for inference"""
    # Calculate monthly payment
    monthly_payment = calculate_monthly_payment(loan_amount, loan_term)
    
    # Get age groups
    age_young, age_prime, age_senior, age_old = encode_age_group(user_info['age'])
    
    # Convert text to numeric values
    gov_emp = 1 if user_info['government_employee'] == 'yes' else 0

    housing_map = {'rent': 0, 'mortgage': 1, 'own': 2}
    credit_map = {'bad': 0, 'fair': 1, 'good': 2, 'excellent': 3}
    education_map = {'basic': 0, 'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
    employment_map = {'unemployed': 0, 'temporary': 1, 'freelancer': 1, 'permanent': 2}
    
    return {
        'government_employee': gov_emp,
        'age_young': age_young,
        'age_prime': age_prime,
        'age_senior': age_senior,
        'age_old': age_old,
        'highest_education': education_map.get(user_info['highest_education'], 0),
        'employment_type': employment_map.get(user_info['employment_type'], 0),
        'len_employment': user_info['len_employment'],
        'size_of_company': user_info['size_of_company'],
        'reported_monthly_income': user_info['reported_monthly_income'],
        'total_existing_debt': user_info['total_existing_debt'],
        'investments_value': user_info['investments_value'],
        'property_owned_value': user_info['property_owned_value'],
        'housing_status': housing_map.get(user_info['housing_status'], 0),
        'credit_history': credit_map.get(user_info['credit_history'], 0),
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'monthly_payment': monthly_payment
    }

def calculate_approval_manually(evidence):
    """Calculate approval probability manually using learned CPD parameters"""
    try:
        # Get the CPD for loan_approved
        cpd_loan_approved = loan_approval_model.get_cpds('loan_approved')
        
        # Get beta coefficients
        beta = cpd_loan_approved.beta
        evidence_vars = cpd_loan_approved.evidence
        
        # Build evidence vector
        evidence_vector = [1]  # intercept
        for var in evidence_vars:
            evidence_vector.append(evidence.get(var, 0))
        
        # Calculate linear combination
        score = np.dot(beta, evidence_vector)
        
        # Convert to probability (sigmoid)
        prob = 1 / (1 + np.exp(-score))
        
        return prob
    except Exception as e:
        print(f"Error in manual calculation: {e}")
        return 0.5

def main():
    # Collect user information
    user_info = collect_user_info()
    
    # Get loan scenarios
    print(f"\n{S_YELLOW}=== Loan Scenarios ==={E_YELLOW}")
    min_amount = float(input("Minimum loan amount (CZK): "))
    max_amount = float(input("Maximum loan amount (CZK): "))
    step_amount = float(input("Step size for amount (CZK): "))
    
    loan_terms = input("Loan terms to test (comma-separated years, e.g. 10,15,20,30): ").strip()
    loan_terms = [int(term.strip()) for term in loan_terms.split(',')]
    
    # Test different loan scenarios
    print(f"\n{S_GREEN}=== Loan Approval Predictions ==={E_GREEN}\n")
    print(f"{'Loan Amount':>15} {'Term':>6} {'Monthly Payment':>15} {'Approval Probability':>20}")
    print("-" * 70)
    
    results = []
    
    for loan_term in loan_terms:
        loan_amount = min_amount
        while loan_amount <= max_amount:
            # Prepare evidence
            evidence = prepare_evidence(user_info, loan_amount, loan_term)
            
            # Calculate approval probability
            try:
                approval_prob = calculate_approval_manually(evidence)
                monthly_payment = evidence['monthly_payment']
                
                # Store result
                results.append({
                    'amount': loan_amount,
                    'term': loan_term,
                    'payment': monthly_payment,
                    'probability': approval_prob
                })
                
                # Display result
                prob_color = S_GREEN if approval_prob > 0.7 else S_YELLOW if approval_prob > 0.4 else S_RED
                print(f"{loan_amount:>15,.0f} {loan_term:>6}y {monthly_payment:>15,.0f} "
                      f"{prob_color}{approval_prob:>19.1%}{E_RED if approval_prob < 0.4 else E_YELLOW if approval_prob < 0.7 else E_GREEN}")
                
            except Exception as e:
                print(f"{S_RED}Error calculating for {loan_amount} CZK, {loan_term}y: {e}{E_RED}")
            
            loan_amount += step_amount
    
    # Find best scenario
    if results:
        best = max(results, key=lambda x: x['probability'])
        print(f"\n{S_CYAN}=== Best Loan Scenario ==={E_CYAN}")
        print(f"Amount: {best['amount']:,.0f} CZK")
        print(f"Term: {best['term']} years")
        print(f"Monthly Payment: {best['payment']:,.0f} CZK")
        print(f"Approval Probability: {best['probability']:.1%}")

if __name__ == "__main__":
    main()