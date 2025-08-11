import numpy as np
import logging
import pandas as pd

logging.getLogger('numexpr').setLevel(logging.WARNING) # INFO:numexpr.utils:NumExpr
from pgmpy.inference import VariableElimination

from data_generation_realistic import encode_age_group
from gaussian_bayesian_network import loan_approval_model
from constants import *


def calculate_monthly_payment(loan_amount, loan_term_years, annual_rate=0.05):
    """Calculate monthly payment using standard amortization formula"""
    if loan_amount <= 0 or loan_term_years <= 0:
        return 0
    
    monthly_rate = annual_rate / 12
    n_payments = loan_term_years * 12
    
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / \
                      ((1 + monthly_rate)**n_payments - 1)
    
    return monthly_payment

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

    avg_salary_str = input(f"Enter the average net salary in the Czechia: (default 50000 CZK) ")
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
                if age < 18:
                    print_wrong_input_message(f"Age must be at least 18. [{i + 1}/3]")
                    continue
                if age > 65:
                    print_wrong_input_message(f"Sorry, we don't provide new mortgages to people over 65. [{i+1}/3]")
                    continue
                if age >= 18 and age <= 65:
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
        
        # DEBUG: Print CPD parameters
        print(f"\n{S_CYAN}=== DEBUG: loan_approved CPD Parameters ==={E_CYAN}")
        print(f"Evidence variables: {evidence_vars}")
        print(f"Beta coefficients shape: {beta.shape}")
        print(f"Beta coefficients: {beta}")
        print(f"Beta intercept (beta[0]): {beta[0]}")
        # LinearGaussianCPD doesn't have variance attribute
        # print(f"Variance: {cpd_loan_approved.variance}")
        
        # Print beta coefficients with variable names
        print(f"\n{S_YELLOW}Beta coefficients breakdown:{E_YELLOW}")
        print(f"  Intercept: {beta[0]:.6f}")
        for i, var in enumerate(evidence_vars):
            print(f"  {var}: {beta[i+1]:.6f}")
        
        # Build evidence vector
        evidence_vector = [1]  # intercept
        for var in evidence_vars:
            evidence_vector.append(evidence.get(var, 0))
        
        # Calculate linear combination
        score = np.dot(beta, evidence_vector)
        
        # DEBUG: Print the raw score before sigmoid
        print(f"\n{S_GREEN}Raw score (linear combination):{E_GREEN} {score:.6f}")
        print(f"This raw score IS the predicted loan_approved value from the Linear Gaussian model")
        
        # The model was trained with loan_approved values between 0 and 1
        # So the raw score should be used directly, not transformed!
        print(f"\n{S_RED}IMPORTANT FINDING:{E_RED}")
        print(f"The training data has loan_approved values between 0 and 1 (probabilities)")
        print(f"LinearGaussianBayesianNetwork predicts these values directly")
        print(f"NO sigmoid transformation is needed!")
        
        # Clip the score to [0, 1] range since it's a probability
        prob_direct = np.clip(score, 0, 1)
        
        # Show both interpretations
        prob_sigmoid = 1 / (1 + np.exp(-score))
        print(f"\n{S_YELLOW}Comparison:{E_YELLOW}")
        print(f"  Direct interpretation (correct): {prob_direct:.6f}")
        print(f"  With sigmoid (incorrect): {prob_sigmoid:.6f}")
        print("-" * 70)
        
        # Return the direct interpretation
        return prob_direct
    except Exception as e:
        print(f"Error in manual calculation: {e}")
        return 0.5

def analyze_loan_approved_values():
    """Analyze the values of loan_approved in the training data"""
    try:
        from data_loader import LoanDataLoader
        loader = LoanDataLoader()
        data = loader.load_data("datasets/mortgage_applications.csv")
        all_data = loader.get_all_data_numeric()
        
        if 'loan_approved' in all_data.columns:
            loan_values = all_data['loan_approved']
            print(f"\n{S_CYAN}=== Training Data Analysis for loan_approved ==={E_CYAN}")
            print(f"Unique values: {loan_values.unique()}")
            print(f"Min value: {loan_values.min()}")
            print(f"Max value: {loan_values.max()}")
            print(f"Mean value: {loan_values.mean():.4f}")
            print(f"Std deviation: {loan_values.std():.4f}")
            print(f"Value counts:\n{loan_values.value_counts()}")
            print("-" * 70)
    except Exception as e:
        print(f"Error analyzing training data: {e}")

def test_inference_methods(evidence):
    """Test different inference methods to understand the model"""
    try:
        print(f"\n{S_CYAN}=== Model Structure ==={E_CYAN}")
        
        # For LinearGaussianBayesianNetwork, show CPD structure
        print(f"Model type: LinearGaussianBayesianNetwork")
        print(f"Number of nodes: {len(loan_approval_model.nodes())}")
        print(f"Number of edges: {len(loan_approval_model.edges())}")
        
        # Show CPDs instead of factors
        cpds = loan_approval_model.get_cpds()
        print(f"Number of CPDs: {len(cpds)}")
        
        # Note: LinearGaussianBayesianNetwork doesn't support standard inference engines
        print(f"\n{S_YELLOW}Note:{E_YELLOW} LinearGaussianBayesianNetwork requires manual computation")
        print("Standard inference engines (VariableElimination) are not supported.")
        
    except Exception as e:
        print(f"Error in model testing: {e}")

def main():
    # Analyze the training data first
    # analyze_loan_approved_values()

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
    first_run = True
    
    for loan_term in loan_terms:
        loan_amount = min_amount
        while loan_amount <= max_amount:
            # Prepare evidence
            evidence = prepare_evidence(user_info, loan_amount, loan_term)
            
            # For the first scenario, test inference methods
            if first_run:
                test_inference_methods(evidence)
                first_run = False
            
            # Calculate approval probability manually
            try:
                # For LinearGaussianBayesianNetwork, we need to compute intermediate values
                # First, calculate all intermediate nodes
                
                # Calculate stability_income
                stability_cpd = loan_approval_model.get_cpds('stability_income')
                stability_vars = ['government_employee', 'age_young', 'age_prime', 'age_senior', 
                                'age_old', 'len_employment', 'size_of_company', 'highest_education', 
                                'employment_type']
                stability_evidence = [1] + [evidence.get(var, 0) for var in stability_vars]
                stability_income = np.dot(stability_cpd.beta, stability_evidence)
                
                # Calculate total_stable_income_monthly
                income_cpd = loan_approval_model.get_cpds('total_stable_income_monthly')
                income_evidence = [1, evidence['reported_monthly_income'], stability_income]
                total_stable_income = np.dot(income_cpd.beta, income_evidence)
                
                # Calculate ratios
                total_debt = evidence.get('total_existing_debt', 0)
                investments = evidence.get('investments_value', 0)
                property_value = evidence.get('property_owned_value', 0)
                
                # Avoid division by zero
                if total_stable_income > 0:
                    ratio_income_debt = total_debt / total_stable_income
                    ratio_payment_to_income = monthly_payment / total_stable_income
                else:
                    ratio_income_debt = 0
                    ratio_payment_to_income = 0
                
                net_worth = investments + property_value
                if net_worth > 0:
                    ratio_debt_net_worth = total_debt / net_worth
                else:
                    ratio_debt_net_worth = 0
                
                # Calculate defaulted score
                defaulted_cpd = loan_approval_model.get_cpds('defaulted')
                defaulted_vars = defaulted_cpd.evidence
                defaulted_evidence = [1]  # intercept
                for var in defaulted_vars:
                    if var == 'total_existing_debt':
                        defaulted_evidence.append(total_debt)
                    elif var == 'total_stable_income_monthly':
                        defaulted_evidence.append(total_stable_income)
                    elif var == 'core_net_worth':
                        defaulted_evidence.append(net_worth)
                    elif var == 'housing_status':
                        defaulted_evidence.append(evidence.get('housing_status', 0))
                    elif var == 'credit_history':
                        defaulted_evidence.append(evidence.get('credit_history', 0))
                    elif var == 'ratio_debt_net_worth':
                        defaulted_evidence.append(ratio_debt_net_worth)
                    elif var == 'ratio_payment_to_income':
                        defaulted_evidence.append(ratio_payment_to_income)
                    else:
                        defaulted_evidence.append(0)
                
                defaulted_score = np.dot(defaulted_cpd.beta, defaulted_evidence)
                
                # Finally, calculate loan approval
                approval_cpd = loan_approval_model.get_cpds('loan_approved')
                approval_evidence = [1]  # intercept
                for var in approval_cpd.evidence:
                    if var == 'ratio_payment_to_income':
                        approval_evidence.append(ratio_payment_to_income)
                    elif var == 'ratio_income_debt':
                        approval_evidence.append(ratio_income_debt)
                    elif var == 'ratio_debt_net_worth':
                        approval_evidence.append(ratio_debt_net_worth)
                    elif var == 'credit_history':
                        approval_evidence.append(evidence.get('credit_history', 0))
                    elif var == 'loan_amount':
                        approval_evidence.append(loan_amount)
                    elif var == 'loan_term':
                        approval_evidence.append(loan_term)
                    elif var == 'defaulted':
                        approval_evidence.append(defaulted_score)
                    else:
                        approval_evidence.append(0)
                
                approval_score = np.dot(approval_cpd.beta, approval_evidence)
                approval_prob = np.clip(approval_score, 0, 1)
                
                # Apply age-based penalty (since model doesn't have enough data for older people)
                age = user_info['age']
                if age >= 50:
                    # Reduce approval probability for older applicants
                    # Age 50: multiply by 0.9, Age 60: multiply by 0.5, Age 65: multiply by 0.2
                    age_factor = max(0.2, 1.0 - (age - 45) * 0.03)
                    approval_prob *= age_factor
                
                # Also penalize if loan term extends past retirement age (65)
                end_age = age + loan_term
                if end_age > 65:
                    # Reduce probability based on how far past retirement the loan extends
                    retirement_factor = max(0.1, 1.0 - (end_age - 65) * 0.05)
                    approval_prob *= retirement_factor
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