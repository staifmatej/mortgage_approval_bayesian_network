import csv
import os
import random
import time
import math


def encode_age_group(age):
    """Convert age to age group indicators - matching the dataset encoding"""
    age_young = 1 if 18 <= age < 25 else 0
    age_prime = 1 if 25 <= age < 40 else 0
    age_senior = 1 if 40 <= age < 50 else 0
    age_old = 1 if age >= 50 else 0
    return age_young, age_prime, age_senior, age_old


class DataGenerator:
    def __init__(self, avg_salary, interest_rate, num_records):
        self.avg_salary = avg_salary
        self.interest_rate = interest_rate
        self.num_records = num_records
        self.csv_path = "datasets/mortgage_applications.csv"

    def initialize_header(self):
        self.create_folder()
        self.seed_random()
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["id",
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
                                       "defaulted",
                                       "loan_approved"])

    def create_folder(self):
        os.makedirs("datasets", exist_ok=True)

    def seed_random(self):
        random.seed(time.time())

    def generate_random_float_from_0_to_1(self):
        return random.random()

    def generate_base_value_property(self, reported_monthly_income):
        base_value_property = reported_monthly_income * self.generate_random_int_from_x_to_y(100, 270)
        base_value_property = math.ceil(base_value_property / 1000) * 1000
        return base_value_property

    def generate_random_investments(self, age, reported_monthly_income):
        """
        Investment simulation where a person saves 'realistic_investment_value' percent of
        their monthly salary each month for 'investment_interval' years with 8% annual return.
        """
        investment_interval = age - 18
        realistic_investment_value = self.generate_random_int_from_x_to_y(10,3000) / 100
        savings_for_investments_per_year = reported_monthly_income * realistic_investment_value * 12
        return (savings_for_investments_per_year * (((1.08)**investment_interval-1)/0.08)*1.08) # SPY has historically 7-9% annually return.

    def generate_random_int_from_x_to_y(self, x_from, y_to):
        return random.randint(x_from, y_to)


    def generate_realistic_data(self):
        self.initialize_header()
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for i in range(self.num_records):

                def generate_id():
                    """Generate deterministically id."""
                    return i + 1

                def generate_government_employee():
                    """ 20% government_employee & 80% prob that person is not government_employee."""
                    government_employee = "yes" if self.generate_random_float_from_0_to_1() < 0.2 else "no"
                    return government_employee

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
                    if r >= 0 and r < 0.2:
                        return "basic"
                    if r >= 0.2 and r < 0.65:
                        return "high_school"
                    if r >= 0.65 and r < 0.85:
                        return "bachelor"
                    if r >= 0.85 and r < 0.95:
                        return "master"
                    if r >= 0.95 and r <= 1:
                        return "phd"

                def generate_age():
                    """Generate random age."""
                    return self.generate_random_int_from_x_to_y(18,65)

                id = generate_id()
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
                        if r >= 0 and r < 0.05: # 5%
                            return "unemployed"
                        if r >= 0.05 and r < 0.25: # 20%
                            return "temporary"
                        if r >= 0.25 and r < 0.45: # 20%
                            return "freelancer"
                        if r >= 0.45 and r <= 1: # 55%
                            return "permanent"

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
                    adult_treshold = 18 # in Czechia is person adult from age of 18.
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
                    len_employment = max_possible_of_len_employment - self.generate_random_int_from_x_to_y(0, max_possible_of_len_employment)
                    if government_employee == "yes":
                        """Bonus years in data simulation for more stable jobs."""
                        if len_employment + len_employment/2 < max_possible_of_len_employment:
                            len_employment += int(max_possible_of_len_employment / 2)

                    if len_employment < 0:
                        """ For people how study PhD. in U.S.A. for example and they are younger than in Czechia is normal. """
                        len_employment = 0
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

                size_of_company = generate_size_of_company()
                age_young, age_prime, age_senior, age_old = encode_age_group(age)

                def generate_study_status():
                    """Study status is for people who are actively still studying."""
                    if age > 26 and highest_education != "master":
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.99: return "no"
                        else: return "yes"
                    if age <= 18 and highest_education == "basic":
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.05: return "no"
                        if r >= 0.05 and r <= 1: return "yes"
                    if age <= 23 and highest_education == "high_school":
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.7: return "no"
                        if r >= 0.7 and r <= 1: return "yes"
                    if age <= 25 and highest_education == "bachelor":
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.5: return "no"
                        if r >= 0.5 and r <= 1: return "yes"
                    if age <= 30 and highest_education == "master":
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.97: return "no"
                        else: return "yes"
                    else: return "no"

                study_status = generate_study_status()

                def generate_reported_monthly_income():
                    """Generate monthly income report with Gaussian curve."""
                    min_salary = int(self.avg_salary / 2)
                    max_salary = self.avg_salary * 10
                    base_salary = self.avg_salary

                    if age < 18:
                        return 0

                    if employment_type == "unemployed":
                        # unemployment benefit
                        benefit = True if self.generate_random_float_from_0_to_1() > 0.7 else False
                        if benefit is False:
                            return 0
                        if benefit is True:
                            return self.generate_random_int_from_x_to_y(0, min_salary / 2.22)
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
                        mean = base_salary * 1.1
                        std_dev = base_salary * 0.25
                    elif highest_education == "master":
                        mean = base_salary * 1.4
                        std_dev = base_salary * 0.3
                    elif highest_education == "phd":
                        mean = base_salary * 1.8
                        std_dev = base_salary * 0.35

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

                    if employment_type == "permanent" and (highest_education == "master" or highest_education == "phd") and len_employment > 20:
                        std_dev *= 0.5

                    # Generate salaries with Gaussian distribution
                    salary = random.gauss(mean, std_dev)

                    salary = max(min_salary, min(max_salary, salary))
                    return math.ceil(salary / 100) * 100

                reported_monthly_income = generate_reported_monthly_income()

                def generate_credit_history():
                    # TODO for CLAUDE CODE: napis mi toto generate_credit_history, cim vyssi plat nad prumerem a cim vyssi vzdelani tim lepsi credit history vyhodnot, penalizuj velky dluh, penalizuj kdyz nema  clovek nemovitost ci investice (oboji dohromady ber jako celek) + odcen pokud pracuje pro stat (ale zde jen trosku  opravdu trosku!) odcen vzdelani hodne! + odcen stability income metriku

                def generate_housing_status():
                    if reported_monthly_income > self.avg_salary * 4:
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.70: return "own"
                        if r >= 0.7 and r < 0.95: return "mortgage"
                        if r >= 0.95 and r <= 1.0: return "rent"

                    if reported_monthly_income > self.avg_salary * 2:
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.40: return "own"
                        if r >= 0.4 and r < 0.90: return "mortgage"
                        if r >= 0.90 and r <= 1.0: return "rent"

                    if reported_monthly_income > self.avg_salary * 1.5:
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.10: return "own"
                        if r >= 0.1 and r < 0.60: return "mortgage"
                        if r >= 0.60 and r <= 1.0: return "rent"

                    if reported_monthly_income < self.avg_salary * 0.25:
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.0001: return "own"
                        if r >= 0.0001 and r < 0.0005: return "mortgage"
                        if r >= 0.0005 and r <= 1.0: return "rent"

                    if reported_monthly_income < self.avg_salary * 0.5:
                        r = self.generate_random_float_from_0_to_1()
                        if r >= 0 and r < 0.01: return "own"
                        if r >= 0.01 and r < 0.09: return "mortgage"
                        if r >= 0.09 and r <= 1.0: return "rent"

                    r = self.generate_random_float_from_0_to_1()
                    if r >= 0 and r < 0.02: return "own"
                    if r >= 0.02 and r < 0.20: return "mortgage"
                    if r >= 0.20 and r <= 1.0: return "rent"

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
                    if highest_education == "master" or highest_education == "phd":
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
                        return property_owned_value * self.generate_base_value_property(reported_monthly_income) * self.generate_random_float_from_0_to_1()
                    else:
                        if self.generate_random_float_from_0_to_1() > 0.50:
                            return property_owned_value * 0.8 * self.generate_base_value_property(reported_monthly_income) * self.generate_random_float_from_0_to_1()
                    return 0

                total_existing_debt = generate_total_existing_debt()

                def generate_core_net_worth():
                    net_worth = 0
                    net_worth += property_owned_value + reported_monthly_income * 3 + investments_value - total_existing_debt
                    return net_worth

                core_net_worth = generate_core_net_worth()

