"""Independent test data generator with different parameters for realistic evaluation."""

import sys
import os
import pandas as pd

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # pylint: disable=import-error
    from data_generation_realistic import DataGenerator
    from utils.constants import S_CYAN, E_CYAN, S_GREEN, E_GREEN
except ImportError:
    # Handle import errors during static analysis
    S_CYAN = E_CYAN = S_GREEN = E_GREEN = ""

class TestDataGenerator:
    """Generate independent test dataset with different economic parameters."""

    def __init__(self, base_avg_salary=35000, base_interest_rate=0.045):
        """Initialize with base parameters from training data."""
        self.base_avg_salary = base_avg_salary
        self.base_interest_rate = base_interest_rate

    def generate_stress_test_data(self, num_records=100000, scenario="economic_downturn"):
        """Generate test data under different economic scenarios."""

        if scenario == "economic_downturn":
            # Economic recession parameters
            test_avg_salary = self.base_avg_salary * 0.85
            test_interest_rate = self.base_interest_rate * 1.3

        elif scenario == "economic_boom":
            # Economic growth parameters
            test_avg_salary = self.base_avg_salary * 1.2  # 20% higher salaries
            test_interest_rate = self.base_interest_rate * 0.8  # 20% lower rates

        elif scenario == "inflation_spike":
            # High inflation scenario
            test_avg_salary = self.base_avg_salary * 1.1  # 10% salary increase
            test_interest_rate = self.base_interest_rate * 1.6  # 60% higher rates

        elif scenario == "conservative_lending":
            # Stricter lending standards
            test_avg_salary = self.base_avg_salary  # Same salaries
            test_interest_rate = self.base_interest_rate * 1.2  # 20% higher rates

        else:  # "baseline"
            # Similar to training but slightly different
            test_avg_salary = self.base_avg_salary * 0.98  # 2% lower
            test_interest_rate = self.base_interest_rate * 1.05  # 5% higher


        # Generate test dataset with modified parameters
        test_generator = DataGenerator(
            avg_salary=test_avg_salary,
            interest_rate=test_interest_rate,
            num_records=num_records
        )

        # Create test dataset filename
        os.makedirs("evaluation", exist_ok=True)
        eval_csv_path = f"evaluation/evaluation_data_{scenario}.csv"

        # Generate data silently
        test_generator.csv_path = eval_csv_path

        test_generator.generate_realistic_data(True)  # True = save to file
        test_generator.remove_wrong_rows(False, None)  # False = no verbose, None = no progress bar

        print(f"{S_GREEN}☑ Generated {num_records:,} test records: {eval_csv_path}{E_GREEN}")

        return eval_csv_path, {
            'scenario': scenario,
            'avg_salary': test_avg_salary,
            'interest_rate': test_interest_rate,
            'salary_change': (test_avg_salary - self.base_avg_salary) / self.base_avg_salary,
            'rate_change': (test_interest_rate - self.base_interest_rate) / self.base_interest_rate
        }

    def generate_multiple_scenarios(self, num_records=3000):
        """Generate test data for multiple economic scenarios."""

        scenarios = ["economic_downturn", "economic_boom", "inflation_spike", "baseline"]
        results = {}

        print(f"{S_CYAN}Generating multiple test scenarios...{E_CYAN}")

        for scenario in scenarios:
            csv_path, scenario_params = self.generate_stress_test_data(num_records, scenario)
            results[scenario] = {
                'csv_path': csv_path,
                'parameters': scenario_params
            }

        print(f"{S_GREEN}☑ Generated {len(scenarios)} independent test datasets{E_GREEN}")
        return results
    def compare_distributions(self, training_csv, test_csv):
        """Compare key distributions between training and test data."""

        print(f"\n{S_CYAN}Comparing data distributions:{E_CYAN}")

        try:
            train_df = pd.read_csv(training_csv)
            test_df = pd.read_csv(test_csv)

            key_metrics = ['reported_monthly_income', 'total_existing_debt',
                          'loan_amount', 'monthly_payment', 'loan_approved']

            for metric in key_metrics:
                if metric in train_df.columns and metric in test_df.columns:
                    train_mean = train_df[metric].mean()
                    test_mean = test_df[metric].mean()
                    diff_pct = ((test_mean - train_mean) / train_mean) * 100

                    print(f"  {metric}:")
                    print(f"    Training: {train_mean:,.0f}")
                    print(f"    Test: {test_mean:,.0f} ({diff_pct:+.1f}%)")

        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
            print(f"Error comparing distributions: {e}")


if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator()

    # Generate single scenario
    test_csv_path, test_params = generator.generate_stress_test_data(5000, "economic_downturn")

    # Compare with training data
    generator.compare_distributions(
        "datasets/mortgage_applications.csv",
        test_csv_path
    )
