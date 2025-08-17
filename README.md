![FYI](FYI.png)

# Linear Gaussian Bayesian Network

## Abstract

This project implements and compares three different
  computational approaches for
  regression models: **Pure Python** (using only for-loops),
  **NumPy** (vectorized operations), and **Numba**
  (JIT-compiled Python). Each engine implements four regression models: **Linear Regression** and 
**Ridge Regression** are implemented from scratch using only basic 
mathematical operations, while **Lasso Regression** and **Elastic Net 
Regression** wrap sklearn implementations to provide consistent interface 
and enable performance comparison across all methods.

  This project extends beyond the scope of the **Linear 
  Algebra II** course at **CTU FIT**, where the least
  squares method was introduced theoretically. As an
  extension of the coursework, I explored practical
  implementation without relying on high-level machine
  learning libraries, demonstrating the mathematical
  foundations learned in class through code.

  The project provides **performance benchmarking** to
  demonstrate the computational advantages of different
  implementation strategies, particularly showcasing
  Numba's JIT compilation performance gains over pure
  Python implementations. Additionally, the program offers
  **curve fitting capabilities** for **sixteen pre-selected
   functions**, allowing users to fit various mathematical
  models to their datasets through an interactive menu
  system.

## Detailed Report

For a more detailed description of the methodology, results, and analysis, please refer to the [staifmatej-report.pdf](staifmatej-report.pdf) file included in this repository.

## Diagram of the Linear Gaussian Bayesian Network


![Diagram of the Linear Gaussian Bayesian Network](mortgage_approval_bayesian_network/diagram_photos/bayesian_network_readme.jpg)

- `🟦 Employment & Demographics` - Basic applicant information including age, highest education level, employment type (permanent, temporary, freelancer, unemployed), government employee status, length of employment, company size, and fundamental loan parameters such as loan term (1–35 years) and requested loan amount.
- `🟩 Financial Inputs` - Core financial data provided by the applicant, specifically their reported monthly income and the average salary benchmark in the country, which serve as the foundation for all financial calculations and comparisons.
- `🟪 Computed Variables` - Derived metrics calculated from input data including monthly payment amount, mortgage end age, years of mortgage continuing after retirement, and income stability score (0–1), which provide deeper insights into the loan's long-term implications.
- `🟨 Risk Assessment` - Comprehensive risk evaluation through various financial ratios and indicators: total stable monthly income, existing debt obligations, property and investment values, net worth calculations, income-to-average-salary ratios, debt-to-net-worth ratios, payment-to-income ratios, income-to-debt coverage ratios, credit history categories (bad, fair, good, excellent), and overall default risk assessment.
- **Final Decision**: The ultimate loan approval decision represented as a probability score (0–1), which synthesizes all input variables, computed metrics, and risk assessments to determine the likelihood of mortgage approval.
  
More detailed information is provided in the report [staifmatej-report.pdf](staifmatej-report.pdf) in this repository.


## Installation & Usage

- Clone the repository using SSH or HTTPS
    - **SSH:** `git@github.com:staifmatej/mortgage_approval_bayesian_network.git`
    - **HTTPS:** `https://github.com/staifmatej/mortgage_approval_bayesian_network.git`

- Navigate to the project directory (to the root folder)

    - `cd mortgage_approval_bayesian_network`

- Create virtual environment and install dependencies:

    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`

- Run program, test unit tests or check for PEP8 score:
 
    - `python main.py` (to start program)
    - `pytest` (to run unit tests)
    - `pylint . --disable=C0301,C0103` (PEP8 score)

## Testing

To run the tests, execute `pytest` directly in the main project directory (**root folder**).

## Codestyle

To check code style compliance, run `pylint . --disable=C0301,C0103` from the **root folder**.
This will analyze all Python files while ignoring line length (C0301) and naming convention (C0103) warnings.
  

**Note**: These pylint warnings are occasionally suppressed
  directly in the code. I tried to minimize pylint warning
  suppressions, but sometimes I determined that suppression
   was the best choice, as making the changes would not
  help improve my program design skills and would only make
   the program structure more chaotic.
