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

-	`Blue nodes` contain data about the mortgage applicant’s age, highest education achieved, employment status and company size (number of employees), length of current employment, and type of employment (permanent, temporary, freelancer, or unemployed).
-	`Purple nodes` always contain numeric values and are derived from other nodes. This category includes metrics such as income stability, which is calculated from the blue nodes and ranges from 0 to 1 — higher values indicate a more stable income for the applicant. Additionally, they include the mortgage payoff date and payment amount. The total sum of numeric asset values is also represented, and in the bottom-left corner, there is a node indicating whether the mortgage is expected to be paid after retirement.
-	`Orange nodes` contain the mortgage parameters directly, specifically the repayment period in years (from 1 to 35) and the amount the person wants to borrow.
-	`Green nodes` contain the applicant’s financial information, including net monthly income, investment amount (if any), property value (if owned), outstanding debt, average salary, and housing status (rent, own, mortgage, homeless).
-	

  
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
