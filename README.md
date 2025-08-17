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

## Node Descriptions

| Node                        | Description                                                |
|-----------------------------|------------------------------------------------------------|
| age                         | applicant’s age                                            |
| highest education           | education level                                            |
| employment type             | permanent, temporary, freelancer, unemployed               |
| government employee         | yes/no flag                                                |
| len employment              | length of current job                                      |
| size company                | company size (# employees)                                 |
| loan term                   | repayment period (1–35 years)                              |
| loan amount                 | requested mortgage amount                                  |
| monthly payment             | calculated monthly installment                            |
| mortgage end age            | applicant’s age at mortgage payoff                         |
| years of mortgage retirement| repayment continuing after retirement                      |
| stability income            | stability metric of income (0–1)                           |
| reported monthly income     | applicant’s declared monthly income                        |
| avg salary                  | average salary in country                                  |
| total stable monthly income | adjusted stable income                                     |
| total existing debt         | applicant’s outstanding debts                              |
| property value              | owned property value                                       |
| investments value           | total investments (if any)                                 |
| core worth                  | net worth (assets − debts)                                 |
| ratio income by avg salary  | income compared to average                                 |
| ratio debt to net worth     | financial leverage indicator                               |
| ratio payment to income     | mortgage burden on income                                  |
| ratio income to debt        | ability to cover debts with income                         |
| credit history              | credit score category (bad, fair, good, excellent)         |
| defaulted                   | default risk, derived from many factors                    |
| loan approved               | final decision probability (0–1)                           |

  
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
