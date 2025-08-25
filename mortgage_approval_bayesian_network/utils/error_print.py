"""Module for formatted error and warning message printing with color support."""
import sys
from utils.constants import S_RED, E_RED, S_YELLOW, E_YELLOW  # pylint: disable=import-error

def print_error_handling(string: str):
    """Print error message in red color and exit program."""
    print(f"{S_RED}ERROR{E_RED}: {string}")
    sys.exit(1)

def print_invalid_input(string: str):
    """Print invalid input warning message in yellow color."""
    print(f"{S_YELLOW}Invalid Input{E_YELLOW}: {string}")

def print_warning_handling(string: str):
    """Print warning message in yellow color without exiting."""
    print(f"{S_YELLOW}WARNING{E_YELLOW}: {string}")
