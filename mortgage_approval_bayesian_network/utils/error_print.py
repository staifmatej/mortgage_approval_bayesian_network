from utils.constants import *

def print_error_handling(string: str):
    print(f"{S_RED}ERROR{E_RED}: {string}")
    exit(1)

def print_invalid_input(string: str):
    print(f"{S_YELLOW}Invalid Input{E_YELLOW}: {string}")

def print_warning_handling(string: str):
    print(f"{S_YELLOW}WARNING{E_YELLOW}: {string}")
