from utils.constants import *

def print_error_handling(string: str):
    print(f"{S_RED}Wrong Input{E_RED}: {string}")
    exit(1)

def print_wrong_handling(string: str):
    print(f"{S_YELLOW}Wrong Input{E_YELLOW}: {string}")

def print_warning_handling(string: str):
    print(f"{S_YELLOW}WARNING{E_YELLOW}: {string}")
