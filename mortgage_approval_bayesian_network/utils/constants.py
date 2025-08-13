"""Module containing color constants for terminal output formatting."""
from colorama import init, Fore, Style

# Initialize colorama: for Windows compatibility.
init(autoreset=True)

# Global constants using colorama.
S_BOLD = Style.BRIGHT
E_BOLD = Style.RESET_ALL
S_RED = Fore.RED
E_RED = Style.RESET_ALL
S_YELLOW = Fore.YELLOW
E_YELLOW = Style.RESET_ALL
S_GREEN = Fore.GREEN
E_GREEN = Style.RESET_ALL
S_CYAN = Fore.CYAN
E_CYAN = Style.RESET_ALL
