"""Pytest configuration module for suppressing package resource warnings."""
import warnings
import sys

# Suppress the pkg_resources warning before it happens
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fs")

# Also try to suppress at import time
if 'fs' in sys.modules:
    warnings.filterwarnings("ignore", category=UserWarning, module="fs.__init__")
