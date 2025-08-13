#!/usr/bin/env python
"""Run pytest with all warnings suppressed"""
import warnings
import sys
import os

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Run pytest
import pytest
sys.exit(pytest.main(['-q']))