
import pytest
import pandas as pd
import sys
import os

# Add src to path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_raw_data, clean_data

@pytest.fixture(scope="session")
def raw_data():
    """Load raw data once for all tests."""
    return load_raw_data()

@pytest.fixture(scope="session")
def cleaned_data(raw_data):
    """Return cleaned data."""
    return clean_data(raw_data)