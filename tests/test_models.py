
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import create_baseline_features
from models import BaselineModel

def test_baseline_model_training_and_prediction():
    """Verify BaselineModel can fit and predict without errors."""
    # Create mock data
    df = pd.DataFrame({
        'Age': [25, 28, 22],
        'Position': ['FW', 'MF', 'DF'],
        'Value': [1_000_000, 2_000_000, 500_000]
    })
    
    X, y = create_baseline_features(df)
    
    model = BaselineModel()
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # Assertions
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions))