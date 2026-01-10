
import pytest
import pandas as pd

def test_load_raw_data_returns_dataframe(raw_data):
    """Verify load_raw_data returns a non-empty DataFrame."""
    assert isinstance(raw_data, pd.DataFrame)
    assert raw_data.shape[0] > 0
    assert 'Value' in raw_data.columns

def test_clean_data_removes_2023_2024_season(cleaned_data):
    """Verify 2023-2024 season is removed."""
    assert '2023-2024' not in cleaned_data['Season'].values

def test_clean_data_no_missing_target_values(cleaned_data):
    """Verify no missing values in target column 'Value'."""
    assert cleaned_data['Value'].isnull().sum() == 0

def test_split_data_temporally():
    """Test temporal split logic without running full pipeline."""
    from data_loader import split_data_temporally
    
    # Create simple test data
    df = pd.DataFrame({
        'Season': ['2018-2019', '2019-2020', '2022-2023'],
        'Value': [1_000_000, 2_000_000, 3_000_000]
    })
    
    train_df, test_df = split_data_temporally(df)
    
    assert len(train_df) == 2
    assert len(test_df) == 1
    assert '2022-2023' in test_df['Season'].values
    