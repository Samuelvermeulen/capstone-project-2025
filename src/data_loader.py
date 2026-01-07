"""
Data loader module for player valuation project.
Handles loading, cleaning, and temporal splitting of data.
Samuel Vermeulen - Capstone Project 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data(file_path="data/raw/PL_players_with_new_columns.csv"):
    """
    Load raw CSV data from the specified path.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Raw dataframe
        
    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        error_msg = f"Data file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data loaded. Shape: {df.shape}")
    
    return df

def clean_data(df):
    """
    Clean the dataset by removing problematic seasons and handling missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    logger.info("Starting data cleaning...")
    
    # Save original shape
    original_shape = df.shape[0]
    
    # 1. Remove 2023-2024 season (100% missing values in target)
    if 'Season' in df.columns:
        seasons_before = df['Season'].unique()
        df = df[df['Season'] != '2023-2024'].copy()
        seasons_after = df['Season'].unique()
        
        removed_count = original_shape - df.shape[0]
        logger.info(f"Removed {removed_count} rows from season 2023-2024")
        logger.info(f"Seasons before: {sorted(seasons_before)}")
        logger.info(f"Seasons after: {sorted(seasons_after)}")
    
    # 2. Check for remaining missing values in target
    if 'Value' in df.columns:
        missing_value = df['Value'].isnull().sum()
        if missing_value > 0:
            logger.warning(f"Still {missing_value} missing values in 'Value' after cleaning")
        else:
            logger.info("No missing values in 'Value' column")
    
    # 3. Basic info about cleaned data
    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Rows removed: {original_shape - df.shape[0]}")
    
    return df

def explore_data(df, sample_size=3):
    """
    Perform exploratory data analysis and print summary.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to explore
    sample_size : int
        Number of sample values to show for categorical columns
    """
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION SUMMARY")
    print("="*60)
    
    # Basic information
    print(f"\n1. Dataset Overview:")
    print(f"   • Total observations: {df.shape[0]}")
    print(f"   • Total features: {df.shape[1]}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\n2. Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} columns")
    
    # Seasons analysis
    if 'Season' in df.columns:
        print(f"\n3. Seasons Analysis:")
        seasons = sorted(df['Season'].unique())
        print(f"   • Available seasons: {len(seasons)}")
        print(f"   • Seasons: {seasons}")
        
        # Players per season
        season_counts = df['Season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"     - {season}: {count} players")
    
    # Target variable analysis
    if 'Value' in df.columns:
        print(f"\n4. Target Variable (Value):")
        print(f"   • Range: €{df['Value'].min():,.0f} - €{df['Value'].max():,.0f}")
        print(f"   • Mean: €{df['Value'].mean():,.0f}")
        print(f"   • Median: €{df['Value'].median():,.0f}")
        print(f"   • Std: €{df['Value'].std():,.0f}")
        
        # Skewness check
        try:
            from scipy.stats import skew
            value_skew = skew(df['Value'].dropna())
            skew_indicator = "✅" if abs(value_skew) < 1 else "⚠️"
            print(f"   • Skewness: {value_skew:.2f} {skew_indicator}")
        except ImportError:
            pass
    
    # Missing values summary
    print(f"\n5. Missing Values Summary:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"   • {col}: {count} missing ({percentage:.1f}%)")
    else:
        print("   ✅ No missing values")
    
    print("\n" + "="*60)

def split_data_temporally(df, train_seasons=None, test_seasons=None):
    """
    Split data into train and test sets based on season.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe
    train_seasons : list, optional
        List of seasons for training (default: 2018-2022)
    test_seasons : list, optional
        List of seasons for testing (default: 2022-2023)
        
    Returns:
    --------
    train_df, test_df : tuple of pandas.DataFrame
        Train and test dataframes
    """
    if train_seasons is None:
        train_seasons = ['2018-2019', '2019-2020', '2020-2021', '2021-2022']
    if test_seasons is None:
        test_seasons = ['2022-2023']
    
    logger.info("Performing temporal split...")
    logger.info(f"Training seasons: {train_seasons}")
    logger.info(f"Test seasons: {test_seasons}")
    
    # Validate seasons are in dataframe
    all_seasons = df['Season'].unique() if 'Season' in df.columns else []
    for season in train_seasons + test_seasons:
        if season not in all_seasons:
            logger.warning(f"Season {season} not found in dataframe")
    
    # Perform split
    train_mask = df['Season'].isin(train_seasons)
    test_mask = df['Season'].isin(test_seasons)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Training set: {train_df.shape[0]} rows")
    logger.info(f"Test set: {test_df.shape[0]} rows")
    
    # Calculate split ratio
    total = train_df.shape[0] + test_df.shape[0]
    if total > 0:
        train_ratio = train_df.shape[0] / total * 100
        test_ratio = test_df.shape[0] / total * 100
        logger.info(f"Split ratio: {train_ratio:.1f}% train, {test_ratio:.1f}% test")
    
    return train_df, test_df

def save_processed_data(train_df, test_df, output_dir="data/processed"):
    """
    Save processed datasets to CSV files.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataframe
    test_df : pandas.DataFrame
        Test dataframe
    output_dir : str
        Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_path = output_path / "train_data.csv"
    test_path = output_path / "test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Training data saved to: {train_path}")
    logger.info(f"Test data saved to: {test_path}")
    
    return train_path, test_path

def get_data_pipeline(clean=True, save=True):
    """
    Complete data pipeline: load, clean, split, and save.
    
    Parameters:
    -----------
    clean : bool
        Whether to clean the data
    save : bool
        Whether to save processed data
        
    Returns:
    --------
    train_df, test_df : tuple of pandas.DataFrame
        Processed train and test dataframes
    """
    logger.info("Starting data pipeline...")
    
    # 1. Load raw data
    df = load_raw_data()
    
    # 2. Clean data
    if clean:
        df = clean_data(df)
    
    # 3. Explore data
    explore_data(df)
    
    # 4. Split temporally
    train_df, test_df = split_data_temporally(df)
    
    # 5. Save processed data
    if save:
        save_processed_data(train_df, test_df)
    
    logger.info("Data pipeline completed successfully!")
    
    return train_df, test_df

if __name__ == "__main__":
    """
    Test the data loader module independently.
    """
    print("Testing data_loader module...\n")
    
    try:
        # Run complete pipeline
        train_df, test_df = get_data_pipeline()
        
        print("\n" + "="*60)
        print("✅ DATA LOADER TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nTraining set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print(f"\nTraining seasons: {sorted(train_df['Season'].unique())}")
        print(f"Test seasons: {sorted(test_df['Season'].unique())}")
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
