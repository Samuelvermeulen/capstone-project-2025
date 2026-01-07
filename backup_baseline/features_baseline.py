"""
Feature engineering module - Version simplifiée et fonctionnelle
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_baseline_features(df):
    """
    Baseline features (Age + Position only) - Version fonctionnelle
    """
    logger.info("Creating baseline features (Age + Position only)...")
    
    # Vérifier les colonnes nécessaires
    required_cols = ['Age', 'Position', 'Value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df_processed = df.copy()
    
    # Imputer l'âge si manquant
    if df_processed['Age'].isnull().any():
        median_age = df_processed['Age'].median()
        df_processed['Age'] = df_processed['Age'].fillna(median_age)
        logger.info(f"Imputed {df_processed['Age'].isnull().sum()} missing Age values")
    
    # One-hot encode Position
    position_dummies = pd.get_dummies(df_processed['Position'], prefix='is')
    
    # Renommer pour clarté
    position_dummies = position_dummies.rename(columns={
        'is_DF': 'is_Defender',
        'is_MF': 'is_Midfielder', 
        'is_FW': 'is_Forward',
        'is_GK': 'is_Goalkeeper'
    })
    
    # S'assurer que toutes les positions existent
    expected_positions = ['is_Defender', 'is_Midfielder', 'is_Forward', 'is_Goalkeeper']
    for pos in expected_positions:
        if pos not in position_dummies.columns:
            position_dummies[pos] = 0
    
    # Combiner Age et Position
    X_baseline = pd.concat([
        df_processed[['Age']].reset_index(drop=True),
        position_dummies.reset_index(drop=True)
    ], axis=1)
    
    # Extraire la target
    y = df_processed['Value'].copy()
    
    logger.info(f"Baseline features created. Shape: {X_baseline.shape}")
    return X_baseline, y

def prepare_baseline_data(train_df, test_df):
    """
    Préparer les données baseline pour train et test
    """
    logger.info("Preparing baseline data...")
    
    X_train, y_train = create_baseline_features(train_df)
    X_test, y_test = create_baseline_features(test_df)
    
    # Assurer la cohérence des colonnes
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    missing_in_test = train_cols - test_cols
    if missing_in_test:
        for col in missing_in_test:
            X_test[col] = 0
    
    X_test = X_test[X_train.columns]
    
    return X_train, y_train, X_test, y_test
