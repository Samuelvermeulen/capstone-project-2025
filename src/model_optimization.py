"""
Phase 6 - Model Optimization and Feature Analysis
Samuel Vermeulen - Capstone Project 2025

Steps:
6.1 - Hyperparameter Tuning (RandomizedSearchCV)
6.2 - Feature Importance Analysis
6.3 - Error and Residual Analysis
6.4 - Simple Baseline Models
6.5 - Synthesis and Deployment Preparation
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import logging
import os
from datetime import datetime
from typing import Dict, Tuple, List

# Machine Learning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# XGBoost
from xgboost import XGBRegressor

# SHAP (optional - for feature interpretation)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Using XGBoost feature importance instead.")
    print("To install: pip install shap")

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plot Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Phase 6 - Imports and configuration complete")

# ============================================================================
# 2. DATA LOADING FUNCTIONS (reuse from Phase 5)
# ============================================================================

def load_and_prepare_data(data_dir: str = "data/processed") -> Tuple:
    """
    Load and prepare data for optimization.
    Reuses functions from Phase 5.
    """
    logger.info(f"📥 Loading data from {data_dir}...")
    
    try:
        # Load transformed data
        X_train = pd.read_csv(f"{data_dir}/X_train_transformed.csv")
        X_test = pd.read_csv(f"{data_dir}/X_test_transformed.csv")
        
        y_train_log = pd.read_csv(f"{data_dir}/y_train_log.csv")['Value_log']
        y_test_log = pd.read_csv(f"{data_dir}/y_test_log.csv")['Value_log']
        
        y_train_orig = pd.read_csv(f"{data_dir}/y_train_original.csv")['Value']
        y_test_orig = pd.read_csv(f"{data_dir}/y_test_original.csv")['Value']
        
        with open(f"{data_dir}/feature_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ Data loaded successfully")
        logger.info(f"   • X_train: {X_train.shape}")
        logger.info(f"   • X_test: {X_test.shape}")
        
        return X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig, metadata
        
    except Exception as e:
        logger.error(f"❌ Error during loading: {e}")
        raise

def align_features(X_train, X_test):
    """
    Align features between train and test (as in Phase 5).
    """
    # Reuse function from Phase 5
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    all_columns = sorted(train_cols.union(test_cols))
    
    X_train_aligned = pd.DataFrame(0, index=X_train.index, columns=all_columns)
    X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=all_columns)
    
    for col in all_columns:
        if col in X_train.columns:
            X_train_aligned[col] = X_train[col]
        if col in X_test.columns:
            X_test_aligned[col] = X_test[col]
    
    return X_train_aligned, X_test_aligned

# ============================================================================
# 3. HYPERPARAMETER TUNING (RANDOMIZEDSEARCHCV)
# ============================================================================

def hyperparameter_tuning_xgboost(X_train, y_train_log, n_iter=50, cv_splits=3):
    """
    Optimize XGBoost hyperparameters with RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train_log: Target in log scale
        n_iter: Number of iterations for random search
        cv_splits: Number of folds for time-series cross-validation
    
    Returns:
        best_model: Best optimized XGBoost model
        best_params: Best parameters found
        cv_results: Complete search results
    """
    logger.info("🎯 Starting XGBoost hyperparameter optimization...")
    
    # ------------------------------------------------------------
    # 3.1 HYPERPARAMETER SEARCH SPACE DEFINITION
    # ------------------------------------------------------------
    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6, 7, 9],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 0.5, 1, 5, 10]
    }
    
    logger.info(f"📋 Search space with {len(param_distributions)} parameters")
    logger.info(f"   • Number of iterations: {n_iter}")
    logger.info(f"   • CV folds: {cv_splits}")
    
    # ------------------------------------------------------------
    # 3.2 MODEL AND SEARCH INITIALIZATION
    # ------------------------------------------------------------
    base_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',  # Minimize MSE
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # ------------------------------------------------------------
    # 3.3 EXECUTING THE SEARCH
    # ------------------------------------------------------------
    logger.info("🔄 Starting random search...")
    start_time = datetime.now()
    
    random_search.fit(X_train, y_train_log)
    
    search_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ Search completed in {search_time:.1f} seconds")
    
    # ------------------------------------------------------------
    # 3.4 RESULTS ANALYSIS
    # ------------------------------------------------------------
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # Convert to positive MSE
    
    logger.info("🏆 BEST PARAMETERS FOUND:")
    for param, value in best_params.items():
        logger.info(f"   • {param}: {value}")
    
    logger.info(f"   • Best MSE (log scale): {best_score:.6f}")
    logger.info(f"   • Best RMSE (log scale): {np.sqrt(best_score):.6f}")
    
    # ------------------------------------------------------------
    # 3.5 RESULTS VISUALIZATION AND SAVING
    # ------------------------------------------------------------
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    
    # Save results
    results_dir = "results/hyperparameter_tuning"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_results_df.to_csv(f"{results_dir}/cv_results_{timestamp}.csv", index=False)
    
    # Create summary file
    with open(f"{results_dir}/best_params_{timestamp}.txt", 'w') as f:
        f.write("BEST XGBOOST HYPERPARAMETERS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Search time: {search_time:.1f} seconds\n")
        f.write(f"Number of iterations: {n_iter}\n")
        f.write(f"Best MSE: {best_score:.6f}\n")
        f.write(f"Best RMSE: {np.sqrt(best_score):.6f}\n\n")
        f.write("Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
    
    logger.info(f"📁 Results saved in: {results_dir}")
    
    return best_model, best_params, cv_results_df

def evaluate_optimized_model(best_model, X_train, X_test, y_train_log, y_test_log, 
                           y_train_orig, y_test_orig):
    """
    Evaluate the optimized model and compare with baseline.
    
    Returns:
        metrics_dict: Dictionary with all metrics
    """
    logger.info("📊 Evaluating optimized model...")
    
    # ------------------------------------------------------------
    # 6.1.6 PREDICTIONS AND METRICS
    # ------------------------------------------------------------
    # Predictions on train and test
    y_pred_train_log = best_model.predict(X_train)
    y_pred_test_log = best_model.predict(X_test)
    
    # Convert to euros
    y_pred_train_euros = np.expm1(y_pred_train_log)
    y_pred_test_euros = np.expm1(y_pred_test_log)
    
    # ------------------------------------------------------------
    # 6.1.7 METRICS CALCULATION
    # ------------------------------------------------------------
    metrics = {
        'train': {
            'rmse_log': np.sqrt(mean_squared_error(y_train_log, y_pred_train_log)),
            'mae_log': mean_absolute_error(y_train_log, y_pred_train_log),
            'r2_log': r2_score(y_train_log, y_pred_train_log),
            'rmse_euros': np.sqrt(mean_squared_error(y_train_orig, y_pred_train_euros)),
            'mae_euros': mean_absolute_error(y_train_orig, y_pred_train_euros),
            'r2_euros': r2_score(y_train_orig, y_pred_train_euros)
        },
        'test': {
            'rmse_log': np.sqrt(mean_squared_error(y_test_log, y_pred_test_log)),
            'mae_log': mean_absolute_error(y_test_log, y_pred_test_log),
            'r2_log': r2_score(y_test_log, y_pred_test_log),
            'rmse_euros': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_euros)),
            'mae_euros': mean_absolute_error(y_test_orig, y_pred_test_euros),
            'r2_euros': r2_score(y_test_orig, y_pred_test_euros)
        }
    }
    
    # ------------------------------------------------------------
    # 6.1.8 RESULTS DISPLAY
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📈 OPTIMIZED MODEL PERFORMANCE")
    print("=" * 70)
    
    print(f"\n🎯 TRAINING METRICS (2018-2022):")
    print(f"   • RMSE: €{metrics['train']['rmse_euros']:,.0f}")
    print(f"   • MAE: €{metrics['train']['mae_euros']:,.0f}")
    print(f"   • R²: {metrics['train']['r2_euros']:.4f}")
    
    print(f"\n📊 TEST METRICS (2022-2023):")
    print(f"   • RMSE: €{metrics['test']['rmse_euros']:,.0f}")
    print(f"   • MAE: €{metrics['test']['mae_euros']:,.0f}")
    print(f"   • R²: {metrics['test']['r2_euros']:.4f}")
    
    # Comparison with Phase 5
    print(f"\n🔄 COMPARISON WITH PHASE 5:")
    print(f"   • Phase 5 (default XGBoost): R² = 0.459, RMSE = €17,383,166")
    print(f"   • Phase 6 (optimized XGBoost): R² = {metrics['test']['r2_euros']:.3f}, "
          f"RMSE = €{metrics['test']['rmse_euros']:,.0f}")
    
    improvement_r2 = (metrics['test']['r2_euros'] - 0.459) * 100
    improvement_rmse = ((17383166 - metrics['test']['rmse_euros']) / 17383166) * 100
    
    print(f"   • R² improvement: {improvement_r2:+.1f}%")
    print(f"   • RMSE improvement: {improvement_rmse:+.1f}%")
    
    return metrics
#######################################
#### Test this steps ####
#######################################
def test_step_6_1(n_iter=20):
    """
    Test function for Step 6.1 - Hyperparameter Tuning
    """
    print("=" * 70)
    print("🧪 TEST STEP 6.1 - HYPERPARAMETER TUNING XGBOOST")
    print("=" * 70)
    
    try:
        # ------------------------------------------------------------
        # 1. DATA LOADING
        # ------------------------------------------------------------
        print("\n1. 📥 Loading data...")
        X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig, metadata = load_and_prepare_data()
        
        # Feature alignment
        X_train_aligned, X_test_aligned = align_features(X_train, X_test)
        
        print(f"   ✓ Data loaded: Train={X_train_aligned.shape}, Test={X_test_aligned.shape}")
        
        # ------------------------------------------------------------
        # 2. HYPERPARAMETER OPTIMIZATION
        # ------------------------------------------------------------
        print(f"\n2. 🎯 Hyperparameter optimization ({n_iter} iterations)...")
        print("   This step may take a few minutes...")
        
        best_model, best_params, cv_results = hyperparameter_tuning_xgboost(
            X_train_aligned, y_train_log, n_iter=n_iter, cv_splits=3
        )
        
        # ------------------------------------------------------------
        # 3. OPTIMIZED MODEL EVALUATION
        # ------------------------------------------------------------
        print("\n3. 📊 Evaluating optimized model...")
        metrics = evaluate_optimized_model(
            best_model, X_train_aligned, X_test_aligned,
            y_train_log, y_test_log, y_train_orig, y_test_orig
        )
        
        # ------------------------------------------------------------
        # 4. SAVING OPTIMIZED MODEL
        # ------------------------------------------------------------
        print("\n4. 💾 Saving optimized model...")
        model_dir = "models/optimized"
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/xgboost_optimized_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        
        # Save metrics
        metrics_path = f"{model_dir}/metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"   ✓ Model saved: {model_path}")
        print(f"   ✓ Metrics saved: {metrics_path}")
        
        # ------------------------------------------------------------
        # 5. STEP SUMMARY
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print("✅ STEP 6.1 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📋 SUMMARY:")
        print(f"   • Best R² achieved: {metrics['test']['r2_euros']:.4f}")
        print(f"   • Best RMSE achieved: €{metrics['test']['rmse_euros']:,.0f}")
        print(f"   • Number of hyperparameters tested: {n_iter}")
        print(f"   • Optimized model saved for next steps")
        
        return True, best_model, metrics
        
    except Exception as e:
        print(f"\n❌ ERROR IN STEP 6.1: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point to test step 6.1
    """
    print("=" * 70)
    print("PHASE 6.1 - HYPERPARAMETER TUNING XGBOOST")
    print("=" * 70)
    
    # Test with reduced iterations to start
    # You can increase n_iter for more exhaustive search
    success, optimized_model, metrics = test_step_6_1(n_iter=20)
    
    if success:
        print("\n🎯 READY FOR STEP 6.2 - FEATURE IMPORTANCE ANALYSIS")
        print("   Next steps:")
        print("   1. Feature importance analysis")
        print("   2. Validation of your hypotheses:")
        print("      • Minutes played in top 10 club")
        print("      • Goals important for strikers")
    else:
        print("\n❌ STEP 6.1 FAILED. Check errors above.")