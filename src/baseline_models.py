
"""
PHASE 6.4 - Simple Baseline Models Comparison
Capstone Project 2025This module compares the optimized XGBoost model against simple baseline models
to measure the "value added" by model complexity.

Models to implement:
1. Linear Regression (with same 25 features)
2. Random Forest (with reasonable default hyperparameters)
3. Naïve baseline (predict median value for all players)

We'll compare metrics (RMSE, MAE, R²) in original € scale.
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import logging
from datetime import datetime
from pathlib import Path

# Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Statistical test
from scipy.stats import ttest_rel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Step 6.4.1: File setup complete")

# ============================================================================
# 2. DATA LOADING FUNCTIONS
# ============================================================================

def load_data_for_baselines():
    """
    Load the data needed for baseline model comparison.
    
    Returns:
        X_train_aligned: Training features (aligned)
        X_test_aligned: Test features (aligned)
        y_train_original: Training target values in €
        y_test_original: Test target values in €
        y_train_log: Training target values (log-transformed)
        y_test_log: Test target values (log-transformed)
        feature_names: List of feature names
    """
    logger.info("📥 Loading data for baseline comparison...")
    
    try:
        # 1. Load transformed features
        X_train = pd.read_csv("data/processed/X_train_transformed.csv")
        X_test = pd.read_csv("data/processed/X_test_transformed.csv")
        
        # 2. Load target values
        y_train_log = pd.read_csv("data/processed/y_train_log.csv").values.flatten()
        y_test_log = pd.read_csv("data/processed/y_test_log.csv").values.flatten()
        
        # 3. Convert log values back to original scale
        y_train_original = np.expm1(y_train_log)
        y_test_original = np.expm1(y_test_log)
        
        # 4. Load original training data for naïve baseline
        train_data = pd.read_csv("data/processed/train_data.csv")
        
        # 5. Align features between train and test
        all_columns = sorted(set(X_train.columns).union(set(X_test.columns)))
        
        X_train_aligned = pd.DataFrame(0, index=X_train.index, columns=all_columns)
        X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=all_columns)
        
        for col in all_columns:
            if col in X_train.columns:
                X_train_aligned[col] = X_train[col]
            if col in X_test.columns:
                X_test_aligned[col] = X_test[col]
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   • X_train shape: {X_train_aligned.shape}")
        logger.info(f"   • X_test shape: {X_test_aligned.shape}")
        logger.info(f"   • y_train shape: {y_train_original.shape}")
        logger.info(f"   • y_test shape: {y_test_original.shape}")
        
        return (X_train_aligned, X_test_aligned, 
                y_train_original, y_test_original,
                y_train_log, y_test_log,
                all_columns, train_data)
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

print("✅ Step 6.4.2: Data loading function ready")

# ============================================================================
# 3. MODEL LOADING FUNCTIONS
# ============================================================================

def load_optimized_xgboost():
    """
    Load the optimized XGBoost model from Phase 6.1 for comparison.
    
    Returns:
        xgboost_model: Trained XGBoost model
        xgboost_info: Information about the model
    """
    logger.info("🤖 Loading optimized XGBoost model for comparison...")
    
    # Check multiple possible locations
    possible_paths = [
        "models/optimized/xgboost_optimized.joblib",
        "models/xgboost_optimized.joblib",
        "models/best_model.joblib"
    ]
    
    xgboost_model = None
    model_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        logger.error("❌ No XGBoost model file found.")
        # Try to find any .joblib file
        import glob
        model_files = glob.glob("models/**/*.joblib", recursive=True)
        if model_files:
            model_path = model_files[0]
            logger.info(f"   Found alternative model: {model_path}")
        else:
            return None, None
    
    # Load the model
    xgboost_model = joblib.load(model_path)
    
    logger.info(f"✅ XGBoost model loaded from: {model_path}")
    
    return xgboost_model, model_path

print("✅ Step 6.4.3: Model loading function ready")

# ============================================================================
# 4. BASELINE MODEL DEFINITIONS
# ============================================================================

def initialize_baseline_models():
    """
    Initialize all baseline models to compare.
    
    Returns:
        models_dict: Dictionary of model names and initialized models
    """
    logger.info("🎯 Initializing baseline models...")
    
    models_dict = {
        # 1. Naïve Baseline 1: Predict median value for all players
        'Naïve_Median': DummyRegressor(strategy='median'),
        
        # 2. Naïve Baseline 2: Predict mean value for all players
        'Naïve_Mean': DummyRegressor(strategy='mean'),
        
        # 3. Simple Linear Regression
        'Linear_Regression': LinearRegression(),
        
        # 4. Random Forest with default parameters
        'Random_Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    logger.info(f"✅ {len(models_dict)} baseline models initialized")
    
    return models_dict

print("✅ Step 6.4.4: Baseline models initialization ready")

# ============================================================================
# 5. MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(models_dict, X_train, y_train_log, X_test, y_test_log, y_test_original):
    """
    Train and evaluate all baseline models.
    
    Args:
        models_dict: Dictionary of models
        X_train: Training features
        y_train_log: Training target (log scale)
        X_test: Test features
        y_test_log: Test target (log scale)
        y_test_original: Test target (original € scale)
    
    Returns:
        results_df: DataFrame with evaluation metrics
        predictions_dict: Dictionary of predictions for each model
    """
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING BASELINE MODELS")
    print("=" * 70)
    
    results = []
    predictions_dict = {}
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, model in models_dict.items():
        print(f"\n🔧 Training {model_name}...")
        
        try:
            # Special handling for different models
            if model_name in ['Linear_Regression']:
                # Linear models use scaled features
                X_train_used = X_train_scaled
                X_test_used = X_test_scaled
            else:
                # Tree models use original features
                X_train_used = X_train.values
                X_test_used = X_test.values
            
            # Train model
            model.fit(X_train_used, y_train_log)
            
            # Make predictions
            y_pred_log = model.predict(X_test_used)
            y_pred_original = np.expm1(y_pred_log)
            
            # Store predictions
            predictions_dict[model_name] = {
                'log_predictions': y_pred_log,
                'original_predictions': y_pred_original,
                'model_object': model
            }
            
            # Calculate metrics in original scale
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)
            
            # Calculate percentage error
            percentage_error = np.mean(np.abs(y_test_original - y_pred_original) / y_test_original) * 100
            
            # Store results
            results.append({
                'Model': model_name,
                'RMSE_€': rmse,
                'MAE_€': mae,
                'R2': r2,
                'Mean_Percentage_Error': percentage_error,
                'MSE_€': mse
            })
            
            print(f"   ✓ {model_name} trained successfully")
            print(f"     RMSE: €{rmse:,.0f}")
            print(f"     MAE: €{mae:,.0f}")
            print(f"     R²: {r2:.3f}")
            print(f"     Mean % Error: {percentage_error:.1f}%")
            
        except Exception as e:
            print(f"   ✗ Error training {model_name}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, predictions_dict

print("✅ Step 6.4.5: Model training and evaluation function ready")

# ============================================================================
# 6. XGBOOST EVALUATION (FOR COMPARISON)
# ============================================================================

def evaluate_xgboost_model(xgboost_model, X_test, y_test_original, y_test_log):
    """
    Evaluate the optimized XGBoost model for comparison.
    
    Args:
        xgboost_model: Trained XGBoost model
        X_test: Test features
        y_test_original: Test target (original € scale)
        y_test_log: Test target (log scale)
    
    Returns:
        xgboost_metrics: Dictionary of XGBoost metrics
        xgboost_predictions: Dictionary of XGBoost predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING OPTIMIZED XGBOOST MODEL")
    print("=" * 70)
    
    try:
        # Make predictions
        y_pred_log = xgboost_model.predict(X_test)
        y_pred_original = np.expm1(y_pred_log)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        percentage_error = np.mean(np.abs(y_test_original - y_pred_original) / y_test_original) * 100
        
        # Store results
        xgboost_metrics = {
            'Model': 'XGBoost_Optimized',
            'RMSE_€': rmse,
            'MAE_€': mae,
            'R2': r2,
            'Mean_Percentage_Error': percentage_error,
            'MSE_€': mse
        }
        
        xgboost_predictions = {
            'log_predictions': y_pred_log,
            'original_predictions': y_pred_original,
            'model_object': xgboost_model
        }
        
        print(f"✓ XGBoost evaluated successfully")
        print(f"  RMSE: €{rmse:,.0f}")
        print(f"  MAE: €{mae:,.0f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Mean % Error: {percentage_error:.1f}%")
        
        return xgboost_metrics, xgboost_predictions
        
    except Exception as e:
        print(f"✗ Error evaluating XGBoost: {e}")
        return None, None

print("✅ Step 6.4.6: XGBoost evaluation function ready")

# ============================================================================
# 7. RESULTS COMPARISON AND VISUALIZATION
# ============================================================================

def compare_and_visualize_results(baseline_results, xgboost_metrics, predictions_dict):
    """
    Compare results and create visualizations.
    
    Args:
        baseline_results: DataFrame with baseline model results
        xgboost_metrics: Dictionary with XGBoost metrics
        predictions_dict: Dictionary of predictions for all models
    """
    print("\n" + "=" * 70)
    print("COMPARING RESULTS AND CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Create directory for results
    results_dir = "results/baseline_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all results
    if xgboost_metrics:
        all_results = pd.concat([
            baseline_results,
            pd.DataFrame([xgboost_metrics])
        ], ignore_index=True)
    else:
        all_results = baseline_results
    
    # Sort by RMSE (best first)
    all_results = all_results.sort_values('RMSE_€', ascending=True).reset_index(drop=True)
    
    # Save results to CSV
    results_csv_path = f"{results_dir}/baseline_comparison_{timestamp}.csv"
    all_results.to_csv(results_csv_path, index=False)
    
    print(f"\n📊 MODEL COMPARISON RESULTS:")
    print("-" * 80)
    print(f"{'Model':<25} {'RMSE (€)':>15} {'MAE (€)':>15} {'R²':>8} {'% Error':>10}")
    print("-" * 80)
    
    for _, row in all_results.iterrows():
        print(f"{row['Model']:<25} {row['RMSE_€']:>15,.0f} {row['MAE_€']:>15,.0f} {row['R2']:>8.3f} {row['Mean_Percentage_Error']:>10.1f}")
    
    # Calculate improvement over best baseline
    if xgboost_metrics and len(baseline_results) > 0:
        best_baseline_rmse = baseline_results['RMSE_€'].min()
        xgboost_rmse = xgboost_metrics['RMSE_€']
        improvement = ((best_baseline_rmse - xgboost_rmse) / best_baseline_rmse) * 100
        
        print(f"\n📈 XGBoost Improvement:")
        print(f"  • Best baseline RMSE: €{best_baseline_rmse:,.0f}")
        print(f"  • XGBoost RMSE: €{xgboost_rmse:,.0f}")
        print(f"  • Improvement: {improvement:.1f}%")
    
    # Create visualizations
    create_comparison_visualizations(all_results, results_dir, timestamp)
    
    return all_results

def create_comparison_visualizations(results_df, results_dir, timestamp):
    """
    Create visualizations for model comparison.
    
    Args:
        results_df: DataFrame with all model results
        results_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
    """
    logger.info("📊 Creating comparison visualizations...")
    
    # 1. Bar chart: RMSE comparison
    plt.figure(figsize=(12, 6))
    
    # Sort by RMSE for better visualization
    sorted_results = results_df.sort_values('RMSE_€', ascending=True)
    
    bars = plt.bar(range(len(sorted_results)), sorted_results['RMSE_€'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(sorted_results))))
    
    # Color XGBoost differently
    for i, (bar, model_name) in enumerate(zip(bars, sorted_results['Model'])):
        if 'XGBoost' in model_name:
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    plt.xticks(range(len(sorted_results)), sorted_results['Model'], rotation=45, ha='right')
    plt.ylabel('RMSE (€)')
    plt.title('Model Comparison: Root Mean Square Error')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(sorted_results['RMSE_€']):
        plt.text(i, v + 0.5e6, f'€{v:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot1_path = f"{results_dir}/rmse_comparison_{timestamp}.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart: R² comparison
    plt.figure(figsize=(12, 6))
    
    sorted_results_r2 = results_df.sort_values('R2', ascending=False)
    
    bars = plt.bar(range(len(sorted_results_r2)), sorted_results_r2['R2'],
                   color=plt.cm.plasma(np.linspace(0, 1, len(sorted_results_r2))))
    
    # Color XGBoost differently
    for i, (bar, model_name) in enumerate(zip(bars, sorted_results_r2['Model'])):
        if 'XGBoost' in model_name:
            bar.set_color('green')
            bar.set_alpha(0.8)
    
    plt.xticks(range(len(sorted_results_r2)), sorted_results_r2['Model'], rotation=45, ha='right')
    plt.ylabel('R² Score')
    plt.title('Model Comparison: R² Score (Higher is Better)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(sorted_results_r2['R2']):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot2_path = f"{results_dir}/r2_comparison_{timestamp}.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Radar chart for multiple metrics
    try:
        create_radar_chart(results_df, results_dir, timestamp)
    except Exception as e:
        logger.warning(f"Could not create radar chart: {e}")
    
    logger.info(f"✅ Visualizations saved to {results_dir}")

def create_radar_chart(results_df, results_dir, timestamp):
    """
    Create a radar chart comparing multiple metrics.
    
    Args:
        results_df: DataFrame with model results
        results_dir: Directory to save visualization
        timestamp: Timestamp for file naming
    """
    # Normalize metrics for radar chart
    metrics = ['RMSE_€', 'MAE_€', 'R2', 'Mean_Percentage_Error']
    
    # For RMSE, MAE, and % Error: lower is better, so invert
    # For R2: higher is better
    
    normalized_data = {}
    
    for model in results_df['Model']:
        model_data = results_df[results_df['Model'] == model].iloc[0]
        normalized_values = []
        
        for metric in metrics:
            value = model_data[metric]
            
            if metric in ['RMSE_€', 'MAE_€', 'Mean_Percentage_Error']:
                # Invert: lower is better
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                if max_val > min_val:
                    normalized = 1 - ((value - min_val) / (max_val - min_val))
                else:
                    normalized = 0.5
            else:  # R2
                # Higher is better
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                if max_val > min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
            
            normalized_values.append(normalized)
        
        normalized_data[model] = normalized_values
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for model, values in normalized_data.items():
        values += values[:1]  # Close the loop
        if 'XGBoost' in model:
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color='red')
            ax.fill(angles, values, alpha=0.1, color='red')
        else:
            ax.plot(angles, values, linewidth=1, linestyle='dashed', label=model, alpha=0.7)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['RMSE', 'MAE', 'R²', '% Error'])
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Radar Chart (Normalized Metrics)', size=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/radar_chart_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Step 6.4.7: Results comparison and visualization functions ready")

# ============================================================================
# 8. STATISTICAL TEST FUNCTION
# ============================================================================

def perform_statistical_tests(predictions_dict, y_test_original, results_dir, timestamp):
    """
    Perform statistical tests to compare model predictions.
    
    Args:
        predictions_dict: Dictionary containing predictions from all models
        y_test_original: True target values in original scale
        results_dir: Directory to save test results
        timestamp: Timestamp for file naming
    """
    print("\n" + "=" * 70)
    print("PERFORMING STATISTICAL TESTS")
    print("=" * 70)
    
    test_results = []
    
    # Get all model names
    model_names = list(predictions_dict.keys())
    
    if len(model_names) < 2:
        print("⚠️  Not enough models for statistical comparison")
        return
    
    # Get XGBoost model if available
    xgboost_key = None
    for key in model_names:
        if 'XGBoost' in key:
            xgboost_key = key
            break
    
    # Calculate absolute errors for each model
    absolute_errors = {}
    for model_name in model_names:
        if 'original_predictions' in predictions_dict[model_name]:
            preds = predictions_dict[model_name]['original_predictions']
            abs_errors = np.abs(y_test_original - preds)
            absolute_errors[model_name] = abs_errors
    
    # Perform paired t-tests comparing each model with XGBoost
    if xgboost_key and len(absolute_errors) > 1:
        print(f"\n📊 Performing paired t-tests (compared to {xgboost_key}):")
        print("-" * 50)
        
        xgboost_errors = absolute_errors[xgboost_key]
        
        for model_name, model_errors in absolute_errors.items():
            if model_name != xgboost_key:
                try:
                    # Perform paired t-test
                    t_stat, p_value = ttest_rel(xgboost_errors, model_errors)
                    
                    # Determine significance
                    significance = ""
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    elif p_value < 0.1:
                        significance = "."
                    
                    # Determine if XGBoost is better
                    xgboost_mean = np.mean(xgboost_errors)
                    model_mean = np.mean(model_errors)
                    difference = model_mean - xgboost_mean
                    percent_difference = (difference / model_mean) * 100
                    
                    comparison = ""
                    if p_value < 0.05:
                        if difference > 0:
                            comparison = f"XGBoost is significantly better ({percent_difference:.1f}% lower error)"
                        else:
                            comparison = f"{model_name} is significantly better ({abs(percent_difference):.1f}% lower error)"
                    else:
                        comparison = "No significant difference"
                    
                    print(f"{xgboost_key} vs {model_name}:")
                    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.6f} {significance}")
                    print(f"  Mean error {xgboost_key}: €{xgboost_mean:,.0f}")
                    print(f"  Mean error {model_name}: €{model_mean:,.0f}")
                    print(f"  {comparison}")
                    print()
                    
                    # Store test results
                    test_results.append({
                        'Comparison': f"{xgboost_key} vs {model_name}",
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significance': significance,
                        f'{xgboost_key}_mean_error': xgboost_mean,
                        f'{model_name}_mean_error': model_mean,
                        'error_difference': difference,
                        'percent_difference': percent_difference,
                        'interpretation': comparison
                    })
                    
                except Exception as e:
                    print(f"  Error comparing {xgboost_key} with {model_name}: {e}")
    
    # Compare best baseline with XGBoost
    if xgboost_key:
        baseline_models = [m for m in model_names if 'XGBoost' not in m]
        if baseline_models:
            # Find best baseline (lowest mean error)
            best_baseline = min(baseline_models, 
                               key=lambda x: np.mean(absolute_errors[x]) if x in absolute_errors else float('inf'))
            
            if best_baseline in absolute_errors:
                best_baseline_errors = absolute_errors[best_baseline]
                xgboost_errors = absolute_errors[xgboost_key]
                
                try:
                    t_stat, p_value = ttest_rel(xgboost_errors, best_baseline_errors)
                    
                    xgboost_mean = np.mean(xgboost_errors)
                    baseline_mean = np.mean(best_baseline_errors)
                    difference = baseline_mean - xgboost_mean
                    percent_difference = (difference / baseline_mean) * 100
                    
                    print(f"\n📈 KEY COMPARISON: {xgboost_key} vs Best Baseline ({best_baseline}):")
                    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.6f}")
                    print(f"  Mean error {xgboost_key}: €{xgboost_mean:,.0f}")
                    print(f"  Mean error {best_baseline}: €{baseline_mean:,.0f}")
                    
                    if p_value < 0.05:
                        if difference > 0:
                            print(f"  ✅ XGBoost is statistically significantly better!")
                            print(f"     Improvement: {percent_difference:.1f}% lower error (p < 0.05)")
                        else:
                            print(f"  ⚠️  {best_baseline} is statistically significantly better!")
                    else:
                        print(f"  ℹ️  No statistically significant difference found (p = {p_value:.4f})")
                    
                except Exception as e:
                    print(f"  Error in key comparison: {e}")
    
    # Save test results to CSV
    if test_results:
        test_results_df = pd.DataFrame(test_results)
        test_results_path = f"{results_dir}/statistical_tests_{timestamp}.csv"
        test_results_df.to_csv(test_results_path, index=False)
        print(f"\n✅ Statistical test results saved to: {test_results_path}")
    
    return test_results

print("✅ Step 6.4.8: Statistical test function ready")

# ============================================================================
# 9. GENERATE COMPREHENSIVE REPORT
# ============================================================================

def generate_baseline_report(all_results, xgboost_metrics, baseline_results, test_results=None):
    """
    Generate a comprehensive report of baseline model comparison.
    
    Args:
        all_results: DataFrame with all model results
        xgboost_metrics: XGBoost metrics dictionary
        baseline_results: Baseline model results DataFrame
        test_results: Results from statistical tests
    """
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "reports/baseline_comparison"
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = f"{report_dir}/baseline_comparison_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BASELINE MODELS COMPARISON REPORT - PHASE 6.4\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of models compared: {len(all_results)}\n")
        f.write(f"Test set size: Based on Phase 5 (382 players from 2022-2023 season)\n\n")
        
        # 1. Executive Summary
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        best_model = all_results.loc[all_results['RMSE_€'].idxmin(), 'Model']
        best_rmse = all_results['RMSE_€'].min()
        worst_model = all_results.loc[all_results['RMSE_€'].idxmax(), 'Model']
        worst_rmse = all_results['RMSE_€'].max()
        
        f.write(f"• Best model: {best_model} (RMSE: €{best_rmse:,.0f})\n")
        f.write(f"• Worst model: {worst_model} (RMSE: €{worst_rmse:,.0f})\n")
        
        if xgboost_metrics and len(baseline_results) > 0:
            best_baseline_rmse = baseline_results['RMSE_€'].min()
            xgboost_rmse = xgboost_metrics['RMSE_€']
            improvement = ((best_baseline_rmse - xgboost_rmse) / best_baseline_rmse) * 100
            
            f.write(f"• XGBoost improvement over best baseline: {improvement:.1f}%\n")
            f.write(f"• Value of complexity: {'Significant' if improvement > 5 else 'Moderate' if improvement > 2 else 'Minimal'}\n")
        
        # Add statistical significance if available
        if test_results:
            f.write("\n• Statistical Significance Summary:\n")
            for test in test_results:
                if 'KEY COMPARISON' in test.get('Comparison', ''):
                    f.write(f"  - {test['interpretation']}\n")
        
        f.write("\n")
        
        # 2. Detailed Results
        f.write("2. DETAILED MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Model':<25} {'RMSE (€)':>15} {'MAE (€)':>15} {'R²':>8} {'% Error':>10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in all_results.iterrows():
            f.write(f"{row['Model']:<25} {row['RMSE_€']:>15,.0f} {row['MAE_€']:>15,.0f} {row['R2']:>8.3f} {row['Mean_Percentage_Error']:>10.1f}\n")
        
        f.write("\n")
        
        # 3. Statistical Test Results
        if test_results:
            f.write("3. STATISTICAL TEST RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write("Paired t-tests comparing absolute errors:\n")
            f.write("Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.1\n\n")
            
            for test in test_results:
                f.write(f"{test['Comparison']}:\n")
                f.write(f"  t-statistic = {test['t_statistic']:.4f}, p-value = {test['p_value']:.6f} {test.get('significance', '')}\n")
                f.write(f"  Interpretation: {test['interpretation']}\n\n")
        
        # 4. Key Insights
        f.write("4. KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Compare Naïve vs Complex models
        naïve_models = all_results[all_results['Model'].str.contains('Naïve')]
        complex_models = all_results[~all_results['Model'].str.contains('Naïve')]
        
        if len(naïve_models) > 0 and len(complex_models) > 0:
            best_naïve_rmse = naïve_models['RMSE_€'].min()
            best_complex_rmse = complex_models['RMSE_€'].min()
            complex_improvement = ((best_naïve_rmse - best_complex_rmse) / best_naïve_rmse) * 100
            
            f.write(f"• Improvement of complex models over naïve baseline: {complex_improvement:.1f}%\n")
        
        # Check if XGBoost is worth the complexity
        if 'XGBoost_Optimized' in all_results['Model'].values:
            xgboost_row = all_results[all_results['Model'] == 'XGBoost_Optimized'].iloc[0]
            linear_row = all_results[all_results['Model'] == 'Linear_Regression'].iloc[0] if 'Linear_Regression' in all_results['Model'].values else None
            
            if linear_row is not None:
                linear_improvement = ((linear_row['RMSE_€'] - xgboost_row['RMSE_€']) / linear_row['RMSE_€']) * 100
                f.write(f"• XGBoost improvement over Linear Regression: {linear_improvement:.1f}%\n")
        
        f.write("\n")
        
        # 5. Recommendations
        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if xgboost_metrics and len(baseline_results) > 0:
            best_baseline_rmse = baseline_results['RMSE_€'].min()
            xgboost_rmse = xgboost_metrics['RMSE_€']
            improvement = ((best_baseline_rmse - xgboost_rmse) / best_baseline_rmse) * 100
            
            # Check statistical significance if available
            is_significant = False
            if test_results:
                for test in test_results:
                    if 'KEY COMPARISON' in test.get('Comparison', ''):
                        if test['p_value'] < 0.05 and test.get('percent_difference', 0) > 0:
                            is_significant = True
            
            if improvement > 10 and is_significant:
                f.write("• STRONGLY RECOMMEND using XGBoost: Significant improvement (>10%) over baselines with statistical significance\n")
            elif improvement > 5 and is_significant:
                f.write("• RECOMMEND using XGBoost: Meaningful improvement (5-10%) over baselines with statistical significance\n")
            elif improvement > 2 and is_significant:
                f.write("• CONSIDER using XGBoost: Small improvement (2-5%) with statistical significance, trade-off with complexity\n")
            elif is_significant:
                f.write("• XGBoost shows statistically significant improvement, but magnitude is small\n")
            elif improvement > 5:
                f.write("• XGBoost shows meaningful improvement but not statistically significant\n")
            else:
                f.write("• CONSIDER simpler model: XGBoost improvement minimal and not statistically significant\n")
        else:
            f.write("• Use the model with best performance given your constraints (interpretability vs accuracy)\n")
        
        f.write("• For production: Consider trade-off between model complexity and maintainability\n")
        f.write("• For exploration: Simpler models can provide better interpretability\n")
        
        f.write("\n")
        
        # 6. Files Generated
        f.write("6. FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("• results/baseline_comparison/baseline_comparison_YYYYMMDD_HHMMSS.csv\n")
        f.write("• results/baseline_comparison/rmse_comparison_YYYYMMDD_HHMMSS.png\n")
        f.write("• results/baseline_comparison/r2_comparison_YYYYMMDD_HHMMSS.png\n")
        f.write("• results/baseline_comparison/radar_chart_comparison_YYYYMMDD_HHMMSS.png\n")
        if test_results:
            f.write("• results/baseline_comparison/statistical_tests_YYYYMMDD_HHMMSS.csv\n")
        f.write(f"• {report_path}\n")
    
    print(f"✓ Comprehensive report saved to: {report_path}")
    
    # Print summary to console
    print("\n📋 BASELINE COMPARISON SUMMARY:")
    print("-" * 40)
    print(f"• Best model: {best_model} (RMSE: €{best_rmse:,.0f})")
    print(f"• Worst model: {worst_model} (RMSE: €{worst_rmse:,.0f})")
    
    if xgboost_metrics and len(baseline_results) > 0:
        print(f"• XGBoost improvement: {improvement:.1f}% over best baseline")
        print(f"• Value of complexity: {'High' if improvement > 10 else 'Medium' if improvement > 5 else 'Low'}")
    
    print(f"• Reports and visualizations saved in: results/baseline_comparison/")
    
    return report_path

print("✅ Step 6.4.9: Report generation function ready")

# ============================================================================
# 10. MAIN EXECUTION FUNCTION
# ============================================================================

def run_baseline_comparison():
    """
    Main function to run the complete baseline models comparison.
    """
    print("=" * 70)
    print("📊 STEP 6.4 - SIMPLE BASELINE MODELS COMPARISON")
    print("=" * 70)
    
    try:
        # Create directories
        os.makedirs("results/baseline_comparison", exist_ok=True)
        os.makedirs("reports/baseline_comparison", exist_ok=True)
        
        # ------------------------------------------------------------
        # 1. LOAD DATA
        # ------------------------------------------------------------
        print("\n1. 📥 Loading data...")
        (X_train_aligned, X_test_aligned, 
         y_train_original, y_test_original,
         y_train_log, y_test_log,
         feature_names, train_data) = load_data_for_baselines()
        
        # ------------------------------------------------------------
        # 2. LOAD XGBOOST MODEL
        # ------------------------------------------------------------
        print("\n2. 🤖 Loading optimized XGBoost model...")
        xgboost_model, model_path = load_optimized_xgboost()
        
        if xgboost_model is None:
            print("⚠️  Proceeding without XGBoost model for comparison")
        
        # ------------------------------------------------------------
        # 3. INITIALIZE BASELINE MODELS
        # ------------------------------------------------------------
        print("\n3. 🎯 Initializing baseline models...")
        baseline_models = initialize_baseline_models()
        
        # ------------------------------------------------------------
        # 4. TRAIN AND EVALUATE BASELINE MODELS
        # ------------------------------------------------------------
        print("\n4. 🔧 Training and evaluating baseline models...")
        baseline_results, baseline_predictions = train_and_evaluate_models(
            baseline_models, X_train_aligned, y_train_log, 
            X_test_aligned, y_test_log, y_test_original
        )
        
        # ------------------------------------------------------------
        # 5. EVALUATE XGBOOST MODEL
        # ------------------------------------------------------------
        print("\n5. ⚡ Evaluating XGBoost model...")
        xgboost_metrics = None
        xgboost_predictions = None
        
        if xgboost_model is not None:
            xgboost_metrics, xgboost_predictions = evaluate_xgboost_model(
                xgboost_model, X_test_aligned, y_test_original, y_test_log
            )
            
            # Add XGBoost predictions to predictions dict
            if xgboost_predictions:
                baseline_predictions['XGBoost_Optimized'] = xgboost_predictions
        
        # ------------------------------------------------------------
        # 6. PERFORM STATISTICAL TESTS
        # ------------------------------------------------------------
        print("\n6. 📈 Performing statistical tests...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/baseline_comparison"
        
        test_results = perform_statistical_tests(
            baseline_predictions, y_test_original, results_dir, timestamp
        )
        
        # ------------------------------------------------------------
        # 7. COMPARE RESULTS AND CREATE VISUALIZATIONS
        # ------------------------------------------------------------
        print("\n7. 📊 Comparing results and creating visualizations...")
        all_results = compare_and_visualize_results(
            baseline_results, xgboost_metrics, baseline_predictions
        )
        
        # ------------------------------------------------------------
        # 8. GENERATE COMPREHENSIVE REPORT
        # ------------------------------------------------------------
        print("\n8. 📋 Generating comprehensive report...")
        report_path = generate_baseline_report(
            all_results, xgboost_metrics, baseline_results, test_results
        )
        
        # ------------------------------------------------------------
        # 9. COMPLETION
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print("✅ STEP 6.4 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   6.5 - Synthesis and Deployment Preparation")
        
        print(f"\n📁 FILES CREATED:")
        print(f"   • results/baseline_comparison/baseline_comparison_*.csv")
        print(f"   • results/baseline_comparison/*_comparison_*.png (3 visualization files)")
        if test_results:
            print(f"   • results/baseline_comparison/statistical_tests_*.csv")
        print(f"   • {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR IN STEP 6.4: {e}")
        import traceback
        traceback.print_exc()
        return False

print("✅ Step 6.4.10: Main execution function ready")

# ============================================================================
# 11. EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":
    # Run the baseline comparison
    success = run_baseline_comparison()
    
    if success:
        print("\n🎉 Baseline models comparison complete!")
        print("   Check the 'results/baseline_comparison' folder for all results.")
    else:
        print("\n❌ Baseline models comparison failed.")
        