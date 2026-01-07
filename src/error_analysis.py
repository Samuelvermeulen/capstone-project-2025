
"""
PHASE 6.3 - Error and Residual Analysis
Capstone Project 2025

This module performs comprehensive error analysis on model predictions.
We'll analyze:
1. Error distribution and statistics
2. Residual analysis by segments (position, age, club)
3. Identification of worst prediction cases
4. Visualization of errors for reporting
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
from scipy import stats
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Step 6.3.1: File setup complete")

# ============================================================================
# 2. DATA LOADING FUNCTIONS
# ============================================================================

def load_data_for_error_analysis():
    """
    Load the data needed for error analysis.
    
    Returns:
        X_test_aligned: Test features (aligned)
        y_test_original: Original target values in €
        y_test_log: Log-transformed target values
        test_metadata: DataFrame with player metadata
    """
    logger.info("📥 Loading data for error analysis...")
    
    try:
        # 1. Load test data
        X_test = pd.read_csv("data/processed/X_test_transformed.csv")
        test_data = pd.read_csv("data/processed/test_data.csv")
        
        # 2. Load target values
        y_test_log = pd.read_csv("data/processed/y_test_log.csv").values.flatten()
        
        # 3. Convert log values back to original scale
        y_test_original = np.expm1(y_test_log)
        
        # 4. Load feature alignment metadata
        with open("data/processed/feature_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # 5. Align features (same as in Phase 5)
        X_train = pd.read_csv("data/processed/X_train_transformed.csv")
        all_columns = sorted(set(X_train.columns).union(set(X_test.columns)))
        
        X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=all_columns)
        for col in all_columns:
            if col in X_test.columns:
                X_test_aligned[col] = X_test[col]
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   • X_test shape: {X_test_aligned.shape}")
        logger.info(f"   • y_test shape: {y_test_original.shape}")
        logger.info(f"   • Test metadata: {len(test_data)} players")
        
        return X_test_aligned, y_test_original, y_test_log, test_data
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

print("✅ Step 6.3.2: Data loading function ready")

# ============================================================================
# 3. MODEL LOADING FUNCTIONS
# ============================================================================

def load_best_model():
    """
    Load the best performing model (XGBoost from Phase 5/6.1).
    
    Returns:
        model: Trained XGBoost model
        model_info: Information about the model
    """
    logger.info("🤖 Loading optimized XGBoost model...")
    
    # Check multiple possible locations
    possible_paths = [
        "models/optimized/xgboost_optimized.joblib",
        "models/xgboost_optimized.joblib",
        "models/xgboost_best.joblib",
        "models/best_model.joblib"
    ]
    
    model = None
    model_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        logger.error("❌ No model file found. Please run Phase 6.1 first.")
        # Try to find any .joblib file
        import glob
        model_files = glob.glob("models/**/*.joblib", recursive=True)
        if model_files:
            model_path = model_files[0]
            logger.info(f"   Found alternative model: {model_path}")
        else:
            return None, None
    
    # Load the model
    model = joblib.load(model_path)
    
    logger.info(f"✅ Model loaded from: {model_path}")
    logger.info(f"   • Model type: {type(model).__name__}")
    
    return model, model_path

print("✅ Step 6.3.3: Model loading function ready")

# ============================================================================
# 4. RESIDUALS CALCULATION (Step 6.3.1)
# ============================================================================

def calculate_residuals_and_errors(model, X_test, y_test_original, y_test_log, test_metadata):
    """
    STEP 6.3.1: Calculate prediction residuals and errors.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test_original: Original target values (€)
        y_test_log: Log-transformed target values
        test_metadata: DataFrame with player metadata
    
    Returns:
        error_df: DataFrame with all error metrics
        predictions: Model predictions
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.1: CALCULATING RESIDUALS AND ERRORS")
    print("=" * 70)
    
    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred_log = model.predict(X_test)
    y_pred_original = np.expm1(y_pred_log)  # Convert back to original scale
    
    # Calculate errors
    logger.info("Calculating error metrics...")
    residuals_original = y_test_original - y_pred_original
    residuals_log = y_test_log - y_pred_log
    percentage_error = (np.abs(residuals_original) / y_test_original) * 100
    
    # Create comprehensive error DataFrame
    error_df = pd.DataFrame({
        'player': test_metadata['Player'].values,
        'club': test_metadata['Club'].values,
        'position': test_metadata['Position'].values,
        'age': test_metadata['Age'].values,
        'true_value': y_test_original,
        'predicted_value': y_pred_original,
        'residual': residuals_original,
        'abs_residual': np.abs(residuals_original),
        'percentage_error': percentage_error,
        'over_under': np.where(residuals_original > 0, 'Underestimated', 'Overestimated')
    })
    
    # Add value categories for segmentation
    error_df['value_category'] = pd.qcut(
        error_df['true_value'], 
        q=4, 
        labels=['Low (Q1)', 'Medium-Low (Q2)', 'Medium-High (Q3)', 'High (Q4)']
    )
    
    # Save error data
    error_df.to_csv("reports/error_analysis/error_data.csv", index=False)
    
    print(f"✓ Error DataFrame created with {len(error_df)} players")
    print(f"✓ Error data saved to: reports/error_analysis/error_data.csv")
    
    # Display basic statistics
    print("\n📊 BASIC ERROR STATISTICS:")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE): €{error_df['abs_residual'].mean():,.0f}")
    print(f"Median Absolute Error: €{error_df['abs_residual'].median():,.0f}")
    print(f"Root Mean Square Error (RMSE): €{np.sqrt(np.mean(residuals_original**2)):,.0f}")
    print(f"Mean Percentage Error: {percentage_error.mean():.1f}%")
    print(f"Mean Residual: €{residuals_original.mean():,.0f} "
          f"({'Overestimation' if residuals_original.mean() < 0 else 'Underestimation'})")
    
    return error_df, {
        'y_pred_original': y_pred_original,
        'y_pred_log': y_pred_log,
        'residuals_original': residuals_original,
        'residuals_log': residuals_log,
        'percentage_error': percentage_error
    }

print("✅ Step 6.3.4: Residuals calculation function ready")

# ============================================================================
# 5. ERROR DISTRIBUTION ANALYSIS (Step 6.3.2)
# ============================================================================

def analyze_error_distribution(error_df, residuals_original):
    """
    STEP 6.3.2: Analyze error distribution and normality.
    
    Args:
        error_df: DataFrame with error metrics
        residuals_original: Array of residuals in original scale
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.2: ANALYZING ERROR DISTRIBUTION")
    print("=" * 70)
    
    # Create directory for plots
    os.makedirs("reports/error_analysis/figures", exist_ok=True)
    
    # Statistical tests
    print("\n📈 DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    # Shapiro-Wilk test for normality (limited to 5000 samples)
    sample_size = min(5000, len(residuals_original))
    sample_residuals = residuals_original[:sample_size] if len(residuals_original) > 5000 else residuals_original
    
    stat, p_value = stats.shapiro(sample_residuals)
    print(f"Shapiro-Wilk Normality Test:")
    print(f"  • Statistic: {stat:.4f}")
    print(f"  • p-value: {p_value:.4e}")
    print(f"  • Conclusion: {'Residuals are normal' if p_value > 0.05 else 'Residuals are NOT normal'}")
    
    # Skewness and kurtosis
    skewness = stats.skew(residuals_original)
    kurtosis = stats.kurtosis(residuals_original)
    print(f"\nSkewness and Kurtosis:")
    print(f"  • Skewness: {skewness:.3f} {'(right skewed)' if skewness > 0 else '(left skewed)'}")
    print(f"  • Kurtosis: {kurtosis:.3f} {'(leptokurtic)' if kurtosis > 0 else '(platykurtic)'}")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram with KDE
    axes[0, 0].hist(residuals_original, bins=50, density=True, alpha=0.6, color='skyblue')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
    axes[0, 0].set_xlabel('Residual (€)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Residuals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    stats.probplot(residuals_original, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: Residuals vs Normal Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot of residuals
    axes[1, 0].boxplot(residuals_original, vert=False)
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Residual (€)')
    axes[1, 0].set_title('Box Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_residuals = np.sort(residuals_original)
    cdf = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
    axes[1, 1].plot(sorted_residuals, cdf, linewidth=2)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Residual (€)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/error_analysis/figures/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Distribution plots saved to: reports/error_analysis/figures/error_distribution.png")
    
    return {
        'shapiro_p_value': p_value,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_normal': p_value > 0.05
    }

print("✅ Step 6.3.5: Error distribution analysis function ready")

# ============================================================================
# 6. SEGMENT ANALYSIS (Step 6.3.3)
# ============================================================================

def analyze_errors_by_segments(error_df):
    """
    STEP 6.3.3: Analyze errors by different segments (position, age, club, value).
    
    Args:
        error_df: DataFrame with error metrics
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.3: ANALYZING ERRORS BY SEGMENTS")
    print("=" * 70)
    
    # Create age groups
    error_df['age_group'] = pd.cut(
        error_df['age'],
        bins=[0, 21, 25, 30, 100],
        labels=['Under 21', '21-25', '26-30', 'Over 30']
    )
    
    # Define top clubs (same as in feature analysis)
    top_clubs = [
        'Manchester City', 'Liverpool', 'Chelsea', 'Manchester United',
        'Tottenham', 'Arsenal', 'Leicester City', 'West Ham',
        'Aston Villa', 'Everton'
    ]
    
    error_df['club_category'] = error_df['club'].apply(
        lambda x: 'Top Club' if x in top_clubs else 'Other Club'
    )
    
    print("\n📊 ERROR ANALYSIS BY SEGMENT:")
    print("-" * 40)
    
    # 1. By Position
    print("\n1. BY POSITION:")
    pos_stats = error_df.groupby('position').agg({
        'abs_residual': ['mean', 'median', 'count'],
        'percentage_error': ['mean', 'median'],
        'residual': ['mean', 'std']
    }).round(2)
    
    for pos in error_df['position'].unique():
        pos_data = error_df[error_df['position'] == pos]
        print(f"   • {pos}: {len(pos_data)} players")
        print(f"     MAE: €{pos_data['abs_residual'].mean():,.0f}, "
              f"Mean % Error: {pos_data['percentage_error'].mean():.1f}%")
    
    # 2. By Age Group
    print("\n2. BY AGE GROUP:")
    age_stats = error_df.groupby('age_group').agg({
        'abs_residual': 'mean',
        'percentage_error': 'mean',
        'true_value': 'mean'
    }).round(2)
    
    for age_group in error_df['age_group'].unique():
        age_data = error_df[error_df['age_group'] == age_group]
        print(f"   • {age_group}: {len(age_data)} players")
        print(f"     MAE: €{age_data['abs_residual'].mean():,.0f}, "
              f"Avg Value: €{age_data['true_value'].mean():,.0f}")
    
    # 3. By Club Category
    print("\n3. BY CLUB CATEGORY:")
    club_stats = error_df.groupby('club_category').agg({
        'abs_residual': 'mean',
        'percentage_error': 'mean',
        'true_value': 'mean'
    }).round(2)
    
    for category in error_df['club_category'].unique():
        cat_data = error_df[error_df['club_category'] == category]
        print(f"   • {category}: {len(cat_data)} players")
        print(f"     MAE: €{cat_data['abs_residual'].mean():,.0f}, "
              f"Avg Value: €{cat_data['true_value'].mean():,.0f}")
    
    # 4. By Value Category
    print("\n4. BY VALUE CATEGORY:")
    value_stats = error_df.groupby('value_category').agg({
        'abs_residual': 'mean',
        'percentage_error': 'mean',
        'true_value': ['min', 'max', 'mean']
    }).round(2)
    
    # Create segment visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position boxplot
    sns.boxplot(x='position', y='residual', data=error_df, ax=axes[0, 0])
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals by Position')
    axes[0, 0].set_ylabel('Residual (€)')
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    # Age group boxplot
    sns.boxplot(x='age_group', y='percentage_error', data=error_df, ax=axes[0, 1])
    axes[0, 1].set_title('Percentage Error by Age Group')
    axes[0, 1].set_ylabel('Error (%)')
    
    # Club category boxplot
    sns.boxplot(x='club_category', y='abs_residual', data=error_df, ax=axes[1, 0])
    axes[1, 0].set_title('Absolute Error by Club Category')
    axes[1, 0].set_ylabel('Absolute Error (€)')
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    # Value category boxplot
    sns.boxplot(x='value_category', y='percentage_error', data=error_df, ax=axes[1, 1])
    axes[1, 1].set_title('Percentage Error by Value Category')
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("reports/error_analysis/figures/errors_by_segments.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Segment analysis plots saved to: reports/error_analysis/figures/errors_by_segments.png")
    
    return {
        'position_stats': pos_stats,
        'age_stats': age_stats,
        'club_stats': club_stats,
        'value_stats': value_stats
    }

print("✅ Step 6.3.6: Segment analysis function ready")

# ============================================================================
# 7. OUTLIER IDENTIFICATION (Step 6.3.4)
# ============================================================================

def identify_outliers(error_df, n=20):
    """
    STEP 6.3.4: Identify worst prediction cases (outliers).
    
    Args:
        error_df: DataFrame with error metrics
        n: Number of worst cases to identify
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.4: IDENTIFYING WORST PREDICTIONS (OUTLIERS)")
    print("=" * 70)
    
    # Filter out extremely low values for percentage error calculation
    significant_errors = error_df[error_df['true_value'] > 1_000_000].copy()
    
    # Identify worst cases
    print("\n🔍 WORST PREDICTIONS ANALYSIS:")
    print("-" * 40)
    
    # 1. Worst absolute errors
    worst_absolute = error_df.nlargest(n, 'abs_residual')[[
        'player', 'club', 'position', 'age',
        'true_value', 'predicted_value', 'residual', 'percentage_error'
    ]].copy()
    
    worst_absolute['error_magnitude'] = 'Absolute'
    
    print(f"\n1. Top {n} WORST ABSOLUTE ERRORS:")
    print("-" * 40)
    for i, row in worst_absolute.iterrows():
        print(f"   {i+1:2d}. {row['player']:20s} ({row['position']}, {row['club']})")
        print(f"       True: €{row['true_value']:,.0f}, Pred: €{row['predicted_value']:,.0f}")
        print(f"       Error: €{row['residual']:,.0f} ({row['percentage_error']:.1f}%)")
    
    # 2. Worst percentage errors (for players > 1M€)
    if len(significant_errors) > 0:
        worst_percentage = significant_errors.nlargest(n, 'percentage_error')[[
            'player', 'club', 'position', 'age',
            'true_value', 'predicted_value', 'residual', 'percentage_error'
        ]].copy()
        
        worst_percentage['error_magnitude'] = 'Percentage'
        
        print(f"\n2. Top {n} WORST PERCENTAGE ERRORS (>€1M players):")
        print("-" * 40)
        for i, row in worst_percentage.iterrows():
            print(f"   {i+1:2d}. {row['player']:20s} ({row['position']}, {row['club']})")
            print(f"       True: €{row['true_value']:,.0f}, Pred: €{row['predicted_value']:,.0f}")
            print(f"       Error: {row['percentage_error']:.1f}% (€{row['residual']:,.0f})")
    else:
        worst_percentage = pd.DataFrame()
        print("\n⚠️  No players with value > €1M for percentage error analysis")
    
    # 3. Largest underestimations
    underestimations = error_df[error_df['residual'] > 0].nlargest(n, 'residual')[[
        'player', 'club', 'position', 'age',
        'true_value', 'predicted_value', 'residual', 'percentage_error'
    ]].copy()
    
    underestimations['error_type'] = 'Underestimation'
    
    # 4. Largest overestimations
    overestimations = error_df[error_df['residual'] < 0].nsmallest(n, 'residual')[[
        'player', 'club', 'position', 'age',
        'true_value', 'predicted_value', 'residual', 'percentage_error'
    ]].copy()
    
    overestimations['error_type'] = 'Overestimation'
    
    # Save outliers to CSV
    worst_absolute.to_csv("reports/error_analysis/worst_absolute_errors.csv", index=False)
    
    if not worst_percentage.empty:
        worst_percentage.to_csv("reports/error_analysis/worst_percentage_errors.csv", index=False)
    
    underestimations.to_csv("reports/error_analysis/largest_underestimations.csv", index=False)
    overestimations.to_csv("reports/error_analysis/largest_overestimations.csv", index=False)
    
    # Analyze patterns in worst predictions
    print("\n📈 PATTERNS IN WORST PREDICTIONS:")
    print("-" * 40)
    
    if not worst_absolute.empty:
        print(f"• Most common position in worst errors: {worst_absolute['position'].mode()[0]}")
        print(f"• Average age in worst errors: {worst_absolute['age'].mean():.1f} years")
        print(f"• Most common club in worst errors: {worst_absolute['club'].mode()[0]}")
    
    print(f"\n✓ Outlier data saved to reports/error_analysis/")
    
    return {
        'worst_absolute': worst_absolute,
        'worst_percentage': worst_percentage,
        'underestimations': underestimations,
        'overestimations': overestimations
    }

print("✅ Step 6.3.7: Outlier identification function ready")

# ============================================================================
# 8. FINAL VISUALIZATIONS (Step 6.3.5)
# ============================================================================

def create_final_visualizations(error_df, predictions_dict):
    """
    STEP 6.3.5: Create comprehensive visualizations for reporting.
    
    Args:
        error_df: DataFrame with error metrics
        predictions_dict: Dictionary with prediction data
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.5: CREATING FINAL VISUALIZATIONS")
    print("=" * 70)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. True vs Predicted Values (scatter)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(error_df['true_value'], error_df['predicted_value'], 
                alpha=0.6, c='blue', s=30)
    
    # Add perfect prediction line
    max_val = max(error_df['true_value'].max(), error_df['predicted_value'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True Value (€)')
    ax1.set_ylabel('Predicted Value (€)')
    ax1.set_title('True vs Predicted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted Values
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(error_df['predicted_value'], predictions_dict['residuals_original'], 
                alpha=0.6, c='green', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Predicted Value (€)')
    ax2.set_ylabel('Residual (€)')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Percentage Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(error_df['percentage_error'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=error_df['percentage_error'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {error_df["percentage_error"].mean():.1f}%')
    
    ax3.set_xlabel('Percentage Error (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Percentage Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap: Mean Error by Position and Age Group
    ax4 = plt.subplot(2, 3, 4)
    
    # Create age groups for heatmap
    error_df['age_group_heatmap'] = pd.cut(
        error_df['age'],
        bins=[0, 21, 24, 27, 30, 100],
        labels=['<21', '21-24', '25-27', '28-30', '>30']
    )
    
    heatmap_data = error_df.pivot_table(
        values='abs_residual',
        index='position',
        columns='age_group_heatmap',
        aggfunc='mean'
    )
    
    im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(heatmap_data.columns)))
    ax4.set_yticks(range(len(heatmap_data.index)))
    ax4.set_xticklabels(heatmap_data.columns)
    ax4.set_yticklabels(heatmap_data.index)
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Position')
    ax4.set_title('Mean Absolute Error by Position & Age')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Mean Absolute Error (€)')
    
    # 5. Error by Value Category
    ax5 = plt.subplot(2, 3, 5)
    
    value_categories = ['Low (Q1)', 'Medium-Low (Q2)', 'Medium-High (Q3)', 'High (Q4)']
    mean_errors = []
    median_errors = []
    
    for category in value_categories:
        cat_data = error_df[error_df['value_category'] == category]
        mean_errors.append(cat_data['percentage_error'].mean())
        median_errors.append(cat_data['percentage_error'].median())
    
    x = range(len(value_categories))
    width = 0.35
    
    ax5.bar([i - width/2 for i in x], mean_errors, width, label='Mean Error', alpha=0.8)
    ax5.bar([i + width/2 for i in x], median_errors, width, label='Median Error', alpha=0.8)
    
    ax5.set_xlabel('Value Category')
    ax5.set_ylabel('Percentage Error (%)')
    ax5.set_title('Error by Value Category')
    ax5.set_xticks(x)
    ax5.set_xticklabels(value_categories, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative Error Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Sort by absolute error
    sorted_errors = error_df.sort_values('abs_residual', ascending=False).reset_index(drop=True)
    cumulative_error = sorted_errors['abs_residual'].cumsum()
    cumulative_percentage = cumulative_error / cumulative_error.max() * 100
    
    ax6.plot(range(len(sorted_errors)), cumulative_percentage, linewidth=2)
    ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% of total error')
    
    # Find where 50% of error occurs
    idx_50 = np.argmax(cumulative_percentage >= 50)
    ax6.axvline(x=idx_50, color='green', linestyle='--', alpha=0.5, 
                label=f'50% at {idx_50} players ({idx_50/len(sorted_errors)*100:.1f}%)')
    
    ax6.set_xlabel('Number of Players (sorted by error)')
    ax6.set_ylabel('Cumulative Error (%)')
    ax6.set_title('Cumulative Distribution of Errors')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/error_analysis/figures/final_visualizations.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Final visualizations saved to: reports/error_analysis/figures/final_visualizations.png")
    
    return fig

print("✅ Step 6.3.8: Final visualizations function ready")

# ============================================================================
# 9. SYNTHESIS AND REPORT (Step 6.3.6)
# ============================================================================

def generate_error_analysis_report(error_df, analysis_results):
    """
    STEP 6.3.6: Generate comprehensive error analysis report.
    
    Args:
        error_df: DataFrame with error metrics
        analysis_results: Dictionary with all analysis results
    """
    print("\n" + "=" * 70)
    print("STEP 6.3.6: GENERATING ERROR ANALYSIS REPORT")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/error_analysis/error_analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ERROR ANALYSIS REPORT - PHASE 6.3\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: XGBoost Optimized\n")
        f.write(f"Test set size: {len(error_df)} players\n")
        f.write(f"Time period: Season 2022-2023\n\n")
        
        # 1. Executive Summary
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Mean Absolute Error (MAE): €{error_df['abs_residual'].mean():,.0f}\n")
        f.write(f"• Root Mean Square Error (RMSE): €{np.sqrt(np.mean(analysis_results['predictions']['residuals_original']**2)):,.0f}\n")
        f.write(f"• Mean Percentage Error: {error_df['percentage_error'].mean():.1f}%\n")
        f.write(f"• Model bias: {'Underestimates' if analysis_results['predictions']['residuals_original'].mean() > 0 else 'Overestimates'} "
                f"by €{analysis_results['predictions']['residuals_original'].mean():,.0f} on average\n\n")
        
        # 2. Key Findings
        f.write("2. KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        # Worst performing segment
        pos_stats = analysis_results['segment_analysis']['position_stats']
        worst_pos = pos_stats[('abs_residual', 'mean')].idxmax()
        f.write(f"• Worst performing position: {worst_pos} "
                f"(MAE: €{pos_stats.loc[worst_pos, ('abs_residual', 'mean')]:,.0f})\n")
        
        # Error distribution
        if analysis_results['distribution_analysis']['is_normal']:
            f.write("• Error distribution: Approximately normal\n")
        else:
            f.write("• Error distribution: Not normal (p-value: "
                    f"{analysis_results['distribution_analysis']['shapiro_p_value']:.4e})\n")
        
        f.write(f"• Error skewness: {analysis_results['distribution_analysis']['skewness']:.3f}\n")
        
        # 3. Recommendations
        f.write("\n3. RECOMMENDATIONS FOR MODEL IMPROVEMENT\n")
        f.write("-" * 40 + "\n")
        
        # Based on worst performing segment
        f.write(f"• Focus on improving predictions for {worst_pos} position\n")
        
        # Based on value categories
        value_stats = analysis_results['segment_analysis']['value_stats']
        if 'percentage_error' in value_stats.columns:
            worst_value_cat = value_stats[('percentage_error', 'mean')].idxmax()
            f.write(f"• High relative errors for {worst_value_cat} players\n")
        
        # Based on outliers
        if 'worst_absolute' in analysis_results['outlier_analysis']:
            outliers = analysis_results['outlier_analysis']['worst_absolute']
            if not outliers.empty:
                common_club = outliers['club'].mode()[0] if not outliers['club'].mode().empty else 'N/A'
                f.write(f"• Review predictions for players from {common_club}\n")
        
        f.write("• Consider adding position-specific features\n")
        f.write("• Investigate high-error players for systematic patterns\n\n")
        
        # 4. Detailed Statistics
        f.write("4. DETAILED STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        f.write("\nError Statistics:\n")
        f.write(f"{'Metric':<25} {'Value':>15}\n")
        f.write("-" * 40 + "\n")
        metrics = [
            ('MAE', f"€{error_df['abs_residual'].mean():,.0f}"),
            ('Median Absolute Error', f"€{error_df['abs_residual'].median():,.0f}"),
            ('RMSE', f"€{np.sqrt(np.mean(analysis_results['predictions']['residuals_original']**2)):,.0f}"),
            ('Mean % Error', f"{error_df['percentage_error'].mean():.1f}%"),
            ('Median % Error', f"{error_df['percentage_error'].median():.1f}%"),
            ('Std of Residuals', f"€{analysis_results['predictions']['residuals_original'].std():,.0f}")
        ]
        
        for metric, value in metrics:
            f.write(f"{metric:<25} {value:>15}\n")
        
        # 5. Files Generated
        f.write("\n5. FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("• reports/error_analysis/error_data.csv\n")
        f.write("• reports/error_analysis/worst_absolute_errors.csv\n")
        f.write("• reports/error_analysis/worst_percentage_errors.csv\n")
        f.write("• reports/error_analysis/largest_underestimations.csv\n")
        f.write("• reports/error_analysis/largest_overestimations.csv\n")
        f.write("• reports/error_analysis/figures/error_distribution.png\n")
        f.write("• reports/error_analysis/figures/errors_by_segments.png\n")
        f.write("• reports/error_analysis/figures/final_visualizations.png\n")
    
    print(f"\n✓ Comprehensive report saved to: {report_path}")
    
    # Print summary to console
    print("\n📋 ERROR ANALYSIS SUMMARY:")
    print("-" * 40)
    print(f"• Test set size: {len(error_df)} players")
    print(f"• MAE: €{error_df['abs_residual'].mean():,.0f}")
    print(f"• Mean % Error: {error_df['percentage_error'].mean():.1f}%")
    print(f"• Model bias: {'Underestimates' if analysis_results['predictions']['residuals_original'].mean() > 0 else 'Overestimates'}")
    print(f"• Error distribution: {'Normal' if analysis_results['distribution_analysis']['is_normal'] else 'Not normal'}")
    print(f"• Reports and visualizations saved in: reports/error_analysis/")
    
    return report_path

print("✅ Step 6.3.9: Report generation function ready")

# ============================================================================
# 10. MAIN EXECUTION FUNCTION
# ============================================================================

def run_error_analysis():
    """
    Main function to run the complete error analysis pipeline.
    """
    print("=" * 70)
    print("🔍 STEP 6.3 - ERROR AND RESIDUAL ANALYSIS")
    print("=" * 70)
    
    try:
        # Create directories
        os.makedirs("reports/error_analysis/figures", exist_ok=True)
        
        # ------------------------------------------------------------
        # 1. LOAD DATA AND MODEL
        # ------------------------------------------------------------
        print("\n1. 📥 Loading data and model...")
        X_test, y_test_original, y_test_log, test_metadata = load_data_for_error_analysis()
        model, model_path = load_best_model()
        
        if model is None:
            print("❌ Could not load model. Exiting.")
            return False
        
        # ------------------------------------------------------------
        # 2. CALCULATE RESIDUALS (Step 6.3.1)
        # ------------------------------------------------------------
        print("\n2. 📊 Calculating residuals and errors...")
        error_df, predictions_dict = calculate_residuals_and_errors(
            model, X_test, y_test_original, y_test_log, test_metadata
        )
        
        # ------------------------------------------------------------
        # 3. ANALYZE ERROR DISTRIBUTION (Step 6.3.2)
        # ------------------------------------------------------------
        print("\n3. 📈 Analyzing error distribution...")
        distribution_results = analyze_error_distribution(error_df, predictions_dict['residuals_original'])
        
        # ------------------------------------------------------------
        # 4. ANALYZE ERRORS BY SEGMENTS (Step 6.3.3)
        # ------------------------------------------------------------
        print("\n4. 🎯 Analyzing errors by segments...")
        segment_results = analyze_errors_by_segments(error_df)
        
        # ------------------------------------------------------------
        # 5. IDENTIFY OUTLIERS (Step 6.3.4)
        # ------------------------------------------------------------
        print("\n5. ⚠️ Identifying worst predictions...")
        outlier_results = identify_outliers(error_df, n=20)
        
        # ------------------------------------------------------------
        # 6. CREATE FINAL VISUALIZATIONS (Step 6.3.5)
        # ------------------------------------------------------------
        print("\n6. 📊 Creating final visualizations...")
        create_final_visualizations(error_df, predictions_dict)
        
        # ------------------------------------------------------------
        # 7. GENERATE REPORT (Step 6.3.6)
        # ------------------------------------------------------------
        print("\n7. 📋 Generating comprehensive report...")
        analysis_results = {
            'predictions': predictions_dict,
            'distribution_analysis': distribution_results,
            'segment_analysis': segment_results,
            'outlier_analysis': outlier_results
        }
        
        report_path = generate_error_analysis_report(error_df, analysis_results)
        
        # ------------------------------------------------------------
        # 8. COMPLETION
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print("✅ STEP 6.3 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   6.4 - Simple Baseline Models Comparison")
        print(f"   6.5 - Synthesis and Deployment Preparation")
        
        print(f"\n📁 FILES CREATED:")
        print(f"   • reports/error_analysis/error_data.csv")
        print(f"   • reports/error_analysis/worst_*_errors.csv (4 files)")
        print(f"   • reports/error_analysis/figures/*.png (3 visualization files)")
        print(f"   • {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR IN STEP 6.3: {e}")
        import traceback
        traceback.print_exc()
        return False

print("✅ Step 6.3.10: Main execution function ready")

# ============================================================================
# 11. EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":
    # Run the error analysis
    success = run_error_analysis()
    
    if success:
        print("\n🎉 Error analysis complete!")
        print("   Check the 'reports/error_analysis' folder for all results.")
    else:
        print("\n❌ Error analysis failed.")