"""
PHASE 6.2 - Feature Importance Analysis
Samuel Vermeulen - Capstone Project 2025

This module analyzes which features are most important for predicting player value.
We'll validate two key hypotheses:
1. Minutes played in top 10 clubs significantly increases player value
2. Goals are more important for strikers than other positions
"""

# ============================================================================
# 1. IMPORTS AND SETUP : This sets up the basic structure of our feature analysis module. We import necessary libraries and configure logging
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

# Machine Learning
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Step 6.2.1: File setup complete")

# ============================================================================
# 2. DATA LOADING FUNCTIONS : This function loads the 25 features we created in Phase 4 and aligns them between training and test sets
# ============================================================================

def load_data_for_analysis():
    """
    Load the data needed for feature importance analysis.
    
    Returns:
        X_train: Training features (aligned)
        X_test: Test features (aligned) 
        feature_names: List of feature names
        metadata: Feature engineering metadata
    """
    logger.info("📥 Loading data for feature analysis...")
    
    try:
        # 1. Load transformed features from Phase 4
        X_train = pd.read_csv("data/processed/X_train_transformed.csv")
        X_test = pd.read_csv("data/processed/X_test_transformed.csv")
        
        # 2. Load metadata
        with open("data/processed/feature_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # 3. Align features (in case of differences)
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
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   • Training shape: {X_train_aligned.shape}")
        logger.info(f"   • Test shape: {X_test_aligned.shape}")
        logger.info(f"   • Number of features: {len(all_columns)}")
        
        return X_train_aligned, X_test_aligned, all_columns, metadata
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

print("✅ Step 6.2.2: Data loading function ready")

# ============================================================================
# 3. MODEL LOADING FUNCTIONS : This function finds and loads the XGBoost model we optimized in Step 6.1
# ============================================================================

def load_optimized_model():
    """
    Load the XGBoost model optimized in Step 6.1.
    
    Returns:
        model: Trained XGBoost model
        model_info: Information about when it was trained
    """
    logger.info("🤖 Loading optimized XGBoost model...")
    
    # Define where models are saved
    model_dir = "models/optimized"
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ Model directory not found: {model_dir}")
        logger.error("Please run Step 6.1 first to train and save the model")
        return None, None
    
    # Find the most recent model
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    if not model_files:
        logger.error("❌ No model files found in directory")
        return None, None
    
    # Sort by timestamp (newest first)
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(model_dir, latest_model)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Extract timestamp from filename
    # Format: xgboost_optimized_YYYYMMDD_HHMMSS.joblib
    timestamp_str = latest_model.replace('xgboost_optimized_', '').replace('.joblib', '')
    
    logger.info(f"✅ Model loaded: {latest_model}")
    logger.info(f"   • Trained on: {timestamp_str[:8]} at {timestamp_str[9:]}")
    
    return model, timestamp_str

print("✅ Step 6.2.3: Model loading function ready")


# ============================================================================
# 4. FEATURE IMPORTANCE CALCULATION : This function uses XGBoost's built-in method to calculate how important each feature is for making predictions. The "gain" importance measures how much each feature reduces the prediction error
# ============================================================================

def calculate_feature_importance(model, X_train, feature_names):
    """
    Calculate feature importance using XGBoost's built-in methods.
    
    XGBoost provides three types of importance:
    1. weight: How many times a feature is used to split the data
    2. gain: Average gain (reduction in loss) when using the feature
    3. cover: Average coverage (number of samples) affected by splits
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        feature_names: List of feature names
    
    Returns:
        importance_df: DataFrame with importance scores
    """
    logger.info("🔍 Calculating feature importance...")
    
    # Get the booster object from the model
    booster = model.get_booster()
    
    # Get importance scores (we'll use 'gain' as it's most relevant for predictions)
    importance_scores = booster.get_score(importance_type='gain')
    
    # Create a DataFrame
    importance_data = []
    
    for feature in feature_names:
        # Get importance score (0 if feature wasn't used)
        score = importance_scores.get(feature, 0)
        importance_data.append({
            'feature': feature,
            'gain_importance': score
        })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Calculate percentage importance
    total_importance = importance_df['gain_importance'].sum()
    if total_importance > 0:
        importance_df['gain_percentage'] = (importance_df['gain_importance'] / total_importance) * 100
    
    # Sort by importance (highest first)
    importance_df = importance_df.sort_values('gain_importance', ascending=False)
    
    logger.info(f"✅ Feature importance calculated")
    logger.info(f"   • Top feature: {importance_df.iloc[0]['feature']}")
    logger.info(f"   • Top importance: {importance_df.iloc[0]['gain_percentage']:.1f}%")
    
    return importance_df

print("✅ Step 6.2.4: Feature importance calculation ready")

# ============================================================================
# 5. DISPLAY TOP FEATURES : This function shows us which features are most important in a clean, readable table format
# ============================================================================

def display_top_features(importance_df, top_n=15):
    """
    Display the top N most important features in a readable format.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to display
    """
    print("\n" + "=" * 70)
    print("🏆 TOP FEATURES BY IMPORTANCE")
    print("=" * 70)
    
    # Get top N features
    top_features = importance_df.head(top_n).copy()
    
    # Add ranking
    top_features['rank'] = range(1, len(top_features) + 1)
    
    # Categorize features for better understanding
    def categorize_feature(feature_name):
        if 'club_' in feature_name:
            return 'Club Feature'
        elif 'is_' in feature_name:
            return 'Position Feature'
        elif 'Goals' in feature_name:
            return 'Goals'
        elif 'Assists' in feature_name:
            return 'Assists'
        elif 'Minutes' in feature_name:
            return 'Minutes Played'
        elif 'Age' in feature_name:
            return 'Age'
        elif 'Height' in feature_name or 'Weight' in feature_name:
            return 'Physical Attribute'
        elif 'Nation' in feature_name:
            return 'Nationality'
        else:
            return 'Other'
    
    top_features['category'] = top_features['feature'].apply(categorize_feature)
    
    # Display table
    print(f"\n{'Rank':<5} {'Feature':<35} {'Importance %':<12} {'Category':<20}")
    print("-" * 80)
    
    for _, row in top_features.iterrows():
        feature_display = row['feature']
        if len(feature_display) > 32:
            feature_display = feature_display[:29] + "..."
        
        print(f"{row['rank']:<5} {feature_display:<35} {row['gain_percentage']:>10.1f}% {row['category']:<20}")
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print(f"   • Total features analyzed: {len(importance_df)}")
    print(f"   • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['gain_percentage']:.1f}%)")
    print(f"   • Top 5 features account for: {importance_df.head(5)['gain_percentage'].sum():.1f}% of importance")
    print(f"   • Top 10 features account for: {importance_df.head(10)['gain_percentage'].sum():.1f}% of importance")
    
    return top_features

print("✅ Step 6.2.5: Top features display function ready")

# ============================================================================
# 6. VALIDATE HYPOTHESIS 1: TOP CLUBS : This function checks if your first hypothesis is correct - that playing for top clubs increases player value
# ============================================================================

def validate_top_club_hypothesis(importance_df):
    """
    Validate Hypothesis 1: Minutes played in top 10 clubs significantly increases value.
    
    We'll check:
    1. Are top club features in the top important features?
    2. What's the combined importance of top club features?
    3. Which specific clubs are most important?
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING HYPOTHESIS 1: TOP CLUBS & MINUTES PLAYED")
    print("=" * 70)
    
    # List of top 10 Premier League clubs (based on 5-Year Ranking)
    top_10_clubs = [
        'Manchester City', 'Liverpool', 'Chelsea', 'Manchester United',
        'Tottenham', 'Arsenal', 'Leicester City', 'West Ham',
        'Aston Villa', 'Everton'
    ]
    
    # Find club features in importance data
    club_features = []
    club_importance = 0
    
    for club in top_10_clubs:
        feature_name = f"club_{club}"
        if feature_name in importance_df['feature'].values:
            imp = importance_df.loc[importance_df['feature'] == feature_name, 'gain_percentage'].values[0]
            club_features.append({
                'club': club,
                'feature': feature_name,
                'importance': imp
            })
            club_importance += imp
    
    # Find minutes played features
    minutes_features = []
    minutes_importance = 0
    
    for feature in importance_df['feature']:
        if 'Minutes' in feature:
            imp = importance_df.loc[importance_df['feature'] == feature, 'gain_percentage'].values[0]
            minutes_features.append({
                'feature': feature,
                'importance': imp
            })
            minutes_importance += imp
    
    # Display results
    print(f"\n📊 HYPOTHESIS 1 RESULTS:")
    print(f"   • Total importance of top club features: {club_importance:.1f}%")
    print(f"   • Total importance of minutes played features: {minutes_importance:.1f}%")
    print(f"   • Combined importance: {(club_importance + minutes_importance):.1f}%")
    
    # Check if top clubs are in top features
    top_10_features = importance_df.head(10)['feature'].tolist()
    top_clubs_in_top_10 = []
    
    for club_data in club_features:
        if club_data['feature'] in top_10_features:
            top_clubs_in_top_10.append(club_data)
    
    print(f"\n🏆 TOP CLUBS IN TOP 10 FEATURES:")
    if top_clubs_in_top_10:
        for i, club_data in enumerate(sorted(top_clubs_in_top_10, key=lambda x: x['importance'], reverse=True), 1):
            print(f"   {i}. {club_data['club']}: {club_data['importance']:.1f}%")
        print(f"   ✅ Hypothesis SUPPORTED: Top clubs are important features")
    else:
        print(f"   ⚠️  No top clubs in top 10 features")
        print(f"   ❌ Hypothesis NOT SUPPORTED by feature ranking")
    
    # Find the most important club
    if club_features:
        most_important_club = max(club_features, key=lambda x: x['importance'])
        print(f"\n👑 MOST IMPORTANT CLUB:")
        print(f"   • {most_important_club['club']}: {most_important_club['importance']:.1f}%")
    
    return {
        'club_importance': club_importance,
        'minutes_importance': minutes_importance,
        'top_clubs_in_top_10': len(top_clubs_in_top_10)
    }

print("✅ Step 6.2.6: Top club hypothesis validation ready")

# ============================================================================
# 7. VALIDATE HYPOTHESIS 2: GOALS FOR STRIKERS : This function checks if your second hypothesis is correct - that goals are particularly important for strikers
# ============================================================================

def validate_goals_hypothesis(importance_df):
    """
    Validate Hypothesis 2: Goals are more important for strikers than other positions.
    
    We'll check:
    1. Importance of goals-related features
    2. Importance of position features (especially strikers/forwards)
    3. Compare goals importance vs position importance
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDATING HYPOTHESIS 2: GOALS FOR STRIKERS")
    print("=" * 70)
    
    # Find goals-related features
    goals_features = []
    goals_importance = 0
    
    goals_keywords = ['Goals', 'goals']  # Case variations
    for feature in importance_df['feature']:
        if any(keyword in feature for keyword in goals_keywords):
            imp = importance_df.loc[importance_df['feature'] == feature, 'gain_percentage'].values[0]
            goals_features.append({
                'feature': feature,
                'importance': imp
            })
            goals_importance += imp
    
    # Find position features
    position_features = []
    position_importance = 0
    
    for feature in importance_df['feature']:
        if feature.startswith('is_'):
            imp = importance_df.loc[importance_df['feature'] == feature, 'gain_percentage'].values[0]
            position_name = feature.replace('is_', '').replace('_', ' ').title()
            position_features.append({
                'feature': feature,
                'position': position_name,
                'importance': imp
            })
            position_importance += imp
    
    # Specifically find striker/forward position
    striker_features = [f for f in position_features if 'Forward' in f['position'] or 'Striker' in f['position']]
    striker_importance = sum([f['importance'] for f in striker_features])
    
    # Display results
    print(f"\n📊 HYPOTHESIS 2 RESULTS:")
    print(f"   • Total importance of goals features: {goals_importance:.1f}%")
    print(f"   • Total importance of position features: {position_importance:.1f}%")
    print(f"   • Importance of striker/forward position: {striker_importance:.1f}%")
    
    print(f"\n⚽ GOALS FEATURES DETAILS:")
    if goals_features:
        goals_features_sorted = sorted(goals_features, key=lambda x: x['importance'], reverse=True)
        for i, goal_feat in enumerate(goals_features_sorted, 1):
            print(f"   {i}. {goal_feat['feature']}: {goal_feat['importance']:.1f}%")
    else:
        print(f"   ⚠️  No goals features found")
    
    print(f"\n🎯 POSITION FEATURES DETAILS:")
    if position_features:
        position_features_sorted = sorted(position_features, key=lambda x: x['importance'], reverse=True)
        for i, pos_feat in enumerate(position_features_sorted, 1):
            print(f"   {i}. {pos_feat['position']}: {pos_feat['importance']:.1f}%")
    else:
        print(f"   ⚠️  No position features found")
    
    # Compare goals vs position importance
    print(f"\n🎯 COMPARISON:")
    if goals_importance > striker_importance:
        print(f"   ✅ Goals ({goals_importance:.1f}%) are MORE important than striker position ({striker_importance:.1f}%)")
        print(f"   → Your hypothesis is SUPPORTED!")
    else:
        print(f"   ⚠️  Striker position ({striker_importance:.1f}%) is MORE important than goals ({goals_importance:.1f}%)")
        print(f"   → Your hypothesis is NOT SUPPORTED")
    
    # Check if goals are in top features
    top_10_features = importance_df.head(10)['feature'].tolist()
    goals_in_top_10 = [f for f in goals_features if f['feature'] in top_10_features]
    
    if goals_in_top_10:
        print(f"\n🏆 GOALS IN TOP 10 FEATURES:")
        for i, goal_feat in enumerate(sorted(goals_in_top_10, key=lambda x: x['importance'], reverse=True), 1):
            print(f"   {i}. {goal_feat['feature']}: {goal_feat['importance']:.1f}%")
    
    return {
        'goals_importance': goals_importance,
        'position_importance': position_importance,
        'striker_importance': striker_importance,
        'goals_in_top_10': len(goals_in_top_10)
    }

print("✅ Step 6.2.7: Goals hypothesis validation ready")

# ============================================================================
# 8. CREATE VISUALIZATIONS : This function creates visual charts to help us understand the feature importance better
# ============================================================================

def create_visualizations(importance_df, top_features):
    """
    Create visualizations to better understand feature importance.
    
    Args:
        importance_df: Full feature importance DataFrame
        top_features: Top N features DataFrame
    """
    logger.info("📊 Creating feature importance visualizations...")
    
    # Create results directory
    results_dir = "results/feature_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Bar chart of top features
    plt.figure(figsize=(12, 8))
    
    # Sort for better visualization (highest at top)
    top_features_sorted = top_features.sort_values('gain_percentage', ascending=True)
    
    # Create color mapping based on category
    color_map = {
        'Club Feature': 'lightblue',
        'Position Feature': 'lightgreen',
        'Goals': 'salmon',
        'Assists': 'orange',
        'Minutes Played': 'gold',
        'Age': 'lightgray',
        'Physical Attribute': 'lightcoral',
        'Nationality': 'violet',
        'Other': 'silver'
    }
    
    colors = [color_map.get(cat, 'gray') for cat in top_features_sorted['category']]
    
    plt.barh(range(len(top_features_sorted)), 
             top_features_sorted['gain_percentage'], 
             color=colors)
    
    plt.yticks(range(len(top_features_sorted)), top_features_sorted['feature'])
    plt.xlabel('Importance (%)')
    plt.title('Top 15 Features by Importance')
    
    # Add value labels
    for i, v in enumerate(top_features_sorted['gain_percentage']):
        plt.text(v + 0.2, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plot1_path = f"{results_dir}/top_features_{timestamp}.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature categories pie chart
    plt.figure(figsize=(10, 8))
    
    # Group by category
    category_summary = top_features.groupby('category')['gain_percentage'].sum().reset_index()
    
    # Only show categories with > 0 importance
    category_summary = category_summary[category_summary['gain_percentage'] > 0]
    
    # Create colors for pie chart
    pie_colors = [color_map.get(cat, 'gray') for cat in category_summary['category']]
    
    plt.pie(category_summary['gain_percentage'], 
            labels=category_summary['category'],
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90)
    
    plt.title('Feature Importance by Category (Top 15 Features)')
    plt.tight_layout()
    plot2_path = f"{results_dir}/feature_categories_{timestamp}.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Visualizations saved:")
    logger.info(f"   • {plot1_path}")
    logger.info(f"   • {plot2_path}")
    
    return {
        'top_features_plot': plot1_path,
        'categories_plot': plot2_path
    }

print("✅ Step 6.2.8: Visualization functions ready")


# ============================================================================
# 9. MAIN EXECUTION FUNCTION
# ============================================================================

def run_feature_analysis():
    """
    Main function to run the complete feature importance analysis.
    """
    print("=" * 70)
    print("🧪 STEP 6.2 - FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    try:
        # ------------------------------------------------------------
        # 1. LOAD DATA AND MODEL
        # ------------------------------------------------------------
        print("\n1. 📥 Loading data and model...")
        
        # Load data
        X_train, X_test, feature_names, metadata = load_data_for_analysis()
        
        # Load optimized model
        model, model_timestamp = load_optimized_model()
        if model is None:
            print("❌ Could not load model. Exiting.")
            return False
        
        # ------------------------------------------------------------
        # 2. CALCULATE FEATURE IMPORTANCE
        # ------------------------------------------------------------
        print("\n2. 🔍 Calculating feature importance...")
        
        importance_df = calculate_feature_importance(model, X_train, feature_names)
        
        # ------------------------------------------------------------
        # 3. DISPLAY TOP FEATURES
        # ------------------------------------------------------------
        print("\n3. 🏆 Displaying top features...")
        
        top_features = display_top_features(importance_df, top_n=15)
        
        # ------------------------------------------------------------
        # 4. VALIDATE HYPOTHESES
        # ------------------------------------------------------------
        print("\n4. 🔬 Validating your hypotheses...")
        
        # Hypothesis 1: Top clubs
        hyp1_results = validate_top_club_hypothesis(importance_df)
        
        # Hypothesis 2: Goals for strikers
        hyp2_results = validate_goals_hypothesis(importance_df)
        
        # ------------------------------------------------------------
        # 5. CREATE VISUALIZATIONS
        # ------------------------------------------------------------
        print("\n5. 📊 Creating visualizations...")
        
        plot_paths = create_visualizations(importance_df, top_features)
        
        # ------------------------------------------------------------
        # 6. SAVE RESULTS
        # ------------------------------------------------------------
        print("\n6. 💾 Saving results...")
        
        # Save importance data
        results_dir = "results/feature_analysis"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        importance_path = f"{results_dir}/feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Create summary report
        summary_path = f"{results_dir}/analysis_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("FEATURE IMPORTANCE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis date: {timestamp}\n")
            f.write(f"Model trained: {model_timestamp}\n\n")
            
            f.write("TOP 5 FEATURES:\n")
            for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                f.write(f"  {i}. {row['feature']}: {row['gain_percentage']:.1f}%\n")
            
            f.write(f"\nHYPOTHESIS 1 RESULTS (Top Clubs):\n")
            f.write(f"  • Club features importance: {hyp1_results['club_importance']:.1f}%\n")
            f.write(f"  • Top clubs in top 10: {hyp1_results['top_clubs_in_top_10']}\n")
            
            f.write(f"\nHYPOTHESIS 2 RESULTS (Goals for Strikers):\n")
            f.write(f"  • Goals importance: {hyp2_results['goals_importance']:.1f}%\n")
            f.write(f"  • Striker position importance: {hyp2_results['striker_importance']:.1f}%\n")
            f.write(f"  • Goals in top 10: {hyp2_results['goals_in_top_10']}\n")
        
        print(f"   ✓ Importance data saved: {importance_path}")
        print(f"   ✓ Summary saved: {summary_path}")
        
        # ------------------------------------------------------------
        # 7. BUSINESS INSIGHTS
        # ------------------------------------------------------------
        print("\n7. 💡 BUSINESS INSIGHTS:")
        
        top_feature = importance_df.iloc[0]['feature']
        top_importance = importance_df.iloc[0]['gain_percentage']
        
        print(f"   • Most important feature: {top_feature} ({top_importance:.1f}%)")
        print(f"   • Club features account for: {hyp1_results['club_importance']:.1f}% of importance")
        print(f"   • Performance features (goals+minutes): {(hyp2_results['goals_importance'] + hyp1_results['minutes_importance']):.1f}%")
        
        if hyp1_results['club_importance'] > hyp2_results['goals_importance']:
            print(f"   → Club matters MORE than individual goals!")
        else:
            print(f"   → Individual goals matter MORE than club!")
        
        # ------------------------------------------------------------
        # 8. COMPLETION
        # ------------------------------------------------------------
        print("\n" + "=" * 70)
        print("✅ STEP 6.2 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   6.3 - Error and Residual Analysis")
        print(f"   6.4 - Simple Baseline Models")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR IN STEP 6.2: {e}")
        import traceback
        traceback.print_exc()
        return False

print("✅ Step 6.2.9: Main execution function ready")

# ============================================================================
# 10. EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":
    # Run the feature analysis
    success = run_feature_analysis()
    
    if success:
        print("\n🎉 Feature importance analysis complete!")
        print("   Check the 'results/feature_analysis' folder for visualizations.")
    else:
        print("\n❌ Feature importance analysis failed.")



