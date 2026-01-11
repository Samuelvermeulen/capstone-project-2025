
#!/usr/bin/env python3
"""
Complete baseline model pipeline.
Samuel Vermeulen - Capstone Project 2025
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 60)
    print("PHASE 3: BASELINE MODEL (Age + Position only)")
    print("=" * 60)
    
    try:
        # Import modules
        from data_loader import get_data_pipeline
        from features import prepare_baseline_data
        from models import BaselineModel
        
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading data...")
        train_df, test_df = get_data_pipeline(clean=True, save=False)
        
        # Step 2: Prepare baseline features
        logger.info("Step 2: Preparing baseline features...")
        X_train, y_train, X_test, y_test = prepare_baseline_data(train_df, test_df)
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"  Training set: {X_train.shape[0]} players")
        print(f"  Test set: {X_test.shape[0]} players")
        print(f"  Features: {list(X_train.columns)}")
        
        # Step 3: Train baseline model
        logger.info("Step 3: Training baseline model...")
        baseline_model = BaselineModel(random_state=42)
        baseline_model.fit(X_train, y_train)
        
        # Step 4: Evaluate baseline model
        logger.info("Step 4: Evaluating baseline model...")
        metrics, predictions = baseline_model.evaluate(X_test, y_test)
        
        print(f"\n📈 BASELINE MODEL PERFORMANCE:")
        print(f"  RMSE: €{metrics['rmse']:,.0f}")
        print(f"  MAE: €{metrics['mae']:,.0f}")
        print(f"  R²: {metrics['r2']:.3f}")
        
        # CORRECTION : Vérifier si 'mape' existe dans les métriques
        if 'mape' in metrics:
            print(f"  MAPE: {metrics['mape']:.1f}%")
        else:
            print(f"  MAPE: N/A (not available)")
        
        # Step 5: Save predictions
        predictions_path = "results/baseline_predictions.csv"
        os.makedirs("results", exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Step 6: Create and save feature importance plot (avec gestion d'erreur)
        logger.info("Creating feature importance plot...")
        plot_path = "results/baseline_feature_importance.png"
        try:
            # Essayer d'appeler la méthode plot_feature_importance
            fig = baseline_model.plot_feature_importance()
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {plot_path}")
        except AttributeError:
            logger.warning("BaselineModel does not have plot_feature_importance method. Creating simple plot...")
            # Créer un graphique simple de feature importance
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Récupérer les feature importances du modèle
            if hasattr(baseline_model.model, 'feature_importances_'):
                importances = baseline_model.model.feature_importances_
                features = X_train.columns
                
                # Créer un DataFrame
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                # Créer le graphique
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df.plot(kind='barh', x='feature', y='importance', ax=ax)
                ax.set_title('Feature Importance - Baseline Model')
                ax.set_xlabel('Importance')
                plt.tight_layout()
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Simple feature importance plot saved to: {plot_path}")
                plt.close(fig)
            else:
                logger.warning("Model does not have feature_importances_ attribute. Skipping plot.")
                plot_path = None
        
        print(f"\n✅ PHASE 3 COMPLETED SUCCESSFULLY!")
        print(f"\n📁 Results saved in 'results/' directory:")
        print(f"  - baseline_predictions.csv")
        if plot_path and os.path.exists(plot_path):
            print(f"  - baseline_feature_importance.png")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in baseline pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import subprocess, sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))
    steps = [
        "run_baseline.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/model_optimization.py",
        "src/feature_analysis.py",
        "src/error_analysis.py",
        "src/baseline_models.py",
        "src/synthesis.py"
    ]
    for s in steps:
        subprocess.run([sys.executable, s], cwd=ROOT, check=True)
    
