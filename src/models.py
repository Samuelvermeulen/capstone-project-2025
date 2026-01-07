"""
Models module for player valuation project.
Phase 5: Implementation of 4 required models
Samuel Vermeulen - Capstone Project 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import joblib

logger = logging.getLogger(__name__)

class ModelPipeline:
    """
    Pipeline for training and evaluating multiple models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize model pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        # Initialize models according to roadmap specifications
        self._initialize_models()
        
        logger.info("Model pipeline initialized with 4 models")
    
    def _initialize_models(self):
        """Initialize the 4 required models with specified parameters."""
        
        # Model 1: Linear Regression (baseline)
        self.models['Linear Regression'] = LinearRegression()
        
        # Model 2: Ridge Regression (alpha=1.0)
        self.models['Ridge Regression'] = Ridge(alpha=1.0, random_state=self.random_state)
        
        # Model 3: Random Forest Regressor
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Model 4: XGBoost Regressor
        self.models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        logger.info("Models initialized: Linear Regression, Ridge, Random Forest, XGBoost")
    
    def prepare_features(self, X: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """
        Prepare features for modeling.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        is_training : bool
            Whether this is training data
            
        Returns:
        --------
        X_scaled : numpy.array
            Scaled features
        """
        if is_training:
            self.feature_names = list(X.columns)
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted scaler on {X.shape[1]} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with is_training=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train all models on the training data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training target (log-transformed values)
        """
        logger.info(f"Training {len(self.models)} models on {X_train.shape[0]} samples...")
        
        # Prepare features
        X_train_scaled = self.prepare_features(X_train, is_training=True)
        
        # Train each model
        training_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Training predictions and metrics
            y_pred_train = model.predict(X_train_scaled)
            
            train_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            }
            
            training_results[name] = train_metrics
            
            logger.info(f"  {name} trained - Train R²: {train_metrics['r2']:.3f}")
        
        self.is_fitted = True
        self.training_results = training_results
        
        logger.info("All models trained successfully")
        return training_results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> Dict:
        """
        Make predictions with trained models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        model_name : str, optional
            Specific model to use (if None, use all models)
            
        Returns:
        --------
        predictions : dict
            Dictionary of model predictions (in log scale)
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        X_scaled = self.prepare_features(X, is_training=False)
        
        # Make predictions
        predictions = {}
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            models_to_predict = {model_name: self.models[model_name]}
        else:
            models_to_predict = self.models
        
        for name, model in models_to_predict.items():
            y_pred_log = model.predict(X_scaled)
            predictions[name] = y_pred_log
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test_log: pd.Series, 
                 y_test_original: pd.Series = None) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test_log : pandas.Series
            Test target in log scale
        y_test_original : pandas.Series, optional
            Test target in original scale (euros)
            
        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame with evaluation metrics for all models
        """
        logger.info(f"Evaluating models on {X_test.shape[0]} test samples...")
        
        # Get predictions from all models
        predictions_log = self.predict(X_test)
        
        # Prepare results
        results = []
        
        for model_name, y_pred_log in predictions_log.items():
            # Metrics in log scale
            metrics_log = {
                'model': model_name,
                'rmse_log': np.sqrt(mean_squared_error(y_test_log, y_pred_log)),
                'mae_log': mean_absolute_error(y_test_log, y_pred_log),
                'r2_log': r2_score(y_test_log, y_pred_log)
            }
            
            # Convert predictions back to euros if original values provided
            if y_test_original is not None:
                y_pred_euros = np.expm1(y_pred_log)
                
                metrics_euros = {
                    'rmse_euros': np.sqrt(mean_squared_error(y_test_original, y_pred_euros)),
                    'mae_euros': mean_absolute_error(y_test_original, y_pred_euros),
                    'r2_euros': r2_score(y_test_original, y_pred_euros)
                }
                
                # Combine metrics
                metrics = {**metrics_log, **metrics_euros}
            else:
                metrics = metrics_log
            
            results.append(metrics)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        logger.info("Evaluation completed")
        return results_df
    
    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to get feature importance from
            
        Returns:
        --------
        importance_df : pandas.DataFrame
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return None
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = 'rmse_euros'):
        """
        Plot comparison of model performance.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            Results from evaluate() method
        metric : str
            Metric to plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort models by metric
        results_sorted = results_df.sort_values(metric)
        
        # Create bar plot
        bars = ax.barh(results_sorted['model'], results_sorted[metric])
        
        # Customize plot
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:,.2f}' if 'euros' in metric else f'{width:.3f}',
                   ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def save_models(self, output_dir: str = "results/models"):
        """
        Save trained models to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            # Create filename
            filename = f"{output_dir}/{name.lower().replace(' ', '_')}.joblib"
            
            # Save model
            joblib.dump(model, filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.joblib")
        logger.info(f"Saved scaler to {output_dir}/scaler.joblib")
        
        logger.info(f"All models saved to {output_dir}")

# Keep BaselineModel class for backward compatibility
class BaselineModel:
    """
    Baseline model using only Age and Position features.
    (For Phase 3 compatibility)
    """
    
    def __init__(self, random_state=42):
        self.model = LinearRegression()
        self.random_state = random_state
        self.is_fitted = False
    
    def fit(self, X, y):
        # Prepare target (log transformation)
        y_log = np.log1p(y)
        self.model.fit(X, y_log)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return metrics, pd.DataFrame({'true_value': y_test, 'predicted_value': y_pred})

if __name__ == "__main__":
    # Test the model pipeline
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data_loader import load_raw_data, clean_data
    from features import prepare_full_data
    
    print("Testing model pipeline with 4 models...")
    
    # Load and prepare data
    df = load_raw_data()
    df = clean_data(df)
    
    # Split into train/test
    train_df = df[df['Season'].isin(['2018-2019', '2019-2020', '2020-2021', '2021-2022'])]
    test_df = df[df['Season'] == '2022-2023']
    
    # Create comprehensive features
    X_train, y_train_log, X_test, y_test_log, fe = prepare_full_data(train_df, test_df)
    
    # Get original target values for evaluation
    y_test_original = test_df['Value'].copy()
    
    # Initialize and train model pipeline
    pipeline = ModelPipeline(random_state=42)
    training_results = pipeline.train(X_train, y_train_log)
    
    # Evaluate on test set
    results_df = pipeline.evaluate(X_test, y_test_log, y_test_original)
    
    print(f"\n✅ Model pipeline test completed!")
    print(f"\nModel Performance (Test Set):")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Show feature importance for tree models
    print(f"\nTop 10 Features - Random Forest:")
    rf_importance = pipeline.get_feature_importance('Random Forest')
    if rf_importance is not None:
        print(rf_importance.head(10).to_string(index=False))
    
    print(f"\nTop 10 Features - XGBoost:")
    xgb_importance = pipeline.get_feature_importance('XGBoost')
    if xgb_importance is not None:
        print(xgb_importance.head(10).to_string(index=False))
