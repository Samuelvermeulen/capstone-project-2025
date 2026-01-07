"""
Data Analysis Example for Capstone Project.
Demonstrates Python fundamentals, data structures, functions, and basic ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. FUNCTIONS (Demonstration des fonctions et récursion)
# ============================================================================

def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number using recursion.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    
    Args:
        n: Number to check
        
    Returns:
        True if prime, False otherwise
    """
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe: handle missing values, remove duplicates.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df


# ============================================================================
# 2. DATA STRUCTURES (Lists, Dictionaries, Tuples)
# ============================================================================

def demonstrate_data_structures():
    """Demonstrate Python data structures."""
    
    # List comprehension
    squares = [x**2 for x in range(10)]
    even_squares = [x for x in squares if x % 2 == 0]
    
    # Dictionary
    student_grades = {
        'Alice': 85,
        'Bob': 92,
        'Charlie': 78,
        'Diana': 95
    }
    
    # Tuple unpacking
    coordinates = [(1, 2), (3, 4), (5, 6)]
    for x, y in coordinates:
        pass  # Just demonstrating unpacking
    
    return squares, even_squares, student_grades


# ============================================================================
# 3. CLASS (Object-Oriented Programming)
# ============================================================================

class DataAnalyzer:
    """Class for data analysis operations."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data.
        
        Args:
            data: DataFrame to analyze
        """
        self.data = data
        self.results = {}
    
    def calculate_statistics(self):
        """Calculate basic statistics for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        self.results['mean'] = numeric_data.mean()
        self.results['std'] = numeric_data.std()
        self.results['min'] = numeric_data.min()
        self.results['max'] = numeric_data.max()
        
        return self.results
    
    def plot_distributions(self, save_path: str = None):
        """Plot distribution of numeric features."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        n_cols = min(3, len(numeric_data.columns))
        
        fig, axes = plt.subplots(1, n_cols, figsize=(15, 4))
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_data.columns[:n_cols]):
            axes[idx].hist(numeric_data[col], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig


# ============================================================================
# 4. MAIN ANALYSIS PIPELINE
# ============================================================================

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic data for analysis.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    # Generate classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some missing values and noise for realism
    mask = np.random.random(df.shape) > 0.95
    df = df.mask(mask)
    
    # Add a categorical column
    categories = ['A', 'B', 'C', 'D']
    df['category'] = np.random.choice(categories, size=len(df))
    
    return df


def train_ml_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a machine learning model.
    
    Returns:
        Trained model and evaluation metrics
    """
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, accuracy, report, feature_importance


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("CAPSTONE PROJECT - DATA SCIENCE & ADVANCED PROGRAMMING")
    print("=" * 70)
    
    # Part 1: Demonstrate functions
    print("\n1. FUNCTION DEMONSTRATION")
    print("-" * 40)
    
    # Fibonacci example
    fib_numbers = [fibonacci(i) for i in range(10)]
    print(f"First 10 Fibonacci numbers: {fib_numbers}")
    
    # Prime numbers example
    primes = [i for i in range(2, 20) if is_prime(i)]
    print(f"Prime numbers under 20: {primes}")
    
    # Part 2: Data structures
    print("\n2. DATA STRUCTURES DEMONSTRATION")
    print("-" * 40)
    squares, even_squares, student_grades = demonstrate_data_structures()
    print(f"Squares: {squares[:5]}...")
    print(f"Even squares: {even_squares[:5]}...")
    print(f"Student grades: {student_grades}")
    
    # Part 3: Generate and analyze data
    print("\n3. DATA GENERATION AND ANALYSIS")
    print("-" * 40)
    
    # Generate sample data
    print("Generating synthetic dataset...")
    data = generate_sample_data(500)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Clean data
    print("\nCleaning data...")
    data_clean = clean_dataframe(data)
    print(f"Missing values after cleaning:\n{data_clean.isnull().sum()}")
    
    # Analyze with class
    print("\nAnalyzing data with DataAnalyzer class...")
    analyzer = DataAnalyzer(data_clean.select_dtypes(include=[np.number]))
    stats = analyzer.calculate_statistics()
    
    print("\nBasic statistics:")
    for stat_name, values in stats.items():
        print(f"\n{stat_name.upper()}:")
        print(values.head())
    
    # Part 4: Machine Learning
    print("\n4. MACHINE LEARNING PIPELINE")
    print("-" * 40)
    
    # Prepare data for ML
    ml_data = data_clean.select_dtypes(include=[np.number])
    X = ml_data.drop('target', axis=1)
    y = ml_data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model, accuracy, report, feature_importance = train_ml_model(
        X_train, y_train, X_test, y_test
    )
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Part 5: Create visualization
    print("\n5. DATA VISUALIZATION")
    print("-" * 40)
    
    # Create and save plot
    plot_path = "results/feature_distributions.png"
    print(f"Creating visualization at {plot_path}...")
    
    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)
    
    # Generate plot
    analyzer.plot_distributions(save_path=plot_path)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved in:")
    print(f"  - results/feature_distributions.png")
    print(f"  - Model trained with {accuracy:.2%} accuracy")
    
    return {
        'data': data_clean,
        'model': model,
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'statistics': stats
    }


if __name__ == "__main__":
    # Execute main analysis
    results = main()
    
    # Demonstrate additional functionality
    print("\nAdditional demonstration: Working with results dictionary")
    print(f"Accuracy stored: {results['accuracy']:.4f}")
    print(f"Number of important features: {len(results['feature_importance'])}")
