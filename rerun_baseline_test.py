#!/usr/bin/env python3
"""
Ré-exécute un test rapide du baseline pour vérifier son fonctionnement.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("🔄 TEST RAPIDE DU BASELINE MODEL")
print("=" * 60)

# Charger les données
print("1. Chargement des données...")
train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

print(f"   • Train: {train_df.shape[0]} joueurs")
print(f"   • Test: {test_df.shape[0]} joueurs")

# Créer features baseline manuellement (au cas où le module est corrompu)
print("\n2. Création des features baseline...")

def create_baseline_features_manual(df):
    """Version manuelle des features baseline"""
    df_processed = df.copy()
    
    # Imputer Age si manquant
    if df_processed['Age'].isnull().any():
        df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
    
    # One-hot encode Position
    position_dummies = pd.get_dummies(df_processed['Position'], prefix='is')
    
    # Renommer pour clarté
    rename_map = {
        'is_DF': 'is_Defender',
        'is_MF': 'is_Midfielder', 
        'is_FW': 'is_Forward',
        'is_GK': 'is_Goalkeeper'
    }
    position_dummies = position_dummies.rename(columns=rename_map)
    
    # S'assurer que toutes les positions existent
    expected = ['is_Defender', 'is_Midfielder', 'is_Forward', 'is_Goalkeeper']
    for pos in expected:
        if pos not in position_dummies.columns:
            position_dummies[pos] = 0
    
    # Combiner Age et Position
    X = pd.concat([df_processed[['Age']], position_dummies], axis=1)
    y = df_processed['Value'].copy()
    
    return X, y

# Créer features
X_train, y_train = create_baseline_features_manual(train_df)
X_test, y_test = create_baseline_features_manual(test_df)

# S'assurer que test a les mêmes colonnes que train
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

print(f"   • Features créées: {list(X_train.columns)}")
print(f"   • Shape X_train: {X_train.shape}")
print(f"   • Shape X_test: {X_test.shape}")

# Entraîner modèle
print("\n3. Entraînement du modèle baseline...")
model = LinearRegression()

# Transformation log de la target
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

# Prédictions
print("4. Prédictions sur le test set...")
y_test_log = np.log1p(y_test)
y_pred_log = model.predict(X_test)

# Convertir en euros
y_pred = np.expm1(y_pred_log)

# Calcul des métriques
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n5. Résultats du test rapide:")
print(f"   • RMSE: €{rmse:,.0f}")
print(f"   • MAE: €{mae:,.0f}")
print(f"   • R²: {r2:.3f}")

# Comparer avec les résultats précédents si disponibles
if os.path.exists("results/baseline_predictions.csv"):
    old_preds = pd.read_csv("results/baseline_predictions.csv")
    old_mae = old_preds['absolute_error'].mean()
    print(f"\n6. Comparaison avec les résultats précédents:")
    print(f"   • MAE actuel: €{mae:,.0f}")
    print(f"   • MAE précédent: €{old_mae:,.0f}")
    diff_pct = ((mae - old_mae) / old_mae) * 100
    print(f"   • Différence: {diff_pct:+.1f}%")
    
    if abs(diff_pct) < 5:
        print(f"   ✅ Les résultats sont cohérents (différence < 5%)")
    else:
        print(f"   ⚠️  Attention: différence significative (> 5%)")

print("\n" + "=" * 60)
print("✅ TEST DU BASELINE COMPLETÉ AVEC SUCCÈS")
print("=" * 60)
