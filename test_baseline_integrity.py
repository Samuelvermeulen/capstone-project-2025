#!/usr/bin/env python3
"""
Test d'intégrité de l'étape 3 (Baseline Model)
"""

import sys
import os
import pandas as pd
import numpy as np

print("🧪 TEST D'INTÉGRITÉ - ÉTAPE 3 (BASELINE)")
print("=" * 60)

# Test 1: Vérifier les données
print("1. Vérification des données...")
try:
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")
    print(f"   ✅ Données chargées")
    print(f"      • Train: {train_df.shape}")
    print(f"      • Test: {test_df.shape}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 2: Vérifier les modules
print("\n2. Vérification des modules Python...")
modules_to_check = ['features', 'models']

for module in modules_to_check:
    module_path = f"src/{module}.py"
    if os.path.exists(module_path):
        with open(module_path, 'r') as f:
            content = f.read()
            if 'def create_baseline_features' in content or 'class BaselineModel' in content:
                print(f"   ✅ {module}.py contient les fonctions baseline")
            else:
                print(f"   ⚠️  {module}.py modifié - fonctions baseline manquantes")
    else:
        print(f"   ❌ {module}.py non trouvé")

# Test 3: Vérifier les résultats
print("\n3. Vérification des résultats baseline...")
results_files = ['baseline_predictions.csv', 'baseline_feature_importance.png']

all_files_exist = True
for file in results_files:
    file_path = f"results/{file}"
    if os.path.exists(file_path):
        print(f"   ✅ {file} présent")
    else:
        print(f"   ❌ {file} manquant")
        all_files_exist = False

# Test 4: Tester une exécution simple du baseline
print("\n4. Test d'exécution simple...")
try:
    # Importer les fonctions baseline si elles existent
    sys.path.insert(0, 'src')
    
    # Essayer d'importer
    try:
        from features import create_baseline_features
        print("   ✅ Fonction create_baseline_features importable")
    except ImportError as e:
        print(f"   ❌ Erreur import features: {e}")
    
    try:
        from models import BaselineModel
        print("   ✅ Classe BaselineModel importable")
    except ImportError as e:
        print(f"   ❌ Erreur import models: {e}")
    
except Exception as e:
    print(f"   ⚠️  Erreur d'exécution: {e}")

# Test 5: Afficher les métriques du baseline si disponibles
print("\n5. Métriques du baseline (si disponibles)...")
if os.path.exists("results/baseline_predictions.csv"):
    try:
        preds = pd.read_csv("results/baseline_predictions.csv")
        mae = preds['absolute_error'].mean()
        median_mae = preds['absolute_error'].median()
        mape = preds['percentage_error'].mean()
        
        print(f"   • MAE: €{mae:,.0f}")
        print(f"   • MAE médiane: €{median_mae:,.0f}")
        print(f"   • MAPE: {mape:.1f}%")
        print(f"   • Nombre de prédictions: {len(preds)}")
    except Exception as e:
        print(f"   ⚠️  Impossible de lire les prédictions: {e}")
else:
    print("   ⚠️  Fichier de prédictions non trouvé")

print("\n" + "=" * 60)
print("📊 RÉSUMÉ DE L'INTÉGRITÉ DE L'ÉTAPE 3:")
if all_files_exist:
    print("✅ L'étape 3 semble intacte et fonctionnelle")
else:
    print("⚠️  Certains fichiers baseline sont manquants ou modifiés")
print("=" * 60)
