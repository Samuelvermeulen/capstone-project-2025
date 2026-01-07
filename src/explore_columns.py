import pandas as pd
import numpy as np

print("Exploration détaillée des colonnes du dataset")
print("=" * 60)

# Charger les données
df = pd.read_csv("data/raw/PL_players_with_new_columns.csv")

print(f"\n📊 INFORMATIONS GÉNÉRALES :")
print(f"• Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"• Période couverte : {df['Season'].min()} à {df['Season'].max()}")
print(f"• Nombre de saisons : {df['Season'].nunique()}")

print(f"\n📋 LISTE COMPLÈTE DES COLONNES :")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col:20} | Type: {df[col].dtype}")

print(f"\n🔍 ANALYSE PAR COLONNE :")
print("-" * 60)

# Analyser chaque colonne
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Valeurs uniques: {df[col].nunique()}")
    
    if df[col].dtype in ['int64', 'float64']:
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Moyenne: {df[col].mean():.2f}")
        print(f"  Valeurs manquantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
    else:
        # Pour les colonnes catégorielles
        sample_values = df[col].dropna().unique()[:5]
        print(f"  Exemples: {sample_values[:3]}")
        print(f"  Valeurs manquantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")

print(f"\n🎯 VARIABLE CIBLE (Value) - ANALYSE DÉTAILLÉE :")
if 'Value' in df.columns:
    print(f"• Type: {df['Value'].dtype}")
    print(f"• Plage: €{df['Value'].min():,.0f} à €{df['Value'].max():,.0f}")
    print(f"• Moyenne: €{df['Value'].mean():,.0f}")
    print(f"• Médiane: €{df['Value'].median():,.0f}")
    print(f"• Écart-type: €{df['Value'].std():,.0f}")
    
    # Distribution par déciles
    print(f"\n• Distribution par déciles :")
    deciles = df['Value'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for q, val in deciles.items():
        print(f"  {int(q*100)}% : €{val:,.0f}")

print(f"\n✅ EXPLORATION TERMINÉE")
