import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("data/raw/PL_players_with_new_columns.csv")

print("🔍 ANALYSE DES PATTERNS DE VALEURS MANQUANTES")
print("=" * 60)

# 1. Identifier les lignes avec Value manquant
missing_value_mask = df['Value'].isnull()
print(f"1. Lignes avec Value manquant: {missing_value_mask.sum()}")

# 2. Vérifier si les autres NA sont sur les mêmes lignes
print("\n2. Co-occurrence des NA:")
print(f"   • Mêmes lignes que Height manquant: {(missing_value_mask & df['Height'].isnull()).sum()}")
print(f"   • Mêmes lignes que Weight manquant: {(missing_value_mask & df['Weight'].isnull()).sum()}")
print(f"   • Mêmes lignes que Preferred Foot manquant: {(missing_value_mask & df['Preferred Foot'].isnull()).sum()}")

# 3. Analyser par saison
print("\n3. Distribution par saison des Value manquants:")
if 'Season' in df.columns:
    missing_by_season = df[missing_value_mask]['Season'].value_counts().sort_index()
    for season, count in missing_by_season.items():
        total_in_season = (df['Season'] == season).sum()
        print(f"   • {season}: {count} manquants ({count/total_in_season*100:.1f}%)")

# 4. Analyser par position
print("\n4. Distribution par position des Value manquants:")
if 'Position' in df.columns:
    missing_by_position = df[missing_value_mask]['Position'].value_counts()
    for position, count in missing_by_position.items():
        total_in_position = (df['Position'] == position).sum()
        print(f"   • {position}: {count} manquants ({count/total_in_position*100:.1f}%)")

# 5. Vérifier les statistiques des lignes avec Value manquant
print("\n5. Caractéristiques des joueurs sans Value:")
if missing_value_mask.any():
    missing_df = df[missing_value_mask]
    print(f"   • Moyenne d'âge: {missing_df['Age'].mean():.1f}")
    print(f"   • Moyenne de buts: {missing_df['Goals'].mean():.1f}")
    print(f"   • Nombre de clubs différents: {missing_df['Club'].nunique()}")
