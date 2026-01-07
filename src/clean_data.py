import pandas as pd

print("Nettoyage du dataset basé sur l'analyse des NA")
print("=" * 60)

# Charger les données
df = pd.read_csv("data/raw/PL_players_with_new_columns.csv")

print(f"Dataset initial: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Saisons présentes: {sorted(df['Season'].unique())}")

# Supprimer la saison 2023-2024
df_clean = df[df['Season'] != '2023-2024'].copy()

print(f"\nAprès suppression de 2023-2024:")
print(f"Lignes restantes: {df_clean.shape[0]}")
print(f"Lignes supprimées: {df.shape[0] - df_clean.shape[0]}")

# Vérifier qu'il n'y a plus de NA dans Value
remaining_na = df_clean['Value'].isnull().sum()
print(f"\nValeurs manquantes restantes dans Value: {remaining_na}")

# Vérifier la distribution par saison
print(f"\nDistribution par saison après nettoyage:")
season_counts = df_clean['Season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count} joueurs")

print("\n✅ Dataset prêt pour l'analyse temporelle (2018-2023)")

# Sauvegarder le dataset nettoyé (optionnel)
df_clean.to_csv("data/processed/PL_players_cleaned.csv", index=False)
print("Dataset nettoyé sauvegardé dans: data/processed/PL_players_cleaned.csv")
