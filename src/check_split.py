import pandas as pd

print("Vérification du split temporel selon le roadmap")
print("=" * 60)

# Charger les données nettoyées
df = pd.read_csv("data/raw/PL_players_with_new_columns.csv")
df = df[df['Season'] != '2023-2024'].copy()

print(f"Dataset nettoyé: {df.shape[0]} joueurs sur {df['Season'].nunique()} saisons")

# Split selon le roadmap
train_seasons = ['2018-2019', '2019-2020', '2020-2021', '2021-2022']
test_seasons = ['2022-2023']

train_df = df[df['Season'].isin(train_seasons)].copy()
test_df = df[df['Season'].isin(test_seasons)].copy()

print(f"\n1. ENSEMBLE D'ENTRAÎNEMENT (2018-2022):")
print(f"   • Saisons: {train_seasons}")
print(f"   • Nombre de joueurs: {train_df.shape[0]}")
print(f"   • Distribution par saison:")
for season in train_seasons:
    count = train_df[train_df['Season'] == season].shape[0]
    print(f"     - {season}: {count} joueurs")

print(f"\n2. ENSEMBLE DE TEST (2022-2023):")
print(f"   • Saison: {test_seasons[0]}")
print(f"   • Nombre de joueurs: {test_df.shape[0]}")

print(f"\n3. RATIO TRAIN/TEST:")
total = train_df.shape[0] + test_df.shape[0]
print(f"   • Train: {train_df.shape[0]}/{total} = {train_df.shape[0]/total*100:.1f}%")
print(f"   • Test: {test_df.shape[0]}/{total} = {test_df.shape[0]/total*100:.1f}%")

print(f"\n4. VÉRIFICATION DES COLONNES CLÉS:")
key_cols = ['Player', 'Season', 'Age', 'Position', 'Value', 'Goals', 'Assists']
for col in key_cols:
    if col in df.columns:
        missing_train = train_df[col].isnull().sum()
        missing_test = test_df[col].isnull().sum()
        print(f"   ✅ {col}: Train ({missing_train} NA), Test ({missing_test} NA)")
    else:
        print(f"   ❌ {col}: COLONNE MANQUANTE")

print(f"\n✅ Split temporel vérifié et prêt pour la modélisation!")
