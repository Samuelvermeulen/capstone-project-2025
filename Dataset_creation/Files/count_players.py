

import pandas as pd

# Charger le dataset
print("📂 Chargement du dataset...")
try:
    df = pd.read_csv('PL_players_with_new_columns.csv')
    print("✅ Dataset chargé avec succès")
except FileNotFoundError:
    print("❌ Fichier 'PL_players_with_new_columns.csv' non trouvé")
    print("📋 Liste des fichiers CSV disponibles:")
    import os
    files = os.listdir()
    csv_files = [f for f in files if f.endswith('.csv')]
    for i, f in enumerate(csv_files):
        print(f"  {i+1}. {f}")
    
    if csv_files:
        choice = input("Entrez le numéro du fichier à analyser: ")
        try:
            df = pd.read_csv(csv_files[int(choice)-1])
            print(f"✅ {csv_files[int(choice)-1]} chargé")
        except:
            print("❌ Choix invalide")
            exit()
    else:
        print("❌ Aucun fichier CSV trouvé")
        exit()

# Compter le nombre de joueurs par saison
print("\n📊 NOMBRE DE JOUEURS PAR SAISON:")
print("=" * 40)

players_per_season = df['Season'].value_counts().sort_index()

for season, count in players_per_season.items():
    print(f"🔹 {season}: {count} joueurs")

print("=" * 40)
print(f"🎯 TOTAL: {len(df)} joueurs")

# Statistiques supplémentaires
print(f"\n📈 STATISTIQUES DÉTAILLÉES:")
print("=" * 40)

# Pourcentage de données remplies par saison pour les nouvelles colonnes
cols_to_check = ['Value', 'Preferred Foot', 'Height', 'Weight']

for season in df['Season'].unique():
    season_data = df[df['Season'] == season]
    total_players = len(season_data)
    
    print(f"\n🔹 Saison {season} ({total_players} joueurs):")
    
    for col in cols_to_check:
        filled_count = season_data[col].notna().sum()
        percentage = (filled_count / total_players) * 100
        print(f"   • {col}: {filled_count}/{total_players} ({percentage:.1f}%)")

# Résumé global
print(f"\n🎯 RÉSUMÉ GLOBAL:")
print("=" * 40)
print(f"Nombre total de saisons: {len(players_per_season)}")
print(f"Nombre total de joueurs: {len(df)}")
print(f"Saison avec le plus de joueurs: {players_per_season.idxmax()} ({players_per_season.max()} joueurs)")
print(f"Saison avec le moins de joueurs: {players_per_season.idxmin()} ({players_per_season.min()} joueurs)")

# Aperçu des données
print(f"\n👀 APERÇU DES DONNÉES:")
print("=" * 40)
print(df[['Player', 'Season', 'Club', 'Value', 'Preferred Foot', 'Height', 'Weight']].head(10))

