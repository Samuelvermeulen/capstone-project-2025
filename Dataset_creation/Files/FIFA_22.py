
import pandas as pd
import os

# Chemin vers le fichier FIFA22 sur le Bureau
file_path = os.path.join(os.path.expanduser("~"), "Desktop", "Dataset_creation", "FIFA_dataset", "FIFA22_official_data.csv")

# Ouvrir le fichier FIFA22
df = pd.read_csv(file_path)

print("🔍 Ouverture du fichier FIFA22 depuis le Bureau...")
print(f"Dimensions initiales : {df.shape}")

# Identifier la colonne des clubs
club_column = None
for col in df.columns:
    if 'club' in col.lower():
        club_column = col
        break

if not club_column:
    print("❌ Colonne des clubs non trouvée automatiquement")
    # Afficher les colonnes pour choisir manuellement
    print("Colonnes disponibles :")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    club_col_index = int(input("\nEntrez le numéro de la colonne des clubs : "))
    club_column = df.columns[club_col_index]

print(f"🏆 Colonne des clubs utilisée : '{club_column}'")

# Liste des clubs Premier League 2021-2022 (saison FIFA22)
premier_league_clubs = [
    'Arsenal', 'Manchester City', 'Manchester United', 'Newcastle United',
    'Liverpool', 'Tottenham Hotspur', 'Brighton & Hove Albion', 'Aston Villa',
    'Crystal Palace', 'Fulham', 'Chelsea', 'Wolverhampton Wanderers',
    'West Ham United', 'Brentford', 'Norwich City', 'Everton',
    'Watford', 'Burnley', 'Leeds United', 'Southampton',
    'Leicester City',
    'Spurs', 'Man City', 'Man United', 'Newcastle', 'West Ham',
    'Brighton', 'Wolves', 'Norwich', 'Leeds'
]

# Filtrer les joueurs de Premier League
premier_league_df = df[df[club_column].isin(premier_league_clubs)]

print(f"\n📊 Résultats du filtrage :")
print(f"Joueurs totaux dans FIFA22 : {len(df)}")
print(f"Joueurs Premier League 2021-2022 : {len(premier_league_df)}")

# Afficher les clubs trouvés
clubs_found = premier_league_df[club_column].unique()
print(f"\n🏆 Clubs Premier League trouvés ({len(clubs_found)}) :")
for club in sorted(clubs_found):
    count = len(premier_league_df[premier_league_df[club_column] == club])
    print(f"  {club} : {count} joueurs")

# SUPPRIMER LES COLONNES SPÉCIFIÉES (liste complète)
colonnes_a_supprimer = [
    # Première liste
    'Photo', 'Flag', 'Overall', 'Club Logo', 'Wage', 'Special', 
    'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 
    'Body Type', 'Real Face', 'Position', 'Joined', 'Loaned From', 
    'Contract Valid Until', 'Release Clause', 'Kit Number', 'Best Overall Rating',
    
    # Nouvelles colonnes statistiques
    'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
    'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
    'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
    'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
    'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
    'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
    'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes',
    'Best Position', 'DefensiveAwareness',
    
    # Nouvelles colonnes ajoutées
    'Potential', 'Jersey Number', 'ID'
]

# Nettoyer les noms de colonnes (supprimer les espaces, vérifier la casse)
colonnes_disponibles = premier_league_df.columns.tolist()
colonnes_trouvees = []

for col in colonnes_a_supprimer:
    # Chercher la colonne avec différentes variantes
    col_variants = [
        col, 
        col.lower(), 
        col.upper(), 
        col.replace(' ', ''), 
        col.replace(' ', '_'),
        col.replace(' ', '')
    ]
    
    for variant in col_variants:
        if variant in colonnes_disponibles:
            colonnes_trouvees.append(variant)
            break

print(f"\n🗑️  Suppression des colonnes :")
print(f"Colonnes à supprimer trouvées : {len(colonnes_trouvees)}/{len(colonnes_a_supprimer)}")

colonnes_supprimees = []
for col in colonnes_trouvees:
    if col in premier_league_df.columns:
        premier_league_df = premier_league_df.drop(columns=[col])
        colonnes_supprimees.append(col)
        print(f"  ✅ {col} supprimée")

# Afficher un résumé des suppressions par catégorie
print(f"\n📋 RÉSUMÉ DES SUPPRESSIONS :")
print(f"Colonnes d'images/visuelles : {len([c for c in colonnes_supprimees if any(x in c.lower() for x in ['photo', 'flag', 'logo', 'face'])])}")
print(f"Colonnes de contrats/économiques : {len([c for c in colonnes_supprimees if any(x in c.lower() for x in ['wage', 'contract', 'release', 'joined'])])}")
print(f"Colonnes de statistiques techniques : {len([c for c in colonnes_supprimees if any(x in c.lower() for x in ['crossing', 'finishing', 'dribbling', 'passing', 'shooting', 'tackle', 'gk'])])}")
print(f"Colonnes d'attributs physiques : {len([c for c in colonnes_supprimees if any(x in c.lower() for x in ['acceleration', 'sprint', 'agility', 'stamina', 'strength', 'jumping'])])}")
print(f"Colonnes d'identifiants/potentiel : {len([c for c in colonnes_supprimees if any(x in c.lower() for x in ['id', 'potential', 'jersey', 'kit'])])}")

print(f"\n📐 Dimensions après suppression : {premier_league_df.shape}")

# Afficher les colonnes restantes
print(f"\n📋 Colonnes restantes ({len(premier_league_df.columns)}) :")
for col in premier_league_df.columns:
    print(f"  {col}")

# Aperçu des données finales
print(f"\n👀 Aperçu des données finales :")
print(premier_league_df.head())

# Sauvegarder sur le Bureau
output_path = os.path.join(os.path.expanduser("~"), "Desktop", "premier_league_cleaned_2022.csv")
premier_league_df.to_csv(output_path, index=False)
print(f"\n✅ Fichier sauvegardé : {output_path}")
print(f"📊 Données finales : {premier_league_df.shape[0]} joueurs, {premier_league_df.shape[1]} colonnes")