
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz, process
import warnings
import os
warnings.filterwarnings('ignore')

# Charger les datasets
print("📂 Chargement des datasets...")
pl_players = pd.read_csv('../Merged_Dataset/PL_players.csv')
merged_data = pd.read_csv('../Merged_Dataset/premier_league_merged_2018-2023.csv')

print(f"PL_players: {pl_players.shape}")
print(f"Merged data: {merged_data.shape}")

# Convertir les âges en numérique
print("🔧 Conversion des types de données...")
pl_players['Age'] = pd.to_numeric(pl_players['Age'], errors='coerce')
merged_data['Age'] = pd.to_numeric(merged_data['Age'], errors='coerce')

# Normalisation des noms de clubs
club_mapping = {
    'man utd': 'manchester united', 'man united': 'manchester united', 'manchester utd': 'manchester united',
    'man city': 'manchester city', 
    'spurs': 'tottenham hotspur', 'tottenham': 'tottenham hotspur',
    'wolves': 'wolverhampton wanderers', 'wolverhampton': 'wolverhampton wanderers',
    'newcastle': 'newcastle united',
    'leicester': 'leicester city',
    'brighton': 'brighton & hove albion',
    'west ham': 'west ham united',
    'norwich': 'norwich city',
    'southampton': 'southampton',
    'crystal palace': 'crystal palace',
    'everton': 'everton',
    'chelsea': 'chelsea',
    'liverpool': 'liverpool',
    'arsenal': 'arsenal',
    'aston villa': 'aston villa',
    'leeds': 'leeds united',
    'watford': 'watford',
    'burnley': 'burnley',
    'brentford': 'brentford',
    'fulham': 'fulham',
    'bournemouth': 'bournemouth',
    'sheffield united': 'sheffield united',
    'west brom': 'west bromwich albion',
    'sheffield wed': 'sheffield wednesday'
}

# Normalisation des nationalités
nation_mapping = {
    'english': 'england', 'england': 'england',
    'french': 'france', 'france': 'france',
    'spanish': 'spain', 'spain': 'spain',
    'german': 'germany', 'germany': 'germany',
    'italian': 'italy', 'italy': 'italy',
    'portuguese': 'portugal', 'portugal': 'portugal',
    'brazilian': 'brazil', 'brazil': 'brazil',
    'dutch': 'netherlands', 'netherlands': 'netherlands',
    'belgian': 'belgium', 'belgium': 'belgium'
}

def normalize_text(text):
    """Normaliser le texte pour le matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    return text

def normalize_club(club):
    """Normaliser les noms de clubs"""
    if pd.isna(club):
        return ""
    club_norm = normalize_text(club)
    return club_mapping.get(club_norm, club_norm)

def normalize_nation(nation):
    """Normaliser les nationalités"""
    if pd.isna(nation):
        return ""
    nation_norm = normalize_text(nation)
    return nation_mapping.get(nation_norm, nation_norm)

def normalize_player_name(name):
    """Normaliser les noms des joueurs"""
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Supprimer les numéros au début
    name = re.sub(r'^\d+\s*', '', name)
    
    # Gérer les formats communs
    name = re.sub(r'[^\w\s]', ' ', name)  # Remplacer la ponctuation par des espaces
    name = re.sub(r'\s+', ' ', name)  # Supprimer les espaces multiples
    
    return name.strip()

def find_best_match(target, choices, threshold=80):
    """Trouver le meilleur match avec fuzzy matching"""
    if not target or not choices:
        return None, 0
    
    best_match, score = process.extractOne(target, choices, scorer=fuzz.token_sort_ratio)
    return best_match if score >= threshold else None, score

print("🔄 Normalisation des données...")

# Normaliser les colonnes pour le matching
merged_data['player_norm'] = merged_data['Player'].apply(normalize_player_name)
merged_data['club_norm'] = merged_data['Club'].apply(normalize_club)
merged_data['nation_norm'] = merged_data['Nationality'].apply(normalize_nation)

pl_players['player_norm'] = pl_players['Player'].apply(normalize_player_name)
pl_players['club_norm'] = pl_players['Club'].apply(normalize_club)
pl_players['nation_norm'] = pl_players['Nation'].apply(normalize_nation)

# Colonnes à ajouter
cols_to_add = ['Value', 'Preferred Foot', 'Height', 'Weight']

# Initialiser les colonnes dans PL_players
for col in cols_to_add:
    pl_players[col] = np.nan

print("🔍 Début du matching des joueurs...")

# Stratégies de matching par ordre de priorité
matches_found = 0
total_rows = len(pl_players)

for idx, row in pl_players.iterrows():
    if idx % 500 == 0:
        print(f"⏳ Progression: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
    
    player_norm = row['player_norm']
    season = row['Season']
    club_norm = row['club_norm']
    nation_norm = row['nation_norm']
    age = row['Age']
    
    # Si saison 2023-2024, on skip (colonnes restent vides)
    if season == '2023-2024':
        continue
    
    # Filtrer le dataset merged par saison
    merged_subset = merged_data[merged_data['Season'] == season]
    
    if merged_subset.empty:
        continue
    
    match_found = False
    match_row = None
    
    # Stratégie 1: Match parfait (Player + Club + Nation)
    if club_norm and nation_norm:
        perfect_match = merged_subset[
            (merged_subset['player_norm'] == player_norm) &
            (merged_subset['club_norm'] == club_norm) &
            (merged_subset['nation_norm'] == nation_norm)
        ]
        if not perfect_match.empty:
            match_row = perfect_match.iloc[0]
            match_found = True
    
    # Stratégie 2: Match avec Club seulement
    if not match_found and club_norm:
        club_match = merged_subset[
            (merged_subset['player_norm'] == player_norm) &
            (merged_subset['club_norm'] == club_norm)
        ]
        if not club_match.empty:
            match_row = club_match.iloc[0]
            match_found = True
    
    # Stratégie 3: Match avec Nation seulement
    if not match_found and nation_norm:
        nation_match = merged_subset[
            (merged_subset['player_norm'] == player_norm) &
            (merged_subset['nation_norm'] == nation_norm)
        ]
        if not nation_match.empty:
            match_row = nation_match.iloc[0]
            match_found = True
    
    # Stratégie 4: Match avec Player seulement + vérification âge
    if not match_found:
        player_match = merged_subset[merged_subset['player_norm'] == player_norm]
        if not player_match.empty:
            # Prendre le premier match (on suppose peu de doublons)
            candidate = player_match.iloc[0]
            # Vérifier l'âge (±2 ans de tolérance) - gérer les NaN
            if pd.notna(age) and pd.notna(candidate['Age']):
                if abs(candidate['Age'] - age) <= 2:
                    match_row = candidate
                    match_found = True
            else:
                # Si âge manquant dans un des deux, on accepte quand même
                match_row = candidate
                match_found = True
    
    # Stratégie 5: Fuzzy matching en dernier recours
    if not match_found:
        player_names = merged_subset['player_norm'].tolist()
        best_match, score = find_best_match(player_norm, player_names, threshold=85)
        if best_match:
            fuzzy_match = merged_subset[merged_subset['player_norm'] == best_match]
            if not fuzzy_match.empty:
                candidate = fuzzy_match.iloc[0]
                # Vérification additionnelle avec l'âge
                if pd.notna(age) and pd.notna(candidate['Age']):
                    if abs(candidate['Age'] - age) <= 2:
                        match_row = candidate
                        match_found = True
                else:
                    # Si âge manquant, on accepte quand même
                    match_row = candidate
                    match_found = True
    
    # Remplir les données si match trouvé
    if match_found and match_row is not None:
        matches_found += 1
        pl_players.at[idx, 'Value'] = match_row['Value']
        pl_players.at[idx, 'Preferred Foot'] = match_row['Preferred Foot']
        pl_players.at[idx, 'Height'] = match_row['Height']
        pl_players.at[idx, 'Weight'] = match_row['Weight']

print(f"\n✅ Matching terminé !")
print(f"📊 {matches_found}/{total_rows} lignes matchées ({matches_found/total_rows*100:.1f}%)")

# Statistiques par saison
print(f"\n📈 Statistiques par saison:")
for season in pl_players['Season'].unique():
    season_data = pl_players[pl_players['Season'] == season]
    total_season = len(season_data)
    matched_season = season_data[cols_to_add[0]].notna().sum()
    print(f"  {season}: {matched_season}/{total_season} ({matched_season/total_season*100:.1f}%)")

# Réorganiser les colonnes pour avoir les nouvelles colonnes après les existantes
existing_cols = [col for col in pl_players.columns if col not in cols_to_add + ['player_norm', 'club_norm', 'nation_norm']]
new_cols_order = existing_cols + cols_to_add
pl_players_final = pl_players[new_cols_order]

# 🔥 NOUVELLE PARTIE : FILTRATION DES LIGNES
print(f"\n🗑️ Filtration des lignes sans données...")

# Séparer la saison 2023-2024 (à garder entièrement)
season_2023_2024 = pl_players_final[pl_players_final['Season'] == '2023-2024']

# Pour les autres saisons (2018-2019 à 2022-2023), garder seulement les lignes avec AU MOINS UNE donnée dans les 4 colonnes
other_seasons = pl_players_final[pl_players_final['Season'] != '2023-2024']

# Compter avant filtration
before_filter = len(other_seasons)

# Garder seulement les lignes qui ont AU MOINS UNE des 4 colonnes remplies
other_seasons_filtered = other_seasons[
    other_seasons['Value'].notna() | 
    other_seasons['Preferred Foot'].notna() | 
    other_seasons['Height'].notna() | 
    other_seasons['Weight'].notna()
]

after_filter = len(other_seasons_filtered)

print(f"📊 Filtration - Saisons 2018-2023:")
print(f"  Avant filtration: {before_filter} lignes")
print(f"  Après filtration: {after_filter} lignes")
print(f"  Lignes supprimées: {before_filter - after_filter}")

# Recréer le dataset final avec la saison 2023-2024 + les autres saisons filtrées
pl_players_final_filtered = pd.concat([other_seasons_filtered, season_2023_2024], ignore_index=True)

print(f"\n📦 Dataset final:")
print(f"  Saison 2023-2024: {len(season_2023_2024)} lignes (toutes conservées)")
print(f"  Autres saisons: {after_filter} lignes (avec données)")
print(f"  TOTAL: {len(pl_players_final_filtered)} lignes")

# Sauvegarder le résultat
output_file = os.path.join(os.path.expanduser("~"), "Desktop", "Dataset_creation", "Merged_Dataset", "PL_players_with_new_columns.csv")
pl_players_final_filtered.to_csv(output_file, index=False)

print(f"\n💾 Fichier sauvegardé: {output_file}")
print(f"📋 Colonnes finales: {list(pl_players_final_filtered.columns)}")

# Aperçu des données
print(f"\n👀 Aperçu des données fusionnées et filtrées:")
print(pl_players_final_filtered.head(10)[['Player', 'Season', 'Club', 'Value', 'Preferred Foot', 'Height', 'Weight']])

