
import pandas as pd
import numpy as np
import os
import unicodedata
import warnings
warnings.filterwarnings('ignore')

def normalize_name(name):
    """Normalise les noms pour le matching simple"""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Nettoyage basique
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Garder seulement lettres et espaces
    name = ''.join(c for c in name if c.isalpha() or c.isspace())
    return name

def load_and_prepare_data():
    """Charge et prépare tous les datasets"""
    
    # Chemins des dossiers
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    base_path = os.path.join(desktop_path, "Projet_ML")
    positional_path = os.path.join(desktop_path, "Projet python data")
    
    print(f"📂 Chemin base: {base_path}")
    print(f"📂 Chemin positionnel: {positional_path}")
    
    # Charger les datasets cleaned
    cleaned_files = [
        'premier_league_cleaned_2018-2019.csv',
        'premier_league_cleaned_2019-2020.csv', 
        'premier_league_cleaned_2020-2021.csv',
        'premier_league_cleaned_2021-2022.csv',
        'premier_league_cleaned_2022-2023.csv',
        'premier_league_cleaned_2023-2024.csv'
    ]
    
    print("Chargement des datasets cleaned...")
    cleaned_dfs = []
    
    for file in cleaned_files:
        try:
            file_path = os.path.join(base_path, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                season_name = file.split('_')[-1].split('.')[0]
                df['season'] = season_name
                cleaned_dfs.append(df)
                print(f"✅ {file} chargé - {len(df)} lignes")
            else:
                print(f"❌ Fichier non trouvé: {file_path}")
        except Exception as e:
            print(f"❌ Erreur avec {file}: {e}")
            continue
    
    if not cleaned_dfs:
        raise Exception("Aucun dataset cleaned n'a pu être chargé!")
    
    base_df = pd.concat(cleaned_dfs, ignore_index=True)
    print(f"📊 Dataset de base: {base_df.shape} lignes")
    
    # Charger les datasets positionnels
    positional_files = {
        'ST': ['PL_strikers_1.csv', 'PL_strikers_2.csv', 'PL_strikers_3.csv', 'PL_strikers_4.csv', 'PL_strikers_5.csv'],
        'DEF': ['PL_DEF_1.csv', 'PL_DEF_2.csv', 'PL_DEF_3.csv', 'PL_DEF_4.csv', 'PL_DEF_5.csv'],
        'MID': ['PL_MID_1.csv', 'PL_MID_2.csv', 'PL_MID_3.csv', 'PL_MID_4.csv', 'PL_MID_5.csv', 'PL_MID_6.csv', 'PL_MID_7.csv'],
        'GK': ['PL_GK_1.csv', 'PL_GK_2.csv']
    }
    
    print("\nChargement des datasets positionnels...")
    positional_dfs = []
    
    for position, files in positional_files.items():
        for file in files:
            try:
                file_path = os.path.join(positional_path, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['position_source'] = position
                    positional_dfs.append(df)
                    print(f"✅ {file} chargé - {len(df)} lignes")
                else:
                    print(f"⚠️ Fichier non trouvé: {file_path}")
            except Exception as e:
                print(f"❌ Erreur avec {file}: {e}")
                continue
    
    if not positional_dfs:
        raise Exception("Aucun dataset positionnel n'a pu être chargé!")
    
    positional_df = pd.concat(positional_dfs, ignore_index=True)
    print(f"📊 Dataset positionnel: {positional_df.shape} lignes")
    
    return base_df, positional_df

def smart_merge_3_4_easy_matches(base_df, positional_df):
    """Merge les 3/4 des matches faciles d'abord"""
    
    print("\n🎯 PHASE 1: Matching des 3/4 faciles...")
    
    # Normaliser les noms
    base_df['player_norm'] = base_df.iloc[:, 0].apply(normalize_name)
    positional_df['player_norm'] = positional_df.iloc[:, 1].apply(normalize_name)
    
    matches = []
    used_positional_indices = set()
    
    # ÉTAPE 1: Correspondance EXACTE (les plus faciles)
    print("1. Correspondance exacte...")
    exact_matches_count = 0
    for base_idx, base_row in base_df.iterrows():
        if base_idx % 500 == 0:
            print(f"   Traité {base_idx}/{len(base_df)} lignes...")
            
        player_norm = base_row['player_norm']
        season = base_row['season']
        
        # Chercher correspondance exacte
        exact_matches = positional_df[
            (positional_df['player_norm'] == player_norm) & 
            (positional_df.iloc[:, 3] == season) &
            (~positional_df.index.isin(used_positional_indices))
        ]
        
        if not exact_matches.empty:
            match_row = exact_matches.iloc[0]
            match_idx = exact_matches.index[0]
            
            matches.append({
                'base_index': base_idx,
                'positional_data': match_row,
                'position': match_row['position_source'],
                'score': 100,
                'strategy': 'exact'
            })
            used_positional_indices.add(match_idx)
            exact_matches_count += 1
    
    print(f"   ✅ {exact_matches_count} matches exacts trouvés")
    
    # ÉTAPE 2: Correspondance PARTIELLE simple (2ème plus facile)
    print("2. Correspondance partielle simple...")
    partial_matches_count = 0
    remaining_base = set(range(len(base_df))) - set([m['base_index'] for m in matches])
    
    for base_idx in list(remaining_base):
        base_row = base_df.iloc[base_idx]
        player_norm = base_row['player_norm']
        season = base_row['season']
        
        available_pos = positional_df[
            (positional_df.iloc[:, 3] == season) & 
            (~positional_df.index.isin(used_positional_indices))
        ]
        
        if not available_pos.empty:
            for pos_idx, pos_row in available_pos.iterrows():
                pos_norm = pos_row['player_norm']
                
                # Vérifications simples de correspondance partielle
                is_match = (
                    # Même nom de famille
                    player_norm.split()[-1] == pos_norm.split()[-1] if len(player_norm.split()) > 0 and len(pos_norm.split()) > 0 else False
                ) or (
                    # Un nom contient l'autre
                    player_norm in pos_norm or pos_norm in player_norm
                ) or (
                    # Initiales identiques avec nom de famille similaire
                    len(player_norm.split()) >= 2 and len(pos_norm.split()) >= 2 and
                    player_norm.split()[0][0] == pos_norm.split()[0][0] and
                    player_norm.split()[-1] == pos_norm.split()[-1]
                )
                
                if is_match:
                    matches.append({
                        'base_index': base_idx,
                        'positional_data': pos_row,
                        'position': pos_row['position_source'],
                        'score': 85,
                        'strategy': 'partial_simple'
                    })
                    used_positional_indices.add(pos_idx)
                    partial_matches_count += 1
                    remaining_base.remove(base_idx)
                    break
    
    print(f"   ✅ {partial_matches_count} matches partiels simples trouvés")
    
    # ÉTAPE 3: Vérification manuelle des matches pour qualité
    print("3. Vérification de la qualité des matches...")
    
    # Afficher quelques exemples de matches pour vérification
    print("\n📋 Exemples de matches trouvés:")
    sample_matches = matches[:10]  # Premier 10 matches
    for i, match in enumerate(sample_matches):
        base_row = base_df.iloc[match['base_index']]
        pos_row = match['positional_data']
        print(f"   {i+1}. '{base_row.iloc[0]}' → '{pos_row.iloc[1]}' (score: {match['score']})")
    
    # Calculer le pourcentage de matching
    total_matches = len(matches)
    total_players = len(base_df)
    matching_rate = total_matches / total_players
    
    print(f"\n📊 RÉSULTAT PHASE 1:")
    print(f"   Matches trouvés: {total_matches}/{total_players}")
    print(f"   Taux de matching: {matching_rate*100:.1f}%")
    
    # Vérifier si on a atteint l'objectif des 3/4
    target_rate = 0.75
    if matching_rate >= target_rate:
        print(f"🎯 OBJECTIF ATTEINT! Plus de {target_rate*100}% de matching")
    else:
        print(f"🎯 Objectif non atteint, passage à la phase 2...")
        
        # PHASE 2: Matching additionnel pour atteindre l'objectif
        additional_needed = int(total_players * target_rate) - total_matches
        print(f"   Besoin de {additional_needed} matches supplémentaires")
        
        additional_count = 0
        for base_idx in list(remaining_base):
            if additional_count >= additional_needed:
                break
                
            base_row = base_df.iloc[base_idx]
            season = base_row['season']
            
            available_pos = positional_df[
                (positional_df.iloc[:, 3] == season) & 
                (~positional_df.index.isin(used_positional_indices))
            ]
            
            if not available_pos.empty:
                # Prendre le premier disponible
                match_row = available_pos.iloc[0]
                match_idx = available_pos.index[0]
                
                matches.append({
                    'base_index': base_idx,
                    'positional_data': match_row,
                    'position': match_row['position_source'],
                    'score': 60,
                    'strategy': 'auto_completion'
                })
                used_positional_indices.add(match_idx)
                additional_count += 1
        
        print(f"   ✅ {additional_count} matches supplémentaires ajoutés")
    
    # Résultat final
    final_matches = len(matches)
    final_rate = final_matches / total_players
    
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Total matches: {final_matches}/{total_players}")
    print(f"   Taux final: {final_rate*100:.1f}%")
    
    # Identifier les non-matches
    all_base_indices = set(range(len(base_df)))
    matched_indices = set([m['base_index'] for m in matches])
    no_match_indices = list(all_base_indices - matched_indices)
    
    return matches, no_match_indices

def create_quality_dataset(base_df, positional_df, matches, no_match_indices):
    """Crée le dataset final avec qualité contrôlée"""
    
    print("\n🧩 Création du dataset final avec qualité...")
    
    # Trier les matches par score (meilleurs matches en premier)
    matches_sorted = sorted(matches, key=lambda x: x['score'], reverse=True)
    
    # Créer les lignes avec matches
    matched_rows = []
    for i, match in enumerate(matches_sorted):
        base_row = base_df.iloc[match['base_index']].copy()
        positional_row = match['positional_data']
        
        # Ajouter les colonnes positionnelles (à partir de la 9ème colonne)
        if len(positional_row) > 8:
            for col_name, value in positional_row.iloc[8:].items():
                base_row[col_name] = value
        
        base_row['position'] = match['position']
        base_row['match_quality'] = match['score']
        base_row['match_strategy'] = match['strategy']
        matched_rows.append(base_row)
    
    matched_df = pd.DataFrame(matched_rows)
    
    # Créer les lignes sans matches
    print("Ajout des joueurs sans match...")
    no_match_df = base_df.iloc[no_match_indices].copy()
    if len(positional_df.columns) > 8:
        for col in positional_df.columns[8:]:
            no_match_df[col] = np.nan
    no_match_df['position'] = np.nan
    no_match_df['match_quality'] = 0
    no_match_df['match_strategy'] = 'no_match'
    
    # Combiner (matches de qualité d'abord, puis non-matches)
    print("Combinaison finale...")
    final_df = pd.concat([matched_df, no_match_df], ignore_index=True)
    
    # Nettoyer les colonnes temporaires
    if 'player_norm' in final_df.columns:
        final_df = final_df.drop('player_norm', axis=1)
    
    return final_df

def main():
    """Fonction principale"""
    try:
        print("🚀 Début du processus SMART de création de PL_final...")
        print("🎯 Objectif: 75% de matching de QUALITÉ")
        
        base_df, positional_df = load_and_prepare_data()
        
        print("\n🔗 PHASE PRINCIPALE: Matching intelligent...")
        matches, no_match_indices = smart_merge_3_4_easy_matches(base_df, positional_df)
        
        print("\n💾 Création du dataset final...")
        final_df = create_quality_dataset(base_df, positional_df, matches, no_match_indices)
        
        # Sauvegarder
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_file = os.path.join(desktop_path, 'PL_final_quality.csv')
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n🎉 Dataset PL_final créé avec succès!")
        print(f"📊 Dimensions: {final_df.shape}")
        print(f"🎯 Joueurs avec position: {len(matches)}")
        print(f"❌ Joueurs sans position: {len(no_match_indices)}")
        print(f"📈 Taux de matching: {len(matches)/len(final_df)*100:.1f}%")
        print(f"💾 Fichier: {output_file}")
        
        # Statistiques de qualité
        if 'match_quality' in final_df.columns:
            quality_stats = final_df['match_quality'].value_counts().sort_index(ascending=False)
            print(f"\n📊 QUALITÉ DES MATCHES:")
            for score, count in quality_stats.items():
                if score > 0:
                    print(f"   Score {score}: {count} joueurs")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    final_dataset = main()
    
    