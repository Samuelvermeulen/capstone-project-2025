

"""
Feature Engineering Module - Phase 4
Samuel Vermeulen - Capstone Project 2025

Objectif : Construire progressivement les features selon le roadmap
Approche : Une fonction par étape, testée individuellement
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Liste basée sur le 5-Year Ranking 2023, avec les noms EXACTS du dataset
TOP_CLUBS_EURO_RANKING = [
    'Manchester City',  # Rank 1
    'Liverpool',        # Rank 2
    'Chelsea',          # Rank 7
    'Manchester Utd',   # Rank 10 (ATTENTION: différent de 'Manchester United')
    'Arsenal',          # Rank 15
    'Tottenham',        # Rank 16
    'Leicester City',   # Rank 33
    'West Ham',         # Rank 38
    'Wolves',           # Rank 47 (Nom exact du dataset)
    'Newcastle Utd'     # Rank 51 (ATTENTION: différent de 'Newcastle United')
]

# NOUVELLES CATÉGORIES BASÉES SUR LE RAPPORT TECHNIQUE
# Clubs de milieu de tableau (moyenne position 8-14)
MIDDLE_TABLE_CLUBS = [
    'Everton',
    'Aston Villa', 
    'Brighton',
    'Crystal Palace',
    'Southampton',
    'Bournemouth',
    'Leeds United',  # Note: vérifier le nom exact dans le dataset
    'Brentford'
]

# Clubs en lutte contre la relégation (moyenne position ≥15)
RELEGATION_BATTLE_CLUBS = [
    'Norwich',
    'Watford',
    'Burnley',
    'Sheffield United',
    'Fulham',
    'Cardiff',
    'Huddersfield'
]

#### Step 0 ### 

def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données nettoyées depuis data/processed/
    
    Returns:
    --------
    train_df, test_df : tuple de DataFrames
    """
    logger.info("Chargement des données nettoyées...")
    
    try:
        train_df = pd.read_csv("data/processed/train_data.csv")
        test_df = pd.read_csv("data/processed/test_data.csv")
        
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise

###### Step 1 #####  DataFrame Inspection
def inspect_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Affiche un résumé informatif d'un DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à inspecter
    name : str
        Nom pour l'affichage
    """
    print(f"\n{'='*60}")
    print(f"🔍 INSPECTION: {name}")
    print(f"{'='*60}")
    
    print(f"Shape: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📊 Types de données:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} colonnes")
    
    print(f"\n⚠️  Valeurs manquantes:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"  • {col}: {count} ({percentage:.1f}%)")
    else:
        print("  ✅ Aucune valeur manquante")
    
    print(f"\n🎯 Variable cible (Value):")
    if 'Value' in df.columns:
        print(f"  • Min: €{df['Value'].min():,.0f}")
        print(f"  • Max: €{df['Value'].max():,.0f}")
        print(f"  • Mean: €{df['Value'].mean():,.0f}")
        print(f"  • Skewness: {df['Value'].skew():.2f}")
    
    print(f"\n📋 5 premières lignes:")
    print(df.head())
    
    print(f"\n{'='*60}")


######### Step 2 ######## Position encoding 

def encode_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode la colonne Position en variables one-hot (4 catégories).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant la colonne 'Position'
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        DataFrame avec les colonnes one-hot ajoutées
    """
    logger.info("Encodage one-hot de la colonne Position...")
    
    # Vérifier que la colonne Position existe
    if 'Position' not in df.columns:
        logger.error("Colonne 'Position' non trouvée")
        return df
    
    # Créer une copie pour éviter les modifications inplace
    df_encoded = df.copy()
    
    # 1. Vérifier les valeurs uniques
    unique_positions = df_encoded['Position'].unique()
    logger.info(f"Positions uniques trouvées: {unique_positions}")
    
    # 2. Créer les colonnes one-hot
    # Pour chaque position, créer une colonne binaire
    positions_map = {
        'DF': 'is_Defender',
        'MF': 'is_Midfielder',
        'FW': 'is_Forward',
        'GK': 'is_Goalkeeper'
    }
    
    # Initialiser toutes les colonnes à 0
    for col_name in positions_map.values():
        df_encoded[col_name] = 0
    
    # Mettre à 1 pour la position correspondante
    for original, encoded in positions_map.items():
        mask = df_encoded['Position'] == original
        df_encoded.loc[mask, encoded] = 1
    
    # 3. Vérifier la distribution
    logger.info("Distribution après encodage:")
    for encoded_col in positions_map.values():
        count = df_encoded[encoded_col].sum()
        percentage = (count / len(df_encoded)) * 100
        logger.info(f"  • {encoded_col}: {count} joueurs ({percentage:.1f}%)")
    
    # 4. Optionnel: Supprimer la colonne originale
    # df_encoded = df_encoded.drop('Position', axis=1)
    # Note: Pour l'instant, gardons-la pour vérification
    
    return df_encoded

####### Step 3 ##### Test the position encoding

def test_position_encoding():
    """
    Teste l'encodage de la position sur les données d'entraînement.
    """
    print("\n🧪 TEST DE L'ENCODAGE POSITION")
    print("=" * 50)
    
    # Charger les données
    train_df, test_df = load_processed_data()
    
    # Appliquer l'encodage
    train_encoded = encode_position(train_df)
    test_encoded = encode_position(test_df)
    
    # Vérifier les résultats
    print("\n✅ Encodage appliqué avec succès!")
    print(f"\nTrain - Avant: {train_df.shape[1]} colonnes")
    print(f"Train - Après: {train_encoded.shape[1]} colonnes")
    
    print(f"\nTest - Avant: {test_df.shape[1]} colonnes")
    print(f"Test - Après: {test_encoded.shape[1]} colonnes")
    
    # Afficher les nouvelles colonnes
    new_cols = [col for col in train_encoded.columns 
                if col.startswith('is_')]
    
    print(f"\n🎯 Nouvelles colonnes créées: {new_cols}")
    
    # Vérifier quelques exemples
    print(f"\n📋 Exemples (premières 3 lignes):")
    sample_cols = ['Player', 'Position'] + new_cols
    print(train_encoded[sample_cols].head(3))
    
    # Vérifier la cohérence
    print(f"\n🔍 Vérification de cohérence:")
    for idx, row in train_encoded.head(5).iterrows():
        position = row['Position']
        expected_col = f"is_{position}"
        # Adapter le nom attendu
        position_map = {'DF': 'Defender', 'MF': 'Midfielder', 
                       'FW': 'Forward', 'GK': 'Goalkeeper'}
        expected_col = f"is_{position_map.get(position, position)}"
        
        if expected_col in new_cols and row[expected_col] == 1:
            print(f"  ✅ Ligne {idx}: {row['Player']} - {position} → {expected_col}=1")
        else:
            print(f"  ❌ Ligne {idx}: Problème de cohérence")
    
    return train_encoded, test_encoded

######## Step 4 #### missing values treatment 

def handle_missing_values(df, is_training=True, imputation_dict=None):
    """
    Gère les valeurs manquantes dans les colonnes numériques.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame d'entrée
    is_training : bool
        Si True, calcule les médianes sur ce dataset
    imputation_dict : dict, optional
        Dictionnaire des valeurs d'imputation (médianes)
        
    Returns:
    --------
    df_imputed : pandas.DataFrame
        DataFrame avec valeurs imputées
    imputation_dict : dict
        Dictionnaire des valeurs d'imputation utilisées
    """
    logger.info("Gestion des valeurs manquantes...")
    
    # Colonnes avec valeurs manquantes identifiées
    cols_with_missing = ['Minutes_played', 'Goals', 'Assists']
    
    # S'assurer que ces colonnes existent
    existing_cols = [col for col in cols_with_missing if col in df.columns]
    
    if not existing_cols:
        logger.info("Aucune colonne avec valeurs manquantes à traiter")
        return df, imputation_dict or {}
    
    df_imputed = df.copy()
    
    if is_training:
        # En mode entraînement : calculer les médianes
        imputation_dict = {}
        for col in existing_cols:
            median_val = df_imputed[col].median()
            imputation_dict[col] = median_val
            df_imputed[col] = df_imputed[col].fillna(median_val)
            missing_count = df[col].isnull().sum()
            logger.info(f"  • {col}: {missing_count} valeurs manquantes → imputées avec {median_val:.2f}")
    else:
        # En mode test : utiliser les médianes du training
        if imputation_dict is None:
            raise ValueError("imputation_dict requis en mode test")
        
        for col in existing_cols:
            if col in imputation_dict:
                df_imputed[col] = df_imputed[col].fillna(imputation_dict[col])
    
    return df_imputed, imputation_dict

######## Step 5 ###### Creation of ratio

def create_ratios(df):
    """
    Crée les ratios dérivés à partir des statistiques de base.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec colonnes Goals, Assists, Minutes_played
        
    Returns:
    --------
    df_with_ratios : pandas.DataFrame
        DataFrame avec colonnes de ratios ajoutées
    """
    logger.info("Création des ratios dérivés...")
    
    df_ratios = df.copy()
    
    # 1. Éviter la division par zéro
    # Créer une copie sécurisée de Minutes_played
    minutes_safe = df_ratios['Minutes_played'].copy()
    minutes_safe[minutes_safe == 0] = 1  # Remplacer 0 par 1 pour éviter division par zéro
    
    # 2. Calculer les ratios de base
    df_ratios['Goals_per_minute'] = df_ratios['Goals'] / minutes_safe
    df_ratios['Assists_per_minute'] = df_ratios['Assists'] / minutes_safe
    
    # 3. Appliquer transformation log pour réduire l'asymétrie
    df_ratios['Goals_per_minute_log'] = np.log1p(df_ratios['Goals_per_minute'])
    df_ratios['Assists_per_minute_log'] = np.log1p(df_ratios['Assists_per_minute'])
    
    # 4. Statistiques sur les ratios créés
    logger.info("Ratios créés avec succès:")
    for ratio_col in ['Goals_per_minute', 'Assists_per_minute']:
        if ratio_col in df_ratios.columns:
            logger.info(f"  • {ratio_col}: min={df_ratios[ratio_col].min():.6f}, "
                       f"max={df_ratios[ratio_col].max():.6f}, "
                       f"mean={df_ratios[ratio_col].mean():.6f}")
    
    return df_ratios



####### Step 6 ###### Test of ratio 

def test_ratios_and_imputation():
    """
    Teste la création des ratios et l'imputation des valeurs manquantes.
    """
    print("\n🧪 TEST DES RATIOS ET IMPUTATION")
    print("=" * 50)
    
    # Charger les données
    train_df, test_df = load_processed_data()
    
    # Étape 1: Encodage position (déjà testé)
    print("\n1. Encodage de la position...")
    train_encoded = encode_position(train_df)
    test_encoded = encode_position(test_df)
    
    # Étape 2: Gestion des valeurs manquantes
    print("\n2. Imputation des valeurs manquantes...")
    train_imputed, imputation_dict = handle_missing_values(train_encoded, is_training=True)
    test_imputed, _ = handle_missing_values(test_encoded, is_training=False, imputation_dict=imputation_dict)
    
    # Vérifier qu'il n'y a plus de valeurs manquantes
    missing_train = train_imputed[['Minutes_played', 'Goals', 'Assists']].isnull().sum().sum()
    missing_test = test_imputed[['Minutes_played', 'Goals', 'Assists']].isnull().sum().sum()
    
    print(f"   ✅ Train - Valeurs manquantes restantes: {missing_train}")
    print(f"   ✅ Test - Valeurs manquantes restantes: {missing_test}")
    print(f"   📊 Valeurs d'imputation utilisées: {imputation_dict}")
    
    # Étape 3: Création des ratios
    print("\n3. Création des ratios...")
    train_with_ratios = create_ratios(train_imputed)
    test_with_ratios = create_ratios(test_imputed)
    
    # Vérification
    print(f"\n✅ Toutes les étapes appliquées avec succès!")
    print(f"\n📊 Dimensions des datasets:")
    print(f"   Train: {train_with_ratios.shape[0]} lignes × {train_with_ratios.shape[1]} colonnes")
    print(f"   Test: {test_with_ratios.shape[0]} lignes × {test_with_ratios.shape[1]} colonnes")
    
    # Afficher les nouvelles colonnes
    new_cols = [col for col in train_with_ratios.columns 
                if col not in train_df.columns and not col.startswith('is_')]
    print(f"\n🎯 Nouvelles colonnes créées:")
    for col in new_cols:
        print(f"   • {col}")
    
    # Aperçu des ratios
    print(f"\n📋 Exemples de ratios (premières 3 lignes):")
    ratio_cols = ['Goals', 'Minutes_played', 'Goals_per_minute', 'Goals_per_minute_log']
    print(train_with_ratios[ratio_cols].head(3))
    
    # Statistiques des ratios
    print(f"\n📈 Statistiques des ratios (train set):")
    for ratio in ['Goals_per_minute', 'Assists_per_minute']:
        if ratio in train_with_ratios.columns:
            data = train_with_ratios[ratio]
            print(f"   • {ratio}:")
            print(f"      Min: {data.min():.6f}")
            print(f"      Max: {data.max():.6f}")
            print(f"      Moyenne: {data.mean():.6f}")
            print(f"      Médiane: {data.median():.6f}")
    
    return train_with_ratios, test_with_ratios, imputation_dict

###### Step 7 ###### club encoding using ALL 4 categories

def encode_clubs(df, is_training=True, top_clubs=None, middle_clubs=None, relegation_clubs=None):
    """
    Encode les clubs en utilisant les 4 catégories : Top, Middle-table, Relegation-battle, Other.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec colonne 'Club'
    is_training : bool
        Si True, utilise les listes prédéfinies
        Si False, utilise les listes fournies en paramètre
    top_clubs : list, optional
        Liste des clubs Top pour le test set
    middle_clubs : list, optional
        Liste des clubs Middle-table pour le test set (nouveau)
    relegation_clubs : list, optional
        Liste des clubs Relegation-battle pour le test set (nouveau)
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        DataFrame avec encoding des clubs
    club_metadata : dict
        Dictionnaire contenant les listes de clubs utilisées
    """
    logger.info("Encodage des clubs avec 4 catégories...")
    
    if 'Club' not in df.columns:
        logger.warning("Colonne 'Club' non trouvée")
        # Retourner des métadonnées vides pour compatibilité
        if is_training:
            return df, {'top_clubs': TOP_CLUBS_EURO_RANKING,
                       'middle_clubs': MIDDLE_TABLE_CLUBS,
                       'relegation_clubs': RELEGATION_BATTLE_CLUBS}
        else:
            return df, {'top_clubs': top_clubs or [],
                       'middle_clubs': middle_clubs or [],
                       'relegation_clubs': relegation_clubs or []}
    
    df_encoded = df.copy()
    
    # Déterminer les listes de clubs à utiliser
    if is_training:
        # En mode entraînement : utiliser nos listes prédéfinies
        top_clubs_used = TOP_CLUBS_EURO_RANKING
        middle_clubs_used = MIDDLE_TABLE_CLUBS
        relegation_clubs_used = RELEGATION_BATTLE_CLUBS
        logger.info("Mode entraînement : utilisation des listes prédéfinies")
    else:
        # En mode test : utiliser les listes passées en paramètre
        # Si certaines listes ne sont pas fournies, utiliser des listes vides
        top_clubs_used = top_clubs or []
        middle_clubs_used = middle_clubs or []
        relegation_clubs_used = relegation_clubs or []
        logger.info("Mode test : utilisation des listes fournies")
    
    # Vérifier quels clubs de la liste sont présents (pour le logging)
    def check_presence(club_list, category_name):
        present = [club for club in club_list if club in df_encoded['Club'].values]
        missing = [club for club in club_list if club not in df_encoded['Club'].values]
        logger.info(f"{category_name}: {len(present)}/{len(club_list)} présents")
        if missing:
            logger.debug(f"  Absents: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    check_presence(top_clubs_used, "Top clubs")
    check_presence(middle_clubs_used, "Middle-table clubs")
    check_presence(relegation_clubs_used, "Relegation-battle clubs")
    
    # Créer la fonction de catégorisation
    def categorize_club(club_name):
        if club_name in top_clubs_used:
            return 'Top_Club'
        elif club_name in middle_clubs_used:
            return 'Middle_Table_Club'
        elif club_name in relegation_clubs_used:
            return 'Relegation_Battle_Club'
        else:
            return 'Other_Club'
    
    # Appliquer la catégorisation
    df_encoded['Club_encoded'] = df_encoded['Club'].apply(categorize_club)
    
    # Statistiques sur la répartition
    logger.info("Répartition après catégorisation:")
    categories = ['Top_Club', 'Middle_Table_Club', 'Relegation_Battle_Club', 'Other_Club']
    for category in categories:
        count = (df_encoded['Club_encoded'] == category).sum()
        percentage = (count / len(df_encoded)) * 100
        logger.info(f"  • {category}: {count} joueurs ({percentage:.1f}%)")
    
    # Créer les variables dummies (one-hot encoding)
    club_dummies = pd.get_dummies(df_encoded['Club_encoded'], prefix='club')
    
    # Ajouter les dummies au dataframe principal
    df_encoded = pd.concat([df_encoded, club_dummies], axis=1)
    
    logger.info(f"Encodage terminé. {len(club_dummies.columns)} colonnes club créées.")
    
    # Afficher les colonnes créées
    if len(club_dummies.columns) <= 10:
        logger.info(f"Colonnes créées: {list(club_dummies.columns)}")
    
    # Préparer les métadonnées pour reproduction
    club_metadata = {
        'top_clubs': top_clubs_used,
        'middle_clubs': middle_clubs_used,
        'relegation_clubs': relegation_clubs_used,
        'club_categories': categories
    }
    
    return df_encoded, club_metadata

###### Step 8 ##### Preparation for the log transformation 

def prepare_target(df):
    """
    Prépare la variable cible avec transformation log.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec colonne 'Value'
        
    Returns:
    --------
    y_log : pandas.Series
        Variable cible transformée (log(1 + Value))
    y_original : pandas.Series
        Variable cible originale (pour référence)
    """
    logger.info("Préparation de la variable cible...")
    
    if 'Value' not in df.columns:
        raise ValueError("Colonne 'Value' non trouvée")
    
    y_original = df['Value'].copy()
    
    # Transformation log pour gérer l'asymétrie
    y_log = np.log1p(y_original)
    
    logger.info(f"Transformation: log(1 + Value)")
    logger.info(f"Original - Min: €{y_original.min():,.0f}, Max: €{y_original.max():,.0f}")
    logger.info(f"Log - Min: {y_log.min():.2f}, Max: {y_log.max():.2f}")
    logger.info(f"Skewness original: {y_original.skew():.2f}")
    logger.info(f"Skewness log: {y_log.skew():.2f}")
    
    return y_log, y_original

##### Step 9 ##### final code for the doc (pipeline)

def create_final_features(df, is_training=True, top_clubs=None, imputation_dict=None):
    """
    Pipeline complet de feature engineering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Données d'entrée
    is_training : bool
        Si True, entraîne les transformateurs
    top_clubs : list, optional
        Liste des top clubs (maintenant obsolète, gardé pour compatibilité)
    imputation_dict : dict, optional
        Dictionnaire d'imputation
        
    Returns:
    --------
    X : pandas.DataFrame
        Features finales
    y_log : pandas.Series
        Target transformée
    metadata : dict
        Métadonnées (top_clubs, imputation_dict, etc.)
    """
    logger.info(f"Pipeline de feature engineering (is_training={is_training})")
    
    # Pour la compatibilité: si top_clubs est fourni mais est une liste simple,
    # c'est probablement l'ancien format. Dans ce cas, on l'ignore car
    # encode_clubs s'attend maintenant à 3 listes séparées.
    
    # 1. Encodage position
    df_encoded = encode_position(df)
    
    # 2. Imputation valeurs manquantes
    df_imputed, imputation_dict = handle_missing_values(
        df_encoded, is_training=is_training, imputation_dict=imputation_dict
    )
    
    # 3. Création ratios
    df_ratios = create_ratios(df_imputed)
    
    # 4. Encodage clubs - maintenant avec 4 catégories
    # Pour la compatibilité ascendante, on passe top_clubs comme premier argument
    # mais encode_clubs s'attend à 3 listes séparées
    df_clubs, club_metadata = encode_clubs(
        df_ratios, 
        is_training=is_training,
        top_clubs=top_clubs,  # Conserve pour compatibilité
        middle_clubs=None,     # Seront déterminés automatiquement si is_training=True
        relegation_clubs=None  # Seront déterminés automatiquement si is_training=True
    )
    
    # 5. Sélection des features pour le modèle
    # Garder les features numériques de base
    numerical_features = [
        'Age', 'Minutes_played', 'Goals', 'Assists',
        'Height', 'Weight', 'Matchs_played',
        'Goals_per_minute_log', 'Assists_per_minute_log'
    ]
    
    # Features de position (one-hot)
    position_features = [col for col in df_clubs.columns if col.startswith('is_')]
    
    # Features de club (one-hot) 
    club_features = [col for col in df_clubs.columns if col.startswith('club_')]
    
    # Nation (on garde comme catégorielle pour l'instant)
    if 'Nation' in df_clubs.columns:
        # Pour les modèles d'arbre, on peut encoder en numérique
        df_clubs['Nation_encoded'] = pd.factorize(df_clubs['Nation'])[0]
        numerical_features.append('Nation_encoded')
    
    # Combiner toutes les features
    all_features = numerical_features + position_features + club_features
    
    # Garder seulement les colonnes qui existent
    existing_features = [col for col in all_features if col in df_clubs.columns]
    
    X = df_clubs[existing_features].copy()
    
    # 6. Préparation de la target
    y_log, y_original = prepare_target(df_clubs)
    
    # Metadata pour reproduction
    metadata = {
        'top_clubs': club_metadata.get('top_clubs', []),
        'middle_clubs': club_metadata.get('middle_clubs', []),
        'relegation_clubs': club_metadata.get('relegation_clubs', []),
        'club_categories': club_metadata.get('club_categories', []),
        'imputation_dict': imputation_dict,
        'feature_names': existing_features,
        'n_features': len(existing_features),
        'n_numerical': len([f for f in existing_features if f in numerical_features]),
        'n_position': len([f for f in existing_features if f in position_features]),
        'n_club': len([f for f in existing_features if f in club_features])
    }
    
    logger.info(f"Features finales: {len(existing_features)} colonnes")
    logger.info(f"  • Numériques: {metadata['n_numerical']}")
    logger.info(f"  • Position: {metadata['n_position']}")
    logger.info(f"  • Club (4 catégories): {metadata['n_club']}")
    
    return X, y_log, metadata

##### Step 10 #### Test of the final Pipeline

def test_complete_pipeline():
    """
    Teste le pipeline complet de feature engineering.
    """
    print("\n🧪 TEST COMPLET DU PIPELINE DE FEATURE ENGINEERING")
    print("=" * 60)
    
    # Charger les données
    train_df, test_df = load_processed_data()
    
    print("\n1. Application sur le train set...")
    X_train, y_train_log, metadata = create_final_features(train_df, is_training=True)
    
    print(f"\n✅ Train set transformé:")
    print(f"   • X shape: {X_train.shape}")
    print(f"   • y shape: {y_train_log.shape}")
    print(f"   • Nombre de features: {metadata['n_features']}")
    print(f"   • Catégories club: {metadata['club_categories']}")
    
    print("\n2. Application sur le test set (avec métadonnées du train)...")
    # Extraire les listes de clubs des métadonnées
    X_test, y_test_log, _ = create_final_features(
        test_df, 
        is_training=False,
        top_clubs=metadata['top_clubs'],  # Passé pour compatibilité
        imputation_dict=metadata['imputation_dict']
    )
    
    print(f"\n✅ Test set transformé:")
    print(f"   • X shape: {X_test.shape}")
    print(f"   • y shape: {y_test_log.shape}")
    
    # Vérifier la cohérence des colonnes
    print(f"\n3. Vérification de cohérence...")
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    if train_cols == test_cols:
        print(f"   ✅ Mêmes colonnes dans train et test")
    else:
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        if missing_in_test:
            print(f"   ⚠️  Colonnes manquantes dans test: {missing_in_test}")
        if missing_in_train:
            print(f"   ⚠️  Colonnes manquantes dans train: {missing_in_train}")
    
    # Afficher un échantillon des features
    print(f"\n4. Échantillon des features (5 premières):")
    print(f"   • Train: {list(X_train.columns)[:5]}")
    print(f"   • Test: {list(X_test.columns)[:5]}")
    
    # Statistiques des features
    print(f"\n5. Types de features:")
    feature_types = {
        'Numérique': [col for col in X_train.columns 
                     if not col.startswith('is_') and not col.startswith('club_')],
        'Position': [col for col in X_train.columns if col.startswith('is_')],
        'Club': [col for col in X_train.columns if col.startswith('club_')]
    }
    
    for type_name, features in feature_types.items():
        if features:
            print(f"   • {type_name}: {len(features)} features")
            if len(features) <= 5:  # Afficher si peu de features
                print(f"     {features}")
    
    return X_train, y_train_log, X_test, y_test_log, metadata

##### Step 11 #### function that save the transformed data

def save_processed_data(X_train, y_train, X_test, y_test, metadata, output_dir="data/processed"):
    """
    Sauvegarde les données transformées et les métadonnées.
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        Features d'entraînement et de test
    y_train, y_test : pandas.Series
        Target transformée (log scale)
    metadata : dict
        Métadonnées du pipeline
    output_dir : str
        Dossier de sauvegarde
    """
    import os
    import json
    
    logger.info(f"Sauvegarde des données transformées dans {output_dir}...")
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les DataFrames
    X_train.to_csv(f"{output_dir}/X_train_transformed.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_transformed.csv", index=False)
    
    # Sauvegarder les targets (en log scale)
    y_train.to_csv(f"{output_dir}/y_train_log.csv", index=False, header=['Value_log'])
    y_test.to_csv(f"{output_dir}/y_test_log.csv", index=False, header=['Value_log'])
    
    # Sauvegarder les métadonnées au format JSON
    metadata_serializable = {
        'top_clubs': metadata['top_clubs'],
        'middle_clubs': metadata['middle_clubs'],
        'relegation_clubs': metadata['relegation_clubs'],
        'club_categories': metadata['club_categories'],
        'imputation_dict': {k: float(v) for k, v in metadata['imputation_dict'].items()},
        'feature_names': metadata['feature_names'],
        'n_features': metadata['n_features'],
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(f"{output_dir}/feature_metadata.json", 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    # Sauvegarder aussi les targets originales (pour référence)
    train_df, test_df = load_processed_data()
    train_df['Value'].to_csv(f"{output_dir}/y_train_original.csv", index=False, header=['Value'])
    test_df['Value'].to_csv(f"{output_dir}/y_test_original.csv", index=False, header=['Value'])
    
    logger.info("✅ Données sauvegardées avec succès:")
    logger.info(f"   • X_train: {output_dir}/X_train_transformed.csv ({X_train.shape})")
    logger.info(f"   • X_test: {output_dir}/X_test_transformed.csv ({X_test.shape})")
    logger.info(f"   • y_train_log: {output_dir}/y_train_log.csv")
    logger.info(f"   • y_test_log: {output_dir}/y_test_log.csv")
    logger.info(f"   • Métadonnées: {output_dir}/feature_metadata.json")
    
    return output_dir

##### Step 12 #### Function that execute he pipeline and save the steps 

def run_and_save_pipeline():
    """
    Exécute le pipeline complet et sauvegarde les résultats.
    """
    print("🚀 EXÉCUTION ET SAUVEGARDE DU PIPELINE COMPLET")
    print("=" * 60)
    
    # 1. Charger les données
    train_df, test_df = load_processed_data()
    
    # 2. Exécuter le pipeline sur le train set
    print("\n1. Feature engineering sur le train set...")
    X_train, y_train_log, metadata = create_final_features(train_df, is_training=True)
    
    # 3. Exécuter sur le test set avec les métadonnées du train
    print("\n2. Feature engineering sur le test set...")
    X_test, y_test_log, _ = create_final_features(
        test_df, 
        is_training=False,
        top_clubs=metadata['top_clubs'],
        imputation_dict=metadata['imputation_dict']
    )
    
    # 4. Sauvegarder
    print("\n3. Sauvegarde des données transformées...")
    save_path = save_processed_data(X_train, y_train_log, X_test, y_test_log, metadata)
    
    # 5. Résumé
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ ET DONNÉES SAUVEGARDÉES")
    print("=" * 60)
    
    summary = f"""
📊 RÉSUMÉ FINAL PHASE 4:
   • Features créées: {metadata['n_features']}
   • Train samples: {X_train.shape[0]}
   • Test samples: {X_test.shape[0]}
   • Catégories de clubs: {len(metadata['club_categories'])}
   • Données sauvegardées dans: {save_path}

🔧 TYPES DE FEATURES:
   • Numériques: {metadata['n_numerical']}
   • Position: {metadata['n_position']}
   • Club: {metadata['n_club']}

🎯 NOUVELLES CATÉGORIES DE CLUBS:
   • Top clubs: {len(metadata['top_clubs'])} clubs
   • Middle-table clubs: {len(metadata['middle_clubs'])} clubs
   • Relegation-battle clubs: {len(metadata['relegation_clubs'])} clubs
   • Other clubs: catégorie résiduelle

🎯 PRÊT POUR LA PHASE 5 (MODÉLISATION)!
    """
    print(summary)
    
    return X_train, y_train_log, X_test, y_test_log, metadata

#### Final Test “######

if __name__ == "__main__":
    """
    Point d'entrée pour tester et sauvegarder.
    """
    print("🧪 MODULE FEATURE ENGINEERING - EXÉCUTION COMPLÈTE")
    
    try:
        # Option 1: Exécuter et sauvegarder le pipeline complet
        X_train, y_train_log, X_test, y_test_log, metadata = run_and_save_pipeline()
        
        # Option 2: Tester le chargement des données sauvegardées
        print("\n" + "="*60)
        print("🔍 TEST DE CHARGEMENT DES DONNÉES SAUVEGARDÉES")
        print("="*60)
        
        # Créer une fonction de test rapide
        test_df = pd.read_csv("data/processed/X_train_transformed.csv")
        print(f"✅ Données chargées: {test_df.shape}")
        print(f"   Colonnes: {list(test_df.columns)[:5]}...")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
