import pandas as pd
import numpy as np
import re
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

def clean_height(height_str):
    """Convertir 179cm en 179"""
    if pd.isna(height_str):
        return np.nan
    if isinstance(height_str, str) and 'cm' in height_str:
        return int(height_str.replace('cm', '').strip())
    try:
        return int(height_str)
    except:
        return np.nan

def clean_weight(weight_str):
    """Convertir 69kg en 69"""
    if pd.isna(weight_str):
        return np.nan
    if isinstance(weight_str, str) and 'kg' in weight_str:
        return int(weight_str.replace('kg', '').strip())
    try:
        return int(weight_str)
    except:
        return np.nan

def convert_value(value_str):
    """Convertir €107.5M en 107500000"""
    if pd.isna(value_str) or value_str == '€0' or value_str == '0':
        return 0
    
    value_str = str(value_str).replace('€', '').strip()
    
    # Gérer les valeurs en K (milliers)
    if 'K' in value_str:
        number = float(value_str.replace('K', '').strip())
        return int(number * 1000)
    
    # Gérer les valeurs en M (millions)
    elif 'M' in value_str:
        number = float(value_str.replace('M', '').strip())
        return int(number * 1000000)
    
    # Gérer les valeurs sans suffixe
    else:
        try:
            return int(float(value_str))
        except:
            return 0

def process_season_2018_2021(df, season_name):
    """Traiter les datasets 2018-2019, 2019-2020, 2020-2021"""
    # Renommer les colonnes selon les spécifications
    df = df.rename(columns={
        'long_name': 'Player',
        'age': 'Age', 
        'club_name': 'Club',
        'preferred_foot': 'Preferred Foot',
        'nationality': 'Nationality'
    })
    
    # Pour 2020-2021, renommer value_eur en Value si nécessaire
    if 'value_eur' in df.columns and 'Value' not in df.columns:
        df = df.rename(columns={'value_eur': 'Value'})
    
    # Réorganiser les colonnes
    column_order = ['Player', 'Age', 'Nationality', 'Club', 'Value', 'Preferred Foot']
    
    # Ajouter les autres colonnes (sauf Height et Weight)
    other_cols = [col for col in df.columns if col not in column_order + ['Height', 'Weight', 'height', 'weight']]
    column_order.extend(other_cols)
    
    # Ajouter Height et Weight à la fin
    height_col = 'Height' if 'Height' in df.columns else 'height'
    weight_col = 'Weight' if 'Weight' in df.columns else 'weight'
    
    if height_col in df.columns:
        column_order.append('Height')
    if weight_col in df.columns:
        column_order.append('Weight')
    
    df = df[column_order]
    
    # Nettoyer Height et Weight
    if 'Height' in df.columns:
        df['Height'] = df['Height'].apply(clean_height)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].apply(clean_weight)
    
    # Convertir les valeurs
    if 'Value' in df.columns:
        df['Value'] = df['Value'].apply(convert_value)
    
    # Ajouter la colonne Season
    df['Season'] = season_name
    
    return df

def process_season_2021_2023(df, season_name):
    """Traiter les datasets 2021-2022 et 2022-2023"""
    # Renommer Name en Player pour uniformité
    df = df.rename(columns={'Name': 'Player'})
    
    # Nettoyer Height et Weight
    df['Height'] = df['Height'].apply(clean_height)
    df['Weight'] = df['Weight'].apply(clean_weight)
    
    # Convertir Value
    df['Value'] = df['Value'].apply(convert_value)
    
    # Réorganiser les colonnes pour avoir Player en première position
    cols = ['Player'] + [col for col in df.columns if col != 'Player' and col != 'Season']
    df = df[cols]
    
    # Ajouter la colonne Season
    df['Season'] = season_name
    
    return df

# Chemin vers les fichiers nettoyés
MERGED_DIR = os.path.join(os.path.expanduser("~/Desktop"), "Dataset_creation", "Merged_Dataset")

# Charger et traiter tous les datasets
datasets = []

# Dataset 2018-2019
try:
    df_2018_2019 = pd.read_csv(os.path.join(MERGED_DIR, 'premier_league_cleaned_2018-2019.csv'))
    df_2018_2019 = process_season_2018_2021(df_2018_2019, '2018-2019')
    datasets.append(df_2018_2019)
    print("✅ 2018-2019 traité")
except Exception as e:
    print(f"❌ Erreur 2018-2019: {e}")

# Dataset 2019-2020  
try:
    df_2019_2020 = pd.read_csv(os.path.join(MERGED_DIR, 'premier_league_cleaned_2019-2020.csv'))
    df_2019_2020 = process_season_2018_2021(df_2019_2020, '2019-2020')
    datasets.append(df_2019_2020)
    print("✅ 2019-2020 traité")
except Exception as e:
    print(f"❌ Erreur 2019-2020: {e}")

# Dataset 2020-2021
try:
    df_2020_2021 = pd.read_csv(os.path.join(MERGED_DIR, 'premier_league_cleaned_2020-2021.csv'))
    df_2020_2021 = process_season_2018_2021(df_2020_2021, '2020-2021')
    datasets.append(df_2020_2021)
    print("✅ 2020-2021 traité")
except Exception as e:
    print(f"❌ Erreur 2020-2021: {e}")

# Dataset 2021-2022
try:
    df_2021_2022 = pd.read_csv(os.path.join(MERGED_DIR, 'premier_league_cleaned_2021-2022.csv'))
    df_2021_2022 = process_season_2021_2023(df_2021_2022, '2021-2022')
    datasets.append(df_2021_2022)
    print("✅ 2021-2022 traité")
except Exception as e:
    print(f"❌ Erreur 2021-2022: {e}")

# Dataset 2022-2023
try:
    df_2022_2023 = pd.read_csv(os.path.join(MERGED_DIR, 'premier_league_cleaned_2022-2023.csv'))
    df_2022_2023 = process_season_2021_2023(df_2022_2023, '2022-2023')
    datasets.append(df_2022_2023)
    print("✅ 2022-2023 traité")
except Exception as e:
    print(f"❌ Erreur 2022-2023: {e}")

# Fusionner tous les datasets
if datasets:
    final_df = pd.concat(datasets, ignore_index=True)
    
    # Réorganiser l'ordre final des colonnes pour avoir Season à la fin
    cols = [col for col in final_df.columns if col != 'Season']
    cols.append('Season')
    final_df = final_df[cols]
    
    # Assurer que Player est la première colonne
    if 'Player' in final_df.columns:
        player_col = final_df['Player']
        final_df = final_df.drop('Player', axis=1)
        final_df.insert(0, 'Player', player_col)
    
    # Sauvegarder le dataset fusionné
    output_path = os.path.join(desktop_path, "Dataset_creation", "Merged_Dataset", "premier_league_merged_2018-2023.csv") 
    final_df.to_csv(output_path, index=False)
    
    print(f"\n🎉 Fusion terminée !")
    print(f"📊 Dataset final: {final_df.shape[0]} lignes, {final_df.shape[1]} colonnes")
    print(f"💾 Sauvegardé sous: premier_league_merged_2018-2023.csv")
    
    # Aperçu des données
    print(f"\n📋 Aperçu des colonnes finales:")
    print(final_df.columns.tolist())
    print(f"\n👀 Aperçu des données:")
    print(final_df.head())
    
else:
    print("❌ Aucun dataset n'a pu être chargé")