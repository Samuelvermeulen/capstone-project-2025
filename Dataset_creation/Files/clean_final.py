
import pandas as pd
import numpy as np
import os

def clean_and_merge_columns():
    """Nettoie le dataset et complète les colonnes vides"""
    
    # Charger le fichier
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    input_file = os.path.join(desktop_path, 'PL_final_quality.csv')
    
    print("📁 Chargement du fichier PL_final_quality.csv...")
    df = pd.read_csv(input_file)
    print(f"📊 Dataset original: {df.shape} lignes, {df.shape[1]} colonnes")
    
    # ÉTAPE 1: Supprimer les lignes avec "no_match"
    print("\n🗑️  Suppression des lignes 'no_match'...")
    initial_rows = len(df)
    df = df[df.iloc[:, -1] != 'no_match']  # Dernière colonne = match_strategy
    final_rows = len(df)
    print(f"✅ Lignes supprimées: {initial_rows - final_rows}")
    print(f"✅ Lignes restantes: {final_rows}")
    
    # ÉTAPE 2: Compléter les colonnes vides de gauche avec les données de droite
    print("\n🔄 Complétion des données manquantes...")
    
    # Mapping des colonnes (droite → gauche)
    column_mapping = {
        '_value_eur': 'Value',
        'Name': 'long_name', 
        'Age': 'age',
        'Nationality': 'nationality',
        'Club': 'club_name',
        'Preferred Foot': 'preferred foot'
    }
    
    # Vérifier quelles colonnes existent dans le dataset
    available_columns = df.columns.tolist()
    print(f"📋 Colonnes disponibles: {available_columns}")
    
    # Appliquer le mapping pour chaque paire de colonnes
    for source_col, target_col in column_mapping.items():
        if source_col in df.columns and target_col in df.columns:
            # Compter les valeurs manquantes avant
            missing_before = df[target_col].isna().sum()
            
            # Compléter les valeurs manquantes de la colonne cible avec la colonne source
            df[target_col] = df[target_col].fillna(df[source_col])
            
            # Compter les valeurs remplies
            missing_after = df[target_col].isna().sum()
            filled_count = missing_before - missing_after
            
            print(f"   ✅ {target_col}: {filled_count} valeurs complétées depuis {source_col}")
        else:
            if source_col not in df.columns:
                print(f"   ⚠️ Colonne source '{source_col}' non trouvée")
            if target_col not in df.columns:
                print(f"   ⚠️ Colonne cible '{target_col}' non trouvée")
    
    # ÉTAPE 3: Sauvegarder le résultat
    output_file = os.path.join(desktop_path, 'PL_final_cleaned.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n🎉 Dataset final créé avec succès!")
    print(f"📊 Dimensions finales: {df.shape}")
    print(f"💾 Fichier sauvegardé: {output_file}")
    
    # Aperçu des données
    print(f"\n👀 Aperçu des données:")
    print(df[['long_name', 'age', 'club_name', 'Value', 'position']].head(10))
    
    return df

# Exécution
if __name__ == "__main__":
    final_df = clean_and_merge_columns()


    