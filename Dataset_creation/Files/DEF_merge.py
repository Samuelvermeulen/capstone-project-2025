
import pandas as pd
import os
import glob

def merge_def_files():
    """
    Fusionne tous les fichiers PL_DEF_*.csv du dossier 'Projet python data'
    en utilisant les indices de colonnes pour Player (colonne 2) et Season (colonne 4)
    """
    # Chemin vers le dossier sur le Bureau
    desktop_path = os.path.expanduser("~/Desktop")  # Chemin du bureau
    project_folder = os.path.join(desktop_path, "Dataset_creation/Fbref_Dataset")
    
    print(f"📁 Recherche dans : {project_folder}")
    
    # Vérifier si le dossier existe
    if not os.path.exists(project_folder):
        print(f"❌ Le dossier '{project_folder}' n'existe pas")
        return None
    
    # Trouver les fichiers PL_DEF_*.csv
    def_files = glob.glob(os.path.join(project_folder, "PL_DEF_*.csv"))
    def_files.sort()  # Trier les fichiers
    
    print(f"🎯 Fichiers PL_DEF trouvés : {[os.path.basename(f) for f in def_files]}")
    
    if not def_files:
        print("❌ Aucun fichier PL_DEF_*.csv trouvé")
        return None
    
    # ÉTAPE 1 : METTRE TOUS LES FICHIERS À LA SUITE
    print("\n" + "="*50)
    print("ÉTAPE 1 : MISE À LA SUITE DES FICHIERS")
    print("="*50)
    
    all_data = []    
    for file in def_files:
        try:
            file_name = os.path.basename(file)
            print(f"📖 Lecture de {file_name}...")
            
            # Lire le fichier CSV
            df = pd.read_csv(file)
            print(f"   ✓ Structure : {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Renommer les colonnes importantes pour faciliter le tri
            if len(df.columns) > 4:  # Vérifier qu'il y a au moins 5 colonnes
                # Créer un dictionnaire pour renommer les colonnes
                new_columns = {}
                if len(df.columns) > 1:  # Colonne 2 = Player
                    new_columns[df.columns[1]] = 'Player'
                if len(df.columns) > 3:  # Colonne 4 = Season  
                    new_columns[df.columns[3]] = 'Season'
                
                df = df.rename(columns=new_columns)
                print(f"   🔧 Colonnes renommées : {new_columns}")
            
            all_data.append(df)
            print(f"   ✅ {len(df)} lignes ajoutées")
            
        except Exception as e:
            print(f"   ✗ Erreur avec {file}: {e}")
            return None
    
    # Fusion simple - juste mettre à la suite
    merged_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Étape 1 terminée")
    print(f"📊 Total des lignes après mise à la suite : {len(merged_df)}")
    
    # ÉTAPE 2 : TRANSFORMATIONS AVANT TRI
    print("\n" + "="*50)
    print("ÉTAPE 2 : TRANSFORMATIONS DES COLONNES")
    print("="*50)
    
    # Transformation 1 : Colonne 38 - Garder uniquement les 2 premières lettres
    if len(merged_df.columns) > 38:  # Vérifier que la colonne 38 existe (index 37)
        col_38_name = merged_df.columns[37]
        print(f"🔧 Transformation colonne 38 ({col_38_name}) : garder 2 premières lettres")
        
        def keep_first_two_chars(value):
            if pd.isna(value) or value == '':
                return value
            value_str = str(value)
            return value_str[:2]  # Garde les 2 premiers caractères
        
        merged_df[col_38_name] = merged_df[col_38_name].apply(keep_first_two_chars)
        print(f"   ✅ Colonne 38 transformée")
    else:
        print(f"   ℹ️  Colonne 38 non trouvée (seulement {len(merged_df.columns)} colonnes)")
    
    # Transformation 2 : Supprimer la colonne 39
    if len(merged_df.columns) > 39:  # Vérifier que la colonne 39 existe (index 38)
        col_39_name = merged_df.columns[38]
        print(f"🗑️  Suppression colonne 39 ({col_39_name})")
        merged_df = merged_df.drop(columns=[col_39_name])
        print(f"   ✅ Colonne 39 supprimée")
    else:
        print(f"   ℹ️  Colonne 39 non trouvée (seulement {len(merged_df.columns)} colonnes)")
    
    # ÉTAPE 3 : TRI PAR NOM ET SAISON
    print("\n" + "="*50)
    print("ÉTAPE 3 : TRI PAR NOM ET SAISON")
    print("="*50)
    
    # Vérifier que nous pouvons accéder aux colonnes par index
    print("🔍 Vérification de l'accès aux colonnes...")
    print(f"   Nombre de colonnes : {len(merged_df.columns)}")
    print(f"   Noms des colonnes : {list(merged_df.columns)}")
    
    # Méthode 1 : Si les colonnes ont été renommées
    if 'Player' in merged_df.columns and 'Season' in merged_df.columns:
        print("   ✓ Tri avec colonnes renommées")
        sorted_df = merged_df.sort_values(['Player', 'Season'])
    
    # Méthode 2 : Accès par index des colonnes
    elif len(merged_df.columns) >= 5:  # Au moins 5 colonnes (index 0 à 4)
        print("   ✓ Tri avec indices de colonnes (2=Player, 4=Season)")
        
        # Trier en utilisant les indices de colonnes
        player_col = merged_df.columns[1]  # Colonne 2 (index 1)
        season_col = merged_df.columns[3]  # Colonne 4 (index 3)
        
        print(f"   Colonne Player : {player_col}")
        print(f"   Colonne Season : {season_col}")
        
        sorted_df = merged_df.sort_values([player_col, season_col])
        
        # Renommer pour la clarté
        sorted_df = sorted_df.rename(columns={
            player_col: 'Player',
            season_col: 'Season'
        })
        
    else:
        print("❌ Pas assez de colonnes pour le tri")
        return None
    
    print("✅ Tri effectué avec succès")
    
    # Sauvegarder le fichier final sur le Bureau
    output_path = os.path.join(desktop_path, "Dataset_creation", "Merged_Dataset", "PL_DEF.csv")
    sorted_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # RAPPORT FINAL
    print("\n" + "="*50)
    print("RAPPORT FINAL")
    print("="*50)
    print(f"✅ Fusion et tri terminés avec succès")
    print(f"💾 Fichier sauvegardé : {output_path}")
    print(f"📈 Total des enregistrements : {len(sorted_df)}")
    print(f"🎯 Fichiers fusionnés : {len(def_files)}")
    print(f"🔧 Transformations appliquées :")
    print(f"   - Colonne 38 : garder 2 premières lettres")
    print(f"   - Colonne 39 : supprimée")
    
    # Aperçu du résultat trié
    print("\n🔍 APERÇU DU RÉSULTAT TRIÉ :")
    print("Les 10 premières lignes :")
    print("-" * 60)
    
    # Afficher les données en utilisant les bonnes colonnes
    if 'Player' in sorted_df.columns and 'Season' in sorted_df.columns:
        preview_data = sorted_df[['Player', 'Season']].head(10)
        for idx, row in preview_data.iterrows():
            print(f"  {row['Player']:25} | {row['Season']:12}")
    else:
        # Utiliser les indices si les colonnes ne sont pas renommées
        player_col = sorted_df.columns[1]
        season_col = sorted_df.columns[3]
        preview_data = sorted_df[[player_col, season_col]].head(10)
        for idx, row in preview_data.iterrows():
            print(f"  {row[player_col]:25} | {row[season_col]:12}")
    
    return sorted_df

# Exécuter la fonction
if __name__ == "__main__":
    print("🚀 DÉBUT DE LA FUSION DES FICHIERS PL_DEF")
    print("=" * 60)
    final_data = merge_def_files()
    
    if final_data is not None:
        print(f"\n🎉 OPÉRATION RÉUSSIE !")
        print(f"📁 Le fichier PL_DEF.csv a été créé sur votre Bureau")
        print(f"📊 Il contient {len(final_data)} enregistrements triés par nom et saison")
        print(f"🔧 Transformations appliquées :")
        print(f"   ✓ Colonne 38 : uniquement 2 premières lettres conservées")
        print(f"   ✓ Colonne 39 : supprimée")
    else:
        print("\n💥 ÉCHEC DE L'OPÉRATION")