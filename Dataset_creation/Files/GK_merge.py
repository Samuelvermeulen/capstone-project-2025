
import pandas as pd
import os
import glob

def merge_gk_files():
    """
    Fusionne tous les fichiers PL_GK_*.csv du dossier 'Projet python data'
    Simple fusion et tri par nom et saison, sans transformations
    """
    # Chemin vers le dossier sur le Bureau
    desktop_path = os.path.expanduser("~/Desktop")  # Chemin du bureau
    project_folder = os.path.join(desktop_path, "Dataset_creation/Fbref_Dataset")
    
    print(f"📁 Recherche dans : {project_folder}")
    
    # Vérifier si le dossier existe
    if not os.path.exists(project_folder):
        print(f"❌ Le dossier '{project_folder}' n'existe pas")
        return None
    
    # Trouver les fichiers PL_GK_*.csv
    gk_files = glob.glob(os.path.join(project_folder, "PL_GK_*.csv"))
    gk_files.sort()  # Trier les fichiers
    
    print(f"🎯 Fichiers PL_GK trouvés : {[os.path.basename(f) for f in gk_files]}")
    
    if not gk_files:
        print("❌ Aucun fichier PL_GK_*.csv trouvé")
        return None
    
    # ÉTAPE 1 : METTRE TOUS LES FICHIERS À LA SUITE
    print("\n" + "="*50)
    print("ÉTAPE 1 : MISE À LA SUITE DES FICHIERS")
    print("="*50)
    
    all_data = []    
    for file in gk_files:
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
    
    # ÉTAPE 2 : TRI PAR NOM ET SAISON
    print("\n" + "="*50)
    print("ÉTAPE 2 : TRI PAR NOM ET SAISON")
    print("="*50)
    
    # Vérifier que nous pouvons accéder aux colonnes par index
    print("🔍 Vérification de l'accès aux colonnes...")
    
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
    output_path = os.path.join(desktop_path, "Dataset_creation", "Merged_Dataset", "PL_GK.csv")
    sorted_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # RAPPORT FINAL
    print("\n" + "="*50)
    print("RAPPORT FINAL")
    print("="*50)
    print(f"✅ Fusion et tri terminés avec succès")
    print(f"💾 Fichier sauvegardé : {output_path}")
    print(f"📈 Total des enregistrements : {len(sorted_df)}")
    print(f"🎯 Fichiers fusionnés : {len(gk_files)}")
    print(f"🔧 Aucune transformation appliquée - données originales conservées")
    
    # Aperçu du résultat trié
    print("\n🔍 APERÇU DU RÉSULTAT TRIÉ :")
    print("Les 10 premières lignes :")
    print("-" * 60)
    
    # Afficher les données en utilisant les bonnes colonnes
    if 'Player' in sorted_df.columns and 'Season' in sorted_df.columns:
        preview_data = sorted_df[['Player', 'Season']].head(10)
        for idx, row in preview_data.iterrows():
            print(f"  {row['Player']:25} | {row['Season']:12}")
    
    return sorted_df

# Exécuter la fonction
if __name__ == "__main__":
    print("🚀 DÉBUT DE LA FUSION DES FICHIERS PL_GK")
    print("=" * 60)
    final_data = merge_gk_files()
    
    if final_data is not None:
        print(f"\n🎉 OPÉRATION RÉUSSIE !")
        print(f"📁 Le fichier PL_GK.csv a été créé sur votre Bureau")
        print(f"📊 Il contient {len(final_data)} enregistrements triés par nom et saison")
        print(f"🔧 Données originales conservées - aucune transformation")
    else:
        print("\n💥 ÉCHEC DE L'OPÉRATION")