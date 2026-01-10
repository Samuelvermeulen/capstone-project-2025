
import pandas as pd
import os

# Chemin vers le fichier FIFA 21 sur le Bureau
file_path = os.path.join(os.path.expanduser("~"), "Desktop", "players_21.csv")

try:
    # Ouvrir le fichier FIFA 21
    df = pd.read_csv(file_path)
    
    print("🔍 Ouverture du fichier FIFA 21 depuis le Bureau...")
    print(f"Dimensions initiales : {df.shape}")
    
    # Identifier la colonne des clubs
    club_column = None
    club_keywords = ['club', 'team', 'squad']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in club_keywords):
            club_column = col
            break
    
    if not club_column:
        print("❌ Colonne des clubs non trouvée automatiquement")
        print("Colonnes disponibles :")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        club_col_index = int(input("\nEntrez le numéro de la colonne des clubs : "))
        club_column = df.columns[club_col_index]

    print(f"🏆 Colonne des clubs utilisée : '{club_column}'")
    
    # Liste des clubs Premier League 2020-2021
    premier_league_clubs_2021 = [
        'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea',
        'Leicester City', 'West Ham United', 'Tottenham Hotspur', 'Arsenal',
        'Leeds United', 'Everton', 'Aston Villa', 'Newcastle United',
        'Wolverhampton Wanderers', 'Crystal Palace', 'Southampton',
        'Brighton & Hove Albion', 'Burnley', 'Fulham', 'West Bromwich Albion',
        'Sheffield United', 'Spurs', 'Man City', 'Man United', 'Newcastle',
        'West Ham', 'Brighton', 'Wolves', 'Leeds'
    ]
    
    # Filtrer les joueurs de Premier League
    premier_league_df = df[df[club_column].isin(premier_league_clubs_2021)]

    print(f"\n📊 Résultats du filtrage FIFA 21 :")
    print(f"Joueurs totaux dans FIFA 21 : {len(df)}")
    print(f"Joueurs Premier League 2020-2021 : {len(premier_league_df)}")
    
    # Afficher les clubs trouvés
    clubs_found = premier_league_df[club_column].unique()
    print(f"\n🏆 Clubs Premier League trouvés ({len(clubs_found)}) :")
    for club in sorted(clubs_found):
        count = len(premier_league_df[premier_league_df[club_column] == club])
        print(f"  {club} : {count} joueurs")
    
    # COLONNES À GARDER (UNIQUEMENT celles que vous voulez)
    colonnes_a_garder = [
        'long_name', 
        'age', 
        'height_cm',  # Sera renommé en Height
        'weight_kg',  # Sera renommé en Weight
        'nationality_name', 
        club_column,  # La colonne des clubs qu'on a identifiée
        'value_eur',  # Sera renommé en Value
        'preferred_foot'
    ]
    
    # Vérifier quelles colonnes existent dans le dataset
    colonnes_existantes = []
    colonnes_manquantes = []
    
    for col in colonnes_a_garder:
        if col in premier_league_df.columns:
            colonnes_existantes.append(col)
        else:
            colonnes_manquantes.append(col)
    
    print(f"\n📋 Colonnes à garder :")
    print(f"Colonnes trouvées : {len(colonnes_existantes)}/{len(colonnes_a_garder)}")
    
    if colonnes_manquantes:
        print(f"Colonnes manquantes : {colonnes_manquantes}")
        print("Recherche d'alternatives...")
        
        # Recherche d'alternatives pour les colonnes manquantes
        alternatives = {
            'nationality_name': ['nationality', 'nation_name'],
            'value_eur': ['value', 'eur_value'],
            'long_name': ['name', 'player_name', 'short_name']
        }
        
        for col_manquante in colonnes_manquantes[:]:
            if col_manquante in alternatives:
                for alternative in alternatives[col_manquante]:
                    if alternative in premier_league_df.columns:
                        colonnes_existantes.append(alternative)
                        colonnes_a_garder[colonnes_a_garder.index(col_manquante)] = alternative
                        colonnes_manquantes.remove(col_manquante)
                        print(f"  ✅ Alternative trouvée : '{col_manquante}' → '{alternative}'")
                        break
    
    # GARDER UNIQUEMENT les colonnes spécifiées
    premier_league_df = premier_league_df[colonnes_existantes]
    
    print(f"\n✅ Colonnes conservées ({len(colonnes_existantes)}) :")
    for col in colonnes_existantes:
        print(f"  {col}")
    
    # RENOMMER les colonnes spécifiques
    print(f"\n🔄 Renommage des colonnes :")
    
    renommage_colonnes = {
        'height_cm': 'Height',
        'weight_kg': 'Weight', 
        'value_eur': 'Value',
        club_column: 'club_name'
    }
    
    for ancien_nom, nouveau_nom in renommage_colonnes.items():
        if ancien_nom in premier_league_df.columns:
            premier_league_df = premier_league_df.rename(columns={ancien_nom: nouveau_nom})
            print(f"  ✅ {ancien_nom} renommé en {nouveau_nom}")
    
    print(f"\n📐 Dimensions finales : {premier_league_df.shape}")
    
    # Afficher les colonnes finales
    print(f"\n📋 COLONNES FINALES ({len(premier_league_df.columns)}) :")
    for i, col in enumerate(premier_league_df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Aperçu des données finales
    print(f"\n👀 Aperçu des 5 premières lignes :")
    print(premier_league_df.head())
    
    # Statistiques finales
    print(f"\n📊 STATISTIQUES FINALES :")
    print(f"Nombre total de joueurs : {len(premier_league_df)}")
    print(f"Âge moyen : {premier_league_df['age'].mean():.1f} ans")
    print(f"Valeur moyenne : €{premier_league_df['Value'].mean():,.0f}")
    print(f"Distribution des pieds préférés :")
    print(premier_league_df['preferred_foot'].value_counts())

    # Value adjusted - CORRECTION DE L'INDENTATION
    def ajuster_valeur(valeur_originale, age, position):
        # Facteur de base
        base = valeur_originale * 0.85
        
        # Facteur position
        facteurs_position = {
            'Attaquant': 0.82,
            'Milieu': 0.87, 
            'Défenseur': 0.90,
            'Gardien': 0.92
        }
        
        # Facteur âge
        if age < 21:
            facteur_age = 1.10
        elif age > 33:
            facteur_age = 0.70
        else:
            facteur_age = 1.0
        
        return base * facteurs_position[position] * facteur_age

    # Appliquer l'ajustement des valeurs
    print(f"\n🔄 Application de l'ajustement des valeurs...")
    
    # Pour l'instant, on applique un ajustement simple sans la position
    # (vous devrez identifier la colonne des positions plus tard)
    premier_league_df['Value_Adjusted'] = premier_league_df.apply(
        lambda row: ajuster_valeur(row['Value'], row['age'], 'Milieu'),  # Position par défaut
        axis=1
    )
    
    print(f"💰 Valeurs ajustées appliquées avec succès")
    print(f"Valeur moyenne originale : €{premier_league_df['Value'].mean():,.0f}")
    print(f"Valeur moyenne ajustée : €{premier_league_df['Value_Adjusted'].mean():,.0f}")

    # Sauvegarder sur le Bureau
    output_path = os.path.join(os.path.expanduser("~"), "Desktop", "premier_league_fifa21_essential.csv")
    premier_league_df.to_csv(output_path, index=False)
    print(f"\n✅ Fichier FIFA 21 sauvegardé : {output_path}")
    print(f"📊 Données finales : {premier_league_df.shape[0]} joueurs, {premier_league_df.shape[1]} colonnes")
    
except FileNotFoundError:
    print(f"❌ Fichier non trouvé : {file_path}")
    print("Vérifiez que le fichier 'players_21.csv' est bien sur votre Bureau")
except Exception as e:
    print(f"❌ Erreur lors du traitement : {e}")