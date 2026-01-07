import pandas as pd

print("Test de chargement des données...")
print("=" * 50)

# Essayer de charger le fichier
try:
    df = pd.read_csv("data/raw/PL_players_with_new_columns.csv")
    print(f"✅ Fichier chargé avec succès !")
    print(f"   Shape: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # Afficher les premières colonnes
    print(f"\nPremières colonnes:")
    for i, col in enumerate(df.columns[:10]):
        print(f"   {i+1}. {col}")
    
    # Compter le nombre de colonnes
    print(f"\nTotal colonnes: {len(df.columns)}")
    
except FileNotFoundError as e:
    print(f"❌ Erreur: {e}")
    print("Vérifiez que le fichier CSV est dans data/raw/")
except Exception as e:
    print(f"❌ Autre erreur: {e}")

