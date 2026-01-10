import pandas as pd
import os

def transform_and_merge_datasets():
    """
    Transforme et fusionne tous les datasets PL_*.csv en un seul fichier PL_players.csv
    avec exactement 11 colonnes compactées
    Nation: garder seulement les 3 dernières lettres puis remplacer par noms complets
    """
    desktop_path = os.path.expanduser("~/Desktop")
    output_file = os.path.join(desktop_path, "Dataset_creation", "Merged_Dataset", "PL_players.csv")

    print("🚀 DÉBUT DE LA TRANSFORMATION ET FUSION DES DATASETS")
    print("=" * 60)

    # Dictionnaire de correspondance abréviation -> pays
    country_mapping = {
        # Europe
        'ALB': 'Albania', 'AND': 'Andorra', 'ARM': 'Armenia', 'AUT': 'Austria',
        'BEL': 'Belgium', 'BIH': 'Bosnia and Herzegovina', 'CRO': 'Croatia',
        'CZE': 'Czech Republic', 'DEN': 'Denmark', 'ENG': 'England', 'ESP': 'Spain',
        'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France', 'GER': 'Germany',
        'GRE': 'Greece', 'HUN': 'Hungary', 'IRL': 'Ireland', 'ISL': 'Iceland',
        'ITA': 'Italy', 'KVX': 'Kosovo', 'MKD': 'North Macedonia', 'MNE': 'Montenegro',
        'NED': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'POR': 'Portugal',
        'ROU': 'Romania', 'SCO': 'Scotland', 'SRB': 'Serbia', 'SUI': 'Switzerland',
        'SVK': 'Slovakia', 'SVN': 'Slovenia', 'SWE': 'Sweden', 'TUR': 'Turkey',
        'UKR': 'Ukraine', 'WAL': 'Wales',

        # Afrique
        'ALG': 'Algeria', 'ANG': 'Angola', 'BFA': 'Burkina Faso', 'CIV': 'Ivory Coast',
        'CMR': 'Cameroon', 'COD': 'DR Congo', 'EGY': 'Egypt', 'GAB': 'Gabon',
        'GHA': 'Ghana', 'GNB': 'Guinea-Bissau', 'GUI': 'Guinea', 'MAR': 'Morocco',
        'MLI': 'Mali', 'MTN': 'Mauritania', 'NGA': 'Nigeria', 'SEN': 'Senegal',
        'TUN': 'Tunisia', 'ZAM': 'Zambia', 'ZIM': 'Zimbabwe',

        # Amériques
        'ARG': 'Argentina', 'BOL': 'Bolivia', 'BRA': 'Brazil', 'CAN': 'Canada',
        'CHI': 'Chile', 'COL': 'Colombia', 'CUB': 'Cuba', 'DOM': 'Dominican Republic',
        'ECU': 'Ecuador', 'HAI': 'Haiti', 'JAM': 'Jamaica', 'MEX': 'Mexico',
        'PAR': 'Paraguay', 'PER': 'Peru', 'USA': 'United States', 'URU': 'Uruguay',
        'VEN': 'Venezuela',

        # Asie
        'IRN': 'Iran', 'JPN': 'Japan', 'KOR': 'South Korea', 'PHI': 'Philippines',
        'RSA': 'South Africa',

        # Océanie
        'AUS': 'Australia', 'NZL': 'New Zealand',

        # Autres
        'BAN': 'Bangladesh', 'GUA': 'Guatemala', 'ISR': 'Israel', 'KEN': 'Kenya',
        'NIR': 'Northern Ireland', 'SKN': 'Saint Kitts and Nevis', 'TAN': 'Tanzania',
        'TOG': 'Togo'
    }

    all_transformed_data = []

    datasets_specs = {
        'PL_strickers.csv': {'position_col_index': 28},
        'PL_GK.csv': {'position_col_index': 30},
        'PL_MID.csv': {'position_col_index': 40},
        'PL_DEF.csv': {'position_col_index': 37},
    }

    for dataset_file, specs in datasets_specs.items():
        file_path = os.path.join(desktop_path, "Dataset_creation", "Merged_Dataset", dataset_file)
        print(f"\n📖 Traitement de {dataset_file}...")

        if not os.path.exists(file_path):
            print(f"   ❌ Fichier {dataset_file} non trouvé")
            continue

        try:
            df = pd.read_csv(file_path)
            print(f"   ✅ Fichier lu - {len(df)} lignes, {len(df.columns)} colonnes")

            required_columns = max(6, specs['position_col_index'] + 1, 16)
            if len(df.columns) < required_columns:
                print(f"   ❌ Pas assez de colonnes (nécessaire: {required_columns})")
                continue

            transformed_rows = []

            for _, row in df.iterrows():
                new_row = {}

                new_row['Player'] = row[df.columns[1]] if len(df.columns) > 1 else ''
                new_row['Season'] = row[df.columns[3]] if len(df.columns) > 3 else ''
                new_row['Age'] = row[df.columns[4]] if len(df.columns) > 4 else ''

                nation_value = row[df.columns[5]] if len(df.columns) > 5 else ''
                if pd.notna(nation_value) and nation_value != '':
                    nation_abbr = str(nation_value)[-3:]
                    new_row['Nation'] = country_mapping.get(nation_abbr, nation_abbr)
                else:
                    new_row['Nation'] = nation_value

                new_row['Club'] = row[df.columns[6]] if len(df.columns) > 6 else ''

                playing_time_cols = [col for col in df.columns if 'Playing Time' in str(col)]
                new_row['Matchs_played'] = row[playing_time_cols[0]] if len(playing_time_cols) >= 1 else ''
                new_row['Minutes_played'] = row[playing_time_cols[1]] if len(playing_time_cols) >= 2 else ''
                new_row['Titular_played'] = row[playing_time_cols[3]] if len(playing_time_cols) >= 4 else ''

                performance_cols = [col for col in df.columns if 'Performance' in str(col)]
                new_row['Goals'] = row[performance_cols[0]] if len(performance_cols) >= 1 else ''
                new_row['Assists'] = row[performance_cols[1]] if len(performance_cols) >= 2 else ''

                new_row['Position'] = row[df.columns[specs['position_col_index']]] if len(df.columns) > specs['position_col_index'] else ''

                transformed_rows.append(new_row)

            transformed_df = pd.DataFrame(transformed_rows)
            print(f"   🔧 Transformé - {len(transformed_df)} lignes, {len(transformed_df.columns)} colonnes")
            all_transformed_data.append(transformed_df)

        except Exception as e:
            print(f"   ❌ Erreur avec {dataset_file}: {e}")
            continue

    if all_transformed_data:
        print(f"\n🔄 Fusion des {len(all_transformed_data)} datasets...")
        final_dataset = pd.concat(all_transformed_data, ignore_index=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_dataset.to_csv(output_file, index=False, encoding='utf-8')

        print("\n" + "=" * 50)
        print("✅ TRANSFORMATION ET FUSION TERMINÉES AVEC SUCCÈS")
        print("=" * 50)
        print(f"💾 Fichier sauvegardé: {output_file}")
        print(f"📊 Dataset final: {len(final_dataset)} joueurs")
        print(f"🎯 Colonnes finales (11): Player, Season, Age, Nation, Club, Matchs_played, Minutes_played, Titular_played, Goals, Assists, Position")

        print(f"\n🔍 APERÇU DES DONNÉES (5 premières lignes):")
        print(final_dataset.head())

        return final_dataset
    else:
        print("\n❌ AUCUN DATASET VALIDE À FUSIONNER")
        return None

if __name__ == "__main__":
    final_data = transform_and_merge_datasets()
    if final_data is not None:
        print(f"\n🎉 OPÉRATION RÉUSSIE !")
        print(f"📁 Le fichier PL_players.csv est prêt dans : ~/Desktop/Dataset_creation/Merged_Dataset/")
    else:
        print("\n💥 ÉCHEC DE L'OPÉRATION")