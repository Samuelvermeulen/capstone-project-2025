import os
import sys
import subprocess
import shutil
from pathlib import Path

# 🔧 Configuration des chemins
BASE_DIR = Path(__file__).parent.parent  # dossier parent de Files/
FILES_DIR = BASE_DIR / "Files"
FBREF_DIR = BASE_DIR / "Fbref_Dataset"
FIFA_DIR = BASE_DIR / "FIFA_Dataset"
MERGED_DIR = BASE_DIR / "Merged_Dataset"

# Créer le dossier Merged_Dataset s'il n'existe pas
MERGED_DIR.mkdir(exist_ok=True)

# ✅ Liste des fichiers requis
REQUIRED_FIFA = [
    "players_19.csv", "players_20.csv", "players_21.csv",
    "FIFA22_official_data.csv", "FIFA23_official_data.csv"
]

REQUIRED_FBREF = [
    f"PL_DEF_{i}.csv" for i in range(1, 6)
] + [
    f"PL_MID_{i}.csv" for i in range(1, 8)
] + [
    f"PL_GK_{i}.csv" for i in range(1, 3)
] + [
    f"PL_strickers_{i}.csv" for i in range(1, 6)
]

# ✅ Vérification des fichiers
def check_files(folder, files, label):
    missing = [f for f in files if not (folder / f).exists()]
    if missing:
        print(f"❌ Fichiers {label} manquants :")
        for f in missing:
            print(f"   - {f}")
        sys.exit(1)
    print(f"✅ Tous les fichiers {label} sont présents.")

# ✅ Exécution d’un script Python
def run_script(script_name, cwd=None):
    print(f"\n🔄 Exécution de {script_name}...")
    result = subprocess.run([sys.executable, script_name], cwd=cwd or FILES_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erreur dans {script_name} :")
        print(result.stderr)
        sys.exit(1)
    print(f"✅ {script_name} terminé avec succès.")

# ✅ Déplacer un fichier généré
def move_output(filename, dest=MERGED_DIR):
    src = FILES_DIR / filename
    if src.exists():
        shutil.move(src, dest / filename)
        print(f"📁 Déplacé : {filename} → {dest}")
    else:
        print(f"⚠️  Fichier non trouvé : {filename}")

# ✅ MAIN
def main():
    print("🚀 Lancement du pipeline complet...\n")

    # 1. Vérification des fichiers
    check_files(FIFA_DIR, REQUIRED_FIFA, "FIFA")
    check_files(FBREF_DIR, REQUIRED_FBREF, "Fbref")

    # 2. Traitement FIFA
    fifa_scripts = ["FIFA_19.py", "FIFA_20.py", "FIFA_21.py", "FIFA_22.py", "FIFA_23.py"]
    for script in fifa_scripts:
        run_script(script)

    run_script("FIFA_final.py")
    move_output("premier_league_merged_2018-2023.csv")

    # 3. Traitement Fbref
    fbref_scripts = ["DEF_merge.py", "MF_merge.py", "GK_merge.py", "FW_merge.py"]
    for script in fbref_scripts:
        run_script(script)

    run_script("PL_final.py")
    move_output("PL_players.csv")

    # 4. Fusion finale
    run_script("Final_merge.py")
    move_output("PL_players_with_new_columns.csv")

    print("\n🎉 Pipeline terminé avec succès !")
    print(f"📊 Fichier final disponible dans : {MERGED_DIR / 'PL_players_with_new_columns.csv'}")

if __name__ == "__main__":
    main()
