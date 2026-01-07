#!/usr/bin/env python3
"""
Player Valuation Project - Main Entry Point
Samuel Vermeulen - Capstone Project 2025
"""

import sys
import os

# Ajouter le dossier src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 50)
    print("⚽ PLAYER VALUATION PROJECT - PREMIER LEAGUE")
    print("=" * 50)
    print("Environment: player-valuation (Conda)")
    print("Python version: 3.11")
    print("Project structure ready.")
    
    # Test d'import
    try:
        import data_loader
        print("✅ Module data_loader importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Vérifiez que les fichiers sont dans src/")

if __name__ == "__main__":
    main()
