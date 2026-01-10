# Premier League Player Dataset Pipeline (PL-Pipe)

## What it does
`PL-Pipe` is an end-to-end Python ETL pipeline that:
1. Cleans & filters raw FIFA 19-23 CSV files (player attributes + market values)
2. Merges Fbref positional CSV files (DEF, MID, FW, GK) per season 2018-2023
3. Fuzzy-matches players across sources on name, club, age & season
4. Outputs a single, analysis-ready CSV with 15+ features (Value, Height, Weight, Goals, Assists, Minutes, etc.)

## Entry point
Run only `run_all.py` – it orchestrates every step and produces the final file.

## Quick Install
```bash
# 1. Python ≥ 3.8 required
# 2. One-line deps
pip install pandas numpy fuzzywuzzy python-Levenshtein