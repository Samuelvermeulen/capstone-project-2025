
#!/usr/bin/env python3
"""
Predict script for football player valuation model.
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PlayerValuationPredictor:
    def __init__(self, model_path="../models/xgboost_optimized.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_order = None
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def _load_metadata(self):
        try:
            with open("../data/processed/feature_metadata.json", 'r') as f:
                metadata = json.load(f)
            self.feature_order = metadata.get('feature_order', [])
            print(f"✅ Feature metadata loaded ({len(self.feature_order)} features)")
        except Exception as e:
            print(f"⚠️ Using expected feature order: {e}")
            # EXACT order the model was trained with
            self.feature_order = [
                'Age', 'Assists', 'Assists_per_minute_log', 'Goals', 
                'Goals_per_minute_log', 'Height', 'Matchs_played', 
                'Minutes_played', 'Nation_encoded', 'Weight', 
                'club_Arsenal', 'club_Chelsea', 'club_Leicester City', 
                'club_Liverpool', 'club_Manchester City', 'club_Manchester Utd',
                'club_Newcastle Utd', 'club_Other', 'club_Tottenham', 
                'club_West Ham', 'club_Wolves', 'is_Defender', 'is_Forward', 
                'is_Goalkeeper', 'is_Midfielder'
            ]
    
    def prepare_features(self, player_data):
        if isinstance(player_data, dict):
            player_data = pd.DataFrame([player_data])
        
        df = player_data.copy()
        
        # Required numeric columns with defaults
        numeric_defaults = {
            'Age': 25, 'Goals': 0, 'Assists': 0, 'Minutes_played': 0,
            'Height': 180, 'Weight': 75, 'Matchs_played': 30, 'Nation_encoded': 0
        }
        
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        
        # Create derived features
        df['Goals_per_minute'] = df['Goals'] / df['Minutes_played'].replace(0, 1)
        df['Assists_per_minute'] = df['Assists'] / df['Minutes_played'].replace(0, 1)
        df['Goals_per_minute_log'] = np.log1p(df['Goals_per_minute'])
        df['Assists_per_minute_log'] = np.log1p(df['Assists_per_minute'])
        
        # Position encoding
        position_mapping = {
            'DF': 'is_Defender',
            'MF': 'is_Midfielder', 
            'FW': 'is_Forward',
            'GK': 'is_Goalkeeper'
        }
        
        for pos_short, pos_col in position_mapping.items():
            df[pos_col] = 0
            if 'Position' in df.columns:
                df[pos_col] = (df['Position'].astype(str) == pos_short).astype(int)
        
        # Club encoding
        expected_clubs = [
            'club_Arsenal', 'club_Chelsea', 'club_Leicester City',
            'club_Liverpool', 'club_Manchester City', 'club_Manchester Utd',
            'club_Newcastle Utd', 'club_Tottenham', 'club_West Ham', 'club_Wolves'
        ]
        
        for club_col in expected_clubs:
            df[club_col] = 0
        
        df['club_Other'] = 1  # Default to Other
        
        if 'Club' in df.columns:
            for idx, row in df.iterrows():
                club_name = str(row['Club']).strip()
                found = False
                
                for club_col in expected_clubs:
                    expected_club = club_col.replace('club_', '')
                    if club_name.lower() == expected_club.lower():
                        df.at[idx, club_col] = 1
                        df.at[idx, 'club_Other'] = 0
                        found = True
                        break
        
        # CRITICAL: Use the exact feature order
        if self.feature_order:
            for feature in self.feature_order:
                if feature not in df.columns:
                    df[feature] = 0
            
            features_df = df[self.feature_order]
        else:
            features_df = df.select_dtypes(include=[np.number])
        
        return features_df
    
    def predict(self, player_data):
        features_df = self.prepare_features(player_data)
        predictions_log = self.model.predict(features_df)
        return np.expm1(predictions_log)

def main():
    parser = argparse.ArgumentParser(description='Predict football player market value')
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--player', type=str, help='Player name')
    parser.add_argument('--age', type=int, help='Player age')
    parser.add_argument('--position', type=str, choices=['GK', 'DF', 'MF', 'FW'], help='Position')
    parser.add_argument('--goals', type=int, default=0, help='Goals')
    parser.add_argument('--assists', type=int, default=0, help='Assists')
    parser.add_argument('--minutes', type=int, default=0, help='Minutes played')
    parser.add_argument('--club', type=str, help='Club name')
    
    args = parser.parse_args()
    
    predictor = PlayerValuationPredictor()
    
    if args.input:
        try:
            input_df = pd.read_csv(args.input)
            predictions = predictor.predict(input_df)
            
            result_df = input_df.copy()
            result_df['Predicted_Value_€'] = predictions
            result_df['Predicted_Value_£'] = predictions * 0.85
            
            result_df.to_csv(args.output, index=False)
            print(f"✅ Predictions saved to {args.output}")
            print(f"📊 First 3 predictions:")
            print(result_df[['Player', 'Age', 'Position', 'Predicted_Value_€']].head(3).to_string())
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.player:
        player_data = {
            'Player': args.player,
            'Age': args.age or 25,
            'Position': args.position or 'MF',
            'Goals': args.goals,
            'Assists': args.assists,
            'Minutes_played': args.minutes,
            'Club': args.club or 'Unknown'
        }
        
        prediction = predictor.predict(player_data)[0]
        
        print("\n🎯 PLAYER VALUATION PREDICTION:")
        print(f"   Player: {args.player}")
        print(f"   Age: {args.age}")
        print(f"   Position: {args.position}")
        print(f"   Club: {args.club}")
        print(f"\n   Estimated Market Value: €{prediction:,.0f}")
        print(f"   (£{prediction * 0.85:,.0f})")
        
        result_df = pd.DataFrame([{
            'Player': args.player,
            'Predicted_Value_€': prediction,
            'Predicted_Value_£': prediction * 0.85,
            'Timestamp': datetime.now().isoformat()
        }])
        result_df.to_csv('single_prediction.csv', index=False)
        print(f"\n✅ Saved to single_prediction.csv")
    
    else:
        print("❌ Please provide --input or --player")
        parser.print_help()

if __name__ == "__main__":
    main()