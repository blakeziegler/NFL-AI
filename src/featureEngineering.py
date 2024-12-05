import pandas as pd 
import numpy as np
from typing import Dict, List, Tuple


class NFLFeatureProcessor:
    def __init__(self): 
        self.teams_ratings: Dict[str, float] = {}
        self.league_avg_points = 0

    def process_initial_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        First pass of the feature engineering focusing on the core predictive features
        """

        processed = df.copy()

        # Basic Game info
        processed = self.add_basic_features(processed)

        # Team performance features
        processed = self.add_team_performance(processed)

        # Power ratings
        processed = self._add_power_ratings(processed)

        return processed

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Add fundamental game-level features"""
        
        df['total_points'] = df['score_home'] + df['score_away']
        df['point_differential'] = df['score_home'] - df['score_away']

        df['spread_performance'] = np.where(
            df['team_favorite_id'] == df['team_home'],
            df['point_differential'] - df['spread_favorite'],
            -df['point_differential'] - df['spread_favorite']
        )

        df['over_under_performance'] = df['total_points'] - df['over_under_line']






    