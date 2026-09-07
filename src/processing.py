"""Data processing: merging sources, feature engineering, team mapping"""
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from src.database import Team, Match, get_session, init_db
import yaml
from difflib import get_close_matches
import re


class TeamNameMapper:
    """Map team names across different data sources"""
    
    # Known mappings from various sources to canonical names
    CANONICAL_NAMES = {
        # football-data.co.uk names -> canonical
        "Juventus": "Juventus",
        "Turin": "Torino",
        "Inter": "Inter",
        "AC Milan": "Milan",
        "Roma": "Roma",
        "Lazio": "Lazio",
        "Napoli": "Napoli",
        "Fiorentina": "Fiorentina",
        "Atalanta": "Atalanta",
        "Bologna": "Bologna",
        "Sassuolo": "Sassuolo",
        "Udinese": "Udinese",
        "Sampdoria": "Sampdoria",
        "Genoa": "Genoa",
        "Cagliari": "Cagliari",
        "Verona": "Hellas Verona",
        "Empoli": "Empoli",
        "Spezia": "Spezia",
        "Venezia": "Venezia",
        "Salernitana": "Salernitana",
        "Monza": "Monza",
        "Lecce": "Lecce",
        "Cremonese": "Cremonese",
        "Parma": "Parma",
        "Como": "Como",
        "Brescia": "Brescia",
        "Pescara": "Pescara",
        "Frosinone": "Frosinone",
        "Benevento": "Benevento",
        "Crotone": "Crotone",
        "Carpi": "Carpi",
        "Palermo": "Palermo",
        "Catania": "Catania",
        "Livorno": "Livorno",
        "Siena": "Siena",
        "Chievo": "Chievo",
        "Atalanta": "Atalanta",
    }
    
    # Reverse mapping for API-Football
    API_FOOTBALL_MAPPING = {
        "Juventus FC": "Juventus",
        "Torino FC": "Torino",
        "FC Internazionale Milano": "Inter",
        "AC Milan": "Milan",
        "AS Roma": "Roma",
        "Lazio": "Lazio",
        "SSC Napoli": "Napoli",
        "ACF Fiorentina": "Fiorentina",
        "Atalanta BC": "Atalanta",
        "Bologna FC 1909": "Bologna",
        "US Sassuolo Calcio": "Sassuolo",
        "Udinese Calcio": "Udinese",
        "Sampdoria": "Sampdoria",
        "Genoa CFC": "Genoa",
        "Cagliari Calcio": "Cagliari",
        "Hellas Verona": "Hellas Verona",
        "Empoli": "Empoli",
        "Spezia": "Spezia",
        "Venezia FC": "Venezia",
        "Salernitana": "Salernitana",
        "AC Monza": "Monza",
        "US Lecce": "Lecce",
        "Cremonese": "Cremonese",
        "Parma Calcio 1913": "Parma",
        "Como 1907": "Como",
        "Brescia": "Brescia",
        "Pescara": "Pescara",
        "Frosinone Calcio": "Frosinone",
        "Benevento": "Benevento",
        "Crotone": "Crotone",
        "Carpi": "Carpi",
        "Palermo": "Palermo",
        "Catania": "Catania",
        "Livorno": "Livorno",
        "Siena": "Siena",
        "Chievo": "Chievo",
    }
    
    # football-data.org mapping
    FOOTBALL_DATA_ORG_MAPPING = {        "Juventus FC": "Juventus",
        "Torino FC": "Torino",
        "Inter": "Inter",
        "FC Internazionale Milano": "Inter",
        "AC Milan": "Milan",
        "Milan": "Milan",
        "AS Roma": "Roma",
        "Roma": "Roma",
        "Lazio": "Lazio",
        "SS Lazio": "Lazio",
        "Napoli": "Napoli",
        "SSC Napoli": "Napoli",
        "Fiorentina": "Fiorentina",
        "ACF Fiorentina": "Fiorentina",
        "Atalanta": "Atalanta",
        "Atalanta BC": "Atalanta",
        "Bologna": "Bologna",
        "Bologna FC 1909": "Bologna",
        "Sassuolo": "Sassuolo",
        "US Sassuolo Calcio": "Sassuolo",
        "Udinese": "Udinese",
        "Udinese Calcio": "Udinese",
        "Sampdoria": "Sampdoria",
        "Genoa": "Genoa",
        "Genoa CFC": "Genoa",
        "Cagliari": "Cagliari",
        "Cagliari Calcio": "Cagliari",
        "Hellas Verona": "Hellas Verona",
        "Verona": "Hellas Verona",
        "Empoli": "Empoli",
        "Spezia": "Spezia",
        "Venezia": "Venezia",
        "Venezia FC": "Venezia",
        "Salernitana": "Salernitana",
        "Monza": "Monza",
        "AC Monza": "Monza",
        "Lecce": "Lecce",
        "US Lecce": "Lecce",
        "Cremonese": "Cremonese",
        "Parma": "Parma",
        "Parma Calcio 1913": "Parma",
        "Como": "Como",
        "Como 1907": "Como",
        "Frosinone": "Frosinone",
        "Frosinone Calcio": "Frosinone",
        "Torino FC": "Torino",
        "Torino": "Torino",
        "Juventus FC": "Juventus",
        "Juventus": "Juventus",
        "AC Milan": "Milan",
    }

    # oddspapi.io mapping (Gamdom odds feed)
    ODDSPAPI_MAPPING = {
        "AC Milan": "Milan",
        "AC Monza": "Monza",
        "ACF Fiorentina": "Fiorentina",
        "AS Roma": "Roma",
        "Atalanta BC": "Atalanta",
        "Bologna FC": "Bologna",
        "Cagliari Calcio": "Cagliari",
        "Como 1907": "Como",
        "Frosinone Calcio": "Frosinone",
        "Genoa CFC": "Genoa",
        "Inter Milano": "Inter",
        "Juventus Turin": "Juventus",
        "Lazio Rome": "Lazio",
        "Parma Calcio": "Parma",
        "SSC Napoli": "Napoli",
        "Sassuolo Calcio": "Sassuolo",
        "Torino FC": "Torino",
        "US Lecce": "Lecce",
        "Udinese Calcio": "Udinese",
        "Venezia FC": "Venezia",
    }
    
    def __init__(self):
        self.canonical_to_sources = {}
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self):
        for source_name, canonical in self.CANONICAL_NAMES.items():
            if canonical not in self.canonical_to_sources:
                self.canonical_to_sources[canonical] = []
            self.canonical_to_sources[canonical].append(source_name)
    
    def map_to_canonical(self, name: str, source: str = "football-data.co.uk") -> str:
        """Map a team name from a source to canonical name"""
        if source == "football-data.co.uk" or source == "datahub.io":
            return self.CANONICAL_NAMES.get(name, name)
        elif source == "api-football":
            return self.API_FOOTBALL_MAPPING.get(name, name)
        elif source == "football-data.org":
            return self.FOOTBALL_DATA_ORG_MAPPING.get(name, name)
        elif source == "oddspapi":
            return self.ODDSPAPI_MAPPING.get(name, name)
        else:
            # Fuzzy match
            matches = get_close_matches(name, self.CANONICAL_NAMES.values(), n=1, cutoff=0.8)
            return matches[0] if matches else name
    
    def get_all_names(self, canonical: str) -> List[str]:
        """Get all known variants of a canonical name"""
        return self.canonical_to_sources.get(canonical, [canonical])


class DataProcessor:
    """Process and merge data from multiple sources"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.mapper = TeamNameMapper()
        self.session = get_session(config_path)
    
    def load_raw_data(self, source: str) -> pd.DataFrame:
        """Load raw CSV from data/raw/"""
        from pathlib import Path
        file_map = {
            "football-data.co.uk": "data/raw/football_data_co_uk_all.csv",
            "datahub.io": "data/raw/datahub_io_all.csv",
            "fbref_schedule": "data/raw/fbref_schedule_2024_25.csv",
        }
        path = Path(file_map.get(source, ""))
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    
    def standardize_historical_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize historical match data from football-data.co.uk/datahub"""
        if df.empty:
            return df
        
        # Map team names
        df['home_team_canonical'] = df['home_team'].apply(
            lambda x: self.mapper.map_to_canonical(x, "football-data.co.uk")
        )
        df['away_team_canonical'] = df['away_team'].apply(
            lambda x: self.mapper.map_to_canonical(x, "football-data.co.uk")
        )
        
        # Ensure required columns exist
        required = ['date', 'home_team_canonical', 'away_team_canonical', 
                    'home_goals', 'away_goals', 'result', 'season']
        for col in required:
            if col not in df.columns:
                df[col] = None
        
        # Calculate result if missing
        if 'result' not in df.columns or df['result'].isna().all():
            df['result'] = np.where(
                df['home_goals'] > df['away_goals'], 'H',
                np.where(df['home_goals'] < df['away_goals'], 'A', 'D')
            )
        
        # Add derived features
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        df['over_25'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        
        return df
    
    def merge_fbref_schedule(self, hist_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:
        """Merge FBref schedule (with xG) into historical data"""
        if fbref_df.empty:
            return hist_df
        
        # FBref columns to merge
        fbref_cols = ['date', 'home_team', 'away_team', 'home_xg', 'away_xg', 
                      'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
        
        # Standardize FBref team names
        fbref_df['home_team_canonical'] = fbref_df['home_team'].apply(
            lambda x: self.mapper.map_to_canonical(x, "fbref")
        )
        fbref_df['away_team_canonical'] = fbref_df['away_team'].apply(
            lambda x: self.mapper.map_to_canonical(x, "fbref")
        )
        
        # Merge on date + teams
        merged = hist_df.merge(
            fbref_df[fbref_cols + ['home_team_canonical', 'away_team_canonical']],
            on=['date', 'home_team_canonical', 'away_team_canonical'],
            how='left',
            suffixes=('', '_fbref')
        )
        
        return merged
    
    def load_into_database(self, df: pd.DataFrame):
        """Load processed matches into SQLite database"""
        from src.database import Team, Match
        from datetime import date
        
        # Get or create teams
        team_cache = {}
        for _, row in df.iterrows():
            for team_col in ['home_team_canonical', 'away_team_canonical']:
                team_name = row[team_col]
                if team_name and team_name not in team_cache:
                    team = self.session.query(Team).filter(Team.name == team_name).first()
                    if not team:
                        team = Team(name=team_name)
                        self.session.add(team)
                        self.session.flush()
                    team_cache[team_name] = team.id
        
        # Insert matches
        for _, row in df.iterrows():
            home_id = team_cache.get(row['home_team_canonical'])
            away_id = team_cache.get(row['away_team_canonical'])
            
            if not home_id or not away_id:
                continue
            
            # Check if match exists
            existing = self.session.query(Match).filter(
                Match.season == row['season'],
                Match.home_team_id == home_id,
                Match.away_team_id == away_id
            ).first()
            
            if existing:
                # Update with new data
                for col in ['home_goals', 'away_goals', 'home_xg', 'away_xg',
                           'home_shots', 'away_shots', 'odds_home', 'odds_draw', 'odds_away']:
                    if col in row and pd.notna(row[col]):
                        setattr(existing, col, row[col])
            else:
                match = Match(
                    season=row['season'],
                    date=row['date'],
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals=row.get('home_goals'),
                    away_goals=row.get('away_goals'),
                    home_goals_ht=row.get('home_goals_ht'),
                    away_goals_ht=row.get('away_goals_ht'),
                    result=row.get('result'),
                    home_xg=row.get('home_xg'),
                    away_xg=row.get('away_xg'),
                    home_shots=row.get('home_shots'),
                    away_shots=row.get('away_shots'),
                    home_shots_on_target=row.get('home_shots_on_target'),
                    away_shots_on_target=row.get('away_shots_on_target'),
                    home_corners=row.get('home_corners'),
                    away_corners=row.get('away_corners'),
                    odds_home=row.get('odds_home'),
                    odds_draw=row.get('odds_draw'),
                    odds_away=row.get('odds_away'),
                    odds_over_25=row.get('odds_over_25'),
                    odds_under_25=row.get('odds_under_25'),
                    source=row.get('source', 'merged'),
                    status='FINISHED' if pd.notna(row.get('home_goals')) else 'SCHEDULED',
                )
                self.session.add(match)
        
        self.session.commit()
        print(f"Loaded/updated {len(df)} matches in database")
    
    def get_upcoming_matches(self, days: int = 14) -> pd.DataFrame:
        """Get upcoming matches from database"""
        from datetime import date, timedelta
        from sqlalchemy import and_
        
        cutoff = date.today() + timedelta(days=days)
        
        matches = self.session.query(Match).filter(
            and_(
                Match.date >= date.today(),
                Match.date <= cutoff,
                Match.status.in_(['SCHEDULED', 'TIMED'])
            )
        ).order_by(Match.date).all()
        
        data = []
        for m in matches:
            data.append({
                'id': m.id,
                'date': m.date,
                'home_team': m.home_team.name,
                'away_team': m.away_team.name,
                'season': m.season,
            })
        
        return pd.DataFrame(data)
    
    def get_team_history(self, team_name: str, n_matches: int = 10, before_date: date = None) -> pd.DataFrame:
        """Get recent match history for a team"""
        from sqlalchemy import or_, and_
        
        team = self.session.query(Team).filter(Team.name == team_name).first()
        if not team:
            return pd.DataFrame()
        
        query = self.session.query(Match).filter(
            or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
            Match.status == 'FINISHED'
        )
        
        if before_date:
            query = query.filter(Match.date < before_date)
        
        matches = query.order_by(Match.date.desc()).limit(n_matches).all()
        
        data = []
        for m in matches:
            is_home = m.home_team_id == team.id
            data.append({
                'date': m.date,
                'is_home': is_home,
                'opponent': m.away_team.name if is_home else m.home_team.name,
                'goals_for': m.home_goals if is_home else m.away_goals,
                'goals_against': m.away_goals if is_home else m.home_goals,
                'xg_for': m.home_xg if is_home else m.away_xg,
                'xg_against': m.away_xg if is_home else m.home_xg,
                'result': 'W' if (is_home and m.result == 'H') or (not is_home and m.result == 'A') 
                          else 'L' if (is_home and m.result == 'A') or (not is_home and m.result == 'H') else 'D',
            })
        
        return pd.DataFrame(data)
    
    def close(self):
        self.session.close()


def main():
    processor = DataProcessor()
    
    # Load historical data
    print("Loading historical data...")
    hist_df = processor.load_raw_data("football-data.co.uk")
    if hist_df.empty:
        hist_df = processor.load_raw_data("datahub.io")
    
    if not hist_df.empty:
        print(f"Loaded {len(hist_df)} historical matches")
        hist_df = processor.standardize_historical_matches(hist_df)
        
        # Try to merge FBref data
        fbref_df = processor.load_raw_data("fbref_schedule")
        if not fbref_df.empty:
            hist_df = processor.merge_fbref_schedule(hist_df, fbref_df)
        
        # Load into database
        processor.load_into_database(hist_df)
    
    # Show upcoming matches
    upcoming = processor.get_upcoming_matches(14)
    print(f"\nUpcoming matches: {len(upcoming)}")
    print(upcoming.to_string())
    
    processor.close()


if __name__ == "__main__":
    main()