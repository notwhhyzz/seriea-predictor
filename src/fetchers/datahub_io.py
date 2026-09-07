"""Fetcher for datahub.io (mirrors football-data.co.uk with stable URLs)"""
import pandas as pd
import requests
from io import StringIO
from typing import List, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import yaml


class DataHubIOFetcher:
    """Fetch Serie A data from datahub.io - stable URLs, no rate limits"""
    
    BASE_URL = "https://datahub.io/football/italian-serie-a/_r"
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["datahub_io"]
        self.enabled = self.config["enabled"]
        self.base_url = self.config["base_url"]
    
    def get_available_seasons(self) -> List[str]:
        """Get list of available season files from datapackage.json"""
        try:
            response = requests.get(f"{self.base_url}/datapackage.json", timeout=10)
            if response.status_code == 200:
                pkg = response.json()
                resources = pkg.get('resources', [])
                seasons = []
                for r in resources:
                    name = r.get('name', '')
                    if name.startswith('season-') and name.endswith('.csv'):
                        seasons.append(name.replace('season-', '').replace('.csv', ''))
                return sorted(seasons)
        except Exception as e:
            print(f"Error fetching datapackage: {e}")
        return []
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3)
    )
    def fetch_season(self, season_code: str) -> Optional[pd.DataFrame]:
        """Fetch a single season by code (e.g., '2425', '2324', '9900')"""
        url = f"{self.base_url}/season-{season_code}.csv"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            df = pd.read_csv(StringIO(response.text), encoding='latin-1')
            return df
        except Exception as e:
            print(f"Error fetching season {season_code}: {e}")
            return None
    
    def fetch_all_seasons(self) -> pd.DataFrame:
        """Fetch all available seasons"""
        seasons = self.get_available_seasons()
        print(f"Found {len(seasons)} seasons on datahub.io")
        
        all_dfs = []
        for season in seasons:
            print(f"Fetching season {season}...")
            df = self.fetch_season(season)
            if df is not None and not df.empty:
                # Add season identifier
                df['Season'] = self._season_code_to_name(season)
                all_dfs.append(df)
                print(f"  -> {len(df)} matches")
            else:
                print(f"  -> No data")
            time.sleep(0.5)  # Be nice to the server
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)
    
    def _season_code_to_name(self, code: str) -> str:
        """Convert season code (2425) to name (2024/25)"""
        if len(code) == 4:
            year_start = int(code[:2])
            year_end = int(code[2:])
            # Handle 2-digit years
            if year_start > 50:
                year_start += 1900
            else:
                year_start += 2000
            if year_end > 50:
                year_end += 1900
            else:
                year_end += 2000
            return f"{year_start}/{str(year_end)[-2:]}"
        return code
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same column mapping as football-data.co.uk"""
        col_map = {
            'Div': 'division',
            'Date': 'date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
            'HTHG': 'home_goals_ht',
            'HTAG': 'away_goals_ht',
            'HTR': 'result_ht',
            'Referee': 'referee',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_on_target',
            'AST': 'away_shots_on_target',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HY': 'home_yellow_cards',
            'AY': 'away_yellow_cards',
            'HR': 'home_red_cards',
            'AR': 'away_red_cards',
            'B365H': 'odds_home',
            'B365D': 'odds_draw',
            'B365A': 'odds_away',
            'B365>2.5': 'odds_over_25',
            'B365<2.5': 'odds_under_25',
        }
        
        existing_cols = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.date
        
        if 'result' in df.columns:
            df['result'] = df['result'].map({'H': 'H', 'D': 'D', 'A': 'A'})
        
        df['source'] = 'datahub.io'
        df['status'] = 'FINISHED'
        
        return df


def main():
    fetcher = DataHubIOFetcher()
    df = fetcher.fetch_all_seasons()
    if not df.empty:
        df = fetcher.standardize_columns(df)
        print(f"\nTotal matches: {len(df)}")
        print(f"Seasons: {df['Season'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        from pathlib import Path
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/raw/datahub_io_all.csv", index=False)
        print("Saved to data/raw/datahub_io_all.csv")
    else:
        print("No data fetched!")


if __name__ == "__main__":
    main()