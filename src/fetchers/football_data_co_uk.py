"""Fetcher for football-data.co.uk CSV data"""
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import yaml


class FootballDataCoUKFetcher:
    """Fetch historical Serie A data from football-data.co.uk"""
    
    # Season code mapping
    SEASON_CODES = {
        "2024/25": "2425",
        "2023/24": "2324",
        "2022/23": "2223",
        "2021/22": "2122",
        "2020/21": "2021",
        "2019/20": "1920",
        "2018/19": "1819",
        "2017/18": "1718",
        "2016/17": "1617",
        "2015/16": "1516",
        "2014/15": "1415",
        "2013/14": "1314",
        "2012/13": "1213",
        "2011/12": "1112",
        "2010/11": "1011",
        "2009/10": "0910",
        "2008/09": "0809",
        "2007/08": "0708",
        "2006/07": "0607",
        "2005/06": "0506",
        "2004/05": "0405",
        "2003/04": "0304",
        "2002/03": "0203",
        "2001/02": "0102",
        "2000/01": "0001",
        "1999/00": "9900",
        "1998/99": "9899",
        "1997/98": "9798",
        "1996/97": "9697",
        "1995/96": "9596",
        "1994/95": "9495",
        "1993/94": "9394",
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["football_data_co_uk"]
        self.base_url = self.config["base_url"]
        self.rate_limit = self.config["rate_limit_seconds"]
        self.enabled = self.config["enabled"]
        self.seasons = self.config.get("seasons", list(self.SEASON_CODES.values()))
        self._last_request = 0
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _fetch_csv(self, season_code: str) -> Optional[pd.DataFrame]:
        """Fetch a single season's CSV"""
        self._rate_limit_wait()
        
        # URL format: https://www.football-data.co.uk/mmz4281/2425/I1.csv
        url = f"{self.base_url}/{season_code}/I1.csv"
        
        response = requests.get(url, timeout=30)
        self._last_request = time.time()
        
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None
        
        # Parse CSV - handle encoding issues
        try:
            df = pd.read_csv(StringIO(response.text), encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(response.text), encoding='cp1252')
        
        return df
    
    def fetch_all_seasons(self) -> pd.DataFrame:
        """Fetch all configured seasons and combine"""
        all_dfs = []
        
        for season_name, season_code in self.SEASON_CODES.items():
            if season_code not in self.seasons:
                continue
                
            print(f"Fetching {season_name} ({season_code})...")
            df = self._fetch_csv(season_code)
            
            if df is not None and not df.empty:
                df['Season'] = season_name
                all_dfs.append(df)
                print(f"  -> {len(df)} matches")
            else:
                print(f"  -> No data or error")
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined
    
    def fetch_current_season(self) -> pd.DataFrame:
        """Fetch only the current season (updated weekly)"""
        current_season = list(self.SEASON_CODES.keys())[0]
        current_code = self.SEASON_CODES[current_season]
        print(f"Fetching current season {current_season}...")
        return self._fetch_csv(current_code)
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to our schema"""
        # Column mapping from football-data.co.uk to our schema
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
            # Odds (various bookmakers, we'll use average or B365)
            'B365H': 'odds_home',
            'B365D': 'odds_draw',
            'B365A': 'odds_away',
            'B365>2.5': 'odds_over_25',
            'B365<2.5': 'odds_under_25',
            'B365BTTS': 'odds_btts_yes',  # Might need mapping
        }
        
        # Rename columns that exist
        existing_cols = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Ensure date is parsed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.date
        
        # Standardize result
        if 'result' in df.columns:
            df['result'] = df['result'].map({'H': 'H', 'D': 'D', 'A': 'A'})
        
        # Add source
        df['source'] = 'football-data.co.uk'
        df['status'] = 'FINISHED'
        
        return df


def main():
    fetcher = FootballDataCoUKFetcher()
    df = fetcher.fetch_all_seasons()
    if not df.empty:
        df = fetcher.standardize_columns(df)
        print(f"\nTotal matches: {len(df)}")
        print(f"Seasons: {df['Season'].unique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        # Save to CSV for inspection
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/raw/football_data_co_uk_all.csv", index=False)
        print("Saved to data/raw/football_data_co_uk_all.csv")
    else:
        print("No data fetched!")


if __name__ == "__main__":
    main()