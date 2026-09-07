"""FBref scraper for advanced stats (xG, xA, passing, defense, etc.)"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml
from pathlib import Path


class FBrefScraper:
    """Scrape advanced stats from FBref for Serie A"""
    
    # Serie A competition ID on FBref
    COMPETITION_ID = 11
    
    # Table categories available
    TABLE_CATEGORIES = {
        "standard": "stats_standard",
        "shooting": "stats_shooting",
        "passing": "stats_passing",
        "passing_types": "stats_passing_types",
        "gca": "stats_gca",  # Goal & Shot Creation
        "defense": "stats_defense",
        "possession": "stats_possession",
        "playing_time": "stats_playing_time",
        "misc": "stats_misc",
        "keeper": "stats_keeper",
        "keeper_adv": "stats_keeper_adv",
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["fbref"]
        self.base_url = self.config["base_url"]
        self.competition_id = self.config["competition_id"]
        self.rate_limit = self.config["rate_limit_seconds"]
        self.use_selenium = self.config["use_selenium"]
        self.enabled = self.config["enabled"]
        self._last_request = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
    
    def _get_season_id(self, season: str) -> str:
        """Convert '2024/25' to FBref season ID like '10731'"""
        # FBref season IDs for Serie A:
        season_ids = {
            "2024/25": "10731",
            "2023/24": "23123",
            "2022/23": "21847",
            "2021/22": "21073",
            "2020/21": "11693",
            "2019/20": "10587",
            "2018/19": "9783",
            "2017/18": "9365",
            "2016/17": "8887",
            "2015/16": "8371",
            "2014/15": "7853",
        }
        return season_ids.get(season, "10731")  # Default to current
    
    def _build_url(self, category: str, season: str) -> str:
        season_id = self._get_season_id(season)
        season_name = season.replace("/", "-")
        return f"{self.base_url}/en/comps/{self.competition_id}/{season_id}/{category}/{season_name}-Serie-A-Stats"
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _fetch_table(self, url: str, table_id: str) -> Optional[pd.DataFrame]:
        self._rate_limit_wait()
        
        response = self.session.get(url, timeout=30)
        self._last_request = time.time()
        
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return None
        
        # FBref tables are in HTML comments - need to extract them
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find table in comments
        comments = soup.find_all(string=lambda text: isinstance(text, str) and '<table' in text)
        
        for comment in comments:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id=table_id)
            if table:
                break
        else:
            # Try direct find
            table = soup.find('table', id=table_id)
        
        if not table:
            print(f"Table {table_id} not found in {url}")
            return None
        
        # Parse with pandas
        try:
            df = pd.read_html(str(table))[0]
        except Exception as e:
            print(f"Error parsing table: {e}")
            return None
        
        # Clean multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c != '']).strip('_') for col in df.columns.values]
        
        # Remove header rows that repeat column names
        first_col = df.columns[0]
        df = df[df[first_col] != first_col]
        
        return df
    
    def fetch_team_stats(self, season: str, category: str = "standard") -> Optional[pd.DataFrame]:
        """Fetch team stats for a season and category"""
        if category not in self.TABLE_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.TABLE_CATEGORIES.keys())}")
        
        table_id = self.TABLE_CATEGORIES[category]
        url = self._build_url(category, season)
        
        print(f"Fetching {category} for {season} from FBref...")
        df = self._fetch_table(url, table_id)
        
        if df is not None:
            df['season'] = season
            df['category'] = category
        
        return df
    
    def fetch_all_team_stats(self, season: str) -> Dict[str, pd.DataFrame]:
        """Fetch all team stat categories for a season"""
        results = {}
        for category in self.TABLE_CATEGORIES:
            df = self.fetch_team_stats(season, category)
            if df is not None:
                results[category] = df
            time.sleep(1)  # Extra delay between categories
        return results
    
    def fetch_match_schedule(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch match schedule with xG for a season"""
        season_id = self._get_season_id(season)
        season_name = season.replace("/", "-")
        url = f"{self.base_url}/en/comps/{self.competition_id}/{season_id}/schedule/{season_name}-Serie-A-Scores-and-Fixtures"
        
        table_id = "sched_" + season_id + "_1"
        
        print(f"Fetching match schedule for {season}...")
        df = self._fetch_table(url, table_id)
        
        if df is not None:
            df['season'] = season
            # Clean up column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        return df
    
    def fetch_player_stats(self, season: str, category: str = "standard") -> Optional[pd.DataFrame]:
        """Fetch player stats for a season"""
        # Player stats are on a different URL pattern
        season_id = self._get_season_id(season)
        season_name = season.replace("/", "-")
        url = f"{self.base_url}/en/comps/{self.competition_id}/{season_id}/{category}/players/{season_name}-Serie-A-Stats"
        
        table_id = self.TABLE_CATEGORIES[category]
        
        print(f"Fetching player {category} for {season}...")
        df = self._fetch_table(url, table_id)
        
        if df is not None:
            df['season'] = season
            df['category'] = category
        
        return df


def main():
    scraper = FBrefScraper()
    if not scraper.enabled:
        print("FBref scraping disabled in config")
        return
    
    # Test with current season
    season = "2024/25"
    
    # Fetch match schedule (includes xG)
    schedule = scraper.fetch_match_schedule(season)
    if schedule is not None:
        print(f"\nSchedule: {len(schedule)} matches")
        print(f"Columns: {list(schedule.columns)}")
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        schedule.to_csv("data/raw/fbref_schedule_2024_25.csv", index=False)
    
    # Fetch team standard stats
    standard = scraper.fetch_team_stats(season, "standard")
    if standard is not None:
        print(f"\nTeam standard stats: {len(standard)} teams")
        print(f"Columns: {list(standard.columns)}")
        standard.to_csv("data/raw/fbref_team_standard_2024_25.csv", index=False)


if __name__ == "__main__":
    main()