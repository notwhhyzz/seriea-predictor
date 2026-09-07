"""Fetchers for API-based sources (football-data.org, API-Football)"""
import os
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import yaml


class FootballDataOrgAPI:
    """football-data.org API v4 - Free tier: 10 req/min, Serie A included"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["football_data_org"]
        self.base_url = self.config["base_url"]
        self.competition_code = self.config["competition_code"]
        self.api_key = os.getenv(self.config["api_key_env"])
        self.rate_limit = 60.0 / self.config["rate_limit_per_minute"]  # seconds between requests
        self.enabled = self.config["enabled"] and bool(self.api_key)
        self._last_request = 0
        
        if not self.api_key:
            print(f"Warning: {self.config['api_key_env']} not set. football-data.org API disabled.")
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
    
    def _headers(self):
        return {"X-Auth-Token": self.api_key}
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        if not self.enabled:
            return None
        self._rate_limit_wait()
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=30)
        self._last_request = time.time()
        if response.status_code == 429:
            raise Exception("Rate limited")
        response.raise_for_status()
        return response.json()
    
    def get_competition_info(self) -> Optional[Dict]:
        return self._get(f"/competitions/{self.competition_code}")
    
    def get_current_season_matches(self, status: str = None) -> List[Dict]:
        """Get matches for current season, optionally filtered by status"""
        params = {"competitions": self.competition_code}
        if status:
            params["status"] = status
        # Get current season matches
        data = self._get("/matches", params)
        return data.get("matches", []) if data else []
    
    def get_upcoming_matches(self, days: int = 14) -> List[Dict]:
        """Get upcoming matches in next N days"""
        # API has 10-day limit per request
        all_matches = []
        for start_offset in range(0, days, 10):
            date_from = (date.today() + timedelta(days=start_offset)).isoformat()
            date_to = (date.today() + timedelta(days=min(start_offset + 10, days))).isoformat()
            
            comp_info = self.get_competition_info()
            current_season = None
            if comp_info and "currentSeason" in comp_info:
                current_season = comp_info["currentSeason"]["startDate"][:4]
            
            params = {
                "dateFrom": date_from,
                "dateTo": date_to,
                "status": "SCHEDULED"
            }
            if current_season:
                params["season"] = current_season
            
            data = self._get("/matches", params)
            if data:
                all_matches.extend(data.get("matches", []))
        
        # Filter for Serie A competition
        filtered = [m for m in all_matches if m.get("competition", {}).get("code") == "SA"]
        return filtered
    
    def get_team_matches(self, team_id: int, season: str = None, limit: int = 10) -> List[Dict]:
        """Get recent matches for a team"""
        params = {"limit": limit}
        if season:
            params["season"] = season
        data = self._get(f"/teams/{team_id}/matches", params)
        return data.get("matches", []) if data else []
    
    def get_standings(self) -> List[Dict]:
        data = self._get(f"/competitions/{self.competition_code}/standings")
        if data and "standings" in data:
            for standing in data["standings"]:
                if standing["type"] == "TOTAL":
                    return standing["table"]
        return []
    
    def standardize_match(self, match: Dict) -> Dict:
        """Convert API match to our schema"""
        utc_date = match.get("utcDate", "")
        match_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date() if utc_date else None
        
        home = match["homeTeam"]
        away = match["awayTeam"]
        score = match.get("score", {})
        full_time = score.get("fullTime", {})
        half_time = score.get("halfTime", {})
        
        return {
            "source": "football-data.org",
            "external_id": match.get("id"),
            "season": match.get("season", {}).get("startDate", "")[:4] + "/" + match.get("season", {}).get("endDate", "")[2:4],
            "matchday": match.get("matchday"),
            "date": match_date,
            "home_team": home.get("name"),
            "home_team_id": home.get("id"),
            "away_team": away.get("name"),
            "away_team_id": away.get("id"),
            "home_goals": full_time.get("home"),
            "away_goals": full_time.get("away"),
            "home_goals_ht": half_time.get("home"),
            "away_goals_ht": half_time.get("away"),
            "status": match.get("status"),
            "referee": match.get("referees", [{}])[0].get("name") if match.get("referees") else None,
            "venue": match.get("venue"),
        }


class APIFootball:
    """API-Football (api-sports.io) - Free tier: 100 req/day, includes predictions, odds, stats"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["api_football"]
        self.base_url = self.config["base_url"]
        self.league_id = self.config["league_id"]
        self.api_key = os.getenv(self.config["api_key_env"])
        self.enabled = self.config["enabled"] and bool(self.api_key)
        self._request_count = 0
        self._day_reset = time.time()
        
        if not self.api_key:
            print(f"Warning: {self.config['api_key_env']} not set. API-Football disabled.")
    
    def _check_rate_limit(self):
        # Reset daily counter
        if time.time() - self._day_reset > 86400:
            self._request_count = 0
            self._day_reset = time.time()
        
        if self._request_count >= 95:  # Leave buffer
            raise Exception("Daily rate limit approaching")
    
    def _headers(self):
        return {"x-apisports-key": self.api_key}
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        if not self.enabled:
            return None
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=30)
        self._request_count += 1
        
        if response.status_code == 429:
            raise Exception("Rate limited")
        response.raise_for_status()
        return response.json()
    
    def get_fixtures(self, season: int = 2024, from_date: str = None, to_date: str = None, status: str = None) -> List[Dict]:
        """Get fixtures for Serie A"""
        params = {"league": self.league_id, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status
        
        data = self._get("/fixtures", params)
        return data.get("response", []) if data else []
    
    def get_upcoming_fixtures(self, days: int = 14) -> List[Dict]:
        from_date = date.today().isoformat()
        to_date = (date.today() + timedelta(days=days)).isoformat()
        return self.get_fixtures(from_date=from_date, to_date=to_date, status="NS")  # NS = Not Started
    
    def get_predictions(self, fixture_id: int) -> Optional[Dict]:
        """Get API-Football's own predictions for a fixture"""
        data = self._get("/predictions", {"fixture": fixture_id})
        return data.get("response", [{}])[0] if data else None
    
    def get_odds(self, fixture_id: int, bookmaker: int = None) -> List[Dict]:
        """Get odds for a fixture"""
        params = {"fixture": fixture_id}
        if bookmaker:
            params["bookmaker"] = bookmaker
        data = self._get("/odds", params)
        return data.get("response", []) if data else []
    
    def get_team_statistics(self, team_id: int, season: int = 2024) -> Optional[Dict]:
        data = self._get("/teams/statistics", {"league": self.league_id, "season": season, "team": team_id})
        return data.get("response", {}) if data else {}
    
    def get_standings(self, season: int = 2024) -> List[Dict]:
        data = self._get("/standings", {"league": self.league_id, "season": season})
        if data and data.get("response"):
            return data["response"][0].get("league", {}).get("standings", [[]])[0]
        return []
    
    def standardize_fixture(self, fixture: Dict) -> Dict:
        """Convert API-Football fixture to our schema"""
        fix = fixture.get("fixture", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})
        
        utc_date = fix.get("date", "")
        match_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date() if utc_date else None
        
        home = teams.get("home", {})
        away = teams.get("away", {})
        
        return {
            "source": "api-football",
            "external_id": fix.get("id"),
            "season": str(fix.get("season", 2024)) + "/" + str(fix.get("season", 2024) + 1)[-2:],
            "matchday": fixture.get("league", {}).get("round", "").replace("Regular Season - ", "").replace("Matchday ", ""),
            "date": match_date,
            "home_team": home.get("name"),
            "home_team_id": home.get("id"),
            "away_team": away.get("name"),
            "away_team_id": away.get("id"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_goals_ht": score.get("halftime", {}).get("home"),
            "away_goals_ht": score.get("halftime", {}).get("away"),
            "status": fix.get("status", {}).get("short"),
            "referee": fix.get("referee"),
            "venue": fix.get("venue", {}).get("name"),
        }
    
    def fetch_historical_seasons(self, seasons: list = None) -> list:
        """Fetch multiple historical seasons from API-Football"""
        if seasons is None:
            seasons = [2023, 2022]  # Known working seasons
        
        all_fixtures = []
        for season in seasons:
            print(f"Fetching season {season}/{season+1}...")
            fixtures = self.get_fixtures(season=season, from_date=f'{season}-08-01', to_date=f'{season+1}-06-30')
            all_fixtures.extend(fixtures)
            time.sleep(1)  # Rate limit
        
        return all_fixtures


def main():
    # Test football-data.org
    print("=== Testing football-data.org ===")
    api1 = FootballDataOrgAPI()
    if api1.enabled:
        upcoming = api1.get_upcoming_matches(7)
        print(f"Upcoming matches (7 days): {len(upcoming)}")
        for m in upcoming[:3]:
            std = api1.standardize_match(m)
            print(f"  {std['date']}: {std['home_team']} vs {std['away_team']}")
    
    # Test API-Football
    print("\n=== Testing API-Football ===")
    api2 = APIFootball()
    if api2.enabled:
        upcoming = api2.get_upcoming_fixtures(7)
        print(f"Upcoming fixtures (7 days): {len(upcoming)}")
        for f in upcoming[:3]:
            std = api2.standardize_fixture(f)
            print(f"  {std['date']}: {std['home_team']} vs {std['away_team']}")
        
        # Test historical
        print("\n=== Historical seasons ===")
        fixtures = api2.fetch_historical_seasons([2023, 2022])
        print(f"Total historical fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()