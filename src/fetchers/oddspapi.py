"""Fetcher for oddspapi.io - Gamdom odds for Serie A"""
import os
import requests
from datetime import datetime
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import yaml


class OddsPapiAPI:
    """oddspapi.io v4 - Gamdom pre-match odds, 1X2 / O-U 2.5 / BTTS"""

    BASE_URL = "https://api.oddspapi.io/v4"
    SERIE_A_TOURNAMENT_ID = 23
    BOOKMAKER = "gamdom"

    # Market IDs (from /v4/markets)
    MARKET_1X2 = "101"    # Full Time Result: outcomes 1 / X / 2
    MARKET_BTTS = "104"   # Both Teams To Score: outcomes Yes / No
    MARKET_OU25 = "1010"  # Over Under 2.5: outcomes Over / Under
    MARKET_DC = "101902"  # Double Chance FT: outcomes 1X / 12 / X2 (2X)
    MARKET_SCORER = "10730"  # Anytime Goal Scorer: players dict {id: {playerName, price, active}}

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["data_sources"]["oddspapi"]
        self.api_key = os.getenv(self.config["api_key_env"])
        self.enabled = self.config["enabled"] and bool(self.api_key)
        self.tournament_id = self.config.get("tournament_id", self.SERIE_A_TOURNAMENT_ID)
        self.bookmaker = self.config.get("bookmaker", self.BOOKMAKER)
        self._outcome_names: Dict[str, Dict[str, str]] = {}  # marketId -> {outcomeId: name}

        if not self.api_key:
            print(f"Warning: {self.config['api_key_env']} not set. OddsPapi disabled.")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        if not self.enabled:
            return None
        params = dict(params or {})
        params["apiKey"] = self.api_key
        response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)
        if response.status_code == 429:
            raise Exception("Rate limited")
        response.raise_for_status()
        return response.json()

    # Known outcome names (stable) — avoids extra /markets calls on quota
    STATIC_OUTCOMES = {
        "101": {"101": "1", "102": "X", "103": "2"},
        "104": {"104": "Yes", "105": "No"},
        "101902": {"101902": "1X", "101903": "12", "101904": "X2"},
    }

    def _outcome_name(self, market_id: str, outcome_id: str) -> str:
        """Resolve outcome ID -> name (static map first, /v4/markets cached fallback)."""
        static = self.STATIC_OUTCOMES.get(str(market_id), {})
        if str(outcome_id) in static:
            return static[str(outcome_id)]
        if market_id not in self._outcome_names:
            markets = self._get("/markets") or []
            for m in markets:
                if str(m.get("marketId")) == str(market_id):
                    self._outcome_names[str(market_id)] = {
                        str(o["outcomeId"]): o["outcomeName"]
                        for o in m.get("outcomes", [])
                    }
                    break
        return self._outcome_names.get(str(market_id), {}).get(str(outcome_id), str(outcome_id))

    @staticmethod
    def _price(outcome: dict) -> Optional[float]:
        """Extract decimal price from an outcome node."""
        players = outcome.get("players", {})
        for key in ("0", *sorted(players.keys())):
            p = players.get(key, {})
            if p.get("active", True) and p.get("price"):
                try:
                    return float(p["price"])
                except (TypeError, ValueError):
                    continue
        return None

    def get_fixtures(self) -> List[Dict]:
        """All upcoming Serie A fixtures with team names."""
        data = self._get("/fixtures", {"tournamentId": self.tournament_id, "statusId": 0})
        return data if isinstance(data, list) else []

    def get_gamdom_odds(self) -> List[Dict]:
        """Gamdom odds for all Serie A fixtures: 1X2 / O-U 2.5 / BTTS / DC / scorers.

        Returns list of {home, away, date, odds_1, odds_x, odds_2,
        odds_over25, odds_under25, odds_btts_yes, odds_btts_no,
        odds_dc_1x, odds_dc_12, odds_dc_x2, scorers: [{player, odds}]}.
        Outcome '2X' is normalized to 'X2'.
        """
        fixtures = {f["fixtureId"]: f for f in self.get_fixtures()}
        odds_data = self._get("/odds-by-tournaments", {
            "bookmaker": self.bookmaker,
            "tournamentIds": self.tournament_id,
        }) or []

        results = []
        for entry in odds_data:
            fixture = fixtures.get(entry.get("fixtureId"), {})
            markets = entry.get("bookmakerOdds", {}).get(self.bookmaker, {}).get("markets", {})

            def market_prices(market_id: str) -> Dict[str, float]:
                out = {}
                for oid, node in markets.get(str(market_id), {}).get("outcomes", {}).items():
                    price = self._price(node)
                    if price:
                        name = self._outcome_name(market_id, oid)
                        out["X2" if name == "2X" else name] = price
                return out

            m1x2 = market_prices(self.MARKET_1X2)
            mou = market_prices(self.MARKET_OU25)
            mbtts = market_prices(self.MARKET_BTTS)
            mdc = market_prices(self.MARKET_DC)

            scorers = []
            for oid, node in markets.get(self.MARKET_SCORER, {}).get("outcomes", {}).items():
                for pid, p in (node.get("players", {}) or {}).items():
                    if p.get("active", False) and p.get("price") and p.get("playerName"):
                        try:
                            scorers.append({"player": p["playerName"], "odds": float(p["price"])})
                        except (TypeError, ValueError):
                            continue
            scorers.sort(key=lambda s: s["odds"])

            start = fixture.get("startTime") or entry.get("startTime")
            try:
                match_date = datetime.fromisoformat(start.replace("Z", "+00:00")).date() if start else None
            except (ValueError, AttributeError):
                match_date = None

            results.append({
                "home": fixture.get("participant1Name"),
                "away": fixture.get("participant2Name"),
                "date": match_date,
                "odds_1": m1x2.get("1"),
                "odds_x": m1x2.get("X"),
                "odds_2": m1x2.get("2"),
                "odds_over25": mou.get("Over"),
                "odds_under25": mou.get("Under"),
                "odds_btts_yes": mbtts.get("Yes"),
                "odds_btts_no": mbtts.get("No"),
                "odds_dc_1x": mdc.get("1X"),
                "odds_dc_12": mdc.get("12"),
                "odds_dc_x2": mdc.get("X2"),
                "scorers": scorers[:30],
            })
            time.sleep(0.2)
        return results


def main():
    api = OddsPapiAPI()
    if not api.enabled:
        print("OddsPapi disabled (no key)")
        return
    odds = api.get_gamdom_odds()
    print(f"Fixtures with Gamdom odds: {len(odds)}")
    for o in odds[:5]:
        print(f"  {o['date']} {o['home']} vs {o['away']}: "
              f"1={o['odds_1']} X={o['odds_x']} 2={o['odds_2']} | "
              f"O2.5={o['odds_over25']} U2.5={o['odds_under25']} | "
              f"BTTS Y={o['odds_btts_yes']} N={o['odds_btts_no']} | "
              f"DC 1X={o['odds_dc_1x']} 12={o['odds_dc_12']} X2={o['odds_dc_x2']} | "
              f"scorers={len(o['scorers'])}")


if __name__ == "__main__":
    main()