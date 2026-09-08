"""Automated scheduler for data updates and predictions"""
import os
import yaml
import pandas as pd
from datetime import datetime, time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Load .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerieAScheduler:
    """Manage automated updates"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.scheduler_config = self.config["scheduler"]
        self.scheduler = BlockingScheduler(timezone='UTC')
        self.enabled = self.scheduler_config["enabled"]
    
    def update_historical_data(self):
        """Daily update of historical data (new matches, corrections)"""
        logger.info("Starting historical data update...")
        try:
            from src.fetchers.football_data_co_uk import FootballDataCoUKFetcher
            from src.fetchers.datahub_io import DataHubIOFetcher
            from src.processing import DataProcessor
            
            # Fetch current season (updated weekly)
            fetcher = FootballDataCoUKFetcher()
            current_df = fetcher.fetch_current_season()
            if current_df is not None and not current_df.empty:
                current_df = fetcher.standardize_columns(current_df)
                processor = DataProcessor()
                processor.load_into_database(current_df)
                processor.close()
                logger.info(f"Updated current season: {len(current_df)} matches")
            
            # Also check datahub.io for any updates
            fetcher2 = DataHubIOFetcher()
            current_df2 = fetcher2.fetch_season("2425")  # Current season code
            if current_df2 is not None and not current_df2.empty:
                current_df2 = fetcher2.standardize_columns(current_df2)
                processor = DataProcessor()
                processor.load_into_database(current_df2)
                processor.close()
            
            logger.info("Historical data update completed")
        except Exception as e:
            logger.error(f"Historical update failed: {e}")
    
    def update_upcoming_matches(self):
        """Every 6 hours: fetch upcoming matches from APIs"""
        logger.info("Updating upcoming matches...")
        try:
            from src.fetchers.api_football import FootballDataOrgAPI, APIFootball
            from src.processing import DataProcessor, TeamNameMapper
            from src.database import Match, Team, get_session
            from sqlalchemy import and_
            from datetime import date, timedelta
            
            session = get_session()
            processor = DataProcessor()
            mapper = TeamNameMapper()
            
            # Get upcoming from football-data.org
            api1 = FootballDataOrgAPI()
            if api1.enabled:
                upcoming = api1.get_upcoming_matches(14)
                for match_data in upcoming:
                    std = api1.standardize_match(match_data)
                    # Map to canonical names
                    std['home_team'] = mapper.map_to_canonical(std['home_team'], 'football-data.org')
                    std['away_team'] = mapper.map_to_canonical(std['away_team'], 'football-data.org')
                    
                    # Check if exists
                    existing = session.query(Match).filter(
                        and_(
                            Match.date == std['date'],
                            Match.home_team.has(name=std['home_team']),
                            Match.away_team.has(name=std['away_team'])
                        )
                    ).first()
                    
                    if not existing:
                        # Create/get teams with canonical names
                        for team_name in [std['home_team'], std['away_team']]:
                            team = session.query(Team).filter(Team.name == team_name).first()
                            if not team:
                                team = Team(name=team_name)
                                session.add(team)
                        
                        session.flush()
                        home_team = session.query(Team).filter(Team.name == std['home_team']).first()
                        away_team = session.query(Team).filter(Team.name == std['away_team']).first()
                        
                        match = Match(
                            season=std['season'],
                            date=std['date'],
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            matchday=std['matchday'],
                            status=std['status'],
                            source='football-data.org',
                        )
                        session.add(match)
            
            # Get upcoming from API-Football
            api2 = APIFootball()
            if api2.enabled:
                upcoming = api2.get_upcoming_fixtures(14)
                for fix in upcoming:
                    std = api2.standardize_fixture(fix)
                    # Map to canonical names
                    std['home_team'] = mapper.map_to_canonical(std['home_team'], 'api-football')
                    std['away_team'] = mapper.map_to_canonical(std['away_team'], 'api-football')
                    
                    existing = session.query(Match).filter(
                        and_(
                            Match.date == std['date'],
                            Match.home_team.has(name=std['home_team']),
                            Match.away_team.has(name=std['away_team'])
                        )
                    ).first()
                    
                    if not existing:
                        for team_name in [std['home_team'], std['away_team']]:
                            team = session.query(Team).filter(Team.name == team_name).first()
                            if not team:
                                team = Team(name=team_name)
                                session.add(team)
                        
                        session.flush()
                        home_team = session.query(Team).filter(Team.name == std['home_team']).first()
                        away_team = session.query(Team).filter(Team.name == std['away_team']).first()
                        
                        match = Match(
                            season=std['season'],
                            date=std['date'],
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            matchday=std['matchday'],
                            status=std['status'],
                            source='api-football',
                        )
                        session.add(match)
            
            session.commit()
            session.close()
            processor.close()
            logger.info("Upcoming matches update completed")
            # Ensure fresh matches get predictions + snapshots
            self.update_predictions()
        except Exception as e:
            logger.error(f"Upcoming matches update failed: {e}")
    
    def update_predictions(self):
        """Generate/update predictions for upcoming matches"""
        logger.info("Generating predictions...")
        try:
            from src.models import EnsemblePredictor
            from src.processing import DataProcessor
            from src.database import Match, get_session
            from sqlalchemy import and_
            from datetime import date, timedelta
            
            session = get_session()
            processor = DataProcessor()
            
            # Get upcoming matches without predictions
            upcoming = processor.get_upcoming_matches(14)
            
            if len(upcoming) > 0:
                # Load historical data for training
                hist_matches = session.query(Match).filter(Match.status == 'FINISHED').all()
                hist_data = []
                for m in hist_matches:
                    hist_data.append({
                        'date': m.date,
                        'home_team_canonical': m.home_team.name,
                        'away_team_canonical': m.away_team.name,
                        'home_goals': m.home_goals,
                        'away_goals': m.away_goals,
                    })
                hist_df = pd.DataFrame(hist_data)
                
                if len(hist_df) > 100:
                    # Train ensemble
                    ensemble = EnsemblePredictor()
                    ensemble.fit_all(hist_df)
                    
                    # Predict
                    predictions = ensemble.predict_upcoming(upcoming)
                    
                    # Save predictions to database
                    for _, pred in predictions.iterrows():
                        match = session.query(Match).filter(Match.id == pred['match_id']).first()
                        if match:
                            match.pred_home_win = pred['p_home']
                            match.pred_draw = pred['p_draw']
                            match.pred_away_win = pred['p_away']
                            match.pred_over_25 = pred['p_over_25']
                            match.pred_under_25 = pred['p_under_25']
                            match.pred_btts_yes = pred['p_btts_yes']
                            match.pred_btts_no = pred['p_btts_no']
                            match.pred_model_version = 'ensemble_v1'
                    
                    session.commit()
                    logger.info(f"Generated predictions for {len(predictions)} matches")

            # Snapshot predictions + settle finished logs/slips
            self.log_prediction_snapshots(session)
            self.settle_logs_and_slips(session)

            session.close()
            processor.close()
        except Exception as e:
            logger.error(f"Prediction update failed: {e}")
    
    def update_live_scores(self):
        """Every 15 minutes: sync today's scores + statuses (incl. FT transitions)."""
        logger.info("Checking live scores...")
        try:
            from src.fetchers.api_football import FootballDataOrgAPI
            from src.database import Match, get_session
            from sqlalchemy import and_
            from datetime import date, timedelta

            session = get_session()
            # Window covers today + previous 3 days: catches matches missed
            # by earlier runs (downtime) so they still become FINISHED.
            today_matches = session.query(Match).filter(
                and_(
                    Match.date >= date.today() - timedelta(days=3),
                    Match.date <= date.today(),
                    Match.status.in_(['LIVE', 'IN_PLAY', 'PAUSED', 'TIMED', 'SCHEDULED', 'NS'])
                )
            ).all()

            if not today_matches:
                logger.info("No matches today, skipping live update")
                session.close()
                return

            api = FootballDataOrgAPI()
            if not api.enabled:
                logger.warning("football-data.org disabled, skipping live update")
                session.close()
                return

            # Whole-season fetch (1 call), filter today in Python: catches FT transitions
            season_matches = api.get_current_season_matches()
            window_start = date.today() - timedelta(days=3)
            by_teams = {}
            for m in season_matches:
                try:
                    std = api.standardize_match(m)
                except Exception:
                    continue
                if std.get('date') and window_start <= std['date'] <= date.today():
                    by_teams[(std['date'], std['home_team'], std['away_team'])] = std

            STATUS_MAP = {'FINISHED': 'FINISHED', 'IN_PLAY': 'LIVE', 'PAUSED': 'LIVE',
                          'TIMED': 'TIMED', 'SCHEDULED': 'SCHEDULED',
                          'POSTPONED': 'POSTPONED', 'SUSPENDED': 'POSTPONED', 'CANCELLED': 'CANCELLED'}
            from src.processing import TeamNameMapper
            mapper = TeamNameMapper()
            updated = 0
            for m in today_matches:
                key = (m.date, m.home_team.name, m.away_team.name)
                std = by_teams.get(key)
                if not std:
                    # fallback: map API names to canonical
                    for (d, h, a), s in by_teams.items():
                        if (d == m.date
                                and mapper.map_to_canonical(h, 'football-data.org') == m.home_team.name
                                and mapper.map_to_canonical(a, 'football-data.org') == m.away_team.name):
                            std = s
                            break
                if not std:
                    continue
                if std.get('home_goals') is not None:
                    m.home_goals = std['home_goals']
                    m.away_goals = std['away_goals']
                if std.get('home_goals_ht') is not None:
                    m.home_goals_ht = std['home_goals_ht']
                    m.away_goals_ht = std['away_goals_ht']
                new_status = STATUS_MAP.get(std.get('status'), m.status)
                if new_status != m.status:
                    m.status = new_status
                    if new_status == 'FINISHED':
                        m.result = ('H' if m.home_goals > m.away_goals
                                    else ('A' if m.home_goals < m.away_goals else 'D'))
                updated += 1

            session.commit()
            session.close()
            logger.info(f"Live scores update completed: {updated} matches synced")
            # Newly finished matches feed history -> settle logs/slips
            self.settle_logs_and_slips()
        except Exception as e:
            logger.error(f"Live scores update failed: {e}")

    def update_gamdom_odds(self):
        """Refresh Gamdom odds (via OddsPapi) for upcoming matches"""
        logger.info("Updating Gamdom odds...")
        try:
            from src.fetchers.oddspapi import OddsPapiAPI
            from src.processing import TeamNameMapper
            from src.database import Match, Team, get_session
            from sqlalchemy import and_
            from datetime import datetime

            api = OddsPapiAPI()
            if not api.enabled:
                logger.warning("OddsPapi disabled (no key), skipping odds update")
                return

            mapper = TeamNameMapper()
            odds_list = api.get_gamdom_odds()
            session = get_session()
            updated = 0

            for o in odds_list:
                if not o.get("home") or not o.get("away") or not o.get("date"):
                    continue
                home = mapper.map_to_canonical(o["home"], "oddspapi")
                away = mapper.map_to_canonical(o["away"], "oddspapi")

                match = session.query(Match).filter(
                    and_(
                        Match.date == o["date"],
                        Match.home_team.has(name=home),
                        Match.away_team.has(name=away),
                    )
                ).first()
                if not match:
                    continue

                for col, val in [
                    ("odds_home", o.get("odds_1")), ("odds_draw", o.get("odds_x")),
                    ("odds_away", o.get("odds_2")), ("odds_over_25", o.get("odds_over25")),
                    ("odds_under_25", o.get("odds_under25")),
                    ("odds_btts_yes", o.get("odds_btts_yes")), ("odds_btts_no", o.get("odds_btts_no")),
                ]:
                    if val:
                        setattr(match, col, val)
                match.odds_source = "gamdom"
                match.odds_updated_at = datetime.utcnow()
                updated += 1

            session.commit()
            session.close()
            logger.info(f"Gamdom odds update completed: {updated} matches")
            # Refresh today's snapshots with new odds + settle anything finished
            self.log_prediction_snapshots()
            self.settle_logs_and_slips()
        except Exception as e:
            logger.error(f"Gamdom odds update failed: {e}")

    def log_prediction_snapshots(self, session=None):
        """Snapshot today's predictions+odds for upcoming matches (upsert per match+day)."""
        from src.database import Match, PredictionLog, get_session
        from sqlalchemy import and_
        from datetime import date, datetime

        own_session = session is None
        session = session or get_session()
        try:
            today = date.today()
            matches = session.query(Match).filter(
                and_(
                    Match.date >= today,
                    Match.status.in_(['SCHEDULED', 'TIMED']),
                    Match.pred_home_win.isnot(None),
                )
            ).all()
            for m in matches:
                log = session.query(PredictionLog).filter(
                    and_(PredictionLog.match_id == m.id, PredictionLog.log_date == today)
                ).first()
                if not log:
                    log = PredictionLog(match_id=m.id, log_date=today)
                    session.add(log)
                log.recorded_at = datetime.utcnow()
                log.p_home, log.p_draw, log.p_away = m.pred_home_win, m.pred_draw, m.pred_away_win
                log.p_over25, log.p_under25 = m.pred_over_25, m.pred_under_25
                log.p_btts_yes, log.p_btts_no = m.pred_btts_yes, m.pred_btts_no
                log.odds_1, log.odds_x, log.odds_2 = m.odds_home, m.odds_draw, m.odds_away
                log.odds_over25, log.odds_under25 = m.odds_over_25, m.odds_under_25
                log.odds_btts_yes, log.odds_btts_no = m.odds_btts_yes, m.odds_btts_no
                log.odds_source = m.odds_source
            session.commit()
            logger.info(f"Logged prediction snapshots for {len(matches)} matches")
        finally:
            if own_session:
                session.close()

    @staticmethod
    def _settle_match_result(match):
        """Return (result, total_goals, btts) or None if not finished."""
        if match.status != 'FINISHED' or match.home_goals is None or match.away_goals is None:
            return None
        hg, ag = match.home_goals, match.away_goals
        result = 'H' if hg > ag else ('A' if hg < ag else 'D')
        return result, hg + ag, (hg > 0 and ag > 0)

    def settle_logs_and_slips(self, session=None):
        """Backfill results on prediction logs + settle open bet slips."""
        from src.database import Match, PredictionLog, BetSlip, get_session
        from datetime import datetime

        own_session = session is None
        session = session or get_session()
        try:
            settled_logs = 0
            for log in session.query(PredictionLog).filter(PredictionLog.result.is_(None)).all():
                match = session.query(Match).filter(Match.id == log.match_id).first()
                info = self._settle_match_result(match) if match else None
                if not info:
                    continue
                result, total, btts = info
                log.result, log.home_goals, log.away_goals = result, match.home_goals, match.away_goals
                if log.p_home is not None:
                    pred = max([('H', log.p_home), ('D', log.p_draw or 0), ('A', log.p_away or 0)],
                               key=lambda x: x[1])[0]
                    log.hit_1x2 = (pred == result)
                if log.p_over25 is not None:
                    log.hit_ou25 = ((log.p_over25 > 0.5) == (total > 2.5))
                if log.p_btts_yes is not None:
                    log.hit_btts = ((log.p_btts_yes > 0.5) == btts)
                settled_logs += 1

            settled_slips = 0
            for slip in session.query(BetSlip).filter(BetSlip.status == 'OPEN').all():
                legs = slip.legs or []
                won, pending = True, False
                for leg in legs:
                    match = session.query(Match).filter(Match.id == leg.get('match_id')).first()
                    info = self._settle_match_result(match) if match else None
                    if not info:
                        pending = True
                        break
                    result, total, btts = info
                    sel, mkt = leg.get('selection'), leg.get('market')
                    if mkt == '1X2':
                        ok = ({'1': 'H', 'X': 'D', '2': 'A'}.get(sel) == result)
                    elif mkt == 'O/U 2.5':
                        ok = ((sel == 'Over 2.5') == (total > 2.5))
                    elif mkt == 'BTTS':
                        ok = ((sel == 'GG Sì') == btts)
                    else:
                        ok = False
                    if not ok:
                        won = False
                        break
                if pending:
                    continue
                slip.status = 'WON' if won else 'LOST'
                slip.settled_at = datetime.utcnow()
                slip.profit = round(slip.stake * (slip.total_odds - 1), 2) if won else round(-slip.stake, 2)
                settled_slips += 1

            session.commit()
            logger.info(f"Settled {settled_logs} prediction logs, {settled_slips} slips")
        finally:
            if own_session:
                session.close()

    @staticmethod
    def _int_or_zero(v):
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    @classmethod
    def store_fixture_players(cls, session, match, entries, mapper):
        """Parse /fixtures/players response into Player + PlayerMatch rows.
        Returns number of player-match rows stored. Pure logic, no API calls."""
        from src.database import Player, PlayerMatch
        from sqlalchemy import and_

        stored = 0
        for entry in entries:
            canon = mapper.map_to_canonical((entry.get("team") or {}).get("name", ""), "api-football")
            if canon == match.home_team.name:
                team_id = match.home_team_id
            elif canon == match.away_team.name:
                team_id = match.away_team_id
            else:
                continue
            for p in entry.get("players", []):
                info, stats_list = p.get("player", {}), p.get("statistics", [])
                if not info.get("id") or not stats_list:
                    continue
                st = stats_list[0]
                g = st.get("games", {})
                player = session.query(Player).filter(
                    Player.api_football_id == info["id"]).first()
                if not player:
                    player = Player(name=info.get("name", "?"), api_football_id=info["id"])
                    session.add(player)
                    session.flush()
                player.team_id = team_id
                player.position = (g.get("position") or "")[:20]
                rating = None
                try:
                    rating = float(g["rating"]) if g.get("rating") else None
                except (TypeError, ValueError):
                    pass
                if not session.query(PlayerMatch).filter(
                        and_(PlayerMatch.player_id == player.id,
                             PlayerMatch.match_id == match.id)).first():
                    session.add(PlayerMatch(
                        player_id=player.id, match_id=match.id, team_id=team_id,
                        season=match.season, minutes=cls._int_or_zero(g.get("minutes")),
                        position=(g.get("position") or "")[:10], rating=rating,
                        goals=cls._int_or_zero((st.get("goals") or {}).get("total")),
                        assists=cls._int_or_zero((st.get("goals") or {}).get("assists")),
                        yellow=cls._int_or_zero((st.get("cards") or {}).get("yellow")),
                        red=cls._int_or_zero((st.get("cards") or {}).get("red")),
                        starter=not g.get("substitute", False),
                    ))
                    stored += 1
        return stored

    def sync_player_stats(self, max_fixtures: int = None):
        """Backfill per-player match stats (goals/assists/cards/minutes/rating).

        Date-based discovery (works on free tier): for each recent date with
        unfinished business, 1 call lists the day's fixtures + 1 call per
        fixture for player stats. Stops early when daily quota runs low.
        """
        logger.info("Syncing player stats...")
        try:
            from src.fetchers.api_football import APIFootball
            from src.processing import TeamNameMapper
            from src.database import Match, Team, Player, PlayerMatch, get_session
            from sqlalchemy import and_, exists
            from datetime import date, timedelta

            with open("config.yaml") as f:
                import yaml
                cfg = yaml.safe_load(f).get("players", {})
            max_fixtures = max_fixtures or cfg.get("max_per_run", 30)
            lookback = cfg.get("lookback_days", 7)
            min_quota = cfg.get("min_quota_reserve", 8)

            api = APIFootball()
            if not api.enabled:
                logger.warning("API-Football disabled, skipping player sync")
                return

            mapper = TeamNameMapper()
            session = get_session()

            # Dates needing work: unfinished matches up to today + recently finished
            # without stats yet (backfill window).
            cutoff = date.today() - timedelta(days=lookback)
            pending = (
                session.query(Match)
                .filter(and_(
                    Match.date >= cutoff,
                    Match.date <= date.today(),
                    Match.status.in_(['FINISHED', 'TIMED', 'SCHEDULED', 'LIVE']),
                ))
                .order_by(Match.date.desc())
                .all()
            )
            dates = sorted({m.date for m in pending
                            if m.status != 'FINISHED'
                            or not session.query(exists().where(PlayerMatch.match_id == m.id)).scalar()},
                           reverse=True)
            logger.info(f"Player sync: {len(dates)} dates to check (lookback {lookback}d)")

            def quota_ok():
                q = api.quota_remaining
                return q is None or q == -1 or q >= min_quota

            synced = skipped = 0
            for d in dates:
                if synced >= max_fixtures or not quota_ok():
                    break
                try:
                    day_fixtures = api.get_fixtures_by_date(d)
                except Exception as e:
                    logger.warning(f"Fixture list failed for {d}: {e}")
                    continue
                for fx in day_fixtures:
                    if synced >= max_fixtures or not quota_ok():
                        break
                    if (fx.get("fixture", {}).get("status", {}).get("short") != "FT"):
                        continue
                    teams = fx.get("teams", {})
                    h = mapper.map_to_canonical(teams.get("home", {}).get("name", ""), "api-football")
                    a = mapper.map_to_canonical(teams.get("away", {}).get("name", ""), "api-football")
                    match = session.query(Match).filter(
                        and_(Match.date == d,
                             Match.home_team.has(name=h),
                             Match.away_team.has(name=a))).first()
                    if not match or session.query(
                            exists().where(PlayerMatch.match_id == match.id)).scalar():
                        continue
                    try:
                        entries = api.get_fixture_players(fx["fixture"]["id"])
                    except Exception as e:
                        logger.warning(f"Players failed for fixture {fx['fixture']['id']}: {e}")
                        continue
                    if not entries:
                        skipped += 1
                        continue
                    try:
                        stored = self.store_fixture_players(session, match, entries, mapper)
                        session.commit()
                        if stored:
                            synced += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        session.rollback()
                        logger.warning(f"Player sync commit failed for match {match.id}: {e}")

            session.close()
            logger.info(f"Player stats synced for {synced} fixtures, quota left: {api.quota_remaining}")
        except Exception as e:
            logger.error(f"Player sync failed: {e}")

    def retrain_models_weekly(self):
        """Weekly model retraining with full historical data"""
        logger.info("Weekly model retraining...")
        try:
            from src.models import EnsemblePredictor
            from src.database import Match, get_session
            import pandas as pd
            
            session = get_session()
            hist_matches = session.query(Match).filter(Match.status == 'FINISHED').all()
            hist_data = []
            for m in hist_matches:
                hist_data.append({
                    'date': m.date,
                    'home_team_canonical': m.home_team.name,
                    'away_team_canonical': m.away_team.name,
                    'home_goals': m.home_goals,
                    'away_goals': m.away_goals,
                })
            hist_df = pd.DataFrame(hist_data)
            
            if len(hist_df) > 200:
                ensemble = EnsemblePredictor()
                ensemble.fit_all(hist_df)
                
                # Re-predict all upcoming
                from src.processing import DataProcessor
                processor = DataProcessor()
                upcoming = processor.get_upcoming_matches(14)
                predictions = ensemble.predict_upcoming(upcoming)
                
                for _, pred in predictions.iterrows():
                    match = session.query(Match).filter(Match.id == pred['match_id']).first()
                    if match:
                        match.pred_home_win = pred['p_home']
                        match.pred_draw = pred['p_draw']
                        match.pred_away_win = pred['p_away']
                        match.pred_over_25 = pred['p_over_25']
                        match.pred_under_25 = pred['p_under_25']
                        match.pred_btts_yes = pred['p_btts_yes']
                        match.pred_btts_no = pred['p_btts_no']
                        match.pred_model_version = 'ensemble_v1_weekly'
                
                session.commit()
                processor.close()
                logger.info("Weekly retraining completed")
            
            session.close()
        except Exception as e:
            logger.error(f"Weekly retraining failed: {e}")
    
    def start(self):
        """Start the scheduler"""
        if not self.enabled:
            logger.info("Scheduler disabled in config")
            return
        
        # Daily historical update at 2 AM UTC
        self.scheduler.add_job(
            self.update_historical_data,
            CronTrigger.from_crontab(self.scheduler_config["update_historical_daily"]),
            id='historical_update',
            max_instances=1
        )
        
        # Every 6 hours: upcoming matches
        self.scheduler.add_job(
            self.update_upcoming_matches,
            CronTrigger.from_crontab(self.scheduler_config["update_upcoming_6h"]),
            id='upcoming_update',
            max_instances=1
        )
        
        # Every 15 minutes: live scores (on match days)
        self.scheduler.add_job(
            self.update_live_scores,
            CronTrigger.from_crontab(self.scheduler_config["update_live_15m"]),
            id='live_update',
            max_instances=1
        )
        
        # Weekly retraining (Sunday 3 AM)
        self.scheduler.add_job(
            self.retrain_models_weekly,
            CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='weekly_retrain',
            max_instances=1
        )

        # Gamdom odds refresh (default every 30 min)
        self.scheduler.add_job(
            self.update_gamdom_odds,
            CronTrigger.from_crontab(self.scheduler_config.get("update_odds_30m", "*/30 * * * *")),
            id='odds_update',
            max_instances=1
        )

        # Player stats sync (daily, quota-aware)
        self.scheduler.add_job(
            self.sync_player_stats,
            CronTrigger.from_crontab(self.scheduler_config.get("update_players_daily", "30 3 * * *")),
            id='players_update',
            max_instances=1
        )

        logger.info("Scheduler started")
        self.scheduler.start()


def run_once(job_name: str):
    """Run a specific job once (for testing)"""
    scheduler = SerieAScheduler()
    jobs = {
        'historical': scheduler.update_historical_data,
        'upcoming': scheduler.update_upcoming_matches,
        'predictions': scheduler.update_predictions,
        'live': scheduler.update_live_scores,
        'retrain': scheduler.retrain_models_weekly,
        'odds': scheduler.update_gamdom_odds,
        'settle': scheduler.settle_logs_and_slips,
        'players': scheduler.sync_player_stats,
    }
    if job_name in jobs:
        jobs[job_name]()
    else:
        print(f"Unknown job: {job_name}. Available: {list(jobs.keys())}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_once(sys.argv[1])
    else:
        SerieAScheduler().start()