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
        """Every 15 minutes during match days: update live scores"""
        logger.info("Checking live scores...")
        try:
            from src.fetchers.api_football import FootballDataOrgAPI, APIFootball
            from src.database import Match, get_session
            from sqlalchemy import and_
            from datetime import date
            
            # Only run on match days (check if any match today)
            session = get_session()
            today_matches = session.query(Match).filter(
                and_(
                    Match.date == date.today(),
                    Match.status.in_(['LIVE', 'IN_PLAY', 'SCHEDULED'])
                )
            ).all()
            
            if not today_matches:
                logger.info("No matches today, skipping live update")
                session.close()
                return
            
            # Update from football-data.org
            api1 = FootballDataOrgAPI()
            if api1.enabled:
                live_matches = api1.get_current_season_matches(status='LIVE')
                for match_data in live_matches:
                    std = api1.standardize_match(match_data)
                    # Find and update match
                    match = session.query(Match).filter(
                        and_(
                            Match.date == std['date'],
                            Match.home_team.has(name=std['home_team']),
                            Match.away_team.has(name=std['away_team'])
                        )
                    ).first()
                    if match:
                        match.home_goals = std['home_goals']
                        match.away_goals = std['away_goals']
                        match.home_goals_ht = std['home_goals_ht']
                        match.away_goals_ht = std['away_goals_ht']
                        match.status = std['status']
            
            session.commit()
            session.close()
            logger.info("Live scores update completed")
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