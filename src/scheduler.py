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