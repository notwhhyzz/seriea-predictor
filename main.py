#!/usr/bin/env python3
"""Serie A Predictor - Main entry point"""
import sys
import argparse
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_init(args):
    """Initialize database and fetch initial data"""
    from src.database import init_db
    from src.fetchers.api_football import APIFootball
    from src.processing import DataProcessor, TeamNameMapper
    import pandas as pd
    
    print("Initializing database...")
    init_db()
    
    print("Fetching historical data from API-Football...")
    api = APIFootball()
    if api.enabled:
        fixtures = api.fetch_historical_seasons([2023, 2022])
        print(f"Got {len(fixtures)} historical fixtures")
        
        # Convert to DataFrame
        data = []
        for f in fixtures:
            std = api.standardize_fixture(f)
            data.append(std)
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Map team names to canonical
            mapper = TeamNameMapper()
            df['home_team_canonical'] = df['home_team'].apply(lambda x: mapper.map_to_canonical(x, "api-football"))
            df['away_team_canonical'] = df['away_team'].apply(lambda x: mapper.map_to_canonical(x, "api-football"))
            
            # Add required columns for processing
            df['result'] = df.apply(
                lambda r: 'H' if pd.notna(r['home_goals']) and pd.notna(r['away_goals']) and r['home_goals'] > r['away_goals']
                         else 'A' if pd.notna(r['home_goals']) and pd.notna(r['away_goals']) and r['home_goals'] < r['away_goals']
                         else 'D' if pd.notna(r['home_goals']) and pd.notna(r['away_goals']) else None, axis=1
            )
            df['status'] = df['status'].map({'FT': 'FINISHED', 'NS': 'SCHEDULED', 'LIVE': 'LIVE', 'HT': 'LIVE'}).fillna('SCHEDULED')
            
            processor = DataProcessor()
            processor.load_into_database(df)
            processor.close()
            print(f"Loaded {len(df)} matches into database")
    
    print("Fetching upcoming matches...")
    from src.scheduler import run_once
    run_once('upcoming')
    run_once('predictions')
    
    print("Initialization complete!")


def cmd_update(args):
    """Update data and predictions"""
    from src.scheduler import run_once
    
    if args.historical:
        run_once('historical')
    if args.upcoming:
        run_once('upcoming')
    if args.predictions:
        run_once('predictions')
    if args.live:
        run_once('live')
    if args.retrain:
        run_once('retrain')
    
    if not any([args.historical, args.upcoming, args.predictions, args.live, args.retrain]):
        run_once('historical')
        run_once('upcoming')
        run_once('predictions')


def cmd_predict(args):
    """Generate predictions for upcoming matches"""
    from src.scheduler import run_once
    run_once('predictions')


def cmd_dashboard(args):
    """Launch Streamlit dashboard"""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py",
                    "--server.port", str(args.port), "--server.address", "0.0.0.0"])


def cmd_scheduler(args):
    """Run scheduler (blocking)"""
    from src.scheduler import SerieAScheduler
    scheduler = SerieAScheduler()
    scheduler.start()


def cmd_backtest(args):
    """Run backtest on historical data"""
    from src.models import EnsemblePredictor
    from src.database import Match, get_session
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    
    session = get_session()
    matches = session.query(Match).filter(Match.status == 'FINISHED').all()
    
    data = []
    for m in matches:
        data.append({
            'date': m.date,
            'home_team_canonical': m.home_team.name,
            'away_team_canonical': m.away_team.name,
            'home_goals': m.home_goals,
            'away_goals': m.away_goals,
            'result': m.result,
        })
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Walk-forward validation
    n_splits = 10
    split_size = len(df) // n_splits
    
    results = []
    for i in range(n_splits - 1):
        train_end = (i + 1) * split_size
        test_start = train_end
        test_end = min(test_start + split_size, len(df))
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        if len(train_df) < 100:
            continue
        
        ensemble = EnsemblePredictor()
        ensemble.fit_all(train_df)
        
        for _, row in test_df.iterrows():
            pred = ensemble.predict(row['home_team_canonical'], row['away_team_canonical'])
            pred_result = '1' if pred['p_home'] > max(pred['p_draw'], pred['p_away']) else ('X' if pred['p_draw'] > pred['p_away'] else '2')
            results.append({
                'date': row['date'],
                'actual': row['result'],
                'predicted': pred_result,
                'p_home': pred['p_home'],
                'p_draw': pred['p_draw'],
                'p_away': pred['p_away'],
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        acc = accuracy_score(results_df['actual'], results_df['predicted'])
        print(f"\nBacktest Results ({len(results_df)} matches):")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Correct: {sum(results_df['actual'] == results_df['predicted'])} / {len(results_df)}")
        
        # Per outcome
        for outcome in ['H', 'D', 'A']:
            subset = results_df[results_df['actual'] == outcome]
            if len(subset) > 0:
                acc_o = accuracy_score(subset['actual'], subset['predicted'])
                print(f"  {outcome}: {acc_o:.2%} ({len(subset)} matches)")
        
        # Brier score
        y_true = pd.get_dummies(results_df['actual'])[['H', 'D', 'A']].values
        y_prob = results_df[['p_home', 'p_draw', 'p_away']].values
        bs = brier_score_loss(y_true.ravel(), y_prob.ravel())
        print(f"  Brier Score: {bs:.4f}")
    else:
        print("Not enough data for backtest")


def main():
    parser = argparse.ArgumentParser(description="Serie A Predictor")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init
    subparsers.add_parser('init', help='Initialize database and fetch initial data')
    
    # update
    update_parser = subparsers.add_parser('update', help='Update data and predictions')
    update_parser.add_argument('--historical', action='store_true', help='Update historical data')
    update_parser.add_argument('--upcoming', action='store_true', help='Update upcoming matches')
    update_parser.add_argument('--predictions', action='store_true', help='Update predictions')
    update_parser.add_argument('--live', action='store_true', help='Update live scores')
    update_parser.add_argument('--retrain', action='store_true', help='Retrain models')
    
    # predict
    subparsers.add_parser('predict', help='Generate predictions')
    
    # dashboard
    dash_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dash_parser.add_argument('--port', type=int, default=8501, help='Port')
    
    # scheduler
    subparsers.add_parser('scheduler', help='Run automated scheduler')
    
    # backtest
    subparsers.add_parser('backtest', help='Run backtest on historical data')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        cmd_init(args)
    elif args.command == 'update':
        cmd_update(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'dashboard':
        cmd_dashboard(args)
    elif args.command == 'scheduler':
        cmd_scheduler(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()