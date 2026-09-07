"""Prediction models: Poisson, Dixon-Coles, Elo"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from scipy.optimize import minimize
from typing import Dict, Tuple, List, Optional
from sqlalchemy.orm import Session
from src.database import Match, Team, EloRating, ModelPerformance, get_session
import yaml
import warnings
warnings.filterwarnings('ignore')


class PoissonModel:
    """Poisson distribution model for match prediction"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["models"]["poisson"]
        self.min_matches = self.config["min_matches_for_form"]
        self.home_advantage = self.config["home_advantage_factor"]
        
        self.team_attack = {}
        self.team_defense = {}
        self.league_avg_home = 1.4
        self.league_avg_away = 1.1
    
    def fit(self, matches: pd.DataFrame):
        """Fit attack/defense strengths from historical matches"""
        # Filter finished matches with goals
        df = matches.dropna(subset=['home_goals', 'away_goals']).copy()
        
        # Calculate league averages
        self.league_avg_home = df['home_goals'].mean()
        self.league_avg_away = df['away_goals'].mean()
        
        teams = pd.unique(df[['home_team_canonical', 'away_team_canonical']].values.ravel())
        teams = [t for t in teams if pd.notna(t)]
        
        # Initialize
        for team in teams:
            self.team_attack[team] = 1.0
            self.team_defense[team] = 1.0
        
        # Iterative fitting (similar to Dixon-Coles but simpler)
        for _ in range(100):
            old_attack = self.team_attack.copy()
            old_defense = self.team_defense.copy()
            
            for team in teams:
                # Home matches
                home_matches = df[df['home_team_canonical'] == team]
                away_matches = df[df['away_team_canonical'] == team]
                
                if len(home_matches) > 0:
                    # Attack: goals scored at home / (league_avg_home * avg opponent defense)
                    opp_defense = np.mean([self.team_defense.get(row['away_team_canonical'], 1.0) 
                                           for _, row in home_matches.iterrows()])
                    goals_scored = home_matches['home_goals'].sum()
                    expected = len(home_matches) * self.league_avg_home * opp_defense
                    if expected > 0:
                        self.team_attack[team] = goals_scored / expected
                
                if len(away_matches) > 0:
                    # Defense: goals conceded away / (league_avg_away * avg opponent attack)
                    opp_attack = np.mean([self.team_attack.get(row['home_team_canonical'], 1.0) 
                                          for _, row in away_matches.iterrows()])
                    goals_conceded = away_matches['home_goals'].sum()
                    expected = len(away_matches) * self.league_avg_away * opp_attack
                    if expected > 0:
                        self.team_defense[team] = goals_conceded / expected
            
            # Normalize
            avg_attack = np.mean(list(self.team_attack.values()))
            avg_defense = np.mean(list(self.team_defense.values()))
            for team in teams:
                self.team_attack[team] /= avg_attack
                self.team_defense[team] /= avg_defense
            
            # Check convergence
            diff = sum(abs(self.team_attack[t] - old_attack.get(t, 1)) for t in teams)
            if diff < 0.001:
                break
    
    def predict_match(self, home_team: str, away_team: str, max_goals: int = 10) -> Dict:
        """Predict match outcome probabilities"""
        # Get strengths
        home_att = self.team_attack.get(home_team, 1.0)
        home_def = self.team_defense.get(home_team, 1.0)
        away_att = self.team_attack.get(away_team, 1.0)
        away_def = self.team_defense.get(away_team, 1.0)
        
        # Expected goals
        lambda_home = self.league_avg_home * home_att * away_def * self.home_advantage
        lambda_away = self.league_avg_away * away_att * home_def
        
        # Probability matrix
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                probs[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        # 1X2 probabilities
        p_home = np.sum(np.tril(probs, -1))
        p_draw = np.sum(np.diag(probs))
        p_away = np.sum(np.triu(probs, 1))
        
        # Over/Under 2.5
        p_over_25 = np.sum(probs[np.arange(max_goals + 1)[:, None] + np.arange(max_goals + 1) > 2.5])
        p_under_25 = 1 - p_over_25
        
        # BTTS
        p_btts_yes = 1 - probs[0, 0] - np.sum(probs[1:, 0]) - np.sum(probs[0, 1:])
        p_btts_no = 1 - p_btts_yes
        
        # Most likely scorelines
        scorelines = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scorelines.append(((i, j), probs[i, j]))
        scorelines.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
            'p_over_25': p_over_25,
            'p_under_25': p_under_25,
            'p_btts_yes': p_btts_yes,
            'p_btts_no': p_btts_no,
            'top_scorelines': scorelines[:5],
        }


class DixonColesModel:
    """Dixon-Coles model with time decay and rho correction"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["models"]["dixon_coles"]
        self.rho = self.config["rho"]
        self.time_decay = self.config["time_decay"]
        
        self.team_attack = {}
        self.team_defense = {}
        self.home_advantage = 0.0
        self.league_avg = 0.0
    
    def _tau(self, hx, ax, hx_param, ax_param):
        """Dixon-Coles tau correction for low scores"""
        if hx == 0 and ax == 0:
            return 1 - self.rho * hx_param * ax_param
        elif hx == 1 and ax == 0:
            return 1 + self.rho * ax_param
        elif hx == 0 and ax == 1:
            return 1 + self.rho * hx_param
        elif hx == 1 and ax == 1:
            return 1 - self.rho
        return 1.0
    
    def _log_likelihood(self, params, matches):
        """Negative log-likelihood for optimization"""
        n_teams = len(self.team_attack)
        # params: [attack_1..n, defense_1..n, home_adv, league_avg]
        attack = dict(zip(self.team_attack.keys(), params[:n_teams]))
        defense = dict(zip(self.team_defense.keys(), params[n_teams:2*n_teams]))
        home_adv = params[2*n_teams]
        league_avg = params[2*n_teams + 1]
        
        ll = 0.0
        for _, row in matches.iterrows():
            hx = row['home_goals']
            ax = row['away_goals']
            home = row['home_team_canonical']
            away = row['away_team_canonical']
            
            if pd.isna(hx) or pd.isna(ax):
                continue
            
            hx_param = attack[home] * defense[away] * np.exp(home_adv) * league_avg
            ax_param = attack[away] * defense[home] * league_avg
            
            tau = self._tau(hx, ax, hx_param, ax_param)
            
            if tau <= 0:
                return 1e10
            
            prob = (poisson.pmf(hx, hx_param) * poisson.pmf(ax, ax_param) * tau)
            if prob <= 0:
                return 1e10
            
            # Time weight
            days_ago = (pd.Timestamp.max - pd.Timestamp(row['date'])).days
            weight = np.exp(-self.time_decay * days_ago)
            
            ll -= weight * np.log(prob)
        
        return ll
    
    def fit(self, matches: pd.DataFrame):
        """Fit model parameters"""
        df = matches.dropna(subset=['home_goals', 'away_goals']).copy()
        df['date'] = pd.to_datetime(df['date'])
        
        teams = pd.unique(df[['home_team_canonical', 'away_team_canonical']].values.ravel())
        teams = [t for t in teams if pd.notna(t)]
        
        # Initialize
        for team in teams:
            self.team_attack[team] = 1.0
            self.team_defense[team] = 1.0
        self.home_advantage = 0.3
        self.league_avg = df['home_goals'].mean()
        
        n_teams = len(teams)
        # Initial params
        x0 = np.ones(2 * n_teams + 2)
        x0[:n_teams] = 1.0  # attack
        x0[n_teams:2*n_teams] = 1.0  # defense
        x0[2*n_teams] = 0.3  # home_adv
        x0[2*n_teams + 1] = self.league_avg  # league_avg
        
        # Constraints: average attack = 1, average defense = 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.mean(x[:n_teams]) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.mean(x[n_teams:2*n_teams]) - 1.0},
        ]
        bounds = [(0.01, 5)] * (2*n_teams) + [(-1, 1), (0.1, 5)]
        
        try:
            result = minimize(
                self._log_likelihood, x0, args=(df,),
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                self.team_attack = dict(zip(teams, result.x[:n_teams]))
                self.team_defense = dict(zip(teams, result.x[n_teams:2*n_teams]))
                self.home_advantage = result.x[2*n_teams]
                self.league_avg = result.x[2*n_teams + 1]
        except Exception as e:
            print(f"Dixon-Coles optimization failed: {e}")
    
    def predict_match(self, home_team: str, away_team: str, max_goals: int = 10) -> Dict:
        """Predict using fitted Dixon-Coles parameters"""
        home_att = self.team_attack.get(home_team, 1.0)
        home_def = self.team_defense.get(home_team, 1.0)
        away_att = self.team_attack.get(away_team, 1.0)
        away_def = self.team_defense.get(away_team, 1.0)
        
        lambda_home = self.league_avg * home_att * away_def * np.exp(self.home_advantage)
        lambda_away = self.league_avg * away_att * home_def
        
        # Same probability calculation as Poisson but with DC correction
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._tau(i, j, lambda_home, lambda_away)
                probs[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away) * tau
        
        # Normalize
        probs = probs / probs.sum()
        
        p_home = np.sum(np.tril(probs, -1))
        p_draw = np.sum(np.diag(probs))
        p_away = np.sum(np.triu(probs, 1))
        
        p_over_25 = np.sum(probs[np.arange(max_goals + 1)[:, None] + np.arange(max_goals + 1) > 2.5])
        p_under_25 = 1 - p_over_25
        
        p_btts_yes = 1 - probs[0, 0] - np.sum(probs[1:, 0]) - np.sum(probs[0, 1:])
        p_btts_no = 1 - p_btts_yes
        
        scorelines = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scorelines.append(((i, j), probs[i, j]))
        scorelines.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
            'p_over_25': p_over_25,
            'p_under_25': p_under_25,
            'p_btts_yes': p_btts_yes,
            'p_btts_no': p_btts_no,
            'top_scorelines': scorelines[:5],
        }


class EloModel:
    """Elo rating system for teams"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config["models"]["elo"]
        self.k_factor = self.config["k_factor"]
        self.home_advantage = self.config["home_advantage"]
        self.initial_rating = self.config["initial_rating"]
        
        self.ratings = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A vs team B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a: str, team_b: str, score_a: float, score_b: float, is_home_a: bool = True):
        """Update ratings after a match"""
        ra = self.ratings.get(team_a, self.initial_rating)
        rb = self.ratings.get(team_b, self.initial_rating)
        
        # Home advantage
        if is_home_a:
            ra += self.home_advantage
        else:
            rb += self.home_advantage
        
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        
        # Actual result: 1 for win, 0.5 for draw, 0 for loss
        if score_a > score_b:
            sa, sb = 1.0, 0.0
        elif score_a == score_b:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 0.0, 1.0
        
        self.ratings[team_a] = ra + self.k_factor * (sa - ea)
        self.ratings[team_b] = rb + self.k_factor * (sb - eb)
        
        # Remove home advantage from stored ratings
        if is_home_a:
            self.ratings[team_a] -= self.home_advantage
        else:
            self.ratings[team_b] -= self.home_advantage
    
    def fit(self, matches: pd.DataFrame):
        """Fit Elo ratings from historical matches chronologically"""
        df = matches.dropna(subset=['home_goals', 'away_goals', 'date']).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        teams = pd.unique(df[['home_team_canonical', 'away_team_canonical']].values.ravel())
        teams = [t for t in teams if pd.notna(t)]
        
        for team in teams:
            self.ratings[team] = self.initial_rating
        
        for _, row in df.iterrows():
            self.update_ratings(
                row['home_team_canonical'],
                row['away_team_canonical'],
                row['home_goals'],
                row['away_goals'],
                is_home_a=True
            )
    
    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Predict match outcome from Elo ratings"""
        home_rating = self.ratings.get(home_team, self.initial_rating) + self.home_advantage
        away_rating = self.ratings.get(away_team, self.initial_rating)
        
        # Convert Elo difference to probabilities using logistic curve
        diff = home_rating - away_rating
        
        # Approximate conversion: P(win) = 1 / (1 + 10^(-diff/400))
        p_home_win = self.expected_score(home_rating, away_rating)
        p_away_win = self.expected_score(away_rating, home_rating)
        p_draw = 1 - p_home_win - p_away_win
        
        # Adjust for draw probability (empirical)
        # In football, draw probability is typically ~25-30%
        p_draw = max(0.22, min(0.32, p_draw))
        remaining = 1 - p_draw
        p_home_win = p_home_win / (p_home_win + p_away_win) * remaining
        p_away_win = p_away_win / (p_home_win + p_away_win) * remaining
        
        # For over/under and BTTS, use Poisson approximation from Elo
        # Map Elo diff to expected goals
        avg_goals = 2.6  # Serie A average
        lambda_home = avg_goals / 2 * (1 + diff / 400)
        lambda_away = avg_goals / 2 * (1 - diff / 400)
        lambda_home = max(0.3, min(3.5, lambda_home))
        lambda_away = max(0.3, min(3.5, lambda_away))
        
        max_goals = 10
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                probs[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        p_over_25 = np.sum(probs[np.arange(max_goals + 1)[:, None] + np.arange(max_goals + 1) > 2.5])
        p_under_25 = 1 - p_over_25
        p_btts_yes = 1 - probs[0, 0] - np.sum(probs[1:, 0]) - np.sum(probs[0, 1:])
        p_btts_no = 1 - p_btts_yes
        
        scorelines = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scorelines.append(((i, j), probs[i, j]))
        scorelines.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'home_rating': home_rating - self.home_advantage,
            'away_rating': away_rating,
            'p_home': p_home_win,
            'p_draw': p_draw,
            'p_away': p_away_win,
            'p_over_25': p_over_25,
            'p_under_25': p_under_25,
            'p_btts_yes': p_btts_yes,
            'p_btts_no': p_btts_no,
            'top_scorelines': scorelines[:5],
        }


class EnsemblePredictor:
    """Combine multiple models with weights"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        if self.config["models"]["poisson"]["enabled"]:
            self.models['poisson'] = PoissonModel(config_path)
        if self.config["models"]["dixon_coles"]["enabled"]:
            self.models['dixon_coles'] = DixonColesModel(config_path)
        if self.config["models"]["elo"]["enabled"]:
            self.models['elo'] = EloModel(config_path)
        
        # Weights for ensemble (can be optimized based on backtesting)
        self.weights = {
            'poisson': 0.35,
            'dixon_coles': 0.40,
            'elo': 0.25,
        }
    
    def fit_all(self, matches: pd.DataFrame):
        """Fit all enabled models"""
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(matches)
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Get ensemble prediction"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_match(home_team, away_team)
        
        # Weighted average
        keys = ['p_home', 'p_draw', 'p_away', 'p_over_25', 'p_under_25', 'p_btts_yes', 'p_btts_no']
        ensemble = {}
        for key in keys:
            ensemble[key] = sum(
                self.weights.get(name, 0) * pred[key] 
                for name, pred in predictions.items()
            )
        
        # Top scorelines (from best model)
        best_model = max(self.models.keys(), key=lambda k: self.weights.get(k, 0))
        ensemble['top_scorelines'] = predictions[best_model]['top_scorelines']
        ensemble['models'] = predictions
        
        return ensemble
    
    def predict_upcoming(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Predict all upcoming matches"""
        results = []
        for _, row in matches_df.iterrows():
            pred = self.predict(row['home_team'], row['away_team'])
            results.append({
                'match_id': row['id'],
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                **{k: v for k, v in pred.items() if k != 'models'}
            })
        return pd.DataFrame(results)


def main():
    # Test with sample data
    session = get_session()
    matches = session.query(Match).filter(Match.status == 'FINISHED').all()
    
    if not matches:
        print("No matches in database. Run processing first.")
        return
    
    # Convert to DataFrame
    data = []
    for m in matches:
        data.append({
            'date': m.date,
            'home_team_canonical': m.home_team.name,
            'away_team_canonical': m.away_team.name,
            'home_goals': m.home_goals,
            'away_goals': m.away_goals,
            'home_xg': m.home_xg,
            'away_xg': m.away_xg,
        })
    df = pd.DataFrame(data)
    
    # Train ensemble
    ensemble = EnsemblePredictor()
    ensemble.fit_all(df)
    
    # Test prediction
    upcoming = session.query(Match).filter(Match.status == 'SCHEDULED').limit(5).all()
    for m in upcoming:
        pred = ensemble.predict(m.home_team.name, m.away_team.name)
        print(f"\n{m.home_team.name} vs {m.away_team.name} ({m.date})")
        print(f"  1X2: {pred['p_home']:.2%} / {pred['p_draw']:.2%} / {pred['p_away']:.2%}")
        print(f"  O/U 2.5: {pred['p_over_25']:.2%} / {pred['p_under_25']:.2%}")
        print(f"  BTTS: {pred['p_btts_yes']:.2%} / {pred['p_btts_no']:.2%}")
        print(f"  Top scorelines: {pred['top_scorelines'][:3]}")


if __name__ == "__main__":
    main()