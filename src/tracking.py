"""Tracking helpers: prediction history, edges, bet slips, Kelly staking"""
import pandas as pd
from datetime import date
from typing import List, Dict, Optional
from src.database import Match, PredictionLog, BetSlip, get_session


# Market definitions: (market, selection label, pred column, odds column)
MARKETS = [
    ('1X2', [('1', 'pred_home', 'odds_home'), ('X', 'pred_draw', 'odds_draw'), ('2', 'pred_away', 'odds_away')]),
    ('O/U 2.5', [('Over 2.5', 'pred_over_25', 'odds_over_25'), ('Under 2.5', 'pred_under_25', 'odds_under_25')]),
    ('BTTS', [('GG Sì', 'pred_btts_yes', 'odds_btts_yes'), ('GG No', 'pred_btts_no', 'odds_btts_no')]),
]


def current_edges(min_edge: float = 0.05, days: int = 14) -> List[Dict]:
    """All current model-vs-Gamdom edges above threshold, sorted desc."""
    from sqlalchemy import and_
    from datetime import timedelta
    session = get_session()
    cutoff = date.today() + timedelta(days=days)
    matches = session.query(Match).filter(
        and_(Match.date >= date.today(), Match.date <= cutoff,
             Match.status.in_(['SCHEDULED', 'TIMED']))
    ).order_by(Match.date).all()

    edges = []
    for m in matches:
        for market, legs in MARKETS:
            for label, pred_col, odds_col in legs:
                p = getattr(m, pred_col, None)
                o = getattr(m, odds_col, None)
                if p is None or o is None or o <= 1:
                    continue
                implied = 1 / o
                edge = p - implied
                if edge >= min_edge:
                    edges.append({
                        'match_id': m.id, 'date': m.date,
                        'home': m.home_team.name, 'away': m.away_team.name,
                        'market': market, 'selection': label,
                        'model_prob': round(float(p), 4), 'odds': round(float(o), 2),
                        'implied': round(implied, 4), 'edge': round(edge, 4),
                    })
    session.close()
    return sorted(edges, key=lambda e: e['edge'], reverse=True)


def kelly_stake(prob: float, odds: float, bankroll: float = 100.0,
                frac: float = 0.25, cap_pct: float = 5.0) -> float:
    """Quarter-Kelly stake in units. Returns 0 if no edge."""
    if odds <= 1 or prob <= 0:
        return 0.0
    f = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    if f <= 0:
        return 0.0
    return round(min(f * frac * bankroll, cap_pct / 100 * bankroll), 2)


def create_slip(legs: List[Dict], stake: float, name: Optional[str] = None) -> int:
    """Save a bet slip. legs: [{match_id, date, home, away, market, selection, model_prob, odds}]."""
    total_odds, combined_prob = 1.0, 1.0
    clean_legs = []
    for leg in legs:
        total_odds *= leg['odds']
        combined_prob *= leg['model_prob']
        clean_legs.append({
            'match_id': leg['match_id'], 'date': str(leg.get('date')),
            'home': leg.get('home'), 'away': leg.get('away'),
            'market': leg['market'], 'selection': leg['selection'],
            'model_prob': leg['model_prob'], 'odds': leg['odds'],
        })
    session = get_session()
    slip = BetSlip(
        name=name or f"Slip {date.today().isoformat()}",
        legs=clean_legs,
        total_odds=round(total_odds, 2),
        combined_prob=round(combined_prob, 4),
        stake=float(stake),
        status='OPEN',
    )
    session.add(slip)
    session.commit()
    slip_id = slip.id
    session.close()
    return slip_id


def get_slips(status: Optional[str] = None) -> pd.DataFrame:
    """Bet slips as DataFrame, optionally filtered by status."""
    session = get_session()
    q = session.query(BetSlip).order_by(BetSlip.created_at.desc())
    if status:
        q = q.filter(BetSlip.status == status)
    rows = [{
        'id': s.id, 'created': s.created_at, 'name': s.name,
        'legs': len(s.legs or []), 'quota': s.total_odds,
        'prob': s.combined_prob, 'stake': s.stake,
        'status': s.status, 'profit': s.profit, 'settled': s.settled_at,
        '_legs': s.legs or [],
    } for s in q.all()]
    session.close()
    return pd.DataFrame(rows)


def slip_summary() -> Dict:
    """Win rate, ROI, profit across closed slips."""
    df = get_slips()
    if df.empty:
        return {'total': 0, 'open': 0, 'won': 0, 'lost': 0, 'win_rate': None,
                'staked': 0.0, 'profit': 0.0, 'roi': None}
    closed = df[df['status'].isin(['WON', 'LOST'])]
    staked = float(closed['stake'].sum()) if len(closed) else 0.0
    profit = float(closed['profit'].sum()) if len(closed) else 0.0
    return {
        'total': len(df), 'open': int((df['status'] == 'OPEN').sum()),
        'won': int((df['status'] == 'WON').sum()), 'lost': int((df['status'] == 'LOST').sum()),
        'win_rate': round((df['status'] == 'WON').sum() / len(closed), 3) if len(closed) else None,
        'staked': round(staked, 2), 'profit': round(profit, 2),
        'roi': round(profit / staked, 3) if staked else None,
    }


def get_prediction_logs() -> pd.DataFrame:
    """All prediction snapshots with match names."""
    session = get_session()
    rows = []
    for log in session.query(PredictionLog).order_by(PredictionLog.log_date.desc()).all():
        m = session.query(Match).filter(Match.id == log.match_id).first()
        rows.append({
            'log_date': log.log_date, 'match_date': m.date if m else None,
            'match': f"{m.home_team.name} vs {m.away_team.name}" if m else f"#{log.match_id}",
            'p_home': log.p_home, 'p_draw': log.p_draw, 'p_away': log.p_away,
            'p_over25': log.p_over25, 'p_btts_yes': log.p_btts_yes,
            'odds_1': log.odds_1, 'odds_x': log.odds_x, 'odds_2': log.odds_2,
            'result': log.result, 'score': f"{log.home_goals}-{log.away_goals}" if log.home_goals is not None else None,
            'hit_1x2': log.hit_1x2, 'hit_ou25': log.hit_ou25, 'hit_btts': log.hit_btts,
        })
    session.close()
    return pd.DataFrame(rows)


def history_summary(df: pd.DataFrame) -> Dict:
    """Hit rates per market on settled logs + calibration buckets."""
    out = {'n_total': len(df)}
    closed = df[df['result'].notna()] if len(df) else df
    out['n_closed'] = len(closed)
    out['n_open'] = len(df) - len(closed)
    for col, key in [('hit_1x2', 'acc_1x2'), ('hit_ou25', 'acc_ou25'), ('hit_btts', 'acc_btts')]:
        sub = closed[closed[col].notna()]
        out[key] = round(float(sub[col].mean()), 3) if len(sub) else None
        out[key + '_n'] = len(sub)
    # Calibration: model confidence (max 1X2 prob) vs actual hit rate
    cal = []
    if len(closed):
        tmp = closed.dropna(subset=['p_home', 'p_draw', 'p_away', 'hit_1x2']).copy()
        if len(tmp):
            tmp['conf'] = tmp[['p_home', 'p_draw', 'p_away']].max(axis=1)
            tmp['bucket'] = (tmp['conf'] * 10).astype(int).clip(0, 9).map(lambda b: f"{b*10}-{b*10+10}%")
            g = tmp.groupby('bucket', sort=True).agg(n=('hit_1x2', 'size'), hit=('hit_1x2', 'mean')).reset_index()
            cal = g.to_dict('records')
    out['calibration'] = cal
    return out
