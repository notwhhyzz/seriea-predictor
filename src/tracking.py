"""Tracking helpers: prediction history, edges, bet slips, Kelly staking"""
import pandas as pd
from datetime import date
from typing import List, Dict, Optional
from sqlalchemy import or_
from src.database import Match, PredictionLog, BetSlip, get_session


# Market definitions: (market, selection label, pred column, odds column)
# DC probs are derived (no DB column): dc_1x = p_home + p_draw, etc.
MARKETS = [
    ('1X2', [('1', 'pred_home', 'odds_home'), ('X', 'pred_draw', 'odds_draw'), ('2', 'pred_away', 'odds_away')]),
    ('DC', [('1X', 'pred_dc_1x', 'odds_dc_1x'), ('12', 'pred_dc_12', 'odds_dc_12'), ('X2', 'pred_dc_x2', 'odds_dc_x2')]),
    ('O/U 2.5', [('Over 2.5', 'pred_over_25', 'odds_over_25'), ('Under 2.5', 'pred_under_25', 'odds_under_25')]),
    ('BTTS', [('GG Sì', 'pred_btts_yes', 'odds_btts_yes'), ('GG No', 'pred_btts_no', 'odds_btts_no')]),
]


def dc_probs(p_home: float, p_draw: float, p_away: float) -> dict:
    """Double chance probabilities derived from 1X2."""
    return {'pred_dc_1x': p_home + p_draw, 'pred_dc_12': p_home + p_away, 'pred_dc_x2': p_draw + p_away}


def current_edges(min_edge: float = 0.05, days: int = 14, include_scorers: bool = True) -> List[Dict]:
    """All current model-vs-Gamdom edges above threshold, sorted desc.
    Includes Double Chance (derived) and anytime-scorer legs (experimental Poisson model)."""
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
        probs = {'pred_home': m.pred_home_win, 'pred_draw': m.pred_draw, 'pred_away': m.pred_away_win,
                 'pred_over_25': m.pred_over_25, 'pred_under_25': m.pred_under_25,
                 'pred_btts_yes': m.pred_btts_yes, 'pred_btts_no': m.pred_btts_no}
        if all(v is not None for v in [m.pred_home_win, m.pred_draw, m.pred_away_win]):
            probs.update(dc_probs(m.pred_home_win, m.pred_draw, m.pred_away_win))
        odds_map = {'odds_home': m.odds_home, 'odds_draw': m.odds_draw, 'odds_away': m.odds_away,
                    'odds_dc_1x': m.odds_dc_1x, 'odds_dc_12': m.odds_dc_12, 'odds_dc_x2': m.odds_dc_x2,
                    'odds_over_25': m.odds_over_25, 'odds_under_25': m.odds_under_25,
                    'odds_btts_yes': m.odds_btts_yes, 'odds_btts_no': m.odds_btts_no}
        for market, legs in MARKETS:
            for label, pred_col, odds_col in legs:
                p, o = probs.get(pred_col), odds_map.get(odds_col)
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
        if include_scorers and m.scorer_odds:
            for leg in scorer_edges(session, m, min_edge):
                edges.append(leg)
    session.close()
    return sorted(edges, key=lambda e: e['edge'], reverse=True)


def match_player_name(oddspapi_name: str, candidates: List[str]) -> Optional[str]:
    """Match 'Surname, Name' (Gamdom) to our canonical player names."""
    from difflib import SequenceMatcher
    parts = [p.strip() for p in (oddspapi_name or '').split(',')]
    reordered = f"{parts[1]} {parts[0]}" if len(parts) == 2 else (oddspapi_name or '')
    best, score = None, 0.0
    for c in candidates:
        r = SequenceMatcher(None, reordered.lower(), (c or '').lower()).ratio()
        if r > score:
            best, score = c, r
    return best if score >= 0.75 else None


def anytime_prob(session, player_name: str, team_name: str) -> Optional[float]:
    """Experimental anytime-scorer probability: Poisson from season goals/90
    (shrunk to a 0.12/90 prior) x expected minutes (avg last 5)."""
    import math
    from src.database import Player, PlayerMatch, Team, Match as M
    team = session.query(Team).filter(Team.name == team_name).first()
    player = session.query(Player).filter(Player.name == player_name).first()
    if not team or not player:
        return None
    season = session.query(M.season).filter(
        or_(M.home_team_id == team.id, M.away_team_id == team.id),
        M.status == 'FINISHED').order_by(M.date.desc()).first()
    if not season:
        return None
    rows = (session.query(PlayerMatch, M.date)
            .join(M, PlayerMatch.match_id == M.id)
            .filter(PlayerMatch.player_id == player.id, PlayerMatch.season == season[0])
            .order_by(M.date.desc()).all())
    if not rows:
        return None
    mins_total = sum(r[0].minutes or 0 for r in rows)
    if mins_total < 90:
        return None
    goals_total = sum(r[0].goals or 0 for r in rows)
    rate90 = (goals_total + 0.36) / (mins_total / 90 + 3)
    recent = rows[:5]
    exp_min = sum(r[0].minutes or 0 for r in recent) / max(len(recent), 1)
    exp_min = max(10.0, min(90.0, exp_min))
    return round(1 - math.exp(-rate90 * exp_min / 90), 4)


def match_scorer_board(match_id: int) -> List[Dict]:
    """Gamdom anytime odds + our form (goals last 3) per scorer. For match detail UI."""
    from src.database import Player, PlayerMatch, Match as M
    session = get_session()
    try:
        m = session.query(Match).filter(Match.id == match_id).first()
        if not m or not m.scorer_odds:
            return []
        our_names = [p.name for p in session.query(Player).filter(
            or_(Player.team_id == m.home_team_id, Player.team_id == m.away_team_id)).all()]
        board = []
        for s in (m.scorer_odds or [])[:12]:
            canon = match_player_name(s.get('player', ''), our_names)
            g3, team_side = None, None
            if canon:
                pl = session.query(Player).filter(Player.name == canon).first()
                if pl:
                    team_side = m.home_team.name if pl.team_id == m.home_team_id else m.away_team.name
                    pm = (session.query(PlayerMatch, M.date)
                          .join(M, PlayerMatch.match_id == M.id)
                          .filter(PlayerMatch.player_id == pl.id)
                          .order_by(M.date.desc()).limit(3).all())
                    g3 = sum(r[0].goals or 0 for r in pm)
            board.append({'player': s.get('player'), 'odds': s.get('odds'),
                          'matched': canon, 'team': team_side, 'goals_l3': g3})
        return board
    finally:
        session.close()


def scorer_edges(session, m, min_edge: float) -> List[Dict]:
    """Anytime-scorer legs for a match (experimental model vs Gamdom)."""
    from src.database import Player
    if not m.scorer_odds:
        return []
    our_names = [p.name for p in session.query(Player).filter(
        or_(Player.team_id == m.home_team_id, Player.team_id == m.away_team_id)).all()]
    legs = []
    for s in (m.scorer_odds or [])[:15]:
        canon = match_player_name(s.get('player', ''), our_names)
        if not canon or not s.get('odds') or s['odds'] <= 1:
            continue
        pl = session.query(Player).filter(Player.name == canon).first()
        if not pl:
            continue
        team_name = m.home_team.name if pl.team_id == m.home_team_id else m.away_team.name
        p = anytime_prob(session, canon, team_name)
        if p is None:
            continue
        implied = 1 / s['odds']
        edge = p - implied
        if edge >= min_edge:
            legs.append({
                'match_id': m.id, 'date': m.date,
                'home': m.home_team.name, 'away': m.away_team.name,
                'market': 'MARCATORE', 'selection': f"{canon} segna", 'player': canon,
                'model_prob': p, 'odds': round(float(s['odds']), 2),
                'implied': round(implied, 4), 'edge': round(edge, 4), 'experimental': True,
            })
    return legs


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
            'player': leg.get('player'), 'model_prob': leg['model_prob'], 'odds': leg['odds'],
            'experimental': bool(leg.get('experimental')),
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
