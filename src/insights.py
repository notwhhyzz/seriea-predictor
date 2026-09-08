"""Pre-match insights: suspensions, form players, streaks, H2H.

Serie A disciplinary approximation: ban at 5th/10th/15th yellow; direct red = 1 match.
"""
import pandas as pd
from sqlalchemy import or_, and_
from src.database import Match, Team, Player, PlayerMatch, get_session


def _team_latest_season(session, team_id: int):
    row = (session.query(Match.season)
           .filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                   Match.status == 'FINISHED')
           .order_by(Match.date.desc()).first())
    return row[0] if row else None


def team_finished(session, team_name: str):
    """Finished matches (chronological) for a team in its latest season."""
    team = session.query(Team).filter(Team.name == team_name).first()
    if not team:
        return [], None
    season = _team_latest_season(session, team.id)
    if not season:
        return [], team
    matches = (session.query(Match)
               .filter(or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
                       Match.status == 'FINISHED', Match.season == season)
               .order_by(Match.date).all())
    return matches, team


def team_streaks(team_name: str) -> dict:
    """W/D/L runs, BTTS/Over streaks, goals for/against (last 5)."""
    session = get_session()
    try:
        matches, team = team_finished(session, team_name)
        if not matches:
            return {}
        rows = []
        for m in matches:
            home = m.home_team_id == team.id
            gf, ga = (m.home_goals, m.away_goals) if home else (m.away_goals, m.home_goals)
            rows.append({'date': m.date, 'res': 'W' if gf > ga else ('D' if gf == ga else 'L'),
                         'gf': gf, 'ga': ga})
        df = pd.DataFrame(rows)
        last5 = df.tail(5)
        res = df['res'].tolist()

        def run_len(pred):
            n = 0
            for r in reversed(res):
                if pred(r):
                    n += 1
                else:
                    break
            return n

        return {
            'played': len(df), 'last5': ''.join(last5['res'].tolist()),
            'unbeaten_run': run_len(lambda r: r != 'L'),
            'winless_run': run_len(lambda r: r != 'W'),
            'btts_streak': _streak(df, lambda r: r['gf'] > 0 and r['ga'] > 0),
            'over25_streak': _streak(df, lambda r: r['gf'] + r['ga'] > 2.5),
            'clean_sheets_l5': int((last5['ga'] == 0).sum()),
            'scored_l5': int(last5['gf'].sum()), 'conceded_l5': int(last5['ga'].sum()),
        }
    finally:
        session.close()


def _streak(df, pred):
    n = 0
    for _, r in df.iloc[::-1].iterrows():
        if pred(r):
            n += 1
        else:
            break
    return n


def player_status(team_name: str) -> dict:
    """Suspended / one-yellow-away (diffidati) players + season card table."""
    session = get_session()
    try:
        matches, team = team_finished(session, team_name)
        if not matches or not team:
            return {'suspended': [], 'warned': [], 'cards': []}
        season = matches[0].season
        last_id = matches[-1].id

        pm = (session.query(PlayerMatch, Player.name)
              .join(Player, PlayerMatch.player_id == Player.id)
              .filter(PlayerMatch.team_id == team.id, PlayerMatch.season == season).all())

        from collections import defaultdict
        agg = defaultdict(lambda: {'y': 0, 'r': 0, 'g': 0, 'a': 0, 'apps': 0,
                                   'y_last': 0, 'r_last': 0, 'name': ''})
        for row, name in pm:
            a = agg[row.player_id]
            a['name'] = name
            a['y'] += row.yellow or 0
            a['r'] += row.red or 0
            a['g'] += row.goals or 0
            a['a'] += row.assists or 0
            a['apps'] += 1 if (row.minutes or 0) > 0 else 0
            if row.match_id == last_id:
                a['y_last'] = row.yellow or 0
                a['r_last'] = row.red or 0

        suspended, warned, cards = [], [], []
        for pid, a in agg.items():
            if a['y'] or a['r']:
                cards.append({'player': a['name'], 'yellow': a['y'], 'red': a['r'],
                              'goals': a['g'], 'apps': a['apps']})
            if a['r_last'] > 0 or (a['y_last'] > 0 and a['y'] % 5 == 0):
                suspended.append({'player': a['name'],
                                  'reason': 'rosso diretto' if a['r_last'] > 0 else f"{a['y']}ª ammonizione"})
            elif a['y'] % 5 == 4 and a['y'] > 0:
                warned.append({'player': a['name'], 'yellow': a['y']})
        cards.sort(key=lambda c: (c['yellow'], c['red']), reverse=True)
        return {'suspended': suspended, 'warned': warned, 'cards': cards}
    finally:
        session.close()


def in_form_players(team_name: str, n: int = 5) -> list:
    """Players with scoring streaks or goals in recent matches + avg rating."""
    session = get_session()
    try:
        matches, team = team_finished(session, team_name)
        if not matches or not team:
            return []
        season = matches[0].season
        mids = [m.id for m in matches]
        rows = (session.query(PlayerMatch, Player.name)
                .join(Player, PlayerMatch.player_id == Player.id)
                .filter(PlayerMatch.team_id == team.id, PlayerMatch.match_id.in_(mids))
                .all())
        order = {mid: i for i, mid in enumerate(mids)}
        from collections import defaultdict
        per = defaultdict(list)
        for pm, name in rows:
            per[pm.player_id].append((order[pm.match_id], name, pm.goals or 0,
                                      pm.assists or 0, pm.rating, pm.minutes or 0))
        out = []
        for pid, games in per.items():
            games.sort()
            name = games[0][1]
            recent = games[-3:]
            g3 = sum(g[2] for g in recent)
            streak, ratings = 0, []
            for _, _, goals, _, rating, mins in reversed(games):
                if mins > 0 and goals > 0:
                    streak += 1
                elif mins > 0:
                    break
                if rating and mins > 0:
                    ratings.append(rating)
            avg_r = round(sum(ratings[:5]) / len(ratings[:5]), 2) if ratings else None
            if streak >= 2 or g3 >= 2 or (avg_r and avg_r >= 7.0):
                out.append({'player': name, 'streak': streak, 'goals_l3': g3,
                            'goals_season': sum(g[2] for g in games),
                            'assists_season': sum(g[3] for g in games),
                            'avg_rating_l5': avg_r})
        out.sort(key=lambda x: (x['streak'], x['goals_l3']), reverse=True)
        return out[:n]
    finally:
        session.close()


def season_scorers(team_name: str, n: int = 8) -> list:
    """Top scorers/assists of the team this season."""
    session = get_session()
    try:
        matches, team = team_finished(session, team_name)
        if not matches or not team:
            return []
        season = matches[0].season
        from sqlalchemy import func
        q = (session.query(Player.name, func.sum(PlayerMatch.goals).label('g'),
                           func.sum(PlayerMatch.assists).label('a'),
                           func.sum(PlayerMatch.minutes).label('m'))
             .join(PlayerMatch, PlayerMatch.player_id == Player.id)
             .filter(PlayerMatch.team_id == team.id, PlayerMatch.season == season)
             .group_by(Player.id).order_by(func.sum(PlayerMatch.goals).desc()).limit(n).all())
        return [{'player': name, 'goals': int(g or 0), 'assists': int(a or 0),
                 'minutes': int(m or 0)} for name, g, a, m in q]
    finally:
        session.close()


def player_match_log(player_name: str, team_name: str) -> pd.DataFrame:
    """Per-match log for a player this season."""
    session = get_session()
    try:
        team = session.query(Team).filter(Team.name == team_name).first()
        player = session.query(Player).filter(Player.name == player_name).first()
        if not team or not player:
            return pd.DataFrame()
        matches, _ = team_finished(session, team_name)
        season = matches[0].season if matches else None
        rows = (session.query(PlayerMatch, Match)
                .join(Match, PlayerMatch.match_id == Match.id)
                .filter(PlayerMatch.player_id == player.id,
                        PlayerMatch.team_id == team.id,
                        PlayerMatch.season == season)
                .order_by(Match.date).all())
        data = []
        for pm, m in rows:
            opp = m.away_team.name if m.home_team_id == team.id else m.home_team.name
            data.append({'date': m.date, 'venue': 'H' if m.home_team_id == team.id else 'A',
                         'opponent': opp, 'score': f"{m.home_goals}-{m.away_goals}",
                         'min': pm.minutes, 'goals': pm.goals, 'assists': pm.assists,
                         'yellow': pm.yellow, 'red': pm.red, 'rating': pm.rating})
        return pd.DataFrame(data)
    finally:
        session.close()


def h2h(home: str, away: str, n: int = 5) -> list:
    """Last n meetings between two teams (any venue)."""
    session = get_session()
    try:
        t1 = session.query(Team).filter(Team.name == home).first()
        t2 = session.query(Team).filter(Team.name == away).first()
        if not t1 or not t2:
            return []
        ms = (session.query(Match)
              .filter(or_(and_(Match.home_team_id == t1.id, Match.away_team_id == t2.id),
                          and_(Match.home_team_id == t2.id, Match.away_team_id == t1.id)),
                      Match.status == 'FINISHED')
              .order_by(Match.date.desc()).limit(n).all())
        return [{'date': m.date, 'home': m.home_team.name, 'away': m.away_team.name,
                 'score': f"{m.home_goals}-{m.away_goals}"} for m in reversed(ms)]
    finally:
        session.close()


def match_insights(home: str, away: str) -> list:
    """Bullet insights for an upcoming match. Each: (icon, text)."""
    bullets = []
    for team, tag in [(home, 'casa'), (away, 'trasferta')]:
        st = team_streaks(team)
        if not st:
            continue
        if st.get('unbeaten_run', 0) >= 4:
            bullets.append(('🛡️', f"{team}: imbattuta da {st['unbeaten_run']} partite"))
        if st.get('winless_run', 0) >= 4:
            bullets.append(('📉', f"{team}: senza vittorie da {st['winless_run']} partite ({tag})"))
        if st.get('over25_streak', 0) >= 3:
            bullets.append(('⚽', f"{team}: Over 2.5 in {st['over25_streak']} gare di fila"))
        if st.get('btts_streak', 0) >= 3:
            bullets.append(('🤝', f"{team}: GG in {st['btts_streak']} gare di fila"))
        ps = player_status(team)
        for s in ps.get('suspended', []):
            bullets.append(('🚫', f"{team}: {s['player']} SQUALIFICATO ({s['reason']})"))
        for w in ps.get('warned', [])[:2]:
            bullets.append(('⚠️', f"{team}: {w['player']} diffidato ({w['yellow']} gialli)"))
        for f in in_form_players(team, 2):
            if f['streak'] >= 2:
                bullets.append(('🔥', f"{team}: {f['player']} a segno in {f['streak']} gare di fila"))
            elif f['goals_l3'] >= 2:
                bullets.append(('🔥', f"{team}: {f['player']} {f['goals_l3']} gol nelle ultime 3"))
    h = h2h(home, away, 3)
    if h:
        last = h[-1]
        bullets.append(('🔄', f"H2H recente: {last['home']} {last['score']} {last['away']} ({last['date']})"))
    return bullets
