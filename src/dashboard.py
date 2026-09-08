"""Streamlit dashboard for Serie A predictions"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from sqlalchemy import and_, or_, desc
from src.database import Match, Team, ModelPerformance, get_session
import yaml


st.set_page_config(
    page_title="Serie A Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark pro styling
import plotly.io as pio
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
  /* Typography */
  h1 { letter-spacing: -0.5px; font-weight: 800 !important; }
  h2, h3 { letter-spacing: -0.3px; font-weight: 700 !important; }

  /* Top tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; padding: 4px 0; }
  .stTabs [data-baseweb="tab"] {
    font-size: 0.85rem; font-weight: 600;
    padding: 10px 16px; border-radius: 10px 10px 0 0;
    color: #9AA4B2;
  }
  .stTabs [aria-selected="true"] {
    color: #22C55E !important;
    border-bottom: 2px solid #22C55E !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #151A23;
    border: 1px solid #232B38;
    border-radius: 12px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] { color: #9AA4B2 !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"] { font-weight: 800 !important; }

  /* Stat boxes (loop-safe alternative to st.metric) */
  .stat { background: #151A23; border: 1px solid #232B38; border-radius: 12px;
          padding: 10px 14px; text-align: center; }
  .stat .stat-label { color: #9AA4B2; font-size: 0.75rem; font-weight: 600; }
  .stat .stat-value { font-size: 1.35rem; font-weight: 800; margin-top: 2px; }
  [data-testid="stMetricLabel"] { color: #9AA4B2 !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"] { font-weight: 800 !important; }

  /* Expanders + buttons */
  [data-testid="stExpander"] { border: 1px solid #232B38; border-radius: 12px; }
  .stButton > button { border-radius: 10px; font-weight: 600; }
  .stDownloadButton > button { border-radius: 10px; }

  /* Badges */
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 0.75rem; font-weight: 700; }
  .pill-open { background: #1E3A2B; color: #22C55E; border: 1px solid #22C55E55; }
  .pill-won { background: #123B22; color: #4ADE80; border: 1px solid #4ADE80; }
  .pill-lost { background: #3B1212; color: #F87171; border: 1px solid #F87171; }
  .pill-edge { background: #13233B; color: #60A5FA; border: 1px solid #60A5FA55; }
  .hero {
    background: linear-gradient(135deg, #0F2E1D 0%, #151A23 60%);
    border: 1px solid #22C55E33;
    border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
  }
  .hero h1 { margin: 0; font-size: 1.9rem; }
  .hero p { margin: 4px 0 0 0; color: #9AA4B2; }
  .section-title { margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# Load config
@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

CONFIG = load_config()


@st.cache_resource
def get_db_session():
    return get_session()


def get_upcoming_matches(days: int = 14):
    """Fetch upcoming matches with predictions"""
    session = get_db_session()
    cutoff = date.today() + timedelta(days=days)
    
    matches = session.query(Match).filter(
        and_(
            Match.date >= date.today(),
            Match.date <= cutoff,
            Match.status.in_(['SCHEDULED', 'TIMED'])
        )
    ).order_by(Match.date).all()
    
    data = []
    for m in matches:
        data.append({
            'id': m.id,
            'date': m.date,
            'matchday': m.matchday,
            'home_team': m.home_team.name,
            'away_team': m.away_team.name,
            'pred_home': m.pred_home_win,
            'pred_draw': m.pred_draw,
            'pred_away': m.pred_away_win,
            'pred_over_25': m.pred_over_25,
            'pred_under_25': m.pred_under_25,
            'pred_btts_yes': m.pred_btts_yes,
            'pred_btts_no': m.pred_btts_no,
            'odds_home': m.odds_home,
            'odds_draw': m.odds_draw,
            'odds_away': m.odds_away,
            'odds_over_25': m.odds_over_25,
            'odds_under_25': m.odds_under_25,
            'odds_btts_yes': m.odds_btts_yes,
            'odds_btts_no': m.odds_btts_no,
            'odds_source': getattr(m, 'odds_source', None),
        })
    
    session.close()
    return pd.DataFrame(data)


def get_recent_results(limit: int = 20):
    """Fetch recent finished matches"""
    session = get_db_session()
    matches = session.query(Match).filter(
        Match.status == 'FINISHED'
    ).order_by(desc(Match.date)).limit(limit).all()
    
    data = []
    for m in matches:
        data.append({
            'date': m.date,
            'matchday': m.matchday,
            'home_team': m.home_team.name,
            'away_team': m.away_team.name,
            'home_goals': m.home_goals,
            'away_goals': m.away_goals,
            'result': m.result,
            'home_xg': m.home_xg,
            'away_xg': m.away_xg,
            'pred_home': m.pred_home_win,
            'pred_draw': m.pred_draw,
            'pred_away': m.pred_away_win,
        })
    
    session.close()
    return pd.DataFrame(data)


def get_team_form(team_name: str, n: int = 5):
    """Get last N matches for a team"""
    session = get_db_session()
    team = session.query(Team).filter(Team.name == team_name).first()
    if not team:
        return pd.DataFrame()
    
    matches = session.query(Match).filter(
        or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
        Match.status == 'FINISHED'
    ).order_by(desc(Match.date)).limit(n).all()
    
    data = []
    for m in reversed(matches):  # Chronological
        is_home = m.home_team_id == team.id
        data.append({
            'date': m.date,
            'opponent': m.away_team.name if is_home else m.home_team.name,
            'venue': 'H' if is_home else 'A',
            'goals_for': m.home_goals if is_home else m.away_goals,
            'goals_against': m.away_goals if is_home else m.home_goals,
            'result': 'W' if (is_home and m.result == 'H') or (not is_home and m.result == 'A')
                      else 'L' if (is_home and m.result == 'A') or (not is_home and m.result == 'H') else 'D',
            'xg_for': m.home_xg if is_home else m.away_xg,
            'xg_against': m.away_xg if is_home else m.home_xg,
        })
    
    session.close()
    return pd.DataFrame(data)


def get_standings():
    """Calculate current standings from finished matches"""
    session = get_db_session()
    matches = session.query(Match).filter(Match.status == 'FINISHED').all()
    
    standings = {}
    for m in matches:
        for team_name, goals_for, goals_against, is_home in [
            (m.home_team.name, m.home_goals, m.away_goals, True),
            (m.away_team.name, m.away_goals, m.home_goals, False)
        ]:
            if team_name not in standings:
                standings[team_name] = {'MP': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}
            
            s = standings[team_name]
            s['MP'] += 1
            s['GF'] += goals_for or 0
            s['GA'] += goals_against or 0
            s['GD'] = s['GF'] - s['GA']
            
            if goals_for is not None and goals_against is not None:
                if goals_for > goals_against:
                    s['W'] += 1
                    s['Pts'] += 3
                elif goals_for == goals_against:
                    s['D'] += 1
                    s['Pts'] += 1
                else:
                    s['L'] += 1
    
    df = pd.DataFrame.from_dict(standings, orient='index')
    df = df.sort_values(['Pts', 'GD', 'GF'], ascending=False).reset_index()
    df.columns = ['Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
    df['Pos'] = range(1, len(df) + 1)
    
    session.close()
    return df


@st.cache_data(ttl=6 * 3600)
def get_live_standings():
    """Fetch live current-season standings from football-data.org API.
    Returns (DataFrame, season_label) or (None, None) on failure."""
    try:
        from src.fetchers.api_football import FootballDataOrgAPI
        from src.processing import TeamNameMapper
        api = FootballDataOrgAPI()
        if not api.enabled:
            return None, None
        table = api.get_standings()
        if not table:
            return None, None
        mapper = TeamNameMapper()
        rows = []
        for t in table:
            rows.append({
                'Pos': t.get('position'),
                'Team': mapper.map_to_canonical(t['team'].get('name', ''), 'football-data.org'),
                'MP': t.get('playedGames', 0),
                'W': t.get('won', 0),
                'D': t.get('draw', 0),
                'L': t.get('lost', 0),
                'GF': t.get('goalsFor', 0),
                'GA': t.get('goalsAgainst', 0),
                'GD': t.get('goalDifference', 0),
                'Pts': t.get('points', 0),
            })
        df = pd.DataFrame(rows).sort_values('Pos').reset_index(drop=True)
        comp = api.get_competition_info() or {}
        season = comp.get('currentSeason', {})
        label = f"{season.get('startDate', '')[:4]}/{season.get('endDate', '')[2:4]} (giornata {season.get('currentMatchday', '—')})"
        return df, label
    except Exception:
        return None, None


def plot_match_prediction(match_row):
    """Create a visual prediction chart for a single match"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('1X2', 'Over/Under 2.5', 'BTTS'),
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]]
    )
    
    # 1X2
    fig.add_trace(go.Pie(
        labels=['Home Win', 'Draw', 'Away Win'],
        values=[match_row['pred_home'], match_row['pred_draw'], match_row['pred_away']],
        marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
        hole=0.4,
        textinfo='label+percent',
    ), row=1, col=1)
    
    # Over/Under
    fig.add_trace(go.Pie(
        labels=['Over 2.5', 'Under 2.5'],
        values=[match_row['pred_over_25'], match_row['pred_under_25']],
        marker_colors=['#3498db', '#9b59b6'],
        hole=0.4,
        textinfo='label+percent',
    ), row=1, col=2)
    
    # BTTS
    fig.add_trace(go.Pie(
        labels=['BTTS Yes', 'BTTS No'],
        values=[match_row['pred_btts_yes'], match_row['pred_btts_no']],
        marker_colors=['#e67e22', '#34495e'],
        hole=0.4,
        textinfo='label+percent',
    ), row=1, col=3)
    
    fig.update_layout(height=300, showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
    return fig


def plot_team_form(form_df, team_name):
    """Plot team recent form"""
    if form_df.empty:
        return None
    
    fig = go.Figure()
    
    # Results as bars
    colors = {'W': '#2ecc71', 'D': '#f39c12', 'L': '#e74c3c'}
    fig.add_trace(go.Bar(
        x=form_df['date'].astype(str),
        y=form_df['goals_for'],
        name='Goals For',
        marker_color=[colors.get(r, '#95a5a6') for r in form_df['result']],
        text=form_df['result'],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        x=form_df['date'].astype(str),
        y=-form_df['goals_against'],
        name='Goals Against',
        marker_color='rgba(231, 76, 60, 0.6)',
    ))
    
    # xG line
    if 'xg_for' in form_df.columns and form_df['xg_for'].notna().any():
        fig.add_trace(go.Scatter(
            x=form_df['date'].astype(str),
            y=form_df['xg_for'],
            name='xG For',
            mode='lines+markers',
            line=dict(color='#3498db', dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=form_df['date'].astype(str),
            y=-form_df['xg_against'],
            name='xG Against',
            mode='lines+markers',
            line=dict(color='#e74c3c', dash='dot'),
        ))
    
    fig.update_layout(
        title=f'{team_name} - Last {len(form_df)} Matches',
        barmode='relative',
        height=300,
        yaxis_title='Goals',
        xaxis_title='Date',
    )
    return fig


def stat_box(label: str, value: str):
    """Loop-safe stat display (st.metric has no key support on this version)."""
    st.markdown(f"<div class='stat'><div class='stat-label'>{label}</div>"
                f"<div class='stat-value'>{value}</div></div>", unsafe_allow_html=True)


# Sidebar (status only — navigation moved to top tabs)
st.sidebar.title("⚽ Serie A Predictor")
st.sidebar.caption(f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
try:
    _sess = get_db_session()
    _n_up = _sess.query(Match).filter(Match.status.in_(['SCHEDULED', 'TIMED'])).count()
    _n_odds = _sess.query(Match).filter(Match.odds_source == 'gamdom').count()
    _sess.close()
    st.sidebar.metric("Partite monitorate", _n_up)
    st.sidebar.metric("Con quote Gamdom", _n_odds)
except Exception:
    pass

# Top navigation tabs
tab_dash, tab_next, tab_table, tab_team, tab_value, tab_hist, tab_slips, tab_conf = st.tabs([
    "🏠 Dashboard", "📅 Prossime", "📊 Classifica", "📈 Squadre",
    "🎯 Value Bets", "📜 History", "🧾 Bet Slips", "⚙️ Setup",
])

# Main content
with tab_dash:
    st.markdown("""<div class="hero"><h1>⚽ Serie A Predictor</h1>
    <p>Modello ensemble Poisson + Dixon-Coles + Elo &nbsp;•&nbsp; Quote live Gamdom &nbsp;•&nbsp; Edge detection</p></div>""",
                unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    upcoming = get_upcoming_matches(7)
    recent = get_recent_results(10)
    
    with col1:
        st.metric("Prossime 7 giorni", len(upcoming))
    with col2:
        st.metric("Partite recenti", len(recent))
    with col3:
        if len(recent) > 0:
            scored = recent.dropna(subset=['pred_home', 'pred_draw', 'pred_away'])
            if len(scored) > 0:
                correct = sum(1 for _, r in scored.iterrows()
                             if (r['pred_home'] > 0.5 and r['result'] == 'H') or
                                (r['pred_draw'] > 0.5 and r['result'] == 'D') or
                                (r['pred_away'] > 0.5 and r['result'] == 'A'))
                st.metric("Accuracy recente", f"{correct/len(scored):.1%} ({correct}/{len(scored)})")
            else:
                st.metric("Accuracy recente", "N/D")
    with col4:
        st.metric("Modello", "Ensemble v1")
    
    # Upcoming matches this week
    st.subheader("⚽ Prossime Partite (7 giorni)")
    upcoming_scored = upcoming.dropna(subset=['pred_home', 'pred_draw', 'pred_away']) if len(upcoming) > 0 else upcoming
    if len(upcoming_scored) > 0:
        for _, match in upcoming_scored.iterrows():
            with st.expander(f"{match['date'].strftime('%a %d/%m')} - {match['home_team']} vs {match['away_team']}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(plot_match_prediction(match), use_container_width=True,
                                    key=f"dash_pie_{match['id']}")
                with col2:
                    st.markdown("**Quote (se disponibili)**")
                    if pd.notna(match['odds_home']):
                        st.write(f"1: {match['odds_home']:.2f} | X: {match['odds_draw']:.2f} | 2: {match['odds_away']:.2f}")
                    else:
                        st.write("Quote non disponibili")
                    
                    # Value indicator
                    if pd.notna(match['odds_home']) and pd.notna(match['pred_home']) and match['odds_home'] > 0:
                        implied_home = 1 / match['odds_home']
                        if match['pred_home'] > implied_home + 0.05:
                            st.success(f"💡 Value Home: {match['pred_home']:.1%} vs {implied_home:.1%} implied")
    
    # Recent results with prediction accuracy
    st.subheader("📋 Risultati Recenti")
    if len(recent) > 0:
        display_df = recent[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result',
                            'pred_home', 'pred_draw', 'pred_away']].copy()
        display_df['Prediction'] = display_df.apply(
            lambda r: ('1' if r['pred_home'] > max(r['pred_draw'], r['pred_away'])
                      else 'X' if r['pred_draw'] > r['pred_away'] else '2')
                      if pd.notna(r['pred_home']) and pd.notna(r['pred_draw']) and pd.notna(r['pred_away'])
                      else '—', axis=1
        )
        display_df['Correct'] = display_df.apply(
            lambda r: r['Prediction'] == r['result'] if r['Prediction'] != '—' else '—', axis=1)
        st.dataframe(
            display_df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'Prediction', 'Correct']],
            use_container_width=True,
            hide_index=True
        )

with tab_next:
    st.title("Prossime Partite Serie A")
    
    days = st.slider("Giorni avanti", 1, 30, 14)
    upcoming = get_upcoming_matches(days)
    
    if len(upcoming) == 0:
        st.info("Nessuna partita in programma nel periodo selezionato")
    else:
        # Group by date
        for match_date, group in upcoming.groupby('date'):
            st.subheader(f"📅 {match_date.strftime('%A %d %B %Y')}")
            for _, match in group.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{match['home_team']}** vs **{match['away_team']}**")
                        if pd.notna(match['matchday']):
                            st.caption(f"Giornata {int(match['matchday'])}")
                    
                    with col2:
                        stat_box("1", f"{match['pred_home']:.1%}" if pd.notna(match['pred_home']) else "—")
                    with col3:
                        stat_box("X", f"{match['pred_draw']:.1%}" if pd.notna(match['pred_draw']) else "—")
                    with col4:
                        stat_box("2", f"{match['pred_away']:.1%}" if pd.notna(match['pred_away']) else "—")
                    with col5:
                        stat_box("Ov 2.5", f"{match['pred_over_25']:.1%}" if pd.notna(match['pred_over_25']) else "—")
                    
                    # Detail expander
                    with st.expander("Dettagli →"):
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.plotly_chart(plot_match_prediction(match), use_container_width=True,
                                            key=f"next_pie_{match['id']}")
                        with detail_col2:
                            from src.insights import match_insights
                            st.markdown("**💡 Insight pre-match**")
                            bullets = match_insights(match['home_team'], match['away_team'])
                            if bullets:
                                for icon, txt in bullets:
                                    st.markdown(f"{icon} {txt}")
                            else:
                                st.caption("Dati giocatori non ancora disponibili per queste squadre (sync in corso).")
                    
                    st.divider()

with tab_table:
    st.title("Classifica Serie A")

    standings, season_label = get_live_standings()
    if standings is not None:
        st.caption(f"🔴 Live — stagione {season_label} (football-data.org, aggiornata ogni 6h)")
    else:
        st.caption("⚠️ Live non disponibile — classifica calcolata dallo storico DB")
        standings = get_standings()
    
    # Color code positions
    def style_standings(row):
        styles = [''] * len(row)
        if row['Pos'] <= 4:
            styles[row.index.get_loc('Pos')] = 'background-color: #c6f6d5; font-weight: bold'
        elif row['Pos'] == 5:
            styles[row.index.get_loc('Pos')] = 'background-color: #fefcbf'
        elif row['Pos'] == 6:
            styles[row.index.get_loc('Pos')] = 'background-color: #faf089'
        elif row['Pos'] >= len(standings) - 2:
            styles[row.index.get_loc('Pos')] = 'background-color: #fed7d7'
        return styles
    
    styled = standings.style.apply(style_standings, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # Visualization
    fig = px.bar(standings, x='Team', y='Pts', color='Pts',
                 color_continuous_scale='RdYlGn',
                 title='Punti per Squadra')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True, key='table_bar')

with tab_team:
    st.title("Analisi Squadra")
    
    session = get_db_session()
    teams = [t.name for t in session.query(Team).order_by(Team.name).all()]
    session.close()
    
    selected_team = st.selectbox("Seleziona squadra", teams)
    
    if selected_team:
        from src.insights import player_status, in_form_players, season_scorers, player_match_log

        st.markdown("### 🚫 Disponibilità & 🔥 Forma")
        ps = player_status(selected_team)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**Squalificati**")
            if ps['suspended']:
                for s in ps['suspended']:
                    st.markdown(f"🚫 {s['player']} <span class='pill pill-lost'>OUT</span> ({s['reason']})", unsafe_allow_html=True)
            else:
                st.caption("Nessuno" if ps['cards'] else "Dati in arrivo (sync giocatori in corso)")
        with a2:
            st.markdown("**Diffidati** (1 giallo dalla squalifica)")
            if ps['warned']:
                for w in ps['warned']:
                    st.markdown(f"⚠️ {w['player']} ({w['yellow']} 🟨)")
            else:
                st.caption("Nessuno")
        with a3:
            st.markdown("**In forma**")
            form_p = in_form_players(selected_team)
            if form_p:
                for f in form_p[:4]:
                    detail = f"striscia {f['streak']}🔥" if f['streak'] >= 2 else f"{f['goals_l3']} gol/3"
                    st.markdown(f"🔥 **{f['player']}** — {detail}" + (f" — rating {f['avg_rating_l5']}" if f['avg_rating_l5'] else ""))
            else:
                st.caption("—")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("### 🎯 Cannonieri stagionali")
            scorers = season_scorers(selected_team)
            if scorers:
                st.dataframe(pd.DataFrame(scorers), use_container_width=True, hide_index=True)
            else:
                st.caption("Dati in arrivo")
        with b2:
            st.markdown("### 🧍 Scheda giocatore")
            all_players = sorted({c['player'] for c in ps['cards']} | {f['player'] for f in form_p} | {s['player'] for s in scorers})
            if all_players:
                sel_player = st.selectbox("Giocatore", all_players, key=f"player_{selected_team}")
                plog = player_match_log(sel_player, selected_team)
                if not plog.empty:
                    st.dataframe(plog, use_container_width=True, hide_index=True)
                else:
                    st.caption("Nessun dato")
            else:
                st.caption("Dati in arrivo")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ultime 5 partite")
            form = get_team_form(selected_team, 5)
            if not form.empty:
                st.plotly_chart(plot_team_form(form, selected_team), use_container_width=True, key=f"team_form_{selected_team}")
                st.dataframe(form[['date', 'venue', 'opponent', 'goals_for', 'goals_against', 'result']], 
                           use_container_width=True, hide_index=True)
            else:
                st.info("Nessun dato disponibile")
        
        with col2:
            st.subheader("Prossime partite")
            upcoming = get_upcoming_matches(30)
            team_matches = upcoming[
                (upcoming['home_team'] == selected_team) | 
                (upcoming['away_team'] == selected_team)
            ]
            if len(team_matches) > 0:
                for _, m in team_matches.iterrows():
                    is_home = m['home_team'] == selected_team
                    opponent = m['away_team'] if is_home else m['home_team']
                    venue = "🏠" if is_home else "🚌"
                    st.markdown(f"{venue} **{m['date'].strftime('%d/%m')}** vs {opponent}")
                    if pd.notna(m['pred_home']):
                        st.caption(f"1: {m['pred_home']:.1%} | X: {m['pred_draw']:.1%} | 2: {m['pred_away']:.1%}")
            else:
                st.info("Nessuna partita in programma")

with tab_value:
    st.title("Value Bets Detection")
    st.caption("Modello vs quote Gamdom: edge = probabilità modello − probabilità implicita")

    upcoming = get_upcoming_matches(14)
    # Assicura colonna sorgente quote (default gamdom se quote presenti)
    if 'odds_source' not in upcoming.columns:
        upcoming['odds_source'] = None
    value_bets = []

    MARKETS = [
        ('1X2', [('1', 'pred_home', 'odds_home'), ('X', 'pred_draw', 'odds_draw'), ('2', 'pred_away', 'odds_away')]),
        ('O/U 2.5', [('Over 2.5', 'pred_over_25', 'odds_over_25'), ('Under 2.5', 'pred_under_25', 'odds_under_25')]),
        ('BTTS', [('GG Sì', 'pred_btts_yes', 'odds_btts_yes'), ('GG No', 'pred_btts_no', 'odds_btts_no')]),
    ]

    for _, match in upcoming.iterrows():
        for market, legs in MARKETS:
            for label, pred_col, odds_col in legs:
                model_prob = match.get(pred_col)
                odds = match.get(odds_col)
                if pd.notna(model_prob) and pd.notna(odds) and odds > 1:
                    implied = 1 / odds
                    edge = model_prob - implied
                    if edge > 0.05:  # 5% minimum edge
                        value_bets.append({
                            'Match': f"{match['home_team']} vs {match['away_team']}",
                            'Date': match['date'],
                            'Mercato': market,
                            'Esito': label,
                            'Modello': f"{model_prob:.1%}",
                            'Quota': f"{odds:.2f}",
                            'Implicita': f"{implied:.1%}",
                            'Edge': f"{edge:.1%}",
                            '_edge': edge,
                        })

    if value_bets:
        df = pd.DataFrame(value_bets).sort_values('_edge', ascending=False).drop(columns=['_edge'])
        n_quoted = int(upcoming['odds_home'].notna().sum()) if 'odds_home' in upcoming.columns else 0
        st.success(f"Quote Gamdom su {n_quoted}/{len(upcoming)} partite • {len(df)} edge > 5%")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart (top 15 per leggibilità)
        chart_df = pd.DataFrame(value_bets).sort_values('_edge', ascending=False).head(15)
        chart_df['label'] = chart_df['Match'] + ' — ' + chart_df['Esito']
        fig = px.bar(chart_df, x='label', y='_edge', color='Mercato',
                     title='Top edge modello vs Gamdom')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key='value_bar')
    else:
        st.info("Nessun edge > 5% al momento. Se le quote mancano, premi 'Aggiorna quote Gamdom' in Impostazioni.")

with tab_hist:
    st.markdown("## 📜 History — tutte le previsioni")
    st.caption("Snapshot giornalieri di modello + quote Gamdom. Gli esiti si compilano da soli a fine partita.")
    from src.tracking import get_prediction_logs, history_summary

    logs = get_prediction_logs()
    if len(logs) == 0:
        st.info("Nessuno snapshot ancora. Premi 'Aggiorna Dati Ora' nel Setup o attendi lo scheduler.")
    else:
        summ = history_summary(logs)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Snapshot chiusi", f"{summ['n_closed']}/{summ['n_total']}")
        k2.metric("Hit 1X2", f"{summ['acc_1x2']:.1%}" if summ['acc_1x2'] is not None else "—",
                  help=f"su {summ['acc_1x2_n']} pronostici" if summ['acc_1x2_n'] else None)
        k3.metric("Hit Over/Under", f"{summ['acc_ou25']:.1%}" if summ['acc_ou25'] is not None else "—")
        k4.metric("Hit BTTS", f"{summ['acc_btts']:.1%}" if summ['acc_btts'] is not None else "—")
        k5.metric("In attesa", summ['n_open'])

        if summ['calibration']:
            cal_df = pd.DataFrame(summ['calibration'])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cal_df['bucket'], y=cal_df['hit'], name='Hit rate reale',
                                 marker_color='#22C55E', text=cal_df['n'], textposition='outside'))
            fig.add_trace(go.Scatter(x=cal_df['bucket'],
                                     y=[(int(b.split('-')[0]) + 5) / 100 for b in cal_df['bucket']],
                                     name='Confidenza modello', mode='lines+markers',
                                     line=dict(color='#60A5FA', dash='dash')))
            fig.update_layout(title='Calibration — confidenza modello vs realtà (n = n. casi)',
                              yaxis=dict(tickformat='.0%', range=[0, 1]), height=340)
            st.plotly_chart(fig, use_container_width=True, key='hist_calibration')
        else:
            st.info("La calibration apparirà quando almeno un pronostico sarà chiuso.")

        st.subheader("Registro completo")
        f_status = st.radio("Filtro", ["Tutti", "Chiusi ✅", "Aperti ⏳"], horizontal=True, key="hist_filter")
        view = logs.copy()
        if f_status == "Chiusi ✅":
            view = view[view['result'].notna()]
        elif f_status == "Aperti ⏳":
            view = view[view['result'].isna()]
        show = view[['log_date', 'match', 'match_date', 'p_home', 'p_draw', 'p_away',
                     'odds_1', 'odds_x', 'odds_2', 'result', 'score', 'hit_1x2', 'hit_ou25', 'hit_btts']].copy()
        for c in ['p_home', 'p_draw', 'p_away']:
            show[c] = show[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)

with tab_slips:
    st.markdown("## 🧾 Bet Slips — schedine tracciate")
    st.caption("Suggerite dal tool su base edge Kelly. Solo tracking: lo stato si aggiorna da solo a fine partite.")
    from src.tracking import current_edges, kelly_stake, create_slip, get_slips, slip_summary

    edges = current_edges(min_edge=0.03)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        n_legs = st.slider("Gambe multipla", 2, 5, 3, key="slip_n")
    with c2:
        stake_in = st.number_input("Stake (unità)", 0.5, 20.0, 2.0, 0.5, key="slip_stake")
    with c3:
        st.write("")
        gen = st.button("🎲 Genera slip suggerita", use_container_width=True)

    if gen:
        if len(edges) < n_legs:
            st.warning(f"Solo {len(edges)} edge ≥ 3% disponibili, ne servono {n_legs}.")
        else:
            legs = edges[:n_legs]
            tot_o, tot_p = 1.0, 1.0
            for leg in legs:
                tot_o *= leg['odds']
                tot_p *= leg['model_prob']
            kelly = kelly_stake(tot_p, tot_o)
            st.session_state['suggested'] = {'legs': legs, 'total_odds': round(tot_o, 2),
                                             'combined_prob': round(tot_p, 4), 'kelly': kelly}

    sugg = st.session_state.get('suggested')
    if sugg:
        st.markdown("### Slip suggerita")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Quota totale", f"{sugg['total_odds']:.2f}")
        s2.metric("Prob. stimata", f"{sugg['combined_prob']:.1%}")
        s3.metric("Edge vs implicita", f"{sugg['combined_prob'] - 1/sugg['total_odds']:+.1%}")
        s4.metric("Stake Kelly/4", f"{sugg['kelly']:.2f}u")
        st.dataframe(pd.DataFrame([{
            'Match': f"{l['home']} vs {l['away']}", 'Data': l['date'], 'Mercato': l['market'],
            'Esito': l['selection'], 'Modello': f"{l['model_prob']:.1%}",
            'Quota': l['odds'], 'Edge': f"{l['edge']:+.1%}"} for l in sugg['legs']]),
            use_container_width=True, hide_index=True)
        if st.button("💾 Salva questa slip", key="save_suggested"):
            sid = create_slip(sugg['legs'], stake_in, name=f"Multipla x{sugg['total_odds']:.2f}")
            st.success(f"Slip #{sid} salvata e in tracking.")
            st.session_state.pop('suggested', None)

    st.markdown("### Singole ad alto edge")
    if edges:
        top = pd.DataFrame([{
            'Match': f"{e['home']} vs {e['away']}", 'Mercato': e['market'], 'Esito': e['selection'],
            'Modello': f"{e['model_prob']:.1%}", 'Quota': e['odds'],
            'Edge': f"{e['edge']:+.1%}", 'Kelly/4': f"{kelly_stake(e['model_prob'], e['odds']):.2f}u",
            '_i': i} for i, e in enumerate(edges[:12])])
        st.dataframe(top.drop(columns=['_i']), use_container_width=True, hide_index=True)
        pick = st.selectbox("Salva una singola come slip:", [f"#{r['_i']} {r['Match']} — {r['Esito']} @{r['Quota']}" for _, r in top.iterrows()], key="pick_single")
        if st.button("💾 Salva singola", key="save_single"):
            e = edges[int(pick.split()[0][1:])]
            sid = create_slip([e], stake_in, name=f"Singola {e['selection']} @{e['odds']}")
            st.success(f"Slip #{sid} salvata e in tracking.")
    else:
        st.info("Nessun edge ≥ 3% al momento.")

    st.markdown("### Le mie slip")
    summ = slip_summary()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Totale slip", summ['total'])
    m2.metric("Aperte", summ['open'])
    m3.metric("Win rate", f"{summ['win_rate']:.1%}" if summ['win_rate'] is not None else "—")
    m4.metric("Profitto", f"{summ['profit']:+.2f}u")
    m5.metric("ROI", f"{summ['roi']:+.1%}" if summ['roi'] is not None else "—")

    sub_open, sub_won, sub_lost = st.tabs(["⏳ Aperte", "✅ Vinte", "❌ Perse"])
    for sub, status, pill in [(sub_open, 'OPEN', 'pill-open'), (sub_won, 'WON', 'pill-won'), (sub_lost, 'LOST', 'pill-lost')]:
        with sub:
            df = get_slips(status)
            if len(df) == 0:
                st.caption("Nessuna slip qui.")
            for _, s in df.iterrows():
                p = 'WON' if s['status'] == 'WON' else ('LOST' if s['status'] == 'LOST' else 'OPEN')
                with st.expander(f"#{s['id']} {s['name']} — quota {s['quota']:.2f} — {s['stake']}u"):
                    st.markdown(f"<span class='pill {pill}'>{p}</span> &nbsp; profitto: **{s['profit']:+.2f}u**" if pd.notna(s['profit']) else f"<span class='pill {pill}'>{p}</span>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{
                        'Match': f"{l.get('home')} vs {l.get('away')}", 'Mercato': l.get('market'),
                        'Esito': l.get('selection'), 'Quota': l.get('odds')} for l in s['_legs']]),
                        use_container_width=True, hide_index=True)

    if st.button("🔄 Ricalcola esiti ora", key="settle_now"):
        with st.spinner("Controllo risultati..."):
            from src.scheduler import SerieAScheduler
            SerieAScheduler().settle_logs_and_slips()
        st.success("Esiti aggiornati.")

with tab_conf:
    st.title("Impostazioni")
    
    st.subheader("Configurazione Modelli")
    st.json(CONFIG["models"])
    
    st.subheader("Fonti Dati")
    st.json({k: v.get("enabled", False) for k, v in CONFIG["data_sources"].items()})
    
    st.subheader("Scheduler")
    st.json(CONFIG["scheduler"])
    
    if st.button("Aggiorna Dati Ora"):
        with st.spinner("Aggiornamento in corso..."):
            from src.scheduler import run_once
            run_once('historical')
            run_once('upcoming')
            run_once('predictions')
        st.success("Aggiornamento completato!")

    if st.button("Aggiorna Quote Gamdom"):
        with st.spinner("Scarico quote da Gamdom..."):
            from src.scheduler import run_once
            run_once('odds')
        st.success("Quote aggiornate! Vai su 🎯 Value Bets per gli edge.")

    if st.button("Riatterra Modelli"):
        with st.spinner("Riaddestramento..."):
            from src.scheduler import run_once
            run_once('retrain')
        st.success("Riaddestramento completato!")

# Footer
st.sidebar.divider()
st.sidebar.caption("Serie A Predictor v0.1.0")
st.sidebar.caption("Dati: football-data.co.uk, API-Football, FBref")
st.sidebar.caption("⚠️ Solo scopi educativi/informativi")