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
    initial_sidebar_state="expanded"
)

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
            Match.status == 'SCHEDULED'
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


# Sidebar
st.sidebar.title("⚽ Serie A Predictor")
st.sidebar.caption(f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Navigation
page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "📅 Prossime Partite",
    "📊 Classifica",
    "📈 Analisi Squadra",
    "🎯 Value Bets",
    "⚙️ Impostazioni"
])

# Main content
if page == "🏠 Dashboard":
    st.title("Serie A Predictor - Dashboard")
    
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
                    st.plotly_chart(plot_match_prediction(match), use_container_width=True)
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
            lambda r: r['Prediction'] == r['result'] if r['Prediction'] != '—' else None, axis=1)
        st.dataframe(
            display_df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'Prediction', 'Correct']],
            use_container_width=True,
            hide_index=True
        )

elif page == "📅 Prossime Partite":
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
                        st.metric("1", f"{match['pred_home']:.1%}" if pd.notna(match['pred_home']) else "—")
                    with col3:
                        st.metric("X", f"{match['pred_draw']:.1%}" if pd.notna(match['pred_draw']) else "—")
                    with col4:
                        st.metric("2", f"{match['pred_away']:.1%}" if pd.notna(match['pred_away']) else "—")
                    with col5:
                        st.metric("Ov 2.5", f"{match['pred_over_25']:.1%}" if pd.notna(match['pred_over_25']) else "—")
                    
                    # Detail expander
                    with st.expander("Dettagli →"):
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.plotly_chart(plot_match_prediction(match), use_container_width=True)
                        with detail_col2:
                            st.markdown("**Probabilità Risultati Esatti**")
                            # Would need model to provide this
                            st.caption("Top scorelines from best model")
                    
                    st.divider()

elif page == "📊 Classifica":
    st.title("Classifica Serie A")
    
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
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Analisi Squadra":
    st.title("Analisi Squadra")
    
    session = get_db_session()
    teams = [t.name for t in session.query(Team).order_by(Team.name).all()]
    session.close()
    
    selected_team = st.selectbox("Seleziona squadra", teams)
    
    if selected_team:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ultime 5 partite")
            form = get_team_form(selected_team, 5)
            if not form.empty:
                st.plotly_chart(plot_team_form(form, selected_team), use_container_width=True)
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

elif page == "🎯 Value Bets":
    st.title("Value Bets Detection")
    st.caption("Confronto probabilità modello vs quote implicite (quando disponibili)")
    
    upcoming = get_upcoming_matches(14)
    value_bets = []
    
    for _, match in upcoming.iterrows():
        if pd.notna(match['odds_home']) and match['odds_home'] > 0:
            implied = {
                'home': 1 / match['odds_home'],
                'draw': 1 / match['odds_draw'] if pd.notna(match['odds_draw']) else 0,
                'away': 1 / match['odds_away'] if pd.notna(match['odds_away']) else 0,
            }
            
            for outcome, model_prob in [('home', match['pred_home']), ('draw', match['pred_draw']), ('away', match['pred_away'])]:
                if implied[outcome] > 0:
                    edge = model_prob - implied[outcome]
                    if edge > 0.05:  # 5% minimum edge
                        value_bets.append({
                            'Match': f"{match['home_team']} vs {match['away_team']}",
                            'Date': match['date'],
                            'Outcome': {'home': '1', 'draw': 'X', 'away': '2'}[outcome],
                            'Model %': f"{model_prob:.1%}",
                            'Implied %': f"{implied[outcome]:.1%}",
                            'Edge': f"{edge:.1%}",
                            'Odds': {'home': match['odds_home'], 'draw': match['odds_draw'], 'away': match['odds_away']}[outcome],
                        })
    
    if value_bets:
        df = pd.DataFrame(value_bets)
        df = df.sort_values('Edge', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Chart
        fig = px.bar(df, x='Match', y='Edge', color='Outcome',
                     title='Value Bets - Edge vs Bookmakers',
                     color_discrete_map={'1': '#2ecc71', 'X': '#f39c12', '2': '#e74c3c'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nessun value bet trovato con le quote attuali")

elif page == "⚙️ Impostazioni":
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