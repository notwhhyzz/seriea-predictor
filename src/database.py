"""Database models and connection management"""
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime,
    ForeignKey, UniqueConstraint, Index, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.sqlite import JSON
from datetime import datetime
from pathlib import Path
import yaml

Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    short_name = Column(String(20))
    fbref_id = Column(String(50))
    api_football_id = Column(Integer)
    football_data_org_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    matches_home = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    matches_away = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    elo_ratings = relationship("EloRating", foreign_keys="EloRating.team_id", back_populates="team")


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    season = Column(String(10), nullable=False)  # e.g., "2024/25"
    matchday = Column(Integer)
    date = Column(Date, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_goals_ht = Column(Integer)
    away_goals_ht = Column(Integer)
    result = Column(String(1))  # H, D, A
    referee = Column(String(100))
    attendance = Column(Integer)
    venue = Column(String(100))
    
    # Advanced stats (from FBref/Understat)
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)
    
    # Odds (from football-data.co.uk)
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)
    odds_over_25 = Column(Float)
    odds_under_25 = Column(Float)
    odds_btts_yes = Column(Float)
    odds_btts_no = Column(Float)
    odds_source = Column(String(50))  # e.g. gamdom, b365
    odds_updated_at = Column(DateTime)
    
    # Predictions (our model outputs)
    pred_home_win = Column(Float)
    pred_draw = Column(Float)
    pred_away_win = Column(Float)
    pred_over_25 = Column(Float)
    pred_under_25 = Column(Float)
    pred_btts_yes = Column(Float)
    pred_btts_no = Column(Float)
    pred_model_version = Column(String(20))
    
    # Metadata
    source = Column(String(50))  # football-data.co.uk, api-football, fbref, merged
    status = Column(String(20), default="SCHEDULED")  # SCHEDULED, LIVE, FINISHED, POSTPONED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="matches_home")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="matches_away")
    
    __table_args__ = (
        UniqueConstraint('season', 'matchday', 'home_team_id', 'away_team_id', name='unique_match'),
        Index('idx_matches_date', 'date'),
        Index('idx_matches_season', 'season'),
        Index('idx_matches_status', 'status'),
    )


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"))
    position = Column(String(20))  # GK, DEF, MID, FWD
    nationality = Column(String(50))
    birth_date = Column(Date)
    fbref_id = Column(String(50))
    api_football_id = Column(Integer)
    market_value = Column(Float)  # in millions EUR
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team")
    stats = relationship("PlayerSeasonStats", back_populates="player")


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(String(10), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"))
    
    # Playing time
    matches_played = Column(Integer, default=0)
    starts = Column(Integer, default=0)
    minutes = Column(Integer, default=0)
    nineties = Column(Float, default=0.0)
    
    # Goals/Assists
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    non_penalty_goals = Column(Integer, default=0)
    penalties_scored = Column(Integer, default=0)
    penalties_missed = Column(Integer, default=0)
    
    # Expected
    xg = Column(Float, default=0.0)
    npxg = Column(Float, default=0.0)
    xa = Column(Float, default=0.0)
    
    # Shooting
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    
    # Passing
    passes_completed = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    pass_completion_pct = Column(Float, default=0.0)
    progressive_passes = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)
    
    # Defense
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    
    # Discipline
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    fouls = Column(Integer, default=0)
    fouls_drawn = Column(Integer, default=0)
    
    # Metadata
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player", back_populates="stats")
    team = relationship("Team")
    
    __table_args__ = (
        UniqueConstraint('player_id', 'season', name='unique_player_season'),
    )


class EloRating(Base):
    __tablename__ = "elo_ratings"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(String(10), nullable=False)
    matchday = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    rating_before = Column(Float, nullable=False)
    rating_after = Column(Float, nullable=False)
    opponent_id = Column(Integer, ForeignKey("teams.id"))
    is_home = Column(Boolean, default=True)
    match_result = Column(String(1))  # W, D, L
    created_at = Column(DateTime, default=datetime.utcnow)
    
    team = relationship("Team", foreign_keys=[team_id], back_populates="elo_ratings")
    opponent = relationship("Team", foreign_keys=[opponent_id])
    
    __table_args__ = (
        UniqueConstraint('team_id', 'season', 'matchday', name='unique_elo_per_matchday'),
        Index('idx_elo_team_date', 'team_id', 'date'),
    )


class ModelPerformance(Base):
    __tablename__ = "model_performance"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    season = Column(String(10), nullable=False)
    matchday_start = Column(Integer)
    matchday_end = Column(Integer)
    
    # Accuracy metrics
    accuracy_1x2 = Column(Float)
    log_loss_1x2 = Column(Float)
    brier_score_1x2 = Column(Float)
    
    accuracy_over_under = Column(Float)
    log_loss_over_under = Column(Float)
    
    accuracy_btts = Column(Float)
    log_loss_btts = Column(Float)
    
    # ROI metrics (if betting simulation)
    roi_1x2 = Column(Float)
    roi_over_under = Column(Float)
    roi_btts = Column(Float)
    
    total_predictions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('model_name', 'season', 'matchday_start', 'matchday_end', name='unique_model_performance'),
    )


class PredictionLog(Base):
    """Daily snapshot of model predictions + bookmaker odds per match.
    Result/hit columns are backfilled by settle job once match finishes."""
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    log_date = Column(Date, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    # Model probabilities
    p_home = Column(Float)
    p_draw = Column(Float)
    p_away = Column(Float)
    p_over25 = Column(Float)
    p_under25 = Column(Float)
    p_btts_yes = Column(Float)
    p_btts_no = Column(Float)

    # Bookmaker odds at snapshot time
    odds_1 = Column(Float)
    odds_x = Column(Float)
    odds_2 = Column(Float)
    odds_over25 = Column(Float)
    odds_under25 = Column(Float)
    odds_btts_yes = Column(Float)
    odds_btts_no = Column(Float)
    odds_source = Column(String(50))

    # Settlement (filled when match finishes)
    result = Column(String(1))  # H, D, A
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    hit_1x2 = Column(Boolean)
    hit_ou25 = Column(Boolean)
    hit_btts = Column(Boolean)

    match = relationship("Match")

    __table_args__ = (
        UniqueConstraint('match_id', 'log_date', name='unique_log_per_match_day'),
        Index('idx_log_date', 'log_date'),
    )


class BetSlip(Base):
    """Tracked bet slip: suggested or manual, auto-settled from results."""
    __tablename__ = "bet_slips"
    id = Column(Integer, primary_key=True)
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String(200))
    legs = Column(JSON)  # [{match_id, date, home, away, market, selection, model_prob, odds}]
    total_odds = Column(Float)
    combined_prob = Column(Float)
    stake = Column(Float, default=1.0)  # units
    status = Column(String(10), default="OPEN")  # OPEN, WON, LOST
    settled_at = Column(DateTime)
    profit = Column(Float)  # units (+/-), set on settlement

    __table_args__ = (
        Index('idx_slip_status', 'status'),
    )


class PlayerMatch(Base):
    """Per-player stats for a single match (from API-Football /fixtures/players)."""
    __tablename__ = "player_match"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(String(10))

    minutes = Column(Integer, default=0)
    position = Column(String(10))
    rating = Column(Float)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    yellow = Column(Integer, default=0)
    red = Column(Integer, default=0)
    starter = Column(Boolean, default=False)

    player = relationship("Player")
    match = relationship("Match")
    team = relationship("Team")

    __table_args__ = (
        UniqueConstraint('player_id', 'match_id', name='unique_player_match'),
        Index('idx_pm_team_season', 'team_id', 'season'),
        Index('idx_pm_match', 'match_id'),
    )


_migrated_paths = set()


def get_engine(config_path: str = "config.yaml"):
    """Create SQLAlchemy engine from config"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    db_path = config["database"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=config["database"]["echo"])
    _ensure_migrated(engine, db_path)
    return engine


def _ensure_migrated(engine, db_path: str):
    """create_all + lightweight migration for columns added after first init.
    Runs once per DB path per process; safe to call on every connection."""
    if db_path in _migrated_paths:
        return
    Base.metadata.create_all(engine)
    from sqlalchemy import text
    with engine.begin() as conn:
        existing = {r[1] for r in conn.execute(text("PRAGMA table_info(matches)")).fetchall()}
        if existing and "odds_source" not in existing:
            conn.execute(text("ALTER TABLE matches ADD COLUMN odds_source VARCHAR(50)"))
        if existing and "odds_updated_at" not in existing:
            conn.execute(text("ALTER TABLE matches ADD COLUMN odds_updated_at DATETIME"))
    _migrated_paths.add(db_path)


def init_db(config_path: str = "config.yaml"):
    """Initialize database tables"""
    return get_engine(config_path)


def get_session(config_path: str = "config.yaml"):
    """Get a new database session"""
    engine = get_engine(config_path)
    Session = sessionmaker(bind=engine)
    return Session()