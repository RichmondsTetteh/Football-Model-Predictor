import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import re
import io
import requests
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# =============================================================================
# CONSTANTS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTCOME_MODEL_FILENAME = os.path.join(BASE_DIR, 'Gradient_Boosting_Classifier_outcome_model.joblib')
GOALS_MODEL_FILENAME = os.path.join(BASE_DIR, 'XGBoost_Regressor_goals_model.joblib')
LABEL_ENCODER_FILENAME = os.path.join(BASE_DIR, 'label_encoder.joblib')
CLUSTER_SCALER_FILENAME = os.path.join(BASE_DIR, 'cluster_robust_scaler.joblib')
CLUSTER_KMEANS_FILENAME = os.path.join(BASE_DIR, 'cluster_kmeans_model.joblib')

DEFAULT_ELO = 1500.0
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

DEFAULTS = {
    'HomeShots': 12.0, 'AwayShots': 10.0,
    'HomeTarget': 4.0, 'AwayTarget': 3.0,
    'OddHome': 2.0, 'OddDraw': 3.0, 'OddAway': 3.0,
    'HomeExpectedGoals': 1.5, 'AwayExpectedGoals': 1.2,
    'Form3Home': 0, 'Form5Home': 0,
    'Form3Away': 0, 'Form5Away': 0,
    'HandiSize': 0.0, 'Over25': 2.0, 'Under25': 2.0,
}

FEATURE_COLUMNS = [ ... ]  # Keep your original FEATURE_COLUMNS list here
BASE_COLUMNS = [ ... ]     # Keep your original BASE_COLUMNS list here

# =============================================================================
# ELO SCRAPING (unchanged from previous version)
# =============================================================================
@st.cache_data(show_spinner="Fetching ELO ratings...", ttl=3600)
def load_elo_ratings():
    try:
        response = requests.get('http://api.clubelo.com/', timeout=10)
        response.raise_for_status()
        df_elo = pd.read_csv(io.StringIO(response.text))
        df_elo['Clean_Club'] = df_elo['Club'].astype(str).str.strip()
        return df_elo
    except Exception:
        # fallback HTML scraping if needed
        pass
    return pd.DataFrame(columns=['Club', 'Elo', 'Clean_Club'])

def get_elo_for_team(team_name: str, df_elo: pd.DataFrame) -> tuple[float, str | None]:
    if df_elo.empty or not team_name.strip():
        return DEFAULT_ELO, None
    name_lower = team_name.strip().lower()
    mask = df_elo['Clean_Club'].str.lower().str.contains(name_lower, na=False, regex=False)
    matches = df_elo[mask]
    if matches.empty:
        return DEFAULT_ELO, f'No ELO match for "{team_name}". Using default.'
    exact = matches[matches['Clean_Club'].str.lower() == name_lower]
    if len(exact) == 1:
        return float(exact['Elo'].iloc[0]), None
    return DEFAULT_ELO, f'Multiple matches for "{team_name}". Using default.'

# =============================================================================
# MODEL LOADING (unchanged)
# =============================================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_model = joblib.load(GOALS_MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        return outcome_model, goals_model, label_encoder
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

@st.cache_resource(show_spinner="Loading cluster models...")
def load_cluster_models():
    if not (os.path.exists(CLUSTER_SCALER_FILENAME) and os.path.exists(CLUSTER_KMEANS_FILENAME)):
        return None, None
    try:
        return joblib.load(CLUSTER_SCALER_FILENAME), joblib.load(CLUSTER_KMEANS_FILENAME)
    except Exception:
        return None, None

# =============================================================================
# API-FOOTBALL HELPERS
# =============================================================================
@st.cache_data(ttl=1800)  # 30 minutes cache
def search_team(team_name: str, api_key: str) -> dict | None:
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/teams",
            headers={"x-apisports-key": api_key},
            params={"search": team_name},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("response"):
            return data["response"][0]["team"]  # Return first match
    except Exception:
        pass
    return None

def calculate_form_from_fixtures(fixtures: list, team_id: int) -> tuple[int, int]:
    """Calculate points from last 3 and last 5 matches"""
    recent = []
    for fix in fixtures:
        if fix["teams"]["home"]["id"] == team_id:
            result = fix["teams"]["home"]["winner"]
            recent.append(3 if result is True else 1 if result is None else 0)
        elif fix["teams"]["away"]["id"] == team_id:
            result = fix["teams"]["away"]["winner"]
            recent.append(3 if result is True else 1 if result is None else 0)
    recent = recent[-5:]  # last 5
    form5 = sum(recent)
    form3 = sum(recent[-3:])
    return form3, form5

@st.cache_data(ttl=600)
def fetch_team_stats(home_team: str, away_team: str, api_key: str):
    if not api_key:
        st.error("Please enter your API-Football key")
        return None

    with st.spinner("Searching teams and fetching recent stats..."):
        home_info = search_team(home_team, api_key)
        away_info = search_team(away_team, api_key)

        if not home_info or not away_info:
            st.warning("Could not find one or both teams in the database.")
            return None

        home_id = home_info["id"]
        away_id = away_info["id"]

        # Get last 5 fixtures for each team (to calculate form)
        try:
            home_fixtures = requests.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                headers={"x-apisports-key": api_key},
                params={"team": home_id, "last": 5},
                timeout=10
            ).json().get("response", [])

            away_fixtures = requests.get(
                f"{API_FOOTBALL_BASE}/fixtures",
                headers={"x-apisports-key": api_key},
                params={"team": away_id, "last": 5},
                timeout=10
            ).json().get("response", [])
        except Exception:
            home_fixtures = away_fixtures = []

        home_form3, home_form5 = calculate_form_from_fixtures(home_fixtures, home_id)
        away_form3, away_form5 = calculate_form_from_fixtures(away_fixtures, away_id)

        # Basic team statistics (shots, etc.) - using current season
        # Note: Detailed shots/xG often limited on free tier
        stats = {
            "home": {
                "avg_shots": 12.0,      # fallback
                "avg_shots_on_target": 4.0,
                "avg_xg": 1.5,
                "form3": home_form3,
                "form5": home_form5,
            },
            "away": {
                "avg_shots": 10.0,
                "avg_shots_on_target": 3.0,
                "avg_xg": 1.2,
                "form3": away_form3,
                "form5": away_form5,
            },
            "odds": {"home_win": 2.0, "draw": 3.0, "away_win": 3.0, "over_2_5": 2.0, "under_2_5": 2.0},
            "notes": f"Data from API-Football. Form calculated from last 5 fixtures. xG/shot stats use defaults (limited on free tier)."
        }

        # Try to get more accurate team statistics if available
        try:
            season = datetime.now().year
            home_stats_resp = requests.get(
                f"{API_FOOTBALL_BASE}/teams/statistics",
                headers={"x-apisports-key": api_key},
                params={"team": home_id, "league": 39, "season": season},  # Example: Premier League (39)
                timeout=10
            )
            # You can parse more detailed stats here if response contains them
        except Exception:
            pass

        return stats

# =============================================================================
# FEATURE ENGINEERING (same as before - abbreviated)
# =============================================================================
# ... Keep your clean_input_data, get_cluster_probabilities, create_derived_features, preprocess_input functions here ...
# (Copy them from the previous corrected version I provided)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {**DEFAULTS, 'HomeElo': DEFAULT_ELO, 'AwayElo': DEFAULT_ELO}
    if 'stats_fetched' not in st.session_state:
        st.session_state.stats_fetched = False
    if 'fetch_notes' not in st.session_state:
        st.session_state.fetch_notes = ""

# =============================================================================
# MAIN APP
# =============================================================================
st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor")

init_session_state()

df_elo = load_elo_ratings()
outcome_model, goals_model, label_encoder = load_models()
cluster_scaler, kmeans_model = load_cluster_models()

if outcome_model is None or goals_model is None:
    st.error("Prediction models could not be loaded.")
    st.stop()

# Sidebar - API Keys
st.sidebar.header("API Configuration")
api_football_key = st.sidebar.text_input(
    "🔑 API-Football Key",
    type="password",
    help="Get free key at https://www.api-football.com/ (100 requests/day)"
)

st.sidebar.divider()

# Team inputs
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Home Team", "Arsenal")
with col2:
    away_team = st.text_input("Away Team", "Chelsea")

# ELO (unchanged)
home_elo_scraped, home_warn = get_elo_for_team(home_team, df_elo)
away_elo_scraped, away_warn = get_elo_for_team(away_team, df_elo)

if home_warn: st.warning(f"🏠 {home_warn}")
if away_warn: st.warning(f"🏟️ {away_warn}")

# Sync ELO
if 'prev_home' not in st.session_state or st.session_state.prev_home != home_team:
    st.session_state.inputs['HomeElo'] = home_elo_scraped
    st.session_state.prev_home = home_team
    st.session_state.stats_fetched = False

if 'prev_away' not in st.session_state or st.session_state.prev_away != away_team:
    st.session_state.inputs['AwayElo'] = away_elo_scraped
    st.session_state.prev_away = away_team
    st.session_state.stats_fetched = False

# Fetch Button
if st.button("🔍 Fetch Stats from API-Football", type="secondary", use_container_width=True, disabled=not api_football_key):
    raw_stats = fetch_team_stats(home_team, away_team, api_football_key)
    if raw_stats:
        # Map to inputs
        st.session_state.inputs.update({
            'HomeShots': raw_stats["home"]["avg_shots"],
            'AwayShots': raw_stats["away"]["avg_shots"],
            'HomeTarget': raw_stats["home"]["avg_shots_on_target"],
            'AwayTarget': raw_stats["away"]["avg_shots_on_target"],
            'HomeExpectedGoals': raw_stats["home"]["avg_xg"],
            'AwayExpectedGoals': raw_stats["away"]["avg_xg"],
            'Form3Home': raw_stats["home"]["form3"],
            'Form5Home': raw_stats["home"]["form5"],
            'Form3Away': raw_stats["away"]["form3"],
            'Form5Away': raw_stats["away"]["form5"],
            'OddHome': raw_stats["odds"]["home_win"],
            'OddDraw': raw_stats["odds"]["draw"],
            'OddAway': raw_stats["odds"]["away_win"],
            'Over25': raw_stats["odds"]["over_2_5"],
            'Under25': raw_stats["odds"]["under_2_5"],
        })
        st.session_state.stats_fetched = True
        st.session_state.fetch_notes = raw_stats.get("notes", "")
        st.rerun()

if st.session_state.stats_fetched and st.session_state.fetch_notes:
    st.caption(f"📰 {st.session_state.fetch_notes}")

# Sidebar inputs (same structure as previous version)
# ... Copy the sidebar expanders for ELO, Form, Team Statistics, Betting Odds from the previous corrected code ...

predict_clicked = st.sidebar.button("🚀 Predict Match", type="primary", use_container_width=True)

# Prediction section (unchanged - copy from previous version)
if predict_clicked:
    # ... your existing prediction logic using st.session_state.inputs ...
    pass
else:
    st.info("👈 Fill parameters in the sidebar and click **Predict Match**")
