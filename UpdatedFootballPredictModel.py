import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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

# Popular Leagues
LEAGUES = {
    "Premier League (England)": 39,
    "La Liga (Spain)": 140,
    "Bundesliga (Germany)": 78,
    "Serie A (Italy)": 135,
    "Ligue 1 (France)": 61,
}

DEFAULTS = {
    'HomeShots': 12.0, 'AwayShots': 10.0,
    'HomeTarget': 4.0, 'AwayTarget': 3.0,
    'OddHome': 2.0, 'OddDraw': 3.0, 'OddAway': 3.0,
    'HomeExpectedGoals': 1.5, 'AwayExpectedGoals': 1.2,
    'Form3Home': 0, 'Form5Home': 0,
    'Form3Away': 0, 'Form5Away': 0,
    'HandiSize': 0.0, 'Over25': 2.0, 'Under25': 2.0,
}

FEATURE_COLUMNS = [
    'HomeElo', 'AwayElo', 'EloDiff', 'EloSum', 'Form3Home', 'Form5Home',
    'Form3Away', 'Form5Away', 'Form3Ratio', 'Form5Ratio',
    'HomeExpectedGoals', 'AwayExpectedGoals', 'HomeWinProbability',
    'DrawProbability', 'AwayWinProbability', 'HomeShotEfficiency',
    'AwayShotEfficiency', 'HomeAttackStrength', 'AwayAttackStrength',
    'HandiSize', 'Over25', 'Under25', 'C_LTH', 'C_LTA', 'C_VHD',
    'C_VAD', 'C_HTB', 'C_PHB',
]

BASE_COLUMNS = [
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25',
]

CLUSTER_ZERO = {k: 0.0 for k in ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']}

# =============================================================================
# ELO SCRAPING (Fixed)
# =============================================================================
@st.cache_data(show_spinner="Fetching latest ELO ratings...", ttl=1800)
def load_elo_ratings():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        url = f"http://api.clubelo.com/{today}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df_elo = pd.read_csv(io.StringIO(response.text))
        df_elo['Clean_Club'] = df_elo['Club'].astype(str).str.strip()
        st.success("✅ ELO ratings loaded successfully (today)")
        return df_elo
    except:
        pass
    # Fallback to yesterday
    try:
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"http://api.clubelo.com/{yesterday}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df_elo = pd.read_csv(io.StringIO(response.text))
        df_elo['Clean_Club'] = df_elo['Club'].astype(str).str.strip()
        st.info("✅ ELO ratings loaded (yesterday)")
        return df_elo
    except Exception:
        st.warning("Could not load ELO data. Using defaults.")
        return pd.DataFrame(columns=['Club', 'Elo', 'Clean_Club'])

def get_elo_for_team(team_name: str, df_elo: pd.DataFrame) -> tuple[float, str | None]:
    if df_elo.empty or not team_name.strip():
        return DEFAULT_ELO, None
    name_lower = team_name.strip().lower()
    mask = df_elo['Clean_Club'].str.lower().str.contains(name_lower, na=False, regex=False)
    matches = df_elo[mask]
    if matches.empty:
        return DEFAULT_ELO, f'No ELO match for "{team_name}". Using default {DEFAULT_ELO}.'
    exact = matches[matches['Clean_Club'].str.lower() == name_lower]
    if len(exact) == 1:
        return float(exact['Elo'].iloc[0]), None
    best_match = matches.loc[matches['Elo'].idxmax()]
    return float(best_match['Elo']), f'Using closest match for "{team_name}".'

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading prediction models...")
def load_models():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_model = joblib.load(GOALS_MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        return outcome_model, goals_model, label_encoder
    except Exception as e:
        st.error(f"Failed to load models: {e}")
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
def search_team(team_name: str, api_key: str):
    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/teams",
            headers={"x-apisports-key": api_key},
            params={"search": team_name},
            timeout=12
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("response"):
            return data["response"][0]["team"]
    except Exception:
        pass
    return None

def fetch_team_statistics(team_id: int, league_id: int, api_key: str):
    season = datetime.now().year
    for s in [season, season - 1]:
        try:
            resp = requests.get(
                f"{API_FOOTBALL_BASE}/teams/statistics",
                headers={"x-apisports-key": api_key},
                params={"league": league_id, "team": team_id, "season": s},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("response"):
                return data["response"]
        except Exception:
            continue
    return None

def parse_form(form_string: str) -> tuple[int, int]:
    """Convert form string like 'WDLWWL' into Form3 and Form5 points"""
    if not form_string:
        return 0, 0
    points_map = {'W': 3, 'D': 1, 'L': 0}
    recent_points = [points_map.get(c, 0) for c in form_string.strip()][-10:]
    form5 = sum(recent_points[-5:])
    form3 = sum(recent_points[-3:])
    return form3, form5

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().any():
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(0.0, inplace=True)
    return df

def compute_cluster_input(row: dict):
    return pd.DataFrame([{
        'HomeShots': row.get('HomeShots', 12.0),
        'AwayShots': row.get('AwayShots', 10.0),
        'HomeTarget': row.get('HomeTarget', 4.0),
        'AwayTarget': row.get('AwayTarget', 3.0),
        'HomeElo': row.get('HomeElo', 1500.0),
        'AwayElo': row.get('AwayElo', 1500.0),
    }])

def get_cluster_probabilities(row: dict, scaler, kmeans) -> dict:
    if scaler is None or kmeans is None:
        return CLUSTER_ZERO.copy()
    try:
        X = compute_cluster_input(row)
        X_scaled = scaler.transform(X)
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
        probas = softmax(-distances, axis=1)[0]
        return dict(zip(['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'], probas))
    except Exception:
        return CLUSTER_ZERO.copy()

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    df['EloSum'] = df['HomeElo'] + df['AwayElo']
    df['Form3Ratio'] = df['Form3Home'] / (df['Form3Away'] + 1e-5)
    df['Form5Ratio'] = df['Form5Home'] / (df['Form5Away'] + 1e-5)

    h = 1 / df['OddHome'].replace(0, 1)
    d = 1 / df['OddDraw'].replace(0, 1)
    a = 1 / df['OddAway'].replace(0, 1)
    total = h + d + a
    df['HomeWinProbability'] = h / total
    df['DrawProbability'] = d / total
    df['AwayWinProbability'] = a / total

    df['HomeShotEfficiency'] = df['HomeTarget'] / (df['HomeShots'] + 0.1)
    df['AwayShotEfficiency'] = df['AwayTarget'] / (df['AwayShots'] + 0.1)
    df['HomeAttackStrength'] = df['HomeExpectedGoals'] * df['Form5Home'] / 15.0
    df['AwayAttackStrength'] = df['AwayExpectedGoals'] * df['Form5Away'] / 15.0
    return df

def preprocess_input(raw_row: dict, scaler, kmeans):
    for col in BASE_COLUMNS + ['HomeExpectedGoals', 'AwayExpectedGoals']:
        raw_row.setdefault(col, DEFAULTS.get(col, 0.0))
    df = pd.DataFrame([raw_row])
    df = clean_input_data(df)
    df = create_derived_features(df)
    cluster_dict = get_cluster_probabilities(df.iloc[0].to_dict(), scaler, kmeans)
    for col, val in cluster_dict.items():
        df[col] = val
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]

# =============================================================================
# SESSION STATE
# =============================================================================
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
st.markdown("Improved version using **API-Football /teams/statistics** for better form and xG estimation.")

df_elo = load_elo_ratings()
outcome_model, goals_model, label_encoder = load_models()
cluster_scaler, kmeans_model = load_cluster_models()

if outcome_model is None or goals_model is None:
    st.error("Prediction models could not be loaded. Check file paths.")
    st.stop()

# --------------------- Sidebar ---------------------
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "🔑 API-Football API Key",
    type="password",
    help="Get your free key at https://www.api-football.com/"
)

selected_league = st.sidebar.selectbox("Select League", options=list(LEAGUES.keys()))
league_id = LEAGUES[selected_league]

st.sidebar.divider()

col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Home Team", "Arsenal")
with col2:
    away_team = st.text_input("Away Team", "Chelsea")

# ELO
home_elo_scraped, home_warn = get_elo_for_team(home_team, df_elo)
away_elo_scraped, away_warn = get_elo_for_team(away_team, df_elo)
if home_warn: st.warning(f"🏠 {home_warn}")
if away_warn: st.warning(f"🏟️ {away_warn}")

# Sync ELO when teams change
if 'prev_home' not in st.session_state or st.session_state.prev_home != home_team:
    st.session_state.inputs['HomeElo'] = home_elo_scraped
    st.session_state.prev_home = home_team
    st.session_state.stats_fetched = False

if 'prev_away' not in st.session_state or st.session_state.prev_away != away_team:
    st.session_state.inputs['AwayElo'] = away_elo_scraped
    st.session_state.prev_away = away_team
    st.session_state.stats_fetched = False

# Fetch Button
if st.button("🔍 Fetch Stats from API-Football", type="secondary", use_container_width=True, disabled=not api_key):
    if not api_key:
        st.error("Please enter your API-Football key in the sidebar.")
    else:
        home_info = search_team(home_team, api_key)
        away_info = search_team(away_team, api_key)

        if not home_info or not away_info:
            st.error("Could not find one or both teams. Try more accurate names.")
        else:
            home_id = home_info["id"]
            away_id = away_info["id"]

            home_stats = fetch_team_statistics(home_id, league_id, api_key)
            away_stats = fetch_team_statistics(away_id, league_id, api_key)

            if home_stats and away_stats:
                form3_home, form5_home = parse_form(home_stats.get("form", ""))
                form3_away, form5_away = parse_form(away_stats.get("form", ""))

                # Use average goals as xG proxy
                xg_home = float(home_stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.5))
                xg_away = float(away_stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.2))

                st.session_state.inputs.update({
                    'HomeShots': 12.0,
                    'AwayShots': 10.0,
                    'HomeTarget': 4.0,
                    'AwayTarget': 3.0,
                    'HomeExpectedGoals': round(xg_home, 2),
                    'AwayExpectedGoals': round(xg_away, 2),
                    'Form3Home': form3_home,
                    'Form5Home': form5_home,
                    'Form3Away': form3_away,
                    'Form5Away': form5_away,
                    'OddHome': 2.0,
                    'OddDraw': 3.0,
                    'OddAway': 3.0,
                    'Over25': 2.0,
                    'Under25': 2.0,
                    'HandiSize': 0.0,
                })

                st.session_state.stats_fetched = True
                st.session_state.fetch_notes = f"✅ Data from {selected_league} via API-Football. Form and xG pulled from official team statistics."
                st.rerun()
            else:
                st.warning("Could not retrieve detailed statistics. Using default values.")

if st.session_state.get('fetch_notes'):
    st.caption(f"📰 {st.session_state.fetch_notes}")

# --------------------- Sidebar Inputs ---------------------
st.sidebar.header("Match Parameters")

with st.sidebar.expander("📊 ELO Ratings", expanded=True):
    home_elo = st.number_input("Home ELO", value=float(st.session_state.inputs['HomeElo']), step=1.0)
    away_elo = st.number_input("Away ELO", value=float(st.session_state.inputs['AwayElo']), step=1.0)
    st.session_state.inputs['HomeElo'] = home_elo
    st.session_state.inputs['AwayElo'] = away_elo

with st.sidebar.expander("📈 Recent Form", expanded=True):
    form3_home = st.number_input("Home Form (last 3)", 0, 9, int(st.session_state.inputs.get('Form3Home', 0)))
    form5_home = st.number_input("Home Form (last 5)", 0, 15, int(st.session_state.inputs.get('Form5Home', 0)))
    form3_away = st.number_input("Away Form (last 3)", 0, 9, int(st.session_state.inputs.get('Form3Away', 0)))
    form5_away = st.number_input("Away Form (last 5)", 0, 15, int(st.session_state.inputs.get('Form5Away', 0)))
    st.session_state.inputs.update({
        'Form3Home': form3_home, 'Form5Home': form5_home,
        'Form3Away': form3_away, 'Form5Away': form5_away
    })

with st.sidebar.expander("🔫 Team Statistics", expanded=False):
    home_shots = st.number_input("Home Shots (avg)", 0.0, 30.0, float(st.session_state.inputs.get('HomeShots', 12.0)), step=0.5)
    away_shots = st.number_input("Away Shots (avg)", 0.0, 30.0, float(st.session_state.inputs.get('AwayShots', 10.0)), step=0.5)
    home_target = st.number_input("Home Shots on Target", 0.0, 15.0, float(st.session_state.inputs.get('HomeTarget', 4.0)), step=0.5)
    away_target = st.number_input("Away Shots on Target", 0.0, 15.0, float(st.session_state.inputs.get('AwayTarget', 3.0)), step=0.5)
    home_xg = st.number_input("Home xG", 0.0, 5.0, float(st.session_state.inputs.get('HomeExpectedGoals', 1.5)), step=0.1)
    away_xg = st.number_input("Away xG", 0.0, 5.0, float(st.session_state.inputs.get('AwayExpectedGoals', 1.2)), step=0.1)

    st.session_state.inputs.update({
        'HomeShots': home_shots, 'AwayShots': away_shots,
        'HomeTarget': home_target, 'AwayTarget': away_target,
        'HomeExpectedGoals': home_xg, 'AwayExpectedGoals': away_xg
    })

with st.sidebar.expander("💰 Betting Odds", expanded=False):
    odd_home = st.number_input("Home Win Odds", 1.01, value=float(st.session_state.inputs.get('OddHome', 2.0)), step=0.05)
    odd_draw = st.number_input("Draw Odds", 1.01, value=float(st.session_state.inputs.get('OddDraw', 3.0)), step=0.05)
    odd_away = st.number_input("Away Win Odds", 1.01, value=float(st.session_state.inputs.get('OddAway', 3.0)), step=0.05)
    over_25 = st.number_input("Over 2.5 Odds", 1.01, value=float(st.session_state.inputs.get('Over25', 2.0)), step=0.05)
    under_25 = st.number_input("Under 2.5 Odds", 1.01, value=float(st.session_state.inputs.get('Under25', 2.0)), step=0.05)
    handi = st.number_input("Handicap Size", -2.0, 2.0, float(st.session_state.inputs.get('HandiSize', 0.0)), step=0.25)

    st.session_state.inputs.update({
        'OddHome': odd_home, 'OddDraw': odd_draw, 'OddAway': odd_away,
        'Over25': over_25, 'Under25': under_25, 'HandiSize': handi
    })

if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.session_state.inputs = {**DEFAULTS, 'HomeElo': home_elo_scraped, 'AwayElo': away_elo_scraped}
    st.session_state.stats_fetched = False
    st.rerun()

predict_clicked = st.sidebar.button("🚀 Predict Match", type="primary", use_container_width=True)

# =============================================================================
# PREDICTION
# =============================================================================
if predict_clicked:
    raw_row = {
        'HomeElo': st.session_state.inputs['HomeElo'],
        'AwayElo': st.session_state.inputs['AwayElo'],
        'Form3Home': st.session_state.inputs['Form3Home'],
        'Form5Home': st.session_state.inputs['Form5Home'],
        'Form3Away': st.session_state.inputs['Form3Away'],
        'Form5Away': st.session_state.inputs['Form5Away'],
        'HomeShots': st.session_state.inputs['HomeShots'],
        'AwayShots': st.session_state.inputs['AwayShots'],
        'HomeTarget': st.session_state.inputs['HomeTarget'],
        'AwayTarget': st.session_state.inputs['AwayTarget'],
        'HomeExpectedGoals': st.session_state.inputs['HomeExpectedGoals'],
        'AwayExpectedGoals': st.session_state.inputs['AwayExpectedGoals'],
        'OddHome': st.session_state.inputs['OddHome'],
        'OddDraw': st.session_state.inputs['OddDraw'],
        'OddAway': st.session_state.inputs['OddAway'],
        'HandiSize': st.session_state.inputs['HandiSize'],
        'Over25': st.session_state.inputs['Over25'],
        'Under25': st.session_state.inputs['Under25'],
    }

    processed = preprocess_input(raw_row, cluster_scaler, kmeans_model)

    if processed is not None:
        outcome_encoded = outcome_model.predict(processed)[0]
        goals_pred = float(goals_model.predict(processed)[0])
        outcome_probs = outcome_model.predict_proba(processed)[0] if hasattr(outcome_model, 'predict_proba') else None

        try:
            predicted_outcome = label_encoder.inverse_transform([outcome_encoded])[0]
        except:
            predicted_outcome = str(outcome_encoded)

        outcome_display = {
            'H': f"🏆 {home_team} Win",
            'D': "🤝 Draw",
            'A': f"🏆 {away_team} Win",
        }.get(predicted_outcome, predicted_outcome)

        st.subheader(f"📊 **{home_team}** vs **{away_team}**")

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.metric("Predicted Outcome", outcome_display)
            st.metric("Total Goals", f"{round(goals_pred)}", delta=f"Raw: {goals_pred:.2f}")
            st.metric("Over/Under 2.5", "Over 2.5 ✅" if goals_pred > 2.5 else "Under 2.5 ✅")

        with col2:
            if outcome_probs is not None:
                classes = getattr(label_encoder, 'classes_', ['A', 'D', 'H'])
                prob_dict = {}
                for i, cls in enumerate(classes):
                    label = f"{home_team} Win" if cls == 'H' else "Draw" if cls == 'D' else f"{away_team} Win"
                    prob_dict[label] = outcome_probs[i]
                st.bar_chart(pd.Series(prob_dict))
                st.success(f"Confidence: {max(outcome_probs):.1%}")

        with st.expander("🔬 Advanced: Cluster Probabilities"):
            cluster_vals = processed[['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']].iloc[0]
            st.dataframe(cluster_vals.rename("Soft Probability").to_frame(), use_container_width=True)

        st.caption("Models: Gradient Boosting (outcome) + XGBoost (goals)")

else:
    st.info("👈 Adjust values in the sidebar and click **Predict Match**")
