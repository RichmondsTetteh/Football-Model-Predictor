import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import requests
import zipfile
import warnings
from datetime import datetime
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
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

LEAGUES = {
    "Premier League (England)": 39,
    "La Liga (Spain)": 140,
    "Bundesliga (Germany)": 78,
    "Serie A (Italy)": 135,
    "Ligue 1 (France)": 61,
}

DEFAULTS = {
    'HomeShots': 12.0,
    'AwayShots': 10.0,
    'HomeTarget': 4.0,
    'AwayTarget': 3.0,
    'OddHome': 2.0,
    'OddDraw': 3.0,
    'OddAway': 3.0,
    'HomeExpectedGoals': 1.5,
    'AwayExpectedGoals': 1.2,
    'Form3Home': 0,
    'Form5Home': 0,
    'Form3Away': 0,
    'Form5Away': 0,
    'HandiSize': 0.0,
    'Over25': 2.0,
    'Under25': 2.0,
}

FEATURE_COLUMNS = [
    'HomeElo', 'AwayElo', 'EloDiff', 'EloSum',
    'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'Form3Ratio', 'Form5Ratio',
    'HomeExpectedGoals', 'AwayExpectedGoals',
    'HomeWinProbability', 'DrawProbability', 'AwayWinProbability',
    'HomeShotEfficiency', 'AwayShotEfficiency',
    'HomeAttackStrength', 'AwayAttackStrength',
    'HandiSize', 'Over25', 'Under25',
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB',
]

BASE_COLUMNS = [
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25',
]

CLUSTER_ZERO = {k: 0.0 for k in ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']}


# =============================================================================
# FOOTBALL-DATA.CO.UK DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=False)
def download_and_merge_football_data():
    """Downloads and merges football data from football-data.co.uk."""
    base_url = "https://www.football-data.co.uk/"
    data_page_url = base_url + "data.php"

    all_dfs = []
    unique_csv_urls = set()
    visited_html_pages = set()

    try:
        response = requests.get(data_page_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        html_pages_to_scrape = {data_page_url}

        for link in soup.find_all('a', href=True):
            href = link['href']
            abs_url = urljoin(base_url, href)
            if abs_url.endswith(('.htm', '.php')) and abs_url.startswith(base_url) and abs_url != data_page_url:
                html_pages_to_scrape.add(abs_url)
            elif abs_url.endswith('.csv') and abs_url.startswith(base_url):
                unique_csv_urls.add(abs_url)
            elif abs_url.endswith('.zip') and abs_url.startswith(base_url):
                if 'data.zip' in abs_url or re.search(r'\b[A-Z0-9]{1,2}\.zip$', abs_url):
                    try:
                        zr = requests.get(abs_url, stream=True, timeout=15)
                        zr.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                            for fn in z.namelist():
                                if fn.endswith('.csv') and not fn.startswith('__MACOSX'):
                                    try:
                                        with z.open(fn) as cf:
                                            df_s = pd.read_csv(cf, encoding='ISO-8859-1', on_bad_lines='skip')
                                            df_s['Source_File'] = fn
                                            all_dfs.append(df_s)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        for page_url in list(html_pages_to_scrape):
            if page_url in visited_html_pages:
                continue
            visited_html_pages.add(page_url)
            try:
                pr = requests.get(page_url, timeout=15)
                pr.raise_for_status()
                ps = BeautifulSoup(pr.content, 'html.parser')
                for link in ps.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.csv'):
                        full_url = urljoin(page_url, href)
                        if full_url.startswith(base_url):
                            unique_csv_urls.add(full_url)
                    elif href.endswith('.zip'):
                        if 'data.zip' in href or re.search(r'\b[A-Z0-9]{1,2}\.zip$', href):
                            fzu = urljoin(page_url, href)
                            if fzu.startswith(base_url):
                                try:
                                    zr = requests.get(fzu, stream=True, timeout=15)
                                    zr.raise_for_status()
                                    with zipfile.ZipFile(io.BytesIO(zr.content)) as z:
                                        for fn in z.namelist():
                                            if fn.endswith('.csv') and not fn.startswith('__MACOSX'):
                                                try:
                                                    with z.open(fn) as cf:
                                                        df_s = pd.read_csv(cf, encoding='ISO-8859-1', on_bad_lines='skip')
                                                        df_s['Source_File'] = fn
                                                        all_dfs.append(df_s)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
            except Exception:
                pass

        for csv_url in unique_csv_urls:
            try:
                df_s = pd.read_csv(csv_url, encoding='ISO-8859-1', on_bad_lines='skip')
                df_s['Source_File'] = csv_url.split('/')[-1]
                all_dfs.append(df_s)
            except Exception:
                pass

        if not all_dfs:
            return pd.DataFrame()

        df_full = pd.concat(all_dfs, ignore_index=True, sort=False)
        df_full['Date'] = pd.to_datetime(df_full['Date'], format='%d/%m/%Y', errors='coerce')
        df_full.dropna(subset=['Date'], inplace=True)
        df_full.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
        df_full = df_full.sort_values(by='Date').reset_index(drop=True)

        for col in ['HS', 'AS', 'HST', 'AST']:
            if col in df_full.columns:
                df_full[col] = df_full[col].fillna(0)

        return df_full

    except Exception as e:
        st.error(f"Error loading football data: {e}")
        return pd.DataFrame()


# =============================================================================
# FORM & STATS HELPERS (from first app)
# =============================================================================
def calculate_team_form(team_name, df_data, num_matches):
    team_matches = df_data[(df_data['HomeTeam'] == team_name) | (df_data['AwayTeam'] == team_name)].copy()
    if team_matches.empty:
        return 0, "No matches found", pd.DataFrame()
    recent = team_matches.sort_values('Date', ascending=False).head(num_matches)
    points = 0
    results = []
    for _, row in recent.iterrows():
        if row['HomeTeam'] == team_name:
            if row['FTR'] == 'H': points += 3; results.append('W')
            elif row['FTR'] == 'D': points += 1; results.append('D')
            else: results.append('L')
        else:
            if row['FTR'] == 'A': points += 3; results.append('W')
            elif row['FTR'] == 'D': points += 1; results.append('D')
            else: results.append('L')
    results.reverse()
    return points, " ".join(results), recent


def calculate_average_stats(team_name, recent_matches_df):
    if recent_matches_df.empty:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    sf = sa = stf = sta = 0
    for _, row in recent_matches_df.iterrows():
        if row['HomeTeam'] == team_name:
            sf += row.get('HS', 0); sa += row.get('AS', 0)
            stf += row.get('HST', 0); sta += row.get('AST', 0)
        else:
            sf += row.get('AS', 0); sa += row.get('HS', 0)
            stf += row.get('AST', 0); sta += row.get('HST', 0)
    n = len(recent_matches_df)
    return {
        'Shots For': sf / n, 'Shots Against': sa / n,
        'Shots On Target For': stf / n, 'Shots On Target Against': sta / n
    }


def calculate_season_average_stats(team_name, df_data):
    tm = df_data[(df_data['HomeTeam'] == team_name) | (df_data['AwayTeam'] == team_name)].copy()
    if tm.empty:
        return {'Shots For': 0, 'Shots Against': 0, 'Shots On Target For': 0, 'Shots On Target Against': 0}
    sf = sa = stf = sta = 0
    for _, row in tm.iterrows():
        if row['HomeTeam'] == team_name:
            sf += row.get('HS', 0); sa += row.get('AS', 0)
            stf += row.get('HST', 0); sta += row.get('AST', 0)
        else:
            sf += row.get('AS', 0); sa += row.get('HS', 0)
            stf += row.get('AST', 0); sta += row.get('HST', 0)
    n = len(tm)
    return {
        'Shots For': sf / n, 'Shots Against': sa / n,
        'Shots On Target For': stf / n, 'Shots On Target Against': sta / n
    }


# =============================================================================
# ELO
# =============================================================================
@st.cache_data(show_spinner="Fetching ELO ratings...", ttl=1800)
def load_elo_ratings():
    for delta in [0, 1]:
        try:
            date_str = (datetime.now() - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            resp = requests.get(f"http://api.clubelo.com/{date_str}", timeout=15)
            resp.raise_for_status()
            df_elo = pd.read_csv(io.StringIO(resp.text))
            df_elo['Clean_Club'] = df_elo['Club'].astype(str).str.strip()
            return df_elo
        except Exception:
            continue
    st.warning("Could not load ELO ratings. Defaulting to 1500.")
    return pd.DataFrame(columns=['Club', 'Elo', 'Clean_Club'])


def get_elo_for_team(team_name, df_elo):
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
    if len(matches) > 1:
        candidates = ', '.join(matches['Clean_Club'].tolist()[:5])
        return DEFAULT_ELO, f'"{team_name}" matched multiple ({candidates}). Using default.'
    return float(matches['Elo'].iloc[0]), None


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
def _api_headers(api_key):
    return {"x-apisports-key": api_key}


def search_team(team_name, api_key):
    try:
        resp = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=_api_headers(api_key),
                            params={"search": team_name}, timeout=12)
        resp.raise_for_status()
        results = resp.json().get("response", [])
        if results:
            return results[0]["team"]
    except Exception as e:
        st.warning(f"Team search failed for '{team_name}': {e}")
    return None


def fetch_team_statistics(team_id, league_id, api_key):
    current_year = datetime.now().year
    for season in [current_year, current_year - 1]:
        try:
            resp = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics",
                                headers=_api_headers(api_key),
                                params={"league": league_id, "team": team_id, "season": season},
                                timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            response_data = payload.get("response")
            if response_data and isinstance(response_data, dict):
                return response_data
        except Exception:
            continue
    return None


def parse_form_string(form_string):
    if not form_string:
        return 0, 0
    pts = {'W': 3, 'D': 1, 'L': 0}
    recent = [pts.get(c, 0) for c in form_string.strip()]
    form3 = sum(recent[-3:]) if len(recent) >= 3 else sum(recent)
    form5 = sum(recent[-5:]) if len(recent) >= 5 else sum(recent)
    return form3, form5


def extract_avg_goals_scored(stats, venue="total"):
    try:
        raw = stats["goals"]["for"]["average"][venue]
        return float(raw) if raw is not None else 1.5
    except Exception:
        return 1.5


def extract_avg_goals_conceded(stats, venue="total"):
    try:
        raw = stats["goals"]["against"]["average"][venue]
        return float(raw) if raw is not None else 1.2
    except Exception:
        return 1.2


def estimate_shots_from_goals(avg_goals):
    avg_shots = max(5.0, min(round(avg_goals / 0.10, 1), 28.0))
    avg_shots_target = max(2.0, min(round(avg_goals / 0.35, 1), 14.0))
    return avg_shots, avg_shots_target


def estimate_over25_odd(stats):
    try:
        over_count = stats["goals"]["for"]["under_over"]["2.5"]["over"]
        total_played = stats["fixtures"]["played"]["total"]
        if over_count is None or not total_played:
            return 2.0
        rate = over_count / total_played
        odd = round(1.0 / rate, 2) if rate > 0 else 2.0
        return max(1.10, min(odd, 4.00))
    except Exception:
        return 2.0


def build_fetched_inputs(home_stats, away_stats):
    form3_home, form5_home = parse_form_string(home_stats.get("form", ""))
    form3_away, form5_away = parse_form_string(away_stats.get("form", ""))
    xg_home = extract_avg_goals_scored(home_stats, venue="home")
    xg_away = extract_avg_goals_scored(away_stats, venue="away")
    home_shots, home_target = estimate_shots_from_goals(xg_home)
    away_shots, away_target = estimate_shots_from_goals(xg_away)
    home_o25_odd = estimate_over25_odd(home_stats)
    away_o25_odd = estimate_over25_odd(away_stats)
    combined_over_odd = round((home_o25_odd + away_o25_odd) / 2, 2)
    combined_under_odd = max(1.10, min(round(3.6 / combined_over_odd, 2), 4.00))
    return {
        'Form3Home': form3_home, 'Form5Home': form5_home,
        'Form3Away': form3_away, 'Form5Away': form5_away,
        'HomeExpectedGoals': round(xg_home, 2), 'AwayExpectedGoals': round(xg_away, 2),
        'HomeShots': home_shots, 'AwayShots': away_shots,
        'HomeTarget': home_target, 'AwayTarget': away_target,
        'Over25': combined_over_odd, 'Under25': combined_under_odd,
        'OddHome': DEFAULTS['OddHome'], 'OddDraw': DEFAULTS['OddDraw'],
        'OddAway': DEFAULTS['OddAway'], 'HandiSize': DEFAULTS['HandiSize'],
    }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def clean_input_data(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median() if df[col].notna().any() else 0.0
        df[col].fillna(median, inplace=True)
    return df


def get_cluster_probabilities(row, scaler, kmeans):
    if scaler is None or kmeans is None:
        return CLUSTER_ZERO.copy()
    try:
        X = pd.DataFrame([{
            'HomeShots': row.get('HomeShots', 12.0),
            'AwayShots': row.get('AwayShots', 10.0),
            'HomeTarget': row.get('HomeTarget', 4.0),
            'AwayTarget': row.get('AwayTarget', 3.0),
            'HomeElo': row.get('HomeElo', 1500.0),
            'AwayElo': row.get('AwayElo', 1500.0),
        }])
        X_scaled = scaler.transform(X)
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
        probas = softmax(-distances, axis=1)[0]
        return dict(zip(['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'], probas))
    except Exception:
        return CLUSTER_ZERO.copy()


def create_derived_features(df):
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


def preprocess_input(raw_row, scaler, kmeans):
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
    return df[FEATURE_COLUMNS].fillna(0)


# =============================================================================
# SESSION STATE INIT
# =============================================================================
if 'inputs' not in st.session_state:
    st.session_state.inputs = {**DEFAULTS, 'HomeElo': DEFAULT_ELO, 'AwayElo': DEFAULT_ELO}
if 'stats_fetched' not in st.session_state:
    st.session_state.stats_fetched = False
if 'fetch_notes' not in st.session_state:
    st.session_state.fetch_notes = ""
if 'fd_analysis_done' not in st.session_state:
    st.session_state.fd_analysis_done = False
if 'fd_results' not in st.session_state:
    st.session_state.fd_results = {}

# =============================================================================
# APP LAYOUT
# =============================================================================
st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor")
st.markdown(
    "Predict match outcomes and total goals using Gradient Boosting + XGBoost. "
    "**Analyze Teams** below to auto-fill form and shot stats from historical data, "
    "or use **Fetch Stats** to pull from API-Football."
)

# Load resources
df_elo = load_elo_ratings()
outcome_model, goals_model, label_encoder = load_models()
cluster_scaler, kmeans_model = load_cluster_models()

if outcome_model is None or goals_model is None:
    st.error("Prediction models could not be loaded. Check file paths.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "🔑 API-Football Key", type="password",
    help="Free key at https://www.api-football.com/ — required only for stat fetching.",
)
selected_league = st.sidebar.selectbox("League", options=list(LEAGUES.keys()))
league_id = LEAGUES[selected_league]
st.sidebar.divider()

# ── Team name inputs ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("🏠 Home Team", "Arsenal")
with col2:
    away_team = st.text_input("🏟️ Away Team", "Chelsea")

home_elo_scraped, home_warn = get_elo_for_team(home_team, df_elo)
away_elo_scraped, away_warn = get_elo_for_team(away_team, df_elo)

if home_warn:
    st.warning(f"🏠 {home_warn}")
if away_warn:
    st.warning(f"🏟️ {away_warn}")

if 'prev_home' not in st.session_state or st.session_state.prev_home != home_team:
    st.session_state.inputs['HomeElo'] = home_elo_scraped
    st.session_state.prev_home = home_team
    st.session_state.stats_fetched = False
    st.session_state.fd_analysis_done = False

if 'prev_away' not in st.session_state or st.session_state.prev_away != away_team:
    st.session_state.inputs['AwayElo'] = away_elo_scraped
    st.session_state.prev_away = away_team
    st.session_state.stats_fetched = False
    st.session_state.fd_analysis_done = False

# ── Action buttons row ────────────────────────────────────────────────────────
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

with btn_col1:
    analyze_clicked = st.button(
        "📊 Analyze Teams (Historical Data)",
        type="primary",
        use_container_width=True,
        help="Loads data from football-data.co.uk and calculates form + shot stats"
    )

with btn_col2:
    fetch_clicked = st.button(
        "🔍 Fetch Stats (API-Football)",
        type="secondary",
        use_container_width=True,
        disabled=not bool(api_key),
        help="Pulls form, xG and shot estimates from API-Football." if api_key else "Enter API key in sidebar.",
    )

with btn_col3:
    if st.session_state.stats_fetched:
        st.success("✅ API stats fetched — review sidebar values before predicting.")
    elif st.session_state.fd_analysis_done:
        st.success("✅ Historical stats loaded — inputs updated. Ready to predict.")
    elif not api_key:
        st.info("💡 Add your API-Football key in the sidebar to enable stat fetching.")

# ── Historical data analysis ──────────────────────────────────────────────────
if analyze_clicked:
    with st.spinner('Loading and merging football data from football-data.co.uk... This may take a moment.'):
        df_full = download_and_merge_football_data()

    if df_full.empty:
        st.error("Could not load football data. Check internet connection or data source.")
    else:
        # Calculate form
        pts3_home, form3_str_home, matches3_home = calculate_team_form(home_team, df_full, 3)
        pts5_home, form5_str_home, matches5_home = calculate_team_form(home_team, df_full, 5)
        pts3_away, form3_str_away, matches3_away = calculate_team_form(away_team, df_full, 3)
        pts5_away, form5_str_away, matches5_away = calculate_team_form(away_team, df_full, 5)

        # Calculate weighted shot stats
        stats5_home = calculate_average_stats(home_team, matches5_home)
        stats5_away = calculate_average_stats(away_team, matches5_away)
        season_stats_home = calculate_season_average_stats(home_team, df_full)
        season_stats_away = calculate_season_average_stats(away_team, df_full)

        w_season, w_last5 = 0.70, 0.30
        weighted_home = {
            k: (w_season * season_stats_home[k]) + (w_last5 * stats5_home[k])
            for k in stats5_home
        }
        weighted_away = {
            k: (w_season * season_stats_away[k]) + (w_last5 * stats5_away[k])
            for k in stats5_away
        }

        # Store results for display
        st.session_state.fd_results = {
            'home_team': home_team, 'away_team': away_team,
            'pts3_home': pts3_home, 'form3_str_home': form3_str_home, 'matches3_home': matches3_home,
            'pts5_home': pts5_home, 'form5_str_home': form5_str_home, 'matches5_home': matches5_home,
            'pts3_away': pts3_away, 'form3_str_away': form3_str_away, 'matches3_away': matches3_away,
            'pts5_away': pts5_away, 'form5_str_away': form5_str_away, 'matches5_away': matches5_away,
            'weighted_home': weighted_home, 'weighted_away': weighted_away,
        }
        st.session_state.fd_analysis_done = True

        # Auto-populate prediction inputs from historical data
        # xG estimated from shots on target (shots on target / 3 ≈ goals)
        home_xg_est = round(weighted_home['Shots On Target For'] / 3.0, 2)
        away_xg_est = round(weighted_away['Shots On Target For'] / 3.0, 2)

        st.session_state.inputs.update({
            'Form3Home': pts3_home,
            'Form5Home': pts5_home,
            'Form3Away': pts3_away,
            'Form5Away': pts5_away,
            'HomeShots': round(weighted_home['Shots For'], 1),
            'AwayShots': round(weighted_away['Shots For'], 1),
            'HomeTarget': round(weighted_home['Shots On Target For'], 1),
            'AwayTarget': round(weighted_away['Shots On Target For'], 1),
            'HomeExpectedGoals': max(0.1, home_xg_est),
            'AwayExpectedGoals': max(0.1, away_xg_est),
        })
        st.session_state.stats_fetched = False

        st.rerun()

# ── Display historical analysis results ──────────────────────────────────────
if st.session_state.fd_analysis_done and st.session_state.fd_results:
    r = st.session_state.fd_results
    if r.get('home_team') == home_team and r.get('away_team') == away_team:
        st.markdown("---")
        st.header(f"📋 Historical Analysis: {home_team} vs {away_team}")

        left, right = st.columns(2)

        with left:
            st.subheader(f"🏠 {home_team}")
            st.write(f"**Last 3:** {r['form3_str_home']} — Points: **{r['pts3_home']}**")
            if not r['matches3_home'].empty:
                cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                                if c in r['matches3_home'].columns]
                st.dataframe(r['matches3_home'][cols_to_show], use_container_width=True)

            st.write(f"**Last 5:** {r['form5_str_home']} — Points: **{r['pts5_home']}**")
            if not r['matches5_home'].empty:
                cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                                if c in r['matches5_home'].columns]
                st.dataframe(r['matches5_home'][cols_to_show], use_container_width=True)

            st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
            for k, v in r['weighted_home'].items():
                st.write(f"- {k}: **{v:.2f}**")

        with right:
            st.subheader(f"🏟️ {away_team}")
            st.write(f"**Last 3:** {r['form3_str_away']} — Points: **{r['pts3_away']}**")
            if not r['matches3_away'].empty:
                cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                                if c in r['matches3_away'].columns]
                st.dataframe(r['matches3_away'][cols_to_show], use_container_width=True)

            st.write(f"**Last 5:** {r['form5_str_away']} — Points: **{r['pts5_away']}**")
            if not r['matches5_away'].empty:
                cols_to_show = [c for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                                if c in r['matches5_away'].columns]
                st.dataframe(r['matches5_away'][cols_to_show], use_container_width=True)

            st.markdown("**Weighted Shot Stats (70% season / 30% last 5):**")
            for k, v in r['weighted_away'].items():
                st.write(f"- {k}: **{v:.2f}**")

        st.info(
            "✅ Form points and shot stats from historical data have been automatically applied "
            "to the prediction inputs in the sidebar. Review and adjust as needed, then click **Predict Match**."
        )

# ── API-Football fetch ────────────────────────────────────────────────────────
if fetch_clicked and api_key:
    with st.spinner(f"Fetching stats for {home_team} and {away_team}…"):
        home_info = search_team(home_team, api_key)
        away_info = search_team(away_team, api_key)
        if not home_info or not away_info:
            st.error("Could not find one or both teams. Try the full official name.")
        else:
            home_stats = fetch_team_statistics(home_info["id"], league_id, api_key)
            away_stats = fetch_team_statistics(away_info["id"], league_id, api_key)
            if home_stats and away_stats:
                new_inputs = build_fetched_inputs(home_stats, away_stats)
                st.session_state.inputs.update(new_inputs)
                st.session_state.stats_fetched = True
                form_len = len(home_stats.get("form", ""))
                st.session_state.fetch_notes = (
                    f"Data from {selected_league} via API-Football "
                    f"({home_info['name']} vs {away_info['name']}). "
                    f"Form from last {form_len} recorded matches. "
                    f"xG = venue-split avg goals scored. "
                    f"Shots estimated from scoring rate. "
                    f"Betting odds not supplied — update manually."
                )
                st.rerun()
            else:
                st.warning("Team(s) found but no statistics returned. Check league and season selection.")

if st.session_state.get('fetch_notes'):
    st.caption(f"📰 {st.session_state.fetch_notes}")

# ── Sidebar parameter inputs ──────────────────────────────────────────────────
st.sidebar.header("Match Parameters")

with st.sidebar.expander("📊 ELO Ratings", expanded=True):
    home_elo = st.number_input("Home ELO", value=float(st.session_state.inputs['HomeElo']), step=1.0)
    away_elo = st.number_input("Away ELO", value=float(st.session_state.inputs['AwayElo']), step=1.0)
    st.session_state.inputs['HomeElo'] = home_elo
    st.session_state.inputs['AwayElo'] = away_elo

with st.sidebar.expander("📈 Recent Form", expanded=True):
    form3_home = st.number_input("Home Form (last 3)", 0, 9, int(st.session_state.inputs['Form3Home']), step=1)
    form5_home = st.number_input("Home Form (last 5)", 0, 15, int(st.session_state.inputs['Form5Home']), step=1)
    form3_away = st.number_input("Away Form (last 3)", 0, 9, int(st.session_state.inputs['Form3Away']), step=1)
    form5_away = st.number_input("Away Form (last 5)", 0, 15, int(st.session_state.inputs['Form5Away']), step=1)
    st.session_state.inputs.update({
        'Form3Home': form3_home, 'Form5Home': form5_home,
        'Form3Away': form3_away, 'Form5Away': form5_away,
    })

with st.sidebar.expander("🔫 Team Statistics", expanded=False):
    home_shots = st.number_input("Home Shots (avg)", 0.0, 30.0, float(st.session_state.inputs['HomeShots']), step=0.5)
    away_shots = st.number_input("Away Shots (avg)", 0.0, 30.0, float(st.session_state.inputs['AwayShots']), step=0.5)
    home_target = st.number_input("Home Shots on Target", 0.0, 15.0, float(st.session_state.inputs['HomeTarget']), step=0.5)
    away_target = st.number_input("Away Shots on Target", 0.0, 15.0, float(st.session_state.inputs['AwayTarget']), step=0.5)
    home_xg = st.number_input("Home xG (avg goals)", 0.0, 5.0, float(st.session_state.inputs['HomeExpectedGoals']), step=0.1)
    away_xg = st.number_input("Away xG (avg goals)", 0.0, 5.0, float(st.session_state.inputs['AwayExpectedGoals']), step=0.1)
    st.session_state.inputs.update({
        'HomeShots': home_shots, 'AwayShots': away_shots,
        'HomeTarget': home_target, 'AwayTarget': away_target,
        'HomeExpectedGoals': home_xg, 'AwayExpectedGoals': away_xg,
    })

with st.sidebar.expander("💰 Betting Odds", expanded=False):
    st.caption("Not supplied by API-Football — enter manually or leave as defaults.")
    odd_home = st.number_input("Home Win Odds", 1.01, value=float(st.session_state.inputs['OddHome']), step=0.05)
    odd_draw = st.number_input("Draw Odds", 1.01, value=float(st.session_state.inputs['OddDraw']), step=0.05)
    odd_away = st.number_input("Away Win Odds", 1.01, value=float(st.session_state.inputs['OddAway']), step=0.05)
    over_25 = st.number_input("Over 2.5 Odds", 1.01, value=float(st.session_state.inputs['Over25']), step=0.05)
    under_25 = st.number_input("Under 2.5 Odds", 1.01, value=float(st.session_state.inputs['Under25']), step=0.05)
    handi = st.number_input("Handicap Size", -2.0, 2.0, float(st.session_state.inputs['HandiSize']), step=0.25)
    st.session_state.inputs.update({
        'OddHome': odd_home, 'OddDraw': odd_draw, 'OddAway': odd_away,
        'Over25': over_25, 'Under25': under_25, 'HandiSize': handi,
    })

if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.session_state.inputs = {**DEFAULTS, 'HomeElo': home_elo_scraped, 'AwayElo': away_elo_scraped}
    st.session_state.stats_fetched = False
    st.session_state.fetch_notes = ""
    st.session_state.fd_analysis_done = False
    st.session_state.fd_results = {}
    st.rerun()

predict_clicked = st.sidebar.button("🚀 Predict Match", type="primary", use_container_width=True)

# =============================================================================
# PREDICTION
# =============================================================================
if predict_clicked:
    inp = st.session_state.inputs
    raw_row = {
        'HomeElo': inp['HomeElo'], 'AwayElo': inp['AwayElo'],
        'Form3Home': inp['Form3Home'], 'Form5Home': inp['Form5Home'],
        'Form3Away': inp['Form3Away'], 'Form5Away': inp['Form5Away'],
        'HomeShots': inp['HomeShots'], 'AwayShots': inp['AwayShots'],
        'HomeTarget': inp['HomeTarget'], 'AwayTarget': inp['AwayTarget'],
        'HomeExpectedGoals': inp['HomeExpectedGoals'], 'AwayExpectedGoals': inp['AwayExpectedGoals'],
        'OddHome': inp['OddHome'], 'OddDraw': inp['OddDraw'], 'OddAway': inp['OddAway'],
        'HandiSize': inp['HandiSize'], 'Over25': inp['Over25'], 'Under25': inp['Under25'],
    }

    processed = preprocess_input(raw_row, cluster_scaler, kmeans_model)
    if processed is not None:
        outcome_encoded = outcome_model.predict(processed)[0]
        goals_pred = float(goals_model.predict(processed)[0])
        outcome_probs = (
            outcome_model.predict_proba(processed)[0]
            if hasattr(outcome_model, 'predict_proba') else None
        )
        try:
            predicted_outcome = label_encoder.inverse_transform([outcome_encoded])[0]
        except ValueError:
            predicted_outcome = str(outcome_encoded)

        outcome_display = {
            'H': f"🏆 {home_team} Win",
            'D': "🤝 Draw",
            'A': f"🏆 {away_team} Win",
        }.get(predicted_outcome, predicted_outcome)

        st.markdown("---")
        st.subheader(f"📊 Prediction: **{home_team}** vs **{away_team}**")

        data_source = ""
        if st.session_state.stats_fetched:
            data_source = "ℹ️ Stats sourced from API-Football."
        elif st.session_state.fd_analysis_done:
            data_source = "ℹ️ Stats sourced from football-data.co.uk historical data."
        if data_source:
            st.caption(data_source)

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.metric("Predicted Outcome", outcome_display)
            st.metric(
                "Total Goals", f"{round(goals_pred)} goals",
                delta=f"Raw: {goals_pred:.2f}", delta_color="off",
                help="Rounded prediction. 'Raw' shows the unrounded model output.",
            )
            st.metric("Over/Under 2.5", "Over 2.5 ✅" if goals_pred > 2.5 else "Under 2.5 ✅")

        with col2:
            if outcome_probs is not None:
                classes = getattr(label_encoder, 'classes_', ['A', 'D', 'H'])
                prob_dict = {
                    (f"{home_team} Win" if c == 'H' else "Draw" if c == 'D' else f"{away_team} Win"):
                    outcome_probs[i]
                    for i, c in enumerate(classes)
                }
                st.bar_chart(pd.Series(prob_dict))
                st.success(f"Confidence: {max(outcome_probs):.1%}")

        with st.expander("🔬 Advanced: Cluster Probabilities"):
            st.caption(
                "Soft cluster assignments via softmax(−distances). "
                "Distance-based approximations, not true posterior probabilities."
            )
            cluster_vals = processed[['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']].iloc[0]
            st.dataframe(cluster_vals.rename("Soft Probability").to_frame(), use_container_width=True)

        st.caption("Models: Gradient Boosting (outcome) + XGBoost (goals)")

else:
    if not st.session_state.fd_analysis_done and not st.session_state.stats_fetched:
        st.info("👈 Click **Analyze Teams** to load historical stats, or **Fetch Stats** for API data. Then click **Predict Match** in the sidebar.")
