import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import warnings
import requests # Import the requests library for API calls

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# =============================================================================
# CONSTANTS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

OUTCOME_MODEL_FILENAME = os.path.join(BASE_DIR, 'Gradient_Boosting_Classifier_outcome_model.joblib')
GOALS_MODEL_FILENAME = os.path.join(BASE_DIR, 'XGBoost_Regressor_goals_model.joblib')
LABEL_ENCODER_FILENAME = os.path.join(BASE_DIR, 'label_encoder.joblib')

CLUSTER_SCALER_FILENAME = os.path.join(BASE_DIR, 'cluster_robust_scaler.joblib')
CLUSTER_KMEANS_FILENAME = os.path.join(BASE_DIR, 'cluster_kmeans_model.joblib')

# =============================================================================
# API KEY MANAGEMENT
# =============================================================================
# For Streamlit Cloud deployment, use st.secrets.
# For local testing, you can set it as an environment variable or define it here.
if "API_FOOTBALL_KEY" in st.secrets:
    API_FOOTBALL_KEY = st.secrets["API_FOOTBALL_KEY"]
    st.info("API Key loaded from Streamlit Secrets.")
elif os.environ.get("API_FOOTBALL_KEY"):
    API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY")
    st.info("API Key loaded from environment variable (for local testing).")
else:
    API_FOOTBALL_KEY = "YOUR_API_FOOTBALL_KEY_HERE" # Fallback for local testing without env var
    st.warning("⚠️ API_FOOTBALL_KEY not found in Streamlit Secrets or environment variables. Please add it for live data fetching. Using placeholder.")

# =============================================================================
# RESOURCE LOADING
# =============================================================================
@st.cache_resource
def load_resources():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_model = joblib.load(GOALS_MODEL_FILENAME)

        try:
            label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        except FileNotFoundError:
            label_encoder = LabelEncoder()
            label_encoder.fit(['A', 'D', 'H'])

        return outcome_model, goals_model, label_encoder
    except Exception as e:
        st.error(f"❌ Failed to load main models: {e}")
        st.info(f"Looking for files in: {BASE_DIR}")
        return None, None, None


@st.cache_resource(show_spinner="Loading cluster models...")
def load_cluster_models():
    if os.path.exists(CLUSTER_SCALER_FILENAME) and os.path.exists(CLUSTER_KMEANS_FILENAME):
        try:
            scaler = joblib.load(CLUSTER_SCALER_FILENAME)
            kmeans = joblib.load(CLUSTER_KMEANS_FILENAME)
            st.success("✅ Cluster models loaded successfully!")
            return scaler, kmeans
        except Exception as e:
            st.error(f"❌ Error loading cluster models: {e}")
            return None, None
    else:
        st.error("❌ Cluster models not found.")
        st.info("Make sure cluster_robust_scaler.joblib and cluster_kmeans_model.joblib are in the repository root.")
        return None, None


outcome_model, goals_model, label_encoder = load_resources()
cluster_scaler, kmeans_model = load_cluster_models()

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
feature_columns = [
    'HomeElo', 'AwayElo', 'EloDiff', 'EloSum', 'Form3Home', 'Form5Home',
    'Form3Away', 'Form5Away', 'Form3Ratio', 'Form5Ratio',
    'HomeExpectedGoals', 'AwayExpectedGoals', 'HomeWinProbability',
    'DrawProbability', 'AwayWinProbability', 'HomeShotEfficiency',
    'AwayShotEfficiency', 'HomeAttackStrength', 'AwayAttackStrength',
    'HandiSize', 'Over25', 'Under25', 'C_LTH', 'C_LTA', 'C_VHD',
    'C_VAD', 'C_HTB', 'C_PHB'
]

base_columns = [
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def clean_input_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median() if len(df) > 0 else 0.0)
    return df


def compute_cluster_features(row):
    """Return EXACT raw columns the scaler was fitted on"""
    # Ensure the order of features matches the order used during fitting
    # (HomeShots, AwayShots, HomeTarget, AwayTarget, HomeElo, AwayElo)
    return pd.DataFrame([
        {
            'HomeShots': row.get('HomeShots', 12.0),
            'AwayShots': row.get('AwayShots', 10.0),
            'HomeTarget': row.get('HomeTarget', 4.0),
            'AwayTarget': row.get('AwayTarget', 3.0),
            'HomeElo': row.get('HomeElo', 1500.0),
            'AwayElo': row.get('AwayElo', 1500.0)
        }
    ])


def get_cluster_probabilities(row):
    if cluster_scaler is None or kmeans_model is None:
        return {'C_LTH': 0.0, 'C_LTA': 0.0, 'C_VHD': 0.0,
                'C_VAD': 0.0, 'C_HTB': 0.0, 'C_PHB': 0.0}

    try:
        X = compute_cluster_features(row)
        X_scaled = cluster_scaler.transform(X)

        distances = np.linalg.norm(X_scaled[:, np.newaxis] - kmeans_model.cluster_centers_, axis=2)
        probas = softmax(-distances, axis=1)[0]

        cluster_names = ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']
        return dict(zip(cluster_names, probas))

    except Exception as e:
        st.warning(f"Cluster computation failed: {e}. Using fallback (0.0).")
        return {'C_LTH': 0.0, 'C_LTA': 0.0, 'C_VHD': 0.0,
                'C_VAD': 0.0, 'C_HTB': 0.0, 'C_PHB': 0.0}


def create_input_features(df):
    df = df.copy()
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    df['EloSum'] = df['HomeElo'] + df['AwayElo']
    df['Form3Ratio'] = df['Form3Home'] / (df['Form3Away'] + 0.1)
    df['Form5Ratio'] = df['Form5Home'] / (df['Form5Away'] + 0.1)
    df['HomeWinProbability'] = np.where(df['OddHome'] > 0, 1 / df['OddHome'], 0.5)
    df['DrawProbability'] = np.where(df['OddDraw'] > 0, 1 / df['OddDraw'], 0.25)
    df['AwayWinProbability'] = np.where(df['OddAway'] > 0, 1 / df['OddAway'], 0.25)
    df['HomeShotEfficiency'] = df['HomeTarget'] / (df['HomeShots'] + 0.1)
    df['AwayShotEfficiency'] = df['AwayTarget'] / (df['AwayShots'] + 0.1)
    df['HomeAttackStrength'] = df['HomeExpectedGoals'] * df['Form5Home'] / 15
    df['AwayAttackStrength'] = df['AwayExpectedGoals'] * df['Form5Away'] / 15
    return df


def preprocess_input_data(input_df, feature_columns):
    defaults = {
        'HomeShots': 12.0, 'AwayShots': 10.0, 'HomeTarget': 4.0, 'AwayTarget': 3.0,
        'OddHome': 2.0, 'OddDraw': 3.0, 'OddAway': 3.0,
        'HomeExpectedGoals': 1.5, 'AwayExpectedGoals': 1.2,
        'Form3Home': 0, 'Form5Home': 0, 'Form3Away': 0, 'Form5Away': 0,
        'HandiSize': 0.0, 'Over25': 2.0, 'Under25': 2.0,
    }

    for col in base_columns + ['HomeExpectedGoals', 'AwayExpectedGoals']:
        if col not in input_df.columns:
            input_df[col] = defaults.get(col, 0.0)

    cleaned_df = clean_input_data(input_df)
    featured_df = create_input_features(cleaned_df)

    # Compute clusters automatically
    cluster_dict = get_cluster_probabilities(featured_df.iloc[0])
    for col, val in cluster_dict.items():
        featured_df[col] = val

    missing = [col for col in feature_columns if col not in featured_df.columns]
    if missing:
        st.warning(f"⚠️ Missing features: {missing}. Using 0 as fallback.")
        for col in missing:
            featured_df[col] = 0.0

    processed_features = featured_df[feature_columns].fillna(0)
    return processed_features


# =============================================================================
# API FOOTBALL INTEGRATION FUNCTION
# =============================================================================
@st.cache_data(ttl=3600) # Cache API responses for 1 hour
def fetch_team_stats_from_api_football(home_team: str, away_team: str, api_key: str):
    """
    Fetches team statistics from an API Football service.
    YOU NEED TO CUSTOMIZE THIS FUNCTION based on the specific API Football you use.

    Args:
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.
        api_key (str): Your API Football key.

    Returns:
        dict: A dictionary containing the fetched statistics, or None if an error occurs.
              The keys should match the input_config keys (e.g., 'Form3Home', 'HomeShots').
    """
    st.info(f"Attempting to fetch stats for {home_team} vs {away_team}...")
    # --- REPLACE THIS SECTION WITH YOUR ACTUAL API CALL LOGIC ---
    # Example using a hypothetical API structure (adjust as needed)
    # You might need to first resolve team IDs if the API uses them instead of names.
    try:
        # Example: Replace with your API endpoint and parameters
        # API_URL = "https://api.football-api.com/v2/teams"
        # headers = {'X-RapidAPI-Key': api_key, 'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'}
        # params = {'search': home_team} or similar

        # For demonstration, we'll return dummy data.
        # In a real scenario, you'd make `requests.get` calls here
        # and parse the JSON response.
        # Example: response = requests.get(API_URL, headers=headers, params=params)
        #          response.raise_for_status() # Raise an exception for HTTP errors
        #          data = response.json()

        # Placeholder logic: Simulate API response based on team names
        if home_team.lower() == "team a" and away_team.lower() == "team b":
            st.success("Simulated API data fetched for Team A vs Team B!")
            return {
                'Form3Home': 7, 'Form5Home': 12,
                'Form3Away': 5, 'Form5Away': 8,
                'HomeShots': 15.0, 'AwayShots': 11.0,
                'HomeTarget': 6.0, 'AwayTarget': 4.0,
                'HomeExpectedGoals': 1.8, 'AwayExpectedGoals': 1.3
            }
        elif home_team.lower() == "team c" and away_team.lower() == "team d":
             st.success("Simulated API data fetched for Team C vs Team D!")
             return {
                'Form3Home': 6, 'Form5Home': 9,
                'Form3Away': 3, 'Form5Away': 6,
                'HomeShots': 10.0, 'AwayShots': 14.0,
                'HomeTarget': 3.0, 'AwayTarget': 5.0,
                'HomeExpectedGoals': 1.0, 'AwayExpectedGoals': 1.7
            }
        else:
            st.warning(f"No API data found for '{home_team}' vs '{away_team}'. Using manual inputs.")
            return None # Indicate no data was found for these teams

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Failed: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing API data: {e}")
        return None
    # --- END OF CUSTOMIZABLE SECTION ---


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.title("⚽ Football Match Predictor")
st.markdown("Predict match outcome, total goals, and probabilities using Gradient Boosting + XGBoost models.")

col_team1, col_team2 = st.columns(2)
with col_team1:
    home_team = st.text_input("Home Team", "Team A")
with col_team2:
    away_team = st.text_input("Away Team", "Team B")

if outcome_model is None or goals_model is None:
    st.error("❌ Models could not be loaded.")
    st.stop()

st.sidebar.subheader("Match Statistics")

# Initialize feature_input_values with default values. These will be updated if API data is fetched.
input_config = {
    'HomeElo': {'label': "🏠 Home Elo Rating", 'value': 1500.0, 'step': 1.0},
    'AwayElo': {'label': "🏟️ Away Elo Rating", 'value': 1500.0, 'step': 1.0},
    'Form3Home': {'label': "📈 Home Form (last 3)", 'value': 0, 'min': 0, 'max': 9, 'step': 1},
    'Form5Home': {'label': "📈 Home Form (last 5)", 'value': 0, 'min': 0, 'max': 15, 'step': 1},
    'Form3Away': {'label': "📉 Away Form (last 3)", 'value': 0, 'min': 0, 'max': 9, 'step': 1},
    'Form5Away': {'label': "📉 Away Form (last 5)", 'value': 0, 'min': 0, 'max': 15, 'step': 1},
    'HomeShots': {'label': "🔫 Home Shots (expected)", 'value': 12.0, 'min': 0.0, 'max': 30.0, 'step': 1.0},
    'AwayShots': {'label': "🔫 Away Shots (expected)", 'value': 10.0, 'min': 0.0, 'max': 30.0, 'step': 1.0},
    'HomeTarget': {'label': "🎯 Home Shots on Target", 'value': 4.0, 'min': 0.0, 'max': 15.0, 'step': 1.0},
    'AwayTarget': {'label': "🎯 Away Shots on Target", 'value': 3.0, 'min': 0.0, 'max': 15.0, 'step': 1.0},
    'HomeExpectedGoals': {'label': "🏆 Home Expected Goals", 'value': 1.5, 'min': 0.0, 'max': 5.0, 'step': 0.1},
    'AwayExpectedGoals': {'label': "🏆 Away Expected Goals", 'value': 1.2, 'min': 0.0, 'max': 5.0, 'step': 0.1},
    'OddHome': {'label': "💰 Home Win Odds", 'value': 2.0, 'min': 1.0, 'step': 0.1},
    'OddDraw': {'label': "💰 Draw Odds", 'value': 3.0, 'min': 1.0, 'step': 0.1},
    'OddAway': {'label': "💰 Away Win Odds", 'value': 3.0, 'min': 1.0, 'step': 0.1},
    'HandiSize': {'label': "📏 Handicap Size", 'value': 0.0, 'min': -2.0, 'max': 2.0, 'step': 0.25},
    'Over25': {'label': "📈 Over 2.5 Goals Odds", 'value': 2.0, 'min': 1.0, 'step': 0.1},
    'Under25': {'label': "📉 Under 2.5 Goals Odds", 'value': 2.0, 'min': 1.0, 'step': 0.1},
}

# Use st.session_state to persist feature_input_values across reruns
if 'feature_input_values' not in st.session_state:
    st.session_state.feature_input_values = {col: cfg['value'] for col, cfg in input_config.items()}

# Dynamic default values for the sidebar inputs
feature_input_values_ui = {}
for col, cfg in input_config.items():
    # Use the value from session_state as the default for the number_input
    st.session_state.feature_input_values[col] = st.sidebar.number_input(
        label=cfg['label'],
        value=st.session_state.feature_input_values.get(col, cfg['value']), # Use session state value first
        step=cfg.get('step', 0.1),
        min_value=cfg.get('min', None),
        max_value=cfg.get('max', None),
        key=f"sidebar_input_{col}" # Add a unique key for each widget
    )

# When the predict button is clicked, we'll try to fetch API data FIRST
if st.sidebar.button("🚀 Predict Match", type="primary", width="stretch"):
    with st.spinner("Fetching live data and making predictions..."):
        fetched_stats = fetch_team_stats_from_api_football(home_team, away_team, API_FOOTBALL_KEY)

        if fetched_stats: # If API data was successfully fetched
            # Update the session state with fetched values
            for stat_key, stat_value in fetched_stats.items():
                if stat_key in st.session_state.feature_input_values:
                    st.session_state.feature_input_values[stat_key] = stat_value
            st.success("API data integrated! Generating prediction.")
        else:
            st.info("No API data found or error occurred. Using current sidebar inputs for prediction.")

    # Always use the values from st.session_state for processing
    input_data = {'HomeTeam': home_team, 'AwayTeam': away_team}
    input_data.update(st.session_state.feature_input_values)
    input_df_raw = pd.DataFrame([input_data])

    processed_features = preprocess_input_data(input_df_raw, feature_columns)

    if processed_features is not None:
        outcome_encoded = outcome_model.predict(processed_features)[0]
        goals_pred = goals_model.predict(processed_features)[0]

        outcome_probs = None
        if hasattr(outcome_model, 'predict_proba'):
            outcome_probs = outcome_model.predict_proba(processed_features)[0]

        classes = getattr(label_encoder, 'classes_', outcome_model.classes_)
        try:
            predicted_outcome = label_encoder.inverse_transform([outcome_encoded])[0]
        except:
            predicted_outcome = classes[outcome_encoded]

        outcome_display_map = {
            'H': f"🏆 {home_team} Win",
            'D': "🤝 Draw",
            'A': f"🏆 {away_team} Win"
        }
        display_outcome = outcome_display_map.get(predicted_outcome, predicted_outcome)

        st.subheader(f"📊 Prediction: **{home_team}** vs **{away_team}**")

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.metric("**Predicted Outcome**", display_outcome)
            st.metric("**Total Goals**", f"{round(goals_pred)}")
            over_prob = "Likely **Over 2.5**" if goals_pred > 2.5 else "Likely **Under 2.5**"
            st.metric("**Over/Under 2.5**", over_prob)

        with col2:
            if outcome_probs is not None:
                probs_dict = {}
                for i, cls in enumerate(classes):
                    if cls == 'H':
                        probs_dict[f"{home_team} Win"] = outcome_probs[i]
                    elif cls == 'D':
                        probs_dict["Draw"] = outcome_probs[i]
                    elif cls == 'A':
                        probs_dict[f"{away_team} Win"] = outcome_probs[i]
                probs_series = pd.Series(probs_dict)
                st.bar_chart(probs_series, width="stretch")
                st.success(f"**Confidence:** {max(outcome_probs):.1%}")

        st.caption("Computed Cluster Probabilities:")
        cluster_values = processed_features[['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']].iloc[0]
        st.dataframe(cluster_values.rename("Probability"), width="stretch")

        st.caption("Model used: Gradient Boosting (outcome) + XGBoost (goals) with auto-computed clusters")

else:
    st.info("👈 Fill in the statistics in the sidebar and click **Predict Match**.")

st.caption("✅ Auto cluster computation + clean config")