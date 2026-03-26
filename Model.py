import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

# =============================================================================
# CONSTANTS
# =============================================================================
OUTCOME_MODEL_FILENAME = 'Gradient_Boosting_Classifier_outcome_model.joblib'
GOALS_MODEL_FILENAME = 'XGBoost_Regressor_goals_model.joblib'
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'

# =============================================================================
# RESOURCE LOADING (cached)
# =============================================================================
@st.cache_resource
def load_resources():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_model = joblib.load(GOALS_MODEL_FILENAME)
        
        # Try to load the exact encoder used during training
        try:
            label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        except FileNotFoundError:
            # Fallback only if file is missing (should rarely happen)
            label_encoder = LabelEncoder()
            label_encoder.fit(['A', 'D', 'H'])  # A=0, D=1, H=2 (standard)
        
        return outcome_model, goals_model, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None, None


# Load models once
outcome_model, goals_model, label_encoder = load_resources()

# =============================================================================
# FEATURE DEFINITIONS (must match training exactly)
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
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25',
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'
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


def create_input_features(df):
    df = df.copy()
    
    # Elo features
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    df['EloSum'] = df['HomeElo'] + df['AwayElo']
    
    # Form ratios (exact match to training)
    df['Form3Ratio'] = df['Form3Home'] / (df['Form3Away'] + 0.1)
    df['Form5Ratio'] = df['Form5Home'] / (df['Form5Away'] + 0.1)
    
    # Probabilities from odds
    df['HomeWinProbability'] = np.where(df['OddHome'] > 0, 1 / df['OddHome'], 0.5)
    df['DrawProbability'] = np.where(df['OddDraw'] > 0, 1 / df['OddDraw'], 0.25)
    df['AwayWinProbability'] = np.where(df['OddAway'] > 0, 1 / df['OddAway'], 0.25)
    
    # Shot efficiency
    df['HomeShotEfficiency'] = df['HomeTarget'] / (df['HomeShots'] + 0.1)
    df['AwayShotEfficiency'] = df['AwayTarget'] / (df['AwayShots'] + 0.1)
    
    # Attack strength
    df['HomeAttackStrength'] = df['HomeExpectedGoals'] * df['Form5Home'] / 15
    df['AwayAttackStrength'] = df['AwayExpectedGoals'] * df['Form5Away'] / 15
    
    return df


def preprocess_input_data(input_df, feature_columns):
    # Add any missing base columns with realistic defaults
    defaults = {
        'HomeShots': 12.0, 'AwayShots': 10.0,
        'HomeTarget': 4.0, 'AwayTarget': 3.0,
        'OddHome': 2.0, 'OddDraw': 3.0, 'OddAway': 3.0,
        'HomeExpectedGoals': 1.5, 'AwayExpectedGoals': 1.2,
        'Form3Home': 0, 'Form5Home': 0,
        'Form3Away': 0, 'Form5Away': 0,
        'HandiSize': 0.0, 'Over25': 2.0, 'Under25': 2.0,
        'C_LTH': 0.0, 'C_LTA': 0.0, 'C_VHD': 0.0,
        'C_VAD': 0.0, 'C_HTB': 0.0, 'C_PHB': 0.0,
    }
    
    for col in base_columns + ['HomeExpectedGoals', 'AwayExpectedGoals']:
        if col not in input_df.columns:
            input_df[col] = defaults.get(col, 0.0)
    
    cleaned_df = clean_input_data(input_df)
    featured_df = create_input_features(cleaned_df)
    
    # Final safety net + feature mismatch warning
    missing = [col for col in feature_columns if col not in featured_df.columns]
    if missing:
        st.warning(f"⚠️ Missing features: {missing}. Using 0 as fallback.")
        for col in missing:
            featured_df[col] = 0.0
    
    processed_features = featured_df[feature_columns].fillna(0)
    return processed_features


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.title("⚽ Football Match Predictor")
st.markdown("Predict match outcome, total goals, and probabilities using Gradient Boosting + XGBoost models.")

# Team names (top level)
col_team1, col_team2 = st.columns(2)
with col_team1:
    home_team = st.text_input("Home Team", "Team A")
with col_team2:
    away_team = st.text_input("Away Team", "Team B")

# Only proceed if models loaded successfully
if outcome_model is None or goals_model is None:
    st.error("❌ Models could not be loaded. Please check that the .joblib files are in the same folder as this script.")
    st.stop()

# =============================================================================
# SIDEBAR INPUTS (clean config-based approach)
# =============================================================================
st.sidebar.subheader("Match Statistics")

# Clean, maintainable configuration for all inputs
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
    
    'C_LTH': {'label': "🔬 Cluster: C_LTH", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'C_LTA': {'label': "🔬 Cluster: C_LTA", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'C_VHD': {'label': "🔬 Cluster: C_VHD", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'C_VAD': {'label': "🔬 Cluster: C_VAD", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'C_HTB': {'label': "🔬 Cluster: C_HTB", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'C_PHB': {'label': "🔬 Cluster: C_PHB", 'value': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01},
}

feature_input_values = {}
for col, cfg in input_config.items():
    feature_input_values[col] = st.sidebar.number_input(
        label=cfg['label'],
        value=cfg['value'],
        step=cfg.get('step', 0.1),
        min_value=cfg.get('min', None),
        max_value=cfg.get('max', None),
    )

# =============================================================================
# PREDICTION
# =============================================================================
if st.sidebar.button("🚀 Predict Match", type="primary", width="stretch"):
    # Build raw input DataFrame
    input_data = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
    }
    input_data.update(feature_input_values)
    input_df_raw = pd.DataFrame([input_data])
    
    # Preprocess
    processed_features = preprocess_input_data(input_df_raw, feature_columns)
    
    if processed_features is not None:
        # Make predictions
        outcome_encoded = outcome_model.predict(processed_features)[0]
        goals_pred = goals_model.predict(processed_features)[0]
        
        # Get probabilities if available
        outcome_probs = None
        if hasattr(outcome_model, 'predict_proba'):
            outcome_probs = outcome_model.predict_proba(processed_features)[0]
        
        # Decode outcome safely
        classes = getattr(label_encoder, 'classes_', outcome_model.classes_)
        try:
            predicted_outcome = label_encoder.inverse_transform([outcome_encoded])[0]
        except:
            predicted_outcome = classes[outcome_encoded]
        
        # Friendly display mapping
        outcome_display_map = {
            'H': f"🏆 {home_team} Win",
            'D': "🤝 Draw",
            'A': f"🏆 {away_team} Win"
        }
        display_outcome = outcome_display_map.get(predicted_outcome, predicted_outcome)
        
        # Consistency check
        if outcome_probs is not None:
            argmax_idx = np.argmax(outcome_probs)
            if argmax_idx != outcome_encoded:
                st.error("⚠️ Model inconsistency detected (predict vs proba). Please retrain models.")
        
        # Results
        st.subheader(f"📊 Prediction: **{home_team}** vs **{away_team}**")
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.metric("**Predicted Outcome**", display_outcome)
            st.metric("**Total Goals**", f"{round(goals_pred)}")
            over_prob = "Likely **Over 2.5**" if goals_pred > 2.5 else "Likely **Under 2.5**"
            st.metric("**Over/Under 2.5**", over_prob)
        
        with col2:
            if outcome_probs is not None:
                # Build readable probability series
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
                max_conf = max(outcome_probs)
                st.success(f"**Confidence:** {max_conf:.1%}")
            else:
                st.warning("Probabilities not available from this model.")
        
        # Extra info
        st.caption("Model used: Gradient Boosting (outcome) + XGBoost (goals)")
        
    else:
        st.error("Preprocessing failed. Please check your inputs.")
else:
    st.info("👈 Fill in the statistics in the sidebar and click **Predict Match** to get the result.")

st.caption("✅ Improved version: clean config, robust encoding, feature mismatch protection, better UX.")
