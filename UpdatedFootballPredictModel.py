import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import warnings
import requests
from datetime import datetime, timedelta

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
if "API_FOOTBALL_KEY" in st.secrets:
    API_FOOTBALL_KEY = st.secrets["API_FOOTBALL_KEY"]
    st.success("✅ API Key loaded from Streamlit Secrets.")
elif os.environ.get("API_FOOTBALL_KEY"):
    API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY")
    st.success("✅ API Key loaded from environment variable.")
else:
    API_FOOTBALL_KEY = None
    st.warning("⚠️ API_FOOTBALL_KEY not found. Please add it to fetch live data.")

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
        return None, None, None

@st.cache_resource(show_spinner="Loading cluster models...")
def load_cluster_models():
    if os.path.exists(CLUSTER_SCALER_FILENAME) and os.path.exists(CLUSTER_KMEANS_FILENAME):
        try:
            scaler = joblib.load(CLUSTER_SCALER_FILENAME)
            kmeans = joblib.load(CLUSTER_KMEANS_FILENAME)
            return scaler, kmeans
        except Exception as e:
            st.error(f"❌ Error loading cluster models: {e}")
            return None, None
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

# =============================================================================
# ENHANCED API FOOTBALL INTEGRATION
# =============================================================================
class FootballAPIFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
    
    def find_team_id(self, team_name):
        """Find team ID by name"""
        params = {"search": team_name}
        response = requests.get(f"{self.base_url}/teams", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data['response']:
                # Try exact match first
                for team_info in data['response']:
                    if team_info['team']['name'].lower() == team_name.lower():
                        return team_info['team']['id']
                # Return first match
                return data['response'][0]['team']['id']
        return None
    
    def get_team_form(self, team_id, league_id, season, num_matches=5):
        """Calculate team form from last N matches"""
        params = {
            "team": team_id,
            "league": league_id,
            "season": season,
            "status": "FT"  # Finished matches only
        }
        
        response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data['response']:
                # Sort by date descending and take last N matches
                matches = sorted(data['response'], 
                               key=lambda x: x['fixture']['date'], 
                               reverse=True)[:num_matches]
                
                points = 0
                for match in matches:
                    home_id = match['teams']['home']['id']
                    away_id = match['teams']['away']['id']
                    home_goals = match['goals']['home'] or 0
                    away_goals = match['goals']['away'] or 0
                    
                    if team_id == home_id:
                        if home_goals > away_goals:
                            points += 3
                        elif home_goals == away_goals:
                            points += 1
                    else:
                        if away_goals > home_goals:
                            points += 3
                        elif away_goals == home_goals:
                            points += 1
                
                return points
        return 0
    
    def get_team_average_stats(self, team_id, league_id, season, stat_type):
        """Get average statistics for a team in a season"""
        params = {
            "team": team_id,
            "league": league_id,
            "season": season,
            "status": "FT"
        }
        
        response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data['response']:
                matches = data['response']
                total_value = 0
                match_count = 0
                
                for match in matches:
                    # Get statistics for this match
                    stats_response = requests.get(
                        f"{self.base_url}/fixtures/statistics", 
                        headers=self.headers, 
                        params={"fixture": match['fixture']['id']}
                    )
                    
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()
                        if stats_data and stats_data['response']:
                            for team_stats in stats_data['response']:
                                if team_stats['team']['id'] == team_id:
                                    for stat in team_stats['statistics']:
                                        if stat['type'] == stat_type and stat['value'] is not None:
                                            # Handle percentage strings
                                            if isinstance(stat['value'], str) and '%' in stat['value']:
                                                value = float(stat['value'].replace('%', ''))
                                            else:
                                                try:
                                                    value = float(stat['value'])
                                                except (ValueError, TypeError):
                                                    value = 0
                                            total_value += value
                                            match_count += 1
                                            break
                
                return total_value / match_count if match_count > 0 else 0
        return 0
    
    def get_expected_goals(self, team_id, league_id, season):
        """Get expected goals (xG) for a team"""
        # Try to get from team statistics endpoint
        params = {
            "team": team_id,
            "league": league_id,
            "season": season
        }
        
        response = requests.get(f"{self.base_url}/teams/statistics", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data.get('response'):
                # Look for xG in statistics
                for stat in data['response'].get('statistics', []):
                    if stat.get('type') == 'Expected Goals':
                        value = stat.get('value')
                        if value is not None:
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                pass
        
        # Fallback: calculate from recent matches
        params = {
            "team": team_id,
            "league": league_id,
            "season": season,
            "status": "FT"
        }
        
        response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data['response']:
                matches = data['response'][:10]  # Last 10 matches
                total_goals = 0
                match_count = 0
                
                for match in matches:
                    if team_id == match['teams']['home']['id']:
                        goals = match['goals']['home'] or 0
                    else:
                        goals = match['goals']['away'] or 0
                    total_goals += goals
                    match_count += 1
                
                return total_goals / match_count if match_count > 0 else 1.5
        
        return 1.5  # Default value
    
    def get_head_to_head_stats(self, team1_id, team2_id, league_id, season):
        """Get head-to-head statistics between two teams"""
        params = {
            "season": season,
            "league": league_id,
            "team": team1_id
        }
        
        response = requests.get(f"{self.base_url}/fixtures", headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and data['response']:
                h2h_matches = []
                for fixture in data['response']:
                    home_id = fixture['teams']['home']['id']
                    away_id = fixture['teams']['away']['id']
                    
                    if (home_id == team1_id and away_id == team2_id) or \
                       (home_id == team2_id and away_id == team1_id):
                        
                        # Get statistics for this match
                        stats_response = requests.get(
                            f"{self.base_url}/fixtures/statistics",
                            headers=self.headers,
                            params={"fixture": fixture['fixture']['id']}
                        )
                        
                        if stats_response.status_code == 200:
                            stats_data = stats_response.json()
                            h2h_matches.append(stats_data)
                
                if h2h_matches:
                    # Calculate average stats from H2H matches
                    avg_home_shots = 0
                    avg_away_shots = 0
                    avg_home_target = 0
                    avg_away_target = 0
                    
                    for match_stats in h2h_matches:
                        for team_stats in match_stats.get('response', []):
                            if team_stats['team']['id'] == team1_id:
                                for stat in team_stats['statistics']:
                                    if stat['type'] == 'Total Shots' and stat['value']:
                                        avg_home_shots += float(stat['value']) if stat['value'] else 0
                                    elif stat['type'] == 'Shots on Goal' and stat['value']:
                                        avg_home_target += float(stat['value']) if stat['value'] else 0
                            elif team_stats['team']['id'] == team2_id:
                                for stat in team_stats['statistics']:
                                    if stat['type'] == 'Total Shots' and stat['value']:
                                        avg_away_shots += float(stat['value']) if stat['value'] else 0
                                    elif stat['type'] == 'Shots on Goal' and stat['value']:
                                        avg_away_target += float(stat['value']) if stat['value'] else 0
                    
                    num_matches = len(h2h_matches)
                    return {
                        'HomeShots': avg_home_shots / num_matches if num_matches > 0 else 12,
                        'AwayShots': avg_away_shots / num_matches if num_matches > 0 else 10,
                        'HomeTarget': avg_home_target / num_matches if num_matches > 0 else 4,
                        'AwayTarget': avg_away_target / num_matches if num_matches > 0 else 3
                    }
        
        return None

@st.cache_data(ttl=3600, show_spinner="Fetching live team statistics...")
def fetch_complete_team_stats(home_team, away_team, api_key):
    """Fetch all required statistics from API-Football"""
    
    if not api_key:
        st.error("❌ API key not provided. Cannot fetch live data.")
        return None
    
    fetcher = FootballAPIFetcher(api_key)
    
    # Configuration
    league_id = 39  # Premier League (you can make this dynamic)
    season = 2024   # Current season
    
    try:
        # Find team IDs
        home_id = fetcher.find_team_id(home_team)
        away_id = fetcher.find_team_id(away_team)
        
        if not home_id or not away_id:
            st.warning(f"Could not find team IDs for {home_team} or {away_team}")
            return None
        
        # Fetch form (last 3 and last 5 matches)
        form3_home = fetcher.get_team_form(home_id, league_id, season, 3)
        form5_home = fetcher.get_team_form(home_id, league_id, season, 5)
        form3_away = fetcher.get_team_form(away_id, league_id, season, 3)
        form5_away = fetcher.get_team_form(away_id, league_id, season, 5)
        
        # Fetch average shots and shots on target
        avg_home_shots = fetcher.get_team_average_stats(home_id, league_id, season, "Total Shots")
        avg_away_shots = fetcher.get_team_average_stats(away_id, league_id, season, "Total Shots")
        avg_home_target = fetcher.get_team_average_stats(home_id, league_id, season, "Shots on Goal")
        avg_away_target = fetcher.get_team_average_stats(away_id, league_id, season, "Shots on Goal")
        
        # If averages are zero, try H2H stats
        if avg_home_shots == 0 or avg_away_shots == 0:
            h2h_stats = fetcher.get_head_to_head_stats(home_id, away_id, league_id, season)
            if h2h_stats:
                avg_home_shots = h2h_stats.get('HomeShots', avg_home_shots)
                avg_away_shots = h2h_stats.get('AwayShots', avg_away_shots)
                avg_home_target = h2h_stats.get('HomeTarget', avg_home_target)
                avg_away_target = h2h_stats.get('AwayTarget', avg_away_target)
        
        # Fetch expected goals
        home_xg = fetcher.get_expected_goals(home_id, league_id, season)
        away_xg = fetcher.get_expected_goals(away_id, league_id, season)
        
        # Elo ratings (you might need a separate API or database for these)
        # For now, using approximate values based on team strength
        elo_ratings = {
            'Manchester City': 2050, 'Liverpool': 2030, 'Arsenal': 2010,
            'Chelsea': 1980, 'Manchester United': 1970, 'Tottenham': 1950,
            'Newcastle': 1930, 'Aston Villa': 1900, 'Brighton': 1880,
            'West Ham': 1860, 'Brentford': 1840, 'Crystal Palace': 1820,
            'Wolves': 1800, 'Fulham': 1780, 'Everton': 1760,
            'Nottingham Forest': 1740, 'Bournemouth': 1720, 'Sheffield United': 1700,
            'Luton': 1680, 'Burnley': 1660
        }
        
        home_elo = elo_ratings.get(home_team, 1800)
        away_elo = elo_ratings.get(away_team, 1800)
        
        # Compile all statistics
        stats = {
            'HomeElo': home_elo,
            'AwayElo': away_elo,
            'Form3Home': form3_home,
            'Form5Home': form5_home,
            'Form3Away': form3_away,
            'Form5Away': form5_away,
            'HomeShots': max(avg_home_shots, 8.0),  # Minimum reasonable value
            'AwayShots': max(avg_away_shots, 6.0),
            'HomeTarget': max(avg_home_target, 3.0),
            'AwayTarget': max(avg_away_target, 2.0),
            'HomeExpectedGoals': home_xg,
            'AwayExpectedGoals': away_xg,
            # Default values for other required fields
            'OddHome': 2.0,
            'OddDraw': 3.2,
            'OddAway': 3.5,
            'HandiSize': 0.0,
            'Over25': 1.9,
            'Under25': 1.9
        }
        
        st.success("✅ Successfully fetched live statistics from API-Football!")
        
        # Display fetched stats
        with st.expander("📊 Fetched Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{home_team}**")
                st.metric("Form (last 3)", f"{form3_home}/9 points")
                st.metric("Form (last 5)", f"{form5_home}/15 points")
                st.metric("Avg Shots", f"{avg_home_shots:.1f}")
                st.metric("Avg Shots on Target", f"{avg_home_target:.1f}")
                st.metric("Expected Goals (xG)", f"{home_xg:.2f}")
                st.metric("Elo Rating", f"{home_elo}")
            
            with col2:
                st.markdown(f"**{away_team}**")
                st.metric("Form (last 3)", f"{form3_away}/9 points")
                st.metric("Form (last 5)", f"{form5_away}/15 points")
                st.metric("Avg Shots", f"{avg_away_shots:.1f}")
                st.metric("Avg Shots on Target", f"{avg_away_target:.1f}")
                st.metric("Expected Goals (xG)", f"{away_xg:.2f}")
                st.metric("Elo Rating", f"{away_elo}")
        
        return stats
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# =============================================================================
# HELPER FUNCTIONS (from original code)
# =============================================================================
def clean_input_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median() if len(df) > 0 else 0.0)
    return df

def compute_cluster_features(row):
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
        st.warning(f"Cluster computation failed: {e}")
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
    cleaned_df = clean_input_data(input_df)
    featured_df = create_input_features(cleaned_df)
    
    # Compute clusters automatically
    cluster_dict = get_cluster_probabilities(featured_df.iloc[0])
    for col, val in cluster_dict.items():
        featured_df[col] = val
    
    missing = [col for col in feature_columns if col not in featured_df.columns]
    if missing:
        for col in missing:
            featured_df[col] = 0.0
    
    processed_features = featured_df[feature_columns].fillna(0)
    return processed_features

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Football Match Predictor", layout="wide")
st.title("⚽ Football Match Predictor")
st.markdown("Predict match outcomes using real-time statistics from API-Football")

# Team input
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("🏠 Home Team", "Arsenal")
with col2:
    away_team = st.text_input("🚗 Away Team", "Liverpool")

# Fetch button
if st.button("🔍 Fetch Live Statistics & Predict", type="primary", use_container_width=True):
    if not API_FOOTBALL_KEY:
        st.error("❌ API key not configured. Please add API_FOOTBALL_KEY to your secrets.")
        st.stop()
    
    with st.spinner("Fetching live statistics from API-Football..."):
        # Fetch statistics automatically
        fetched_stats = fetch_complete_team_stats(home_team, away_team, API_FOOTBALL_KEY)
        
        if fetched_stats:
            # Create input dataframe
            input_data = {
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                **fetched_stats
            }
            input_df_raw = pd.DataFrame([input_data])
            
            # Preprocess and predict
            processed_features = preprocess_input_data(input_df_raw, feature_columns)
            
            if processed_features is not None and outcome_model and goals_model:
                # Make predictions
                outcome_encoded = outcome_model.predict(processed_features)[0]
                goals_pred = goals_model.predict(processed_features)[0]
                
                # Get probabilities
                outcome_probs = None
                if hasattr(outcome_model, 'predict_proba'):
                    outcome_probs = outcome_model.predict_proba(processed_features)[0]
                
                # Decode outcome
                classes = getattr(label_encoder, 'classes_', outcome_model.classes_)
                try:
                    predicted_outcome = label_encoder.inverse_transform([outcome_encoded])[0]
                except:
                    predicted_outcome = classes[outcome_encoded]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Results layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🏆 Predicted Winner", 
                             f"{home_team}" if predicted_outcome == 'H' else 
                             f"{away_team}" if predicted_outcome == 'A' else "Draw")
                
                with col2:
                    st.metric("⚽ Predicted Total Goals", f"{round(goals_pred, 1)}")
                
                with col3:
                    over_under = "Over 2.5" if goals_pred > 2.5 else "Under 2.5"
                    st.metric("📈 Over/Under", over_under)
                
                # Probability chart
                if outcome_probs is not None:
                    st.subheader("📊 Outcome Probabilities")
                    probs_dict = {}
                    for i, cls in enumerate(classes):
                        if cls == 'H':
                            probs_dict[f"{home_team} Win"] = outcome_probs[i]
                        elif cls == 'D':
                            probs_dict["Draw"] = outcome_probs[i]
                        elif cls == 'A':
                            probs_dict[f"{away_team} Win"] = outcome_probs[i]
                    
                    probs_series = pd.Series(probs_dict)
                    st.bar_chart(probs_series)
                    st.caption(f"**Confidence:** {max(outcome_probs):.1%}")
                
                # Feature importance
                with st.expander("🔧 Advanced Features"):
                    st.json({k: float(v) for k, v in fetched_stats.items() if isinstance(v, (int, float))})
        else:
            st.error("❌ Failed to fetch statistics. Please check team names and try again.")
else:
    st.info("👈 Enter team names and click 'Fetch Live Statistics & Predict'")

# Footer
st.markdown("---")
st.caption("Data source: API-Football v3 | Model: Gradient Boosting + XGBoost")
