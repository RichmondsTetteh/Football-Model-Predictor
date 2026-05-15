import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# =============================================================================
# CONSTANTS - Adjusted for Colab environment
# =============================================================================
# Model filenames 
OUTCOME_MODEL_FILENAME = 'outcome_model.joblib'
GOALS_HOME_MODEL_FILENAME = 'goals_home_model.joblib' 
GOALS_AWAY_MODEL_FILENAME = 'goals_away_model.joblib' 
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'
FEATURE_COLS_FILENAME = 'feature_cols.joblib'

# DEFAULTS for input features, matching Streamlit app's expectations
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

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS (Copied from Colab notebook cells)
# =============================================================================
def cap(series, lo=None, hi=None):
    """Clip a series with optional lower/upper bounds."""
    if lo is not None:
        series = series.clip(lower=lo)
    if hi is not None:
        series = series.clip(upper=hi)
    return series

def expanding_team_mean(df, team_col, value_col, default):
    """
    Computes the historical average per team in time order,
    shifted by 1 so only PAST matches inform the current row.
    Returns a Series aligned to df.index.
    """
    result = pd.Series(index=df.index, dtype=float)
    for team, grp in df.groupby(team_col, sort=False):
        vals = grp[value_col].copy()
        hist = vals.expanding().mean().shift(1)
        hist.iloc[0] = default
        result.loc[grp.index] = hist.values
    return result.fillna(default)

def build_features(df):
    """
    Engineer all PRE-MATCH features.
    Returns the feature DataFrame and the list of column names.
    (Copied from notebook cell `Xzg4gIEfzRbD`)
    """
    df = df.copy()

    # -- Elo features --
    df["HomeElo"]  = cap(df["HomeElo"],  lo=1000, hi=3000)
    df["AwayElo"]  = cap(df["AwayElo"],  lo=1000, hi=3000)
    df["EloDiff"]  = df["HomeElo"] - df["AwayElo"]
    df["EloRatio"] = df["HomeElo"] / df["AwayElo"].replace(0, np.nan).fillna(1500)

    # -- Form features --
    for col in ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]:
        df[col] = cap(df[col], lo=0, hi=15)

    df["Form3Diff"]  = df["Form3Home"] - df["Form3Away"]
    df["Form5Diff"]  = df["Form5Home"] - df["Form5Away"]
    df["Form5Ratio"] = (df["Form5Home"] + 0.5) / (df["Form5Away"] + 0.5)
    df["Form5Ratio"] = cap(df["Form5Ratio"], lo=0.1, hi=10)

    # -- Historical expected goals (corrected expanding mean) --
    # Note: For new predictions, FTHome/FTAway might not be available. 
    # The training build_features used these to calculate XG. For inference 
    # we rely on 'HomeExpectedGoals'/'AwayExpectedGoals' from raw_row inputs.
    # If you want to use the expanding mean logic, you would need to feed it
    # historical data that includes FTHome/FTAway and apply it iteratively.
    # For this adaptation, we assume 'HomeExpectedGoals'/'AwayExpectedGoals' are given.
    if 'FTHome' in df.columns: # Only calculate if actual goals are present (training data)
        df["HomeXG"] = expanding_team_mean(df, "HomeTeam", "FTHome", default=1.5)
    else:
        df["HomeXG"] = df['HomeExpectedGoals'] # Use provided XG for prediction
        
    if 'FTAway' in df.columns:
        df["AwayXG"] = expanding_team_mean(df, "AwayTeam", "FTAway", default=1.2)
    else:
        df["AwayXG"] = df['AwayExpectedGoals'] # Use provided XG for prediction

    df["XGDiff"] = df["HomeXG"] - df["AwayXG"]

    # -- Attack / defence strength proxies --
    df["HomeAttack"]  = df["HomeXG"] * df["Form5Home"] / 7.5
    df["AwayAttack"]  = df["AwayXG"] * df["Form5Away"] / 7.5
    df["HomeAttack"]  = cap(df["HomeAttack"],  lo=0, hi=5)
    df["AwayAttack"]  = cap(df["AwayAttack"],  lo=0, hi=5)

    # -- Pre-match betting odds → implied probabilities (opening odds only) --
    for col in ["OddHome", "OddDraw", "OddAway"]:
        df[col] = np.where((df[col] > 1) & (df[col] < 1000), df[col], np.nan)

    df["ProbHome"]  = (1 / df["OddHome"]).fillna(0.45)
    df["ProbDraw"]  = (1 / df["OddDraw"]).fillna(0.28)
    df["ProbAway"]  = (1 / df["OddAway"]).fillna(0.27)
    row_sum = df[["ProbHome", "ProbDraw", "ProbAway"]].sum(axis=1).replace(0, 1)
    df["ProbHome"] /= row_sum
    df["ProbDraw"] /= row_sum
    df["ProbAway"] /= row_sum
    df["OddsEdge"]  = df["ProbHome"] - df["ProbAway"]

    # -- Handicap (pre-match market signal) --
    df["HandiSize"] = df["HandiSize"].fillna(0)

    # -- Cluster features (pre-match market cluster signals) --
    # These columns must be present in the input dataframe if they are to be used.
    # If not present, they will be effectively skipped or filled with 0 later.
    cluster_cols = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]
    for col in cluster_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median()).fillna(0)
        else:
            df[col] = 0.0 # Ensure cluster columns are present even if not in raw input

    # -- Temporal features (seasonality / fatigue proxy) --
    # Requires 'MatchDate' column to be datetime objects
    if 'MatchDate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['MatchDate']):
        df["MatchMonth"]    = df["MatchDate"].dt.month
        df["MatchDayOfWeek"]= df["MatchDate"].dt.dayofweek
    else:
        df["MatchMonth"]    = 0 # Default if no MatchDate
        df["MatchDayOfWeek"]= 0 # Default if no MatchDate

    # The actual FEATURE_COLS will be loaded from the saved joblib file
    # and used to select the final feature set for the model.
    feature_cols = [
        "HomeElo", "AwayElo", "EloDiff", "EloRatio",
        "Form3Diff", "Form5Diff", "Form5Ratio",
        "Form3Home", "Form5Home", "Form3Away", "Form5Away",
        "HomeXG", "AwayXG", "XGDiff",
        "HomeAttack", "AwayAttack",
        "ProbHome", "ProbDraw", "ProbAway", "OddsEdge",
        "HandiSize",
        "MatchMonth", "MatchDayOfWeek",
    ] + cluster_cols # Add cluster cols even if they were defaulted to 0.0

    # Replace any remaining inf / nan
    df[feature_cols] = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df, feature_cols # Return all generated features and their names

# =============================================================================
# MODEL LOADING - Adapted for Colab saved artifacts
# =============================================================================
def load_colab_models():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_home_model = joblib.load(GOALS_HOME_MODEL_FILENAME)
        goals_away_model = joblib.load(GOALS_AWAY_MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        feature_cols = joblib.load(FEATURE_COLS_FILENAME)
        return outcome_model, goals_home_model, goals_away_model, label_encoder, feature_cols
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None, None, None, None, None

outcome_model, goals_home_model, goals_away_model, label_encoder, FEATURE_COLS = load_colab_models()

if outcome_model is None or goals_home_model is None or goals_away_model is None:
    print("Prediction models could not be loaded. Check file paths and ensure they exist.")

# =============================================================================
# PREPROCESSING FOR INFERENCE
# =============================================================================
def preprocess_input_for_prediction(raw_row: dict) -> pd.DataFrame:
    """
    Preprocesses a single raw input row (dictionary) for prediction.
    Uses the build_features function from the notebook's training pipeline.
    """
    # Ensure all BASE_COLUMNS and expected goals are present with defaults
    input_data = raw_row.copy()
    for col in list(DEFAULTS.keys()): # Iterate over keys to set defaults
        input_data.setdefault(col, DEFAULTS[col])
    
    # Create a DataFrame from the raw input and apply feature engineering
    # Note: 'MatchDate' might be missing in a single raw_row for new predictions.
    # Handle this by adding a placeholder if necessary or expect it in raw_row.
    df_raw = pd.DataFrame([input_data])
    
    # Build features using the notebook's logic
    df_processed, _ = build_features(df_raw)

    # Ensure all required FEATURE_COLS are present, filling with 0 if missing
    for col in FEATURE_COLS:
        if col not in df_processed.columns:
            df_processed[col] = 0.0
            
    return df_processed[FEATURE_COLS].fillna(0)

# =============================================================================
# PREDICTION FUNCTION - Adapted for Colab
# =============================================================================
def predict_matches_colab(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw matches DataFrame, return it with added prediction columns:
      PredictedOutcome      — H / D / A
      ProbH / ProbD / ProbA — calibrated probabilities
      PredictedHomeGoals    — expected home goals
      PredictedAwayGoals    — expected away goals
      PredictedTotalGoals   — sum
    
    Assumes models and feature_cols are loaded globally.
    """
    if outcome_model is None or goals_home_model is None or goals_away_model is None or label_encoder is None or FEATURE_COLS is None:
        print("Models or feature columns not loaded. Cannot make predictions.")
        return new_df

    new_df = new_df.copy()
    
    # Convert MatchDate if present
    if 'MatchDate' in new_df.columns:
        new_df["MatchDate"] = pd.to_datetime(new_df["MatchDate"], errors="coerce")
        new_df = new_df.sort_values("MatchDate").reset_index(drop=True)

    # Preprocess all rows
    # The build_features function expects a dataframe with 'MatchDate' for temporal features.
    # For new predictions, if 'MatchDate' is not in `new_df`, `build_features` will default 'MatchMonth' and 'MatchDayOfWeek' to 0.
    processed_features, _ = build_features(new_df)
    
    # Ensure the feature matrix for prediction has the exact columns used during training
    X_new = processed_features[FEATURE_COLS].fillna(0)

    # Outcome prediction
    proba = outcome_model.predict_proba(X_new)
    enc_pred = outcome_model.predict(X_new)
    predicted_outcome_labels = label_encoder.inverse_transform(enc_pred)

    new_df["PredictedOutcome"] = predicted_outcome_labels
    for i, cls in enumerate(label_encoder.classes_):
        new_df[f"Prob{cls}"] = proba[:, i]

    # Goals prediction
    new_df["PredictedHomeGoals"] = goals_home_model.predict(X_new).clip(0)
    new_df["PredictedAwayGoals"] = goals_away_model.predict(X_new).clip(0)
    new_df["PredictedTotalGoals"] = new_df["PredictedHomeGoals"] + new_df["PredictedAwayGoals"]

    return new_df

print("Adapted prediction functions loaded. Use `predict_matches_colab(your_dataframe)` to make predictions.")

# --- Demo of the adapted function ---
print("\n--- Demoing predict_matches_colab with a sample from test data ---")
# Use 'processed' and 'split_idx' from previous notebook cells
# Make sure to include all necessary columns expected by build_features, even if empty/0
sample_data_for_demo = data.iloc[split_idx:split_idx + 5].copy()

# Ensure 'HomeExpectedGoals' and 'AwayExpectedGoals' are present in the sample 
# if 'FTHome'/'FTAway' are not there, so build_features can use them.
# For this demo, since 'data' has FTHome/FTAway, build_features will calculate XG.
# For real new data, you would provide these or other relevant pre-match stats.

preds_colab = predict_matches_colab(sample_data_for_demo)

display_cols_demo = [
    "MatchDate", "HomeTeam", "AwayTeam", "FTResult",
    "PredictedOutcome", "ProbA", "ProbD", "ProbH",
    "PredictedHomeGoals", "PredictedAwayGoals", "PredictedTotalGoals",
]

print("\nSample predictions vs actuals (using Colab adapted function):")
print(preds_colab[display_cols_demo].to_string(index=False))
