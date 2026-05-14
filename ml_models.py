"""
ml_models.py
============
Three ML models for the IPL Analytics Dashboard:

1. Win Probability Predictor  — LogisticRegression on match-level features
2. Match Score Predictor      — RandomForestRegressor on innings-level features
3. Best XI Selector           — scoring model (batting + bowling index) per season/team
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
#  1.  WIN PROBABILITY PREDICTOR
# ══════════════════════════════════════════════════════════

def build_win_probability_model(matches: pd.DataFrame, deliveries: pd.DataFrame):
    """
    Train a Logistic Regression model to predict win probability.
    Features: powerplay runs, wickets lost in PP, required run rate,
              current run rate, venue (encoded), batting team (encoded).
    Returns: trained model, scaler, label encoders, feature names, metrics dict
    """

    # ── Build innings-level features ──────────────────────
    records = []

    for match_id in deliveries['match_id'].unique():
        m_del = deliveries[deliveries['match_id'] == match_id]
        m_info = matches[matches['match_id'] == match_id]
        if m_info.empty:
            continue
        m_info = m_info.iloc[0]
        winner = m_info.get('winner', None)
        if pd.isna(winner) or winner == '':
            continue

        for innings in [1, 2]:
            inn_del = m_del[m_del['inning'] == innings] if 'inning' in m_del.columns else pd.DataFrame()
            if inn_del.empty:
                continue

            batting_team = inn_del['batting_team'].iloc[0] if 'batting_team' in inn_del.columns else None
            if batting_team is None:
                continue

            # Powerplay (overs 1-6)
            pp = inn_del[inn_del['over'] <= 6]
            pp_runs = pp['total_runs'].sum() if not pp.empty else 0
            pp_wickets = pp['is_wicket'].sum() if 'is_wicket' in pp.columns and not pp.empty else 0

            # Full innings
            total_runs = inn_del['total_runs'].sum()
            total_wickets = inn_del['is_wicket'].sum() if 'is_wicket' in inn_del.columns else 0
            total_balls = inn_del['is_legal'].sum() if 'is_legal' in inn_del.columns else len(inn_del)
            total_overs = total_balls / 6 if total_balls > 0 else 1

            crr = total_runs / total_overs if total_overs > 0 else 0
            season = m_info.get('season', 0)
            venue = m_info.get('venue', 'Unknown')

            won = 1 if winner == batting_team else 0

            records.append({
                'match_id':    match_id,
                'innings':     innings,
                'batting_team': batting_team,
                'venue':       venue,
                'season':      season,
                'pp_runs':     pp_runs,
                'pp_wickets':  pp_wickets,
                'total_runs':  total_runs,
                'total_wickets': total_wickets,
                'crr':         round(crr, 2),
                'won':         won,
            })

    df = pd.DataFrame(records)
    if df.empty or len(df) < 50:
        return None, None, {}, [], {'accuracy': 0, 'train_size': 0}

    # ── Encode categoricals ────────────────────────────────
    le_team  = LabelEncoder()
    le_venue = LabelEncoder()
    df['team_enc']  = le_team.fit_transform(df['batting_team'].astype(str))
    df['venue_enc'] = le_venue.fit_transform(df['venue'].astype(str))

    features = ['pp_runs', 'pp_wickets', 'crr', 'total_wickets', 'team_enc', 'venue_enc']
    X = df[features].fillna(0)
    y = df['won']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    metrics = {
        'accuracy':   round(acc * 100, 1),
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }

    encoders = {'team': le_team, 'venue': le_venue}
    return model, scaler, encoders, features, metrics


def predict_win_probability(
    model, scaler, encoders, features,
    batting_team, venue,
    pp_runs, pp_wickets, crr, total_wickets
):
    """Return win probability (0–100) for batting team."""
    if model is None:
        return 50.0

    try:
        team_enc  = encoders['team'].transform([batting_team])[0]
    except ValueError:
        team_enc  = 0
    try:
        venue_enc = encoders['venue'].transform([venue])[0]
    except ValueError:
        venue_enc = 0

    row = pd.DataFrame([{
        'pp_runs':        pp_runs,
        'pp_wickets':     pp_wickets,
        'crr':            crr,
        'total_wickets':  total_wickets,
        'team_enc':       team_enc,
        'venue_enc':      venue_enc,
    }])[features]

    row_s = scaler.transform(row)
    prob  = model.predict_proba(row_s)[0][1]
    return round(prob * 100, 1)


# ══════════════════════════════════════════════════════════
#  2.  MATCH SCORE PREDICTOR
# ══════════════════════════════════════════════════════════

def build_score_predictor(matches: pd.DataFrame, deliveries: pd.DataFrame):
    """
    Predict final innings score given first 10 overs of data.
    Features: runs at 10 overs, wickets at 10 overs, venue, batting team.
    Model: RandomForestRegressor
    Returns: model, scaler, encoders, features, metrics
    """
    records = []

    for match_id in deliveries['match_id'].unique():
        m_del  = deliveries[deliveries['match_id'] == match_id]
        m_info = matches[matches['match_id'] == match_id]
        if m_info.empty:
            continue
        m_info = m_info.iloc[0]
        venue  = m_info.get('venue', 'Unknown')

        for innings in [1, 2]:
            inn = m_del[m_del['inning'] == innings] if 'inning' in m_del.columns else pd.DataFrame()
            if inn.empty:
                continue

            batting_team = inn['batting_team'].iloc[0] if 'batting_team' in inn.columns else 'Unknown'

            # First 10 overs
            first10 = inn[inn['over'] <= 10]
            runs_10  = first10['total_runs'].sum()
            wkts_10  = first10['is_wicket'].sum() if 'is_wicket' in first10.columns else 0

            # Full innings total
            final_score = inn['total_runs'].sum()
            if final_score < 30:
                continue  # incomplete innings

            records.append({
                'batting_team': batting_team,
                'venue':        venue,
                'runs_10':      runs_10,
                'wkts_10':      wkts_10,
                'final_score':  final_score,
            })

    df = pd.DataFrame(records)
    if df.empty or len(df) < 50:
        return None, None, {}, [], {'mae': 0, 'train_size': 0}

    le_team  = LabelEncoder()
    le_venue = LabelEncoder()
    df['team_enc']  = le_team.fit_transform(df['batting_team'].astype(str))
    df['venue_enc'] = le_venue.fit_transform(df['venue'].astype(str))

    features = ['runs_10', 'wkts_10', 'team_enc', 'venue_enc']
    X = df[features].fillna(0)
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    metrics = {
        'mae':        round(mae, 1),
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }

    encoders = {'team': le_team, 'venue': le_venue}
    return model, scaler, encoders, features, metrics


def predict_final_score(model, scaler, encoders, features,
                        batting_team, venue, runs_10, wkts_10):
    """Predict final innings score. Returns (predicted, low, high)."""
    if model is None:
        return 160, 145, 175

    try:
        team_enc  = encoders['team'].transform([batting_team])[0]
    except ValueError:
        team_enc  = 0
    try:
        venue_enc = encoders['venue'].transform([venue])[0]
    except ValueError:
        venue_enc = 0

    row = pd.DataFrame([{
        'runs_10':  runs_10,
        'wkts_10':  wkts_10,
        'team_enc': team_enc,
        'venue_enc': venue_enc,
    }])[features]

    row_s = scaler.transform(row)

    # Predict with all trees for confidence interval
    tree_preds = np.array([tree.predict(row_s)[0] for tree in model.estimators_])
    pred  = int(np.mean(tree_preds))
    low   = int(np.percentile(tree_preds, 15))
    high  = int(np.percentile(tree_preds, 85))
    return pred, low, high


# ══════════════════════════════════════════════════════════
#  3.  BEST XI SELECTOR
# ══════════════════════════════════════════════════════════

def select_best_xi(deliveries: pd.DataFrame, matches: pd.DataFrame,
                   season=None, team=None, top_n=11):
    """
    Score every player using a composite batting + bowling index.

    Batting Index  = (runs * SR/100) + (4s * 0.5) + (6s * 1.0) - (dot_pct * 0.3)
    Bowling Index  = (wickets * 20) - (economy * 3) + (dot_pct * 0.5)
    Overall Score  = batting_index + bowling_index  (all-rounders rank high)

    Returns a DataFrame with columns:
    Player, Role, Batting Index, Bowling Index, Overall Score, Runs, SR,
    Wickets, Economy
    """
    df = deliveries.copy()

    if season is not None:
        df = df[df['season'] == season]
    if team is not None:
        df = df[(df['batting_team'] == team) | (df['bowling_team'] == team)] \
            if 'bowling_team' in df.columns \
            else df[df['batting_team'] == team]

    if df.empty:
        return pd.DataFrame()

    # ── Batting stats ──────────────────────────────────────
    bat = df[df['batter'].notna()].groupby('batter').agg(
        runs       = ('batsman_runs', 'sum'),
        balls      = ('is_legal',     'sum'),
        fours      = ('is_four',      'sum'),
        sixes      = ('is_six',       'sum'),
        dots       = ('is_dot',       'sum'),
    ).reset_index().rename(columns={'batter': 'player'})

    bat = bat[bat['balls'] >= 30]  # minimum 30 balls faced
    bat['sr']       = (bat['runs'] / bat['balls'] * 100).round(1)
    bat['dot_pct']  = (bat['dots'] / bat['balls'] * 100).round(1)
    bat['bat_idx']  = (
        bat['runs'] * (bat['sr'] / 100)
        + bat['fours'] * 0.5
        + bat['sixes'] * 1.0
        - bat['dot_pct'] * 0.3
    ).round(1)

    # ── Bowling stats ──────────────────────────────────────
    bowl = df[df['bowler'].notna()].groupby('bowler').agg(
        bowl_balls  = ('is_legal',   'sum'),
        runs_given  = ('total_runs', 'sum'),
        wickets     = ('is_wicket',  'sum'),
        bowl_dots   = ('is_dot',     'sum'),
    ).reset_index().rename(columns={'bowler': 'player'})

    bowl = bowl[bowl['bowl_balls'] >= 30]
    bowl['economy']      = (bowl['runs_given'] / (bowl['bowl_balls'] / 6)).round(2)
    bowl['bowl_dot_pct'] = (bowl['bowl_dots'] / bowl['bowl_balls'] * 100).round(1)
    bowl['bowl_idx']     = (
        bowl['wickets'] * 20
        - bowl['economy'] * 3
        + bowl['bowl_dot_pct'] * 0.5
    ).round(1)

    # ── Merge ─────────────────────────────────────────────
    merged = bat.merge(bowl, on='player', how='outer').fillna(0)
    merged['bat_idx']  = merged['bat_idx'].clip(lower=0)
    merged['bowl_idx'] = merged['bowl_idx'].clip(lower=0)
    merged['overall']  = (merged['bat_idx'] + merged['bowl_idx']).round(1)

    # ── Role classification ────────────────────────────────
    def classify(row):
        has_bat  = row['bat_idx']  > 10
        has_bowl = row['bowl_idx'] > 10
        if has_bat and has_bowl:
            return 'All-Rounder'
        elif has_bat:
            return 'Batter'
        elif has_bowl:
            return 'Bowler'
        return 'Batter'

    merged['role'] = merged.apply(classify, axis=1)

    # ── Pick Best XI — ensure role balance ────────────────
    # Target: ~5 batters, ~3 all-rounders, ~3 bowlers
    merged = merged.sort_values('overall', ascending=False)

    batters     = merged[merged['role'] == 'Batter'].head(5)
    allrounders = merged[merged['role'] == 'All-Rounder'].head(3)
    bowlers     = merged[merged['role'] == 'Bowler'].head(3)

    xi = pd.concat([batters, allrounders, bowlers]).drop_duplicates('player')

    # If we don't have 11, fill with top overall
    if len(xi) < top_n:
        remaining = merged[~merged['player'].isin(xi['player'])].head(top_n - len(xi))
        xi = pd.concat([xi, remaining])

    xi = xi.head(top_n).reset_index(drop=True)
    xi.index = xi.index + 1  # 1-based index

    return xi[[
        'player', 'role', 'bat_idx', 'bowl_idx', 'overall',
        'runs', 'sr', 'wickets', 'economy'
    ]].rename(columns={
        'player':   'Player',
        'role':     'Role',
        'bat_idx':  'Batting Index',
        'bowl_idx': 'Bowling Index',
        'overall':  'Overall Score',
        'runs':     'Runs',
        'sr':       'Strike Rate',
        'wickets':  'Wickets',
        'economy':  'Economy',
    })