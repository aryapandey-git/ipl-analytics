import pandas as pd


def load_raw_data(data_dir="data"):
    """Load raw CSV files from Kaggle IPL dataset."""
    matches    = pd.read_csv(f"{data_dir}/matches.csv")
    deliveries = pd.read_csv(f"{data_dir}/deliveries.csv")
    return matches, deliveries


def clean_data(matches, deliveries):
    """
    Clean, merge, and enrich match + delivery data.
    Returns fully enriched DataFrames ready for analytics.
    """

    # ── Normalise match_id ─────────────────────────────────
    if 'id' in matches.columns:
        matches = matches.rename(columns={'id': 'match_id'})

    # ── Merge season + match metadata into deliveries ──────
    meta_cols = ['match_id', 'season', 'date', 'venue',
                 'team1', 'team2', 'winner', 'city']
    meta_cols = [c for c in meta_cols if c in matches.columns]
    deliveries = deliveries.merge(
        matches[meta_cols], on='match_id', how='left'
    )

    # ── Normalise season (handles "IPL 2020" strings) ──────
    for df in [matches, deliveries]:
        df['season'] = (
            df['season'].astype(str)
            .str.extract(r'(\d{4})')[0]
            .astype(float).astype('Int64')
        )

    # ── Phase column ───────────────────────────────────────
    deliveries['phase'] = deliveries['over'].apply(
        lambda x: 'Powerplay' if x <= 6
        else ('Middle' if x <= 15 else 'Death')
    )

    # ── Ensure total_runs column ───────────────────────────
    if 'total_runs' not in deliveries.columns:
        extra_col = next(
            (c for c in ['extra_runs', 'extras'] if c in deliveries.columns),
            None
        )
        deliveries['total_runs'] = (
            deliveries['batsman_runs'] + deliveries[extra_col]
            if extra_col else deliveries['batsman_runs']
        )

    # ── Wicket flag (handle player_dismissed column) ───────
    if 'player_dismissed' in deliveries.columns:
        deliveries['is_wicket'] = (
            deliveries['player_dismissed'].notna().astype(int)
        )

    # ── Legal delivery flag ────────────────────────────────
    if 'wide_runs' in deliveries.columns and 'noball_runs' in deliveries.columns:
        deliveries['is_legal'] = (
            (deliveries['wide_runs'] == 0) &
            (deliveries['noball_runs'] == 0)
        ).astype(int)
    else:
        deliveries['is_legal'] = 1

    # ── Derived ball-level flags ───────────────────────────
    deliveries['is_four'] = (deliveries['batsman_runs'] == 4).astype(int)
    deliveries['is_six']  = (deliveries['batsman_runs'] == 6).astype(int)
    deliveries['is_dot']  = (
        (deliveries['batsman_runs'] == 0) & (deliveries['is_legal'] == 1)
    ).astype(int)

    return matches, deliveries
