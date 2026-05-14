import pandas as pd
import numpy as np


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def _bat(df, player):
    """Rows where player is the batter."""
    return df[df['batter'] == player]


def _bowl(df, bowler):
    """Rows where player is the bowler."""
    return df[df['bowler'] == bowler]


def _legal_balls(df):
    """Only legal deliveries (exclude wides & no-balls)."""
    return df[df['is_legal'] == 1] if 'is_legal' in df.columns else df


def _overs_from_balls(balls):
    """Convert raw ball count to overs (e.g. 67 → 11.1)."""
    return balls // 6 + (balls % 6) / 10


# ══════════════════════════════════════════════
#  TOP PERFORMERS
# ══════════════════════════════════════════════

def top_run_scorers(df, top_n=10):
    """Top batters by total runs."""
    return (
        df.groupby('batter')['batsman_runs']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'batter': 'Player', 'batsman_runs': 'Runs'})
    )


def top_wicket_takers(df, top_n=10):
    """Top bowlers by total wickets (excludes run-outs)."""
    filtered = df[
        ~df.get('dismissal_kind', pd.Series(dtype=str)).isin(['run out', 'retired hurt'])
    ] if 'dismissal_kind' in df.columns else df

    return (
        filtered.groupby('bowler')['is_wicket']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'bowler': 'Player', 'is_wicket': 'Wickets'})
    )


def top_six_hitters(df, top_n=10):
    """Top six hitters of all time."""
    return (
        df.groupby('batter')['is_six']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'batter': 'Player', 'is_six': 'Sixes'})
    )


def most_economical_bowlers(df, min_overs=20, top_n=10):
    """Most economical bowlers (minimum overs threshold)."""
    legal = _legal_balls(df)
    grp = legal.groupby('bowler').agg(
        Balls  = ('total_runs', 'count'),
        Runs   = ('total_runs', 'sum'),
    ).reset_index()
    grp = grp[grp['Balls'] >= min_overs * 6]
    grp['Overs']   = grp['Balls'].apply(_overs_from_balls)
    grp['Economy'] = (grp['Runs'] / (grp['Balls'] / 6)).round(2)
    return (
        grp[['bowler', 'Overs', 'Runs', 'Economy']]
        .rename(columns={'bowler': 'Player'})
        .sort_values('Economy')
        .head(top_n)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════
#  PLAYER PROFILE  (fixed average formula)
# ══════════════════════════════════════════════

def player_profile(deliveries, matches, player):
    """
    Complete batting profile.
    Average = runs / dismissals  (not runs / innings — that was the ChatGPT bug).
    """
    p = _bat(deliveries, player)
    legal = _legal_balls(p)

    total_runs   = int(p['batsman_runs'].sum())
    total_balls  = int(legal.shape[0])
    innings      = int(p['match_id'].nunique())
    dismissals   = int(p['is_wicket'].sum()) if 'is_wicket' in p.columns else innings
    average      = round(total_runs / dismissals, 2) if dismissals > 0 else total_runs
    strike_rate  = round((total_runs / total_balls) * 100, 2) if total_balls > 0 else 0

    scores = p.groupby('match_id')['batsman_runs'].sum()
    highest   = int(scores.max()) if not scores.empty else 0
    fifties   = int(((scores >= 50) & (scores < 100)).sum())
    hundreds  = int((scores >= 100).sum())
    fours     = int(p['is_four'].sum())
    sixes     = int(p['is_six'].sum())
    dot_pct   = round(p['is_dot'].sum() / total_balls * 100, 1) if total_balls > 0 else 0

    # Season-wise runs for trend chart
    season_runs = (
        p.groupby('season')['batsman_runs'].sum()
        .reset_index()
        .rename(columns={'batsman_runs': 'Runs'})
        .sort_values('season')
    )

    return {
        'Player'         : player,
        'Runs'           : total_runs,
        'Innings'        : innings,
        'Dismissals'     : dismissals,
        'Average'        : average,
        'Strike Rate'    : strike_rate,
        'Highest Score'  : highest,
        '50s'            : fifties,
        '100s'           : hundreds,
        '4s'             : fours,
        '6s'             : sixes,
        'Dot Ball %'     : dot_pct,
        'Season Runs'    : season_runs,
    }


# ══════════════════════════════════════════════
#  PLAYER COMPARISON  (multi-metric clean table)
# ══════════════════════════════════════════════

def compare_players(deliveries, player1, player2):
    """
    Returns a comparison DataFrame — one row per metric, two player columns.
    Fixes the same-Y-axis chart problem from the original code.
    """
    profiles = {}
    for p in [player1, player2]:
        bat = _bat(deliveries, p)
        legal = _legal_balls(bat)
        runs      = int(bat['batsman_runs'].sum())
        balls     = int(legal.shape[0])
        dismissals = int(bat['is_wicket'].sum()) if 'is_wicket' in bat.columns else max(bat['match_id'].nunique(), 1)
        profiles[p] = {
            'Runs'         : runs,
            'Innings'      : int(bat['match_id'].nunique()),
            'Average'      : round(runs / dismissals, 2) if dismissals > 0 else runs,
            'Strike Rate'  : round(runs / balls * 100, 2) if balls > 0 else 0,
            '4s'           : int(bat['is_four'].sum()),
            '6s'           : int(bat['is_six'].sum()),
            'Dot Ball %'   : round(bat['is_dot'].sum() / balls * 100, 1) if balls > 0 else 0,
            'Boundary %'   : round((bat['is_four'].sum() + bat['is_six'].sum()) / balls * 100, 1) if balls > 0 else 0,
        }

    rows = []
    for metric in profiles[player1]:
        rows.append({
            'Metric'  : metric,
            player1   : profiles[player1][metric],
            player2   : profiles[player2][metric],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
#  HEAD-TO-HEAD  (new feature)
# ══════════════════════════════════════════════

def head_to_head(deliveries, batter, bowler):
    """
    How a specific batter performs against a specific bowler.
    """
    h2h = deliveries[
        (deliveries['batter'] == batter) &
        (deliveries['bowler'] == bowler)
    ]
    legal = _legal_balls(h2h)
    runs   = int(h2h['batsman_runs'].sum())
    balls  = int(legal.shape[0])
    wkts   = int(h2h['is_wicket'].sum()) if 'is_wicket' in h2h.columns else 0
    fours  = int(h2h['is_four'].sum())
    sixes  = int(h2h['is_six'].sum())
    sr     = round(runs / balls * 100, 2) if balls > 0 else 0

    return {
        'Batter'       : batter,
        'Bowler'       : bowler,
        'Runs'         : runs,
        'Balls'        : balls,
        'Dismissals'   : wkts,
        'Strike Rate'  : sr,
        '4s'           : fours,
        '6s'           : sixes,
        'Dot Balls'    : int(h2h['is_dot'].sum()),
    }


# ══════════════════════════════════════════════
#  PHASE ANALYSIS  (batting)
# ══════════════════════════════════════════════

def phase_analysis(deliveries, player):
    """Runs, SR, and boundary % across powerplay / middle / death."""
    p = _bat(deliveries, player)
    results = []
    for phase_name, mask in [
        ('Powerplay (1–6)',    p['over'] <= 6),
        ('Middle (7–15)',     (p['over'] > 6)  & (p['over'] <= 15)),
        ('Death (16–20)',      p['over'] > 15),
    ]:
        sub   = p[mask]
        legal = _legal_balls(sub)
        runs  = int(sub['batsman_runs'].sum())
        balls = int(legal.shape[0])
        results.append({
            'Phase'       : phase_name,
            'Runs'        : runs,
            'Balls'       : balls,
            'Strike Rate' : round(runs / balls * 100, 2) if balls > 0 else 0,
            '4s'          : int(sub['is_four'].sum()),
            '6s'          : int(sub['is_six'].sum()),
            'Dot Ball %'  : round(sub['is_dot'].sum() / balls * 100, 1) if balls > 0 else 0,
        })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════
#  BOUNDARY ANALYSIS
# ══════════════════════════════════════════════

def boundary_analysis(deliveries, player):
    p     = _bat(deliveries, player)
    legal = _legal_balls(p)
    balls = int(legal.shape[0])
    fours = int(p['is_four'].sum())
    sixes = int(p['is_six'].sum())
    dots  = int(p['is_dot'].sum())
    return {
        '4s'           : fours,
        '6s'           : sixes,
        'Boundary %'   : round((fours + sixes) / balls * 100, 2) if balls > 0 else 0,
        'Dot Ball %'   : round(dots / balls * 100, 2) if balls > 0 else 0,
        'Boundary Runs': int(fours * 4 + sixes * 6),
    }


# ══════════════════════════════════════════════
#  BOWLING ANALYTICS  (fixed overs bug)
# ══════════════════════════════════════════════

def bowling_analytics(deliveries, bowler):
    """
    Full bowling profile.
    Bug fix: overs are counted from LEGAL balls only (original counted wides).
    """
    b     = _bowl(deliveries, bowler)
    legal = _legal_balls(b)

    wickets      = int(b['is_wicket'].sum())
    runs_conceded= int(b['total_runs'].sum())
    legal_balls  = int(legal.shape[0])
    overs_dec    = round(legal_balls / 6, 2)
    economy      = round(runs_conceded / (legal_balls / 6), 2) if legal_balls > 0 else 0
    dot_balls    = int(b['is_dot'].sum())
    dot_pct      = round(dot_balls / legal_balls * 100, 1) if legal_balls > 0 else 0

    # Best figures in a single match
    match_wkts = b.groupby('match_id')['is_wicket'].sum()
    match_runs = b.groupby('match_id')['total_runs'].sum()
    best_idx   = match_wkts.idxmax() if not match_wkts.empty else None
    best_figures = (
        f"{int(match_wkts[best_idx])}/{int(match_runs[best_idx])}"
        if best_idx is not None else "N/A"
    )

    # Season-wise wickets for trend chart
    season_wkts = (
        b.groupby('season')['is_wicket'].sum()
        .reset_index()
        .rename(columns={'is_wicket': 'Wickets'})
        .sort_values('season')
    )

    return {
        'Bowler'         : bowler,
        'Wickets'        : wickets,
        'Runs Conceded'  : runs_conceded,
        'Overs'          : overs_dec,
        'Economy'        : economy,
        'Dot Balls'      : dot_balls,
        'Dot Ball %'     : dot_pct,
        'Best Figures'   : best_figures,
        'Season Wickets' : season_wkts,
    }


# ══════════════════════════════════════════════
#  BOWLING PHASE ANALYSIS  (new)
# ══════════════════════════════════════════════

def bowling_phase_analysis(deliveries, bowler):
    """Economy and wickets across powerplay / middle / death."""
    b = _bowl(deliveries, bowler)
    results = []
    for phase_name, mask in [
        ('Powerplay (1–6)',   b['over'] <= 6),
        ('Middle (7–15)',    (b['over'] > 6)  & (b['over'] <= 15)),
        ('Death (16–20)',     b['over'] > 15),
    ]:
        sub   = b[mask]
        legal = _legal_balls(sub)
        balls = int(legal.shape[0])
        runs  = int(sub['total_runs'].sum())
        wkts  = int(sub['is_wicket'].sum())
        results.append({
            'Phase'   : phase_name,
            'Balls'   : balls,
            'Runs'    : runs,
            'Wickets' : wkts,
            'Economy' : round(runs / (balls / 6), 2) if balls > 0 else 0,
        })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════
#  VENUE ANALYSIS  (new)
# ══════════════════════════════════════════════

def venue_batting_stats(deliveries, player):
    """Batter's performance broken down by venue."""
    p = _bat(deliveries, player)
    if 'venue' not in p.columns:
        return pd.DataFrame()

    grp = p.groupby('venue').agg(
        Runs       = ('batsman_runs', 'sum'),
        Balls      = ('is_legal', 'sum'),
        Dismissals = ('is_wicket', 'sum'),
        Matches    = ('match_id', 'nunique'),
    ).reset_index()

    grp['Average']     = (grp['Runs'] / grp['Dismissals'].replace(0, 1)).round(2)
    grp['Strike Rate'] = (grp['Runs'] / grp['Balls'].replace(0, 1) * 100).round(2)
    return grp.sort_values('Runs', ascending=False).head(15)


# ══════════════════════════════════════════════
#  SEASON TRENDS  (new)
# ══════════════════════════════════════════════

def season_batting_trend(deliveries, player):
    """Year-by-year batting stats for a player."""
    p = _bat(deliveries, player)
    rows = []
    for season, grp in p.groupby('season'):
        legal      = _legal_balls(grp)
        runs       = int(grp['batsman_runs'].sum())
        balls      = int(legal.shape[0])
        dismissals = int(grp['is_wicket'].sum()) if 'is_wicket' in grp.columns else 1
        rows.append({
            'Season'      : int(season),
            'Runs'        : runs,
            'Innings'     : int(grp['match_id'].nunique()),
            'Average'     : round(runs / dismissals, 2) if dismissals > 0 else runs,
            'Strike Rate' : round(runs / balls * 100, 2) if balls > 0 else 0,
            '6s'          : int(grp['is_six'].sum()),
        })
    return pd.DataFrame(rows).sort_values('Season')


def season_bowling_trend(deliveries, bowler):
    """Year-by-year bowling stats for a bowler."""
    b = _bowl(deliveries, bowler)
    rows = []
    for season, grp in b.groupby('season'):
        legal = _legal_balls(grp)
        balls = int(legal.shape[0])
        runs  = int(grp['total_runs'].sum())
        wkts  = int(grp['is_wicket'].sum())
        rows.append({
            'Season'  : int(season),
            'Wickets' : wkts,
            'Economy' : round(runs / (balls / 6), 2) if balls > 0 else 0,
            'Dot %'   : round(grp['is_dot'].sum() / balls * 100, 1) if balls > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values('Season')


# ══════════════════════════════════════════════
#  DISMISSAL ANALYSIS  (new)
# ══════════════════════════════════════════════

def dismissal_analysis(deliveries, player):
    """What kinds of dismissals does a batter get?"""
    if 'dismissal_kind' not in deliveries.columns:
        return pd.DataFrame()
    p = _bat(deliveries, player)
    dismissed = p[p['is_wicket'] == 1] if 'is_wicket' in p.columns else p[p['dismissal_kind'].notna()]
    return (
        dismissed['dismissal_kind']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Dismissal Type', 'dismissal_kind': 'Count'})
    )


# ══════════════════════════════════════════════
#  TEAM SUMMARY  (new)
# ══════════════════════════════════════════════

def team_season_summary(matches, deliveries, team, season=None):
    """Win/loss record and run stats for a team."""
    m = matches.copy()
    if season:
        m = m[m['season'] == season]

    played = m[(m['team1'] == team) | (m['team2'] == team)]
    wins   = played[played['winner'] == team].shape[0]
    losses = played.shape[0] - wins

    # Runs scored by team
    d = deliveries.copy()
    if season:
        d = d[d['season'] == season]

    batting_df = d[d['batting_team'] == team] if 'batting_team' in d.columns else d
    runs_scored = int(batting_df['batsman_runs'].sum()) if not batting_df.empty else 0

    return {
        'Team'        : team,
        'Season'      : season or 'All',
        'Matches'     : int(played.shape[0]),
        'Wins'        : int(wins),
        'Losses'      : int(losses),
        'Win Rate %'  : round(wins / played.shape[0] * 100, 1) if played.shape[0] > 0 else 0,
        'Runs Scored' : runs_scored,
    }


# ══════════════════════════════════════════════
#  AI PROMPT BUILDER  (new — feeds Claude API)
# ══════════════════════════════════════════════

def build_player_prompt(profile: dict) -> str:
    """Build a rich prompt for Claude AI from a player profile dict."""
    p = profile
    return (
        f"You are an expert IPL cricket analyst. Analyse {p['Player']} based on these stats:\n"
        f"- Runs: {p['Runs']} | Innings: {p['Innings']} | Average: {p['Average']} | SR: {p['Strike Rate']}\n"
        f"- Highest: {p['Highest Score']} | 50s: {p['50s']} | 100s: {p['100s']}\n"
        f"- Fours: {p['4s']} | Sixes: {p['6s']} | Dot Ball %: {p['Dot Ball %']}\n\n"
        f"Write 4 sentences: (1) overall batting style, (2) best phase of play, "
        f"(3) one clear weakness based on dot ball % or average, "
        f"(4) legacy/impact in IPL. Be direct, analytical, no fluff."
    )


def build_comparison_prompt(p1_name, p2_name, comp_df: pd.DataFrame) -> str:
    """Build a comparison prompt for Claude AI."""
    rows = comp_df.set_index('Metric')
    metrics = "\n".join(
        f"  {row}: {p1_name}={rows.loc[row, p1_name]}  vs  {p2_name}={rows.loc[row, p2_name]}"
        for row in rows.index
        if row in ['Runs', 'Average', 'Strike Rate', '4s', '6s', 'Dot Ball %']
    )
    return (
        f"Compare {p1_name} vs {p2_name} in IPL based on:\n{metrics}\n\n"
        f"Give a 3-sentence verdict: who wins on consistency (average), "
        f"who wins on aggression (SR + 6s), and who you'd pick in a knockout match. Be direct."
    )


def build_bowling_prompt(bowl_stats: dict) -> str:
    """Build a bowling analysis prompt for Claude AI."""
    b = bowl_stats
    return (
        f"Analyse IPL bowler {b['Bowler']}:\n"
        f"- Wickets: {b['Wickets']} | Economy: {b['Economy']} | Dot Ball %: {b['Dot Ball %']}\n"
        f"- Best figures: {b['Best Figures']} | Overs bowled: {b['Overs']}\n\n"
        f"Write 3 sentences: (1) bowling style and primary weapon, "
        f"(2) what the economy and dot ball % reveal about their effectiveness, "
        f"(3) their legacy in IPL bowling. Be punchy and analytical."
    )
