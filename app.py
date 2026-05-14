import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

from data_loader import load_raw_data, clean_data
from analytics import (
    top_run_scorers, top_wicket_takers,
    top_six_hitters, most_economical_bowlers,
    compare_players, player_profile,
    phase_analysis, boundary_analysis,
    bowling_analytics, bowling_phase_analysis,
    head_to_head, venue_batting_stats,
    season_batting_trend, season_bowling_trend,
    dismissal_analysis, team_season_summary,
)

st.set_page_config(
    page_title="IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  PALETTE
#  Sidebar bg:       #C4956A  (warm caramel)
#  Sidebar inputs:   #EDD9C0  (light beige — NEW, much lighter)
#  Main BG:          #FAF8F4  (parchment off-white)
#  Card BG:          #F3EDE4  (warm cream)
#  Border:           #DDD0BC  (light tan)
#  Text dark:        #4A2E12  (deep espresso)
#  Text muted:       #8C6845  (mid brown)
#  Accent:           #9B6440  (sienna)
# ══════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=Poppins:wght@300;400;500&display=swap');

  /* ── Base reset ── */
  html, body, [class*="css"] {
      font-family: 'Poppins', sans-serif !important;
      color: #4A2E12 !important;
  }
  .stApp {
      background-color: #FAF8F4 !important;
  }
  .main .block-container {
      background-color: #FAF8F4 !important;
      padding-top: 2rem !important;
  }

  /* ════════════════════════════
     SIDEBAR
  ════════════════════════════ */
  [data-testid="stSidebar"] {
      background-color: #C4956A !important;
      border-right: 1px solid #AD7D53 !important;
  }
  [data-testid="stSidebar"] > div:first-child {
      background-color: #C4956A !important;
  }
  /* All sidebar text dark espresso */
  [data-testid="stSidebar"] * {
      color: #2E1A08 !important;
      font-family: 'Poppins', sans-serif !important;
  }
  /* Sidebar title — Playfair */
  [data-testid="stSidebar"] h1 {
      font-family: 'Playfair Display', serif !important;
      font-weight: 500 !important;
      font-size: 1.4rem !important;
      letter-spacing: 0.02em !important;
      color: #2E1A08 !important;
  }
  /* Sidebar sub-labels */
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
      font-family: 'Poppins', sans-serif !important;
      font-weight: 500 !important;
      font-size: 0.7rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.14em !important;
      color: #2E1A08 !important;
  }
  [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small {
      color: #4A2E12 !important;
      font-size: 0.77rem !important;
      font-weight: 300 !important;
  }
  [data-testid="stSidebar"] hr {
      border-color: #AD7D53 !important;
      opacity: 0.5 !important;
  }

  /* ── Sidebar inputs: LIGHT BEIGE ── */
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
      background-color: #EDD9C0 !important;
      border: 1px solid #D4BA9A !important;
      color: #2E1A08 !important;
      border-radius: 4px !important;
  }
  [data-testid="stSidebar"] .stTextInput > div > div,
  [data-testid="stSidebar"] .stTextInput > div > div > input {
      background-color: #EDD9C0 !important;
      border: 1px solid #D4BA9A !important;
      color: #2E1A08 !important;
      border-radius: 4px !important;
  }
  [data-testid="stSidebar"] input {
      background-color: transparent !important;
      color: #2E1A08 !important;
      font-family: 'Poppins', sans-serif !important;
  }
  [data-testid="stSidebar"] input::placeholder {
      color: #8C6845 !important;
  }
  /* Dropdown arrow / icon area */
  [data-testid="stSidebar"] [data-baseweb="select"] svg {
      fill: #6B4226 !important;
  }
  /* Sidebar alert boxes */
  [data-testid="stSidebar"] .stAlert {
      background-color: rgba(237, 217, 192, 0.55) !important;
      border: 1px solid #D4BA9A !important;
      border-radius: 4px !important;
      color: #2E1A08 !important;
  }

  /* ════════════════════════════
     MAIN CONTENT — TYPOGRAPHY
  ════════════════════════════ */

  /* H1 — main dashboard title — Playfair */
  h1,
  .main h1,
  [data-testid="stMainBlockContainer"] h1,
  section[data-testid="stMain"] h1,
  .stApp h1 {
      font-family: 'Playfair Display', serif !important;
      font-weight: 400 !important;
      font-size: 2.55rem !important;
      letter-spacing: 0.01em !important;
      color: #4A2E12 !important;
      line-height: 1.2 !important;
  }

  /* Hide keyboard_double_arrow collapse button */
  button[data-testid="collapsedControl"],
  [data-testid="collapsedControl"],
  .st-emotion-cache-1dp5vir,
  button[kind="header"],
  [data-testid="stSidebarCollapseButton"] {
      display: none !important;
  }
  /* Hide the collapse icon text that shows as keyboard_double */
  [data-testid="stSidebar"] button span[class*="material"] {
      display: none !important;
  }
  /* H2 — tab section subheadings — Playfair */
  .main h2,
  [data-testid="stMainBlockContainer"] h2,
  section[data-testid="stMain"] h2 {
      font-family: 'Playfair Display', serif !important;
      font-weight: 400 !important;
      font-size: 1.4rem !important;
      letter-spacing: 0.01em !important;
      color: #4A2E12 !important;
  }
  /* H3 and below — Poppins */
  .main h3,
  [data-testid="stMainBlockContainer"] h3,
  section[data-testid="stMain"] h3 {
      font-family: 'Poppins', sans-serif !important;
      font-weight: 500 !important;
      font-size: 0.68rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.15em !important;
      color: #8C6845 !important;
  }

  /* Body / paragraphs / spans — Poppins */
  .main p,
  .main li,
  .main span,
  [data-testid="stMainBlockContainer"] p,
  [data-testid="stMainBlockContainer"] span {
      font-family: 'Poppins', sans-serif !important;
      color: #4A2E12 !important;
  }
  .stCaption, small {
      color: #8C6845 !important;
      font-family: 'Poppins', sans-serif !important;
      font-weight: 300 !important;
      font-size: 0.78rem !important;
  }
  hr { border-color: #DDD0BC !important; opacity: 0.7 !important; }

  /* ════════════════════════════
     TABS
  ════════════════════════════ */
  .stTabs [data-baseweb="tab-list"] {
      background-color: #F3EDE4 !important;
      border-bottom: 1px solid #DDD0BC !important;
      gap: 0 !important;
      padding: 0 4px !important;
      border-radius: 6px 6px 0 0 !important;
  }
  .stTabs [data-baseweb="tab"] {
      font-family: 'Poppins', sans-serif !important;
      font-weight: 400 !important;
      font-size: 0.76rem !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
      color: #8C6845 !important;
      background-color: transparent !important;
      border: none !important;
      padding: 12px 22px !important;
      transition: all 0.2s !important;
  }
  .stTabs [aria-selected="true"] {
      color: #4A2E12 !important;
      border-bottom: 2px solid #9B6440 !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
      color: #4A2E12 !important;
      background-color: rgba(155, 100, 64, 0.07) !important;
  }

  /* ════════════════════════════
     METRIC CARDS
  ════════════════════════════ */
  [data-testid="stMetric"] {
      background-color: #F3EDE4 !important;
      border: 1px solid #DDD0BC !important;
      border-radius: 6px !important;
      padding: 18px 20px !important;
  }
  [data-testid="stMetricLabel"] {
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.65rem !important;
      font-weight: 500 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.13em !important;
      color: #8C6845 !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Playfair Display', serif !important;
      font-size: 1.75rem !important;
      font-weight: 400 !important;
      color: #4A2E12 !important;
  }
  [data-testid="stMetricDelta"] {
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.72rem !important;
      color: #9B6440 !important;
  }

  /* ════════════════════════════
     BUTTONS
  ════════════════════════════ */
  .stButton > button {
      font-family: 'Poppins', sans-serif !important;
      font-weight: 400 !important;
      font-size: 0.73rem !important;
      letter-spacing: 0.11em !important;
      text-transform: uppercase !important;
      color: #4A2E12 !important;
      background-color: transparent !important;
      border: 1px solid #9B6440 !important;
      border-radius: 4px !important;
      padding: 10px 26px !important;
      transition: all 0.2s !important;
  }
  .stButton > button:hover {
      background-color: #9B6440 !important;
      color: #FAF8F4 !important;
  }

  /* ════════════════════════════
     MAIN CONTENT SELECTBOX / TEXT INPUT
     (lighter warm cream — matching card bg)
  ════════════════════════════ */
  .main .stSelectbox > div > div,
  .main [data-baseweb="select"] > div {
      background-color: #F3EDE4 !important;
      border: 1px solid #DDD0BC !important;
      border-radius: 4px !important;
      color: #4A2E12 !important;
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.85rem !important;
  }
  .main .stSelectbox label,
  .main .stTextInput label {
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.66rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.11em !important;
      color: #8C6845 !important;
      font-weight: 500 !important;
  }
  .main .stTextInput > div > div {
      background-color: #F3EDE4 !important;
      border: 1px solid #DDD0BC !important;
      border-radius: 4px !important;
  }
  input[type="password"] {
      background-color: transparent !important;
      color: #4A2E12 !important;
      font-family: 'Poppins', sans-serif !important;
  }

  /* ════════════════════════════
     ALERTS
  ════════════════════════════ */
  .stAlert {
      background-color: #F3EDE4 !important;
      border: 1px solid #DDD0BC !important;
      border-radius: 4px !important;
      color: #4A2E12 !important;
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.83rem !important;
  }

  /* ════════════════════════════
     DATAFRAME / TABLE
  ════════════════════════════ */
  [data-testid="stDataFrame"] {
      border: 1px solid #DDD0BC !important;
      border-radius: 4px !important;
  }
  [data-testid="stDataFrame"] th {
      background-color: #E8DDD0 !important;
      color: #4A2E12 !important;
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.66rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.1em !important;
      font-weight: 500 !important;
  }
  [data-testid="stDataFrame"] td {
      background-color: #FAF8F4 !important;
      color: #4A2E12 !important;
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.82rem !important;
      font-weight: 300 !important;
  }

  /* ════════════════════════════
     SECTION TITLE (custom div)
  ════════════════════════════ */
  .section-title {
      font-family: 'Playfair Display', serif !important;
      font-size: 1.3rem !important;
      font-weight: 400 !important;
      color: #4A2E12 !important;
      letter-spacing: 0.01em !important;
      margin: 24px 0 12px 0 !important;
      padding-bottom: 8px !important;
      border-bottom: 1px solid #DDD0BC !important;
  }

  /* ════════════════════════════
     INSIGHT BOX
  ════════════════════════════ */
  .insight-box {
      background-color: #F3EDE4 !important;
      border: 1px solid #DDD0BC !important;
      border-left: 3px solid #9B6440 !important;
      border-radius: 4px !important;
      padding: 18px 22px !important;
      font-family: 'Poppins', sans-serif !important;
      font-size: 0.84rem !important;
      font-weight: 300 !important;
      line-height: 1.9 !important;
      color: #4A2E12 !important;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #9B6440 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #FAF8F4; }
  ::-webkit-scrollbar-thumb { background: #DDD0BC; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #9B6440; }
</style>
""", unsafe_allow_html=True)


# ── Earthy chart palette ──────────────────────
EARTHY = ["#9B6440", "#C4956A", "#DDD0BC", "#6B3F20", "#E8DDD0", "#7A5030", "#B8896A"]

def apply_theme(fig, height=None):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F3EDE4",
        font=dict(family="Poppins, sans-serif", color="#4A2E12", size=11),
        xaxis=dict(
            gridcolor="#E8DDD0", linecolor="#DDD0BC",
            tickfont=dict(color="#8C6845", size=10),
            title_font=dict(color="#8C6845"),
        ),
        yaxis=dict(
            gridcolor="#E8DDD0", linecolor="#DDD0BC",
            tickfont=dict(color="#8C6845", size=10),
            title_font=dict(color="#8C6845"),
        ),
        legend=dict(font=dict(color="#4A2E12", family="Poppins", size=11)),
        margin=dict(l=0, r=0, t=16, b=0),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════
#  API HELPER
# ══════════════════════════════════════════════




# ══════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════

@st.cache_data
def load_data():
    m, d = load_raw_data("data")
    return clean_data(m, d)

with st.spinner("Loading IPL data..."):
    matches, deliveries = load_data()



# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.subheader("Filters")
    season_options = sorted(matches['season'].dropna().unique().tolist())
    season = st.selectbox("Season", ["All"] + [str(s) for s in season_options])

    if season != "All":
        s = int(season)
        deliveries_f = deliveries[deliveries['season'] == s]
        matches_f    = matches[matches['season'] == s]
    else:
        deliveries_f = deliveries
        matches_f    = matches

    st.divider()
    st.caption(f"{season} season")
    st.caption(f"Matches: {len(matches_f)}")
    st.caption(f"Deliveries: {len(deliveries_f):,}")
    batters = deliveries_f['batter'].nunique()
    bowlers = deliveries_f['bowler'].nunique()
    st.caption(f"Batters: {batters}  ·  Bowlers: {bowlers}")



# ══════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════

st.markdown('<h1 style="font-family: Playfair Display, serif; font-weight: 400; font-size: 2.6rem; color: #4A2E12; letter-spacing: 0.01em; line-height: 1.2;">IPL Analytics Dashboard</h1>', unsafe_allow_html=True)
st.caption("Advanced cricket analytics · Kaggle IPL 2008 – 2024")
st.divider()


# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════

tab_overview, tab_batters, tab_bowlers, tab_compare, tab_h2h, tab_teams = st.tabs([
    "Overview", "Batters", "Bowlers", "Compare", "Head-to-Head", "Teams",
])


# ══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab_overview:
    st.markdown('<div class="section-title">Top Performers</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    runs_df    = top_run_scorers(deliveries_f, top_n=1)
    wickets_df = top_wicket_takers(deliveries_f, top_n=1)
    sixes_df   = top_six_hitters(deliveries_f, top_n=1)
    eco_df     = most_economical_bowlers(deliveries_f, top_n=1)

    with col1:
        st.metric("Top Scorer", runs_df.iloc[0]['Player'], f"{runs_df.iloc[0]['Runs']:,} runs")
    with col2:
        st.metric("Top Wicket-Taker", wickets_df.iloc[0]['Player'], f"{int(wickets_df.iloc[0]['Wickets'])} wkts")
    with col3:
        st.metric("Six Machine", sixes_df.iloc[0]['Player'], f"{int(sixes_df.iloc[0]['Sixes'])} sixes")
    with col4:
        if not eco_df.empty:
            st.metric("Most Economical", eco_df.iloc[0]['Player'], f"{eco_df.iloc[0]['Economy']} eco")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Run Scorers")
        top10_runs = top_run_scorers(deliveries_f, top_n=10)
        fig = px.bar(
            top10_runs, x='Runs', y='Player', orientation='h',
            color='Runs',
            color_continuous_scale=[[0, "#E8DDD0"], [0.5, "#C4956A"], [1, "#6B3F20"]],
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        apply_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Top 10 Wicket Takers")
        top10_wkts = top_wicket_takers(deliveries_f, top_n=10)
        fig2 = px.bar(
            top10_wkts, x='Wickets', y='Player', orientation='h',
            color='Wickets',
            color_continuous_scale=[[0, "#E8DDD0"], [0.5, "#B8896A"], [1, "#5C3020"]],
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        apply_theme(fig2, height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # ── IPL Honours Table ─────────────────────────────────────
    st.markdown('<div class="section-title">IPL Honours — Season by Season</div>', unsafe_allow_html=True)

    # Hardcoded IPL finals history (2008–2024)
    ipl_finals = [
        (2008, "Rajasthan Royals",         "Chennai Super Kings"),
        (2009, "Deccan Chargers",           "Royal Challengers Bangalore"),
        (2010, "Chennai Super Kings",       "Mumbai Indians"),
        (2011, "Chennai Super Kings",       "Royal Challengers Bangalore"),
        (2012, "Kolkata Knight Riders",     "Chennai Super Kings"),
        (2013, "Mumbai Indians",            "Chennai Super Kings"),
        (2014, "Kolkata Knight Riders",     "Kings XI Punjab"),
        (2015, "Mumbai Indians",            "Chennai Super Kings"),
        (2016, "Sunrisers Hyderabad",       "Royal Challengers Bangalore"),
        (2017, "Mumbai Indians",            "Rising Pune Supergiant"),
        (2018, "Chennai Super Kings",       "Sunrisers Hyderabad"),
        (2019, "Mumbai Indians",            "Chennai Super Kings"),
        (2020, "Mumbai Indians",            "Delhi Capitals"),
        (2021, "Chennai Super Kings",       "Kolkata Knight Riders"),
        (2022, "Gujarat Titans",            "Rajasthan Royals"),
        (2023, "Chennai Super Kings",       "Gujarat Titans"),
        (2024, "Kolkata Knight Riders",     "Sunrisers Hyderabad"),
    ]

    finals_df = pd.DataFrame(ipl_finals, columns=["Season", "Champion 🏆", "Runner-up 🥈"])
    finals_df = finals_df.sort_values("Season", ascending=False).reset_index(drop=True)

    # Trophy count summary
    trophy_count = finals_df["Champion 🏆"].value_counts().reset_index()
    trophy_count.columns = ["Team", "Titles"]
    runnerup_count = finals_df["Runner-up 🥈"].value_counts().reset_index()
    runnerup_count.columns = ["Team", "Runner-up Finishes"]
    summary = trophy_count.merge(runnerup_count, on="Team", how="outer").fillna(0)
    summary["Runner-up Finishes"] = summary["Runner-up Finishes"].astype(int)
    summary = summary.sort_values("Titles", ascending=False).reset_index(drop=True)

    honours_col1, honours_col2 = st.columns([3, 2])

    with honours_col1:
        st.subheader("Finals History")
        # Style the dataframe rows
        def highlight_rows(row):
            return ['background-color: #F3EDE4; color: #4A2E12'] * len(row)
        st.dataframe(
            finals_df.set_index("Season"),
            use_container_width=True,
            height=420,
        )

    with honours_col2:
        st.subheader("All-time Trophy Tally")
        fig_trophies = px.bar(
            summary,
            x="Titles",
            y="Team",
            orientation="h",
            color="Titles",
            color_continuous_scale=[[0, "#E8DDD0"], [0.5, "#C4956A"], [1, "#6B3F20"]],
            text="Titles",
        )
        fig_trophies.update_traces(textposition="outside", textfont=dict(color="#4A2E12", family="Poppins", size=11))
        fig_trophies.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            xaxis_title="IPL Titles",
        )
        apply_theme(fig_trophies, height=420)
        st.plotly_chart(fig_trophies, use_container_width=True)

    st.divider()

    # Runner-up appearances bar
    st.subheader("Runner-up Appearances")
    fig_ru = px.bar(
        summary.sort_values("Runner-up Finishes", ascending=False),
        x="Runner-up Finishes",
        y="Team",
        orientation="h",
        color="Runner-up Finishes",
        color_continuous_scale=[[0, "#E8DDD0"], [0.5, "#B8896A"], [1, "#5C3020"]],
        text="Runner-up Finishes",
    )
    fig_ru.update_traces(textposition="outside", textfont=dict(color="#4A2E12", family="Poppins", size=11))
    fig_ru.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        xaxis_title="Runner-up Finishes",
    )
    apply_theme(fig_ru, height=320)
    st.plotly_chart(fig_ru, use_container_width=True)

    st.divider()

    if 'season' in deliveries.columns:
        st.subheader("Season-wise Total Runs Trend")
        season_total = (
            deliveries.groupby('season')['batsman_runs']
            .sum().reset_index()
            .rename(columns={'batsman_runs': 'Total Runs'})
            .sort_values('season')
        )
        fig3 = px.area(
            season_total, x='season', y='Total Runs',
            markers=True, color_discrete_sequence=["#9B6440"],
        )
        fig3.update_traces(
            fillcolor="rgba(155,100,64,0.15)",
            line_color="#9B6440",
            marker=dict(color="#6B3F20", size=6)
        )
        fig3.update_layout(xaxis_title="Season", yaxis_title="Runs")
        apply_theme(fig3, height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — BATTERS
# ══════════════════════════════════════════════

with tab_batters:
    player_list = sorted(deliveries_f['batter'].dropna().unique())
    player1 = st.selectbox("Select Batter", player_list, key="bat_select")
    profile = player_profile(deliveries_f, matches_f, player1)

    st.markdown('<div class="section-title">Career Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Runs",          profile['Runs'])
    m2.metric("Innings",       profile['Innings'])
    m3.metric("Average",       profile['Average'])
    m4.metric("Strike Rate",   profile['Strike Rate'])
    m5.metric("Highest Score", profile['Highest Score'])
    m6.metric("50s / 100s",    f"{profile['50s']} / {profile['100s']}")

    m7, m8, m9 = st.columns(3)
    m7.metric("4s",         profile['4s'])
    m8.metric("6s",         profile['6s'])
    m9.metric("Dot Ball %", f"{profile['Dot Ball %']}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Phase-wise Runs")
        phase_df = phase_analysis(deliveries_f, player1)
        fig_phase = px.pie(phase_df, names='Phase', values='Runs',
                           color_discrete_sequence=EARTHY, hole=0.45)
        fig_phase.update_traces(
            textfont=dict(family="Poppins, sans-serif", color="#FAF8F4", size=11),
            marker=dict(line=dict(color="#FAF8F4", width=1.5))
        )
        apply_theme(fig_phase, height=280)
        st.plotly_chart(fig_phase, use_container_width=True)
        st.dataframe(phase_df.set_index('Phase'), use_container_width=True)

    with col_right:
        st.subheader("Dismissal Types")
        diss_df = dismissal_analysis(deliveries_f, player1)
        if not diss_df.empty:
            diss_df.columns = ['Dismissal', 'Count']
            fig_diss = px.pie(diss_df, names='Dismissal', values='Count',
                              color_discrete_sequence=EARTHY, hole=0.45)
            fig_diss.update_traces(
                textfont=dict(family="Poppins, sans-serif", color="#FAF8F4", size=11),
                marker=dict(line=dict(color="#FAF8F4", width=1.5))
            )
            apply_theme(fig_diss, height=280)
            st.plotly_chart(fig_diss, use_container_width=True)
        else:
            st.info("Dismissal data not available in this dataset.")

    st.subheader("Season-by-Season Batting Trend")
    trend_df = season_batting_trend(deliveries, player1)
    if not trend_df.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=trend_df['Season'], y=trend_df['Runs'],
            name='Runs', marker_color='#C4956A', yaxis='y1'
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_df['Season'], y=trend_df['Strike Rate'],
            name='Strike Rate', mode='lines+markers',
            line=dict(color='#6B3F20', width=2),
            marker=dict(color='#6B3F20', size=6), yaxis='y2'
        ))
        fig_trend.update_layout(
            yaxis=dict(title='Runs', side='left', title_font=dict(color='#8C6845')),
            yaxis2=dict(title='Strike Rate', side='right', overlaying='y',
                        title_font=dict(color='#8C6845')),
            legend=dict(orientation='h', y=1.1),
        )
        apply_theme(fig_trend, height=320)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Performance by Venue")
    venue_df = venue_batting_stats(deliveries_f, player1)
    if not venue_df.empty:
        st.dataframe(
            venue_df[['venue', 'Matches', 'Runs', 'Average', 'Strike Rate']]
            .rename(columns={'venue': 'Venue'}).set_index('Venue'),
            use_container_width=True
        )



# ══════════════════════════════════════════════
#  TAB 3 — BOWLERS
# ══════════════════════════════════════════════

with tab_bowlers:
    bowler_list = sorted(deliveries_f['bowler'].dropna().unique())
    bowler      = st.selectbox("Select Bowler", bowler_list, key="bowl_select")
    bowl        = bowling_analytics(deliveries_f, bowler)

    st.markdown('<div class="section-title">Career Summary</div>', unsafe_allow_html=True)
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Wickets",       bowl['Wickets'])
    b2.metric("Economy",       bowl['Economy'])
    b3.metric("Overs",         bowl['Overs'])
    b4.metric("Runs Conceded", bowl['Runs Conceded'])
    b5.metric("Dot Balls",     bowl['Dot Balls'])
    b6.metric("Best Figures",  bowl['Best Figures'])

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Phase-wise Bowling")
        bp_df = bowling_phase_analysis(deliveries_f, bowler)
        fig_bp = px.bar(bp_df, x='Phase', y=['Wickets', 'Economy'],
                        barmode='group', color_discrete_sequence=['#9B6440', '#C4956A'])
        fig_bp.update_layout(legend=dict(orientation='h', y=1.1))
        apply_theme(fig_bp, height=300)
        st.plotly_chart(fig_bp, use_container_width=True)

    with col_r:
        st.subheader("Dot Ball %")
        dot_pct = bowl['Dot Ball %']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=dot_pct,
            title={'text': "Dot Ball %",
                   'font': {'family': 'Poppins', 'color': '#8C6845', 'size': 13}},
            number={'font': {'family': 'Playfair Display', 'color': '#4A2E12', 'size': 38}},
            gauge={
                'axis': {'range': [0, 80], 'tickcolor': '#8C6845',
                         'tickfont': {'color': '#8C6845', 'family': 'Poppins', 'size': 10}},
                'bar':   {'color': "#9B6440"},
                'bgcolor': "#F3EDE4",
                'bordercolor': "#DDD0BC",
                'steps': [
                    {'range': [0,  30], 'color': "#FAF8F4"},
                    {'range': [30, 50], 'color': "#E8DDD0"},
                    {'range': [50, 80], 'color': "#DDD0BC"},
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Season-by-Season Wickets & Economy")
    bowl_trend = season_bowling_trend(deliveries, bowler)
    if not bowl_trend.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=bowl_trend['Season'], y=bowl_trend['Wickets'],
            name='Wickets', marker_color='#C4956A', yaxis='y1'
        ))
        fig_bt.add_trace(go.Scatter(
            x=bowl_trend['Season'], y=bowl_trend['Economy'],
            name='Economy', mode='lines+markers',
            line=dict(color='#6B3F20', width=2),
            marker=dict(color='#6B3F20', size=6), yaxis='y2'
        ))
        fig_bt.update_layout(
            yaxis=dict(title='Wickets', side='left', title_font=dict(color='#8C6845')),
            yaxis2=dict(title='Economy', side='right', overlaying='y',
                        title_font=dict(color='#8C6845')),
            legend=dict(orientation='h', y=1.1),
        )
        apply_theme(fig_bt, height=320)
        st.plotly_chart(fig_bt, use_container_width=True)



# ══════════════════════════════════════════════
#  TAB 4 — COMPARE
# ══════════════════════════════════════════════

with tab_compare:
    st.markdown('<div class="section-title">Side-by-Side Comparison</div>', unsafe_allow_html=True)

    player_list_all = sorted(deliveries_f['batter'].dropna().unique())
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Player 1", player_list_all, key="cmp_p1")
    with c2:
        p2 = st.selectbox("Player 2", player_list_all,
                          index=min(1, len(player_list_all) - 1), key="cmp_p2")

    comp_df = compare_players(deliveries_f, p1, p2)
    st.divider()

    for _, row in comp_df.iterrows():
        metric = row['Metric']
        v1, v2 = row[p1], row[p2]
        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_a:
            st.metric(f"{p1}", v1,
                      delta=("Higher" if v1 >= v2 else "") if metric != 'Dot Ball %'
                      else ("More dots" if v1 > v2 else ""))
        with col_b:
            st.markdown(
                f"<div style='text-align:center;padding-top:20px;"
                f"font-family:Poppins,sans-serif;font-size:0.63rem;"
                f"text-transform:uppercase;letter-spacing:0.13em;color:#8C6845'>{metric}</div>",
                unsafe_allow_html=True
            )
        with col_c:
            st.metric(f"{p2}", v2,
                      delta=("Higher" if v2 >= v1 else "") if metric != 'Dot Ball %'
                      else ("More dots" if v2 > v1 else ""))

    st.divider()
    st.subheader("Radar Chart")
    radar_metrics = ['Average', 'Strike Rate', '4s', '6s', 'Boundary %']
    comp_idx  = comp_df.set_index('Metric')
    available = [m for m in radar_metrics if m in comp_idx.index]
    if available:
        vals1 = [float(comp_idx.loc[m, p1]) for m in available]
        vals2 = [float(comp_idx.loc[m, p2]) for m in available]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(
            r=vals1 + [vals1[0]], theta=available + [available[0]],
            fill='toself', name=p1,
            line=dict(color='#9B6440'), fillcolor='rgba(155,100,64,0.15)'
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=vals2 + [vals2[0]], theta=available + [available[0]],
            fill='toself', name=p2,
            line=dict(color='#C4956A'), fillcolor='rgba(196,149,106,0.15)'
        ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor='#F3EDE4',
                radialaxis=dict(visible=True, gridcolor='#DDD0BC',
                                tickfont=dict(color='#8C6845', family='Poppins', size=9)),
                angularaxis=dict(tickfont=dict(color='#4A2E12', family='Poppins', size=10)),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=-0.1,
                        font=dict(family='Poppins', color='#4A2E12', size=11)),
            margin=dict(l=40, r=40, t=40, b=40), height=420,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    st.subheader("Phase-by-Phase Comparison")
    ph1 = phase_analysis(deliveries_f, p1).rename(columns={'Runs': f'{p1} Runs'})
    ph2 = phase_analysis(deliveries_f, p2).rename(columns={'Runs': f'{p2} Runs'})
    ph_merged = ph1[['Phase', f'{p1} Runs']].merge(ph2[['Phase', f'{p2} Runs']], on='Phase')
    fig_ph = px.bar(
        ph_merged.melt(id_vars='Phase', var_name='Player', value_name='Runs'),
        x='Phase', y='Runs', color='Player', barmode='group',
        color_discrete_sequence=['#9B6440', '#C4956A'],
    )
    fig_ph.update_layout(legend=dict(orientation='h', y=1.1))
    apply_theme(fig_ph, height=300)
    st.plotly_chart(fig_ph, use_container_width=True)



# ══════════════════════════════════════════════
#  TAB 5 — HEAD-TO-HEAD
# ══════════════════════════════════════════════

with tab_h2h:
    st.markdown('<div class="section-title">Batter vs Bowler</div>', unsafe_allow_html=True)
    st.caption("How a batter performs against a specific bowler")

    h2h_c1, h2h_c2 = st.columns(2)
    with h2h_c1:
        h2h_bat  = st.selectbox("Batter", sorted(deliveries_f['batter'].dropna().unique()), key="h2h_bat")
    with h2h_c2:
        h2h_bowl = st.selectbox("Bowler", sorted(deliveries_f['bowler'].dropna().unique()), key="h2h_bowl")

    h2h_stats = head_to_head(deliveries_f, h2h_bat, h2h_bowl)
    st.divider()

    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Runs",        h2h_stats['Runs'])
    h2.metric("Balls Faced", h2h_stats['Balls'])
    h3.metric("Dismissals",  h2h_stats['Dismissals'])
    h4.metric("Strike Rate", h2h_stats['Strike Rate'])
    h5.metric("4s",          h2h_stats['4s'])
    h6.metric("6s",          h2h_stats['6s'])

    st.divider()

    runs  = h2h_stats['Runs']
    balls = h2h_stats['Balls']
    disms = h2h_stats['Dismissals']
    sr    = h2h_stats['Strike Rate']

    if balls == 0:
        verdict = f"{h2h_bat} and {h2h_bowl} have never faced each other in this dataset."
    elif disms == 0:
        verdict = f"{h2h_bat} dominates — never dismissed by {h2h_bowl} ({runs} runs, SR {sr})"
    elif sr > 150:
        verdict = f"{h2h_bat} attacks — SR {sr} despite {disms} dismissal(s). Batter's edge."
    elif disms >= 3:
        verdict = f"{h2h_bowl} has the edge — dismissed {h2h_bat} {disms} time(s) in {balls} balls."
    else:
        verdict = f"Competitive matchup — {runs} runs off {balls} balls, {disms} dismissal(s)."

    st.info(verdict)



# ══════════════════════════════════════════════
#  TAB 6 — TEAMS
# ══════════════════════════════════════════════

with tab_teams:
    st.markdown('<div class="section-title">Team Analysis</div>', unsafe_allow_html=True)

    all_teams = sorted(pd.concat([matches_f['team1'], matches_f['team2']]).dropna().unique())
    selected_team = st.selectbox("Select Team", all_teams, key="team_sel")

    team_stats = team_season_summary(
        matches_f, deliveries_f, selected_team,
        season if season != "All" else None
    )

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Matches Played", team_stats['Matches'])
    t2.metric("Wins",           team_stats['Wins'])
    t3.metric("Losses",         team_stats['Losses'])
    t4.metric("Win Rate",       f"{team_stats['Win Rate %']}%")
    t5.metric("Runs Scored",    f"{team_stats['Runs Scored']:,}")

    st.divider()
    st.subheader("Win Rate by Season")
    season_win_rows = []
    for s in sorted(matches['season'].dropna().unique()):
        s_matches = matches[
            (matches['season'] == s) &
            ((matches['team1'] == selected_team) | (matches['team2'] == selected_team))
        ]
        played = len(s_matches)
        if played == 0:
            continue
        wins = len(s_matches[s_matches['winner'] == selected_team])
        season_win_rows.append({'Season': int(s), 'Matches': played, 'Wins': wins,
                                 'Win %': round(wins / played * 100, 1)})

    if season_win_rows:
        win_trend = pd.DataFrame(season_win_rows)
        fig_win = px.line(win_trend, x='Season', y='Win %',
                          markers=True, color_discrete_sequence=['#9B6440'])
        fig_win.update_traces(marker=dict(color='#6B3F20', size=7))
        fig_win.add_hline(y=50, line_dash='dash', line_color='#DDD0BC',
                          annotation_text='50%',
                          annotation_font_color='#8C6845',
                          annotation_font_family='Poppins')
        fig_win.update_layout(yaxis=dict(range=[0, 105]))
        apply_theme(fig_win, height=300)
        st.plotly_chart(fig_win, use_container_width=True)
        st.dataframe(win_trend.set_index('Season'), use_container_width=True)


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════

st.divider()
st.caption("IPL Analytics  ·  Data: Kaggle IPL 2008 – 2024  ·  Streamlit + Plotly")