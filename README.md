# 🏏 IPL Analytics Dashboard

A full-stack cricket analytics web app built with Python, Streamlit and Plotly — covering IPL seasons 2008–2024 with interactive dashboards and machine learning predictions.

🔗 **Live App:** [ipl-stats-analysis-system.streamlit.app](https://ipl-stats-analysis-system.streamlit.app)

---

## Features

### 📊 Analytics (6 Tabs)

| Tab | What it shows |
|-----|--------------|
| **Overview** | Top performers, IPL Honours table (all champions & runners-up), season runs trend |
| **Batters** | Career stats, phase analysis, dismissal breakdown, venue performance, season trends |
| **Bowlers** | Wickets, economy, dot ball %, phase-wise bowling, season trends |
| **Compare** | Side-by-side batter comparison with radar chart |
| **Head-to-Head** | Batter vs bowler matchup stats with verdict |
| **Teams** | Win rate by season, team summary stats |

### 🤖 Machine Learning (Predictions Tab)

| Model | Algorithm | What it does |
|-------|-----------|-------------|
| **Win Probability Predictor** | Logistic Regression | Predicts win % based on powerplay score, wickets, run rate & venue |
| **Match Score Predictor** | Random Forest (200 trees) | Predicts final innings score from first 10 overs data with confidence range |
| **Best XI Selector** | Composite Scoring Model | Selects optimal XI using Batting Index + Bowling Index per season/team |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly |
| ML | scikit-learn (LogisticRegression, RandomForestRegressor) |
| Data | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## ML Model Details

### Win Probability — Logistic Regression
**Features:** Powerplay runs, powerplay wickets, current run rate, total wickets lost, batting team (encoded), venue (encoded)

### Score Predictor — Random Forest
**Features:** Runs at 10 overs, wickets at 10 overs, batting team (encoded), venue (encoded)
**Output:** Predicted score + conservative/aggressive range (15th–85th percentile across trees)

### Best XI — Composite Scoring
```
Batting Index  = (Runs x SR/100) + (4s x 0.5) + (6s x 1.0) - (Dot% x 0.3)
Bowling Index  = (Wickets x 20) - (Economy x 3) + (DotBall% x 0.5)
Overall Score  = Batting Index + Bowling Index
```
Role balance enforced: ~5 Batters · ~3 All-Rounders · ~3 Bowlers

---

## Project Structure

```
ipl-analytics/
│
├── app.py              # Main Streamlit dashboard
├── analytics.py        # Analytics & stats functions
├── data_loader.py      # Data loading & cleaning pipeline
├── ml_models.py        # All three ML models
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml     # Streamlit config
│
└── data/               # Add CSV files here (gitignored)
    ├── matches.csv
    └── deliveries.csv
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/aryapandey-git/ipl-analytics.git
cd ipl-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add data
# Download from Kaggle:
# https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
mkdir data
# Place matches.csv and deliveries.csv inside data/

# 4. Run
streamlit run app.py
```

---

## Dataset

- **Source:** [Kaggle — IPL Complete Dataset 2008–2024](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **Matches:** 1,095 games
- **Deliveries:** 260,920 balls
- **Players:** 673 batters · 530 bowlers
- **Seasons:** 2008 – 2024

---

## Design

- **Fonts:** Playfair Display (headings) · Poppins (body)
- **Palette:** Earthy browns — caramel sidebar `#C4956A` · parchment background `#FAF8F4` · sienna accents `#9B6440`

---

## Author

**Arya Pandey**
[GitHub: aryapandey-git](https://github.com/aryapandey-git)