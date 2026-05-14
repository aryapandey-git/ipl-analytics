# 🏏 IPL AI Analytics Dashboard

Advanced cricket analytics powered by real Kaggle data + Claude AI.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Kaggle IPL dataset
Place your CSV files inside a `data/` folder:
```
data/
  matches.csv
  deliveries.csv
```
Download from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Add Claude API key (optional but recommended)
- Get your key at: https://console.anthropic.com
- Paste it in the sidebar when the app opens
- Enables AI insights for players, bowlers, comparisons, and head-to-head

---

## Features

| Tab | Features |
|---|---|
| Overview | Top scorers, wicket-takers, season trends |
| Batters | Full profile, phase analysis, dismissals, venue stats, season trend |
| Bowlers | Full profile, phase bowling, dot ball gauge, season trend |
| Compare | Side-by-side metrics, radar chart, phase comparison |
| Head-to-Head | Batter vs Bowler stats, auto verdict |
| Teams | Win rate, season trends, match records |

## Bug fixes vs original ChatGPT code

- **Average formula fixed**: now `runs / dismissals` (not runs / innings)
- **Bowling overs fixed**: now counts only legal balls (original counted wides as balls)
- **Wickets exclude run-outs**: bowler wickets only count credited dismissals
- **Season normalisation**: handles both "2020" and "IPL 2020" formats
- **Comparison chart fixed**: metrics plotted on separate axes (original put Strike Rate and 4s on same axis)
