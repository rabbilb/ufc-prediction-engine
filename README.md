# UFC Fight Prediction Engine

An end-to-end machine learning pipeline that scrapes UFC fight data, engineers leakage-safe features, and predicts fight outcomes through an XGBoost classifier.
The engine is displayed on a Streamlit application, allowing users to pit two fighters against each other, with win probabilities and a SHAP waterfall plot displayed
for analysis.

**Streamlit App Link:** https://ybrmh2rf4gbxakb9gbz3bs.streamlit.app/

## Overview
Mixed Martial Arts (MMA) is notoriously one of the hardest sports to predict due it being such a high variance sport. The diversity of skillsets, 
propensity for quick, violent moments, and hidden factors like weight cuts and mental state make MMA a coin flip problem. Knowing this, the goal was to
build a statistical model that could pull as much signal from historical fight data to make the most informed decision for each bout.

To tackle this problem, the project scrapes data from `ufcstats.com` and engineers historical features using only information available prior to the fight it describes.
The result is an XGBoost classifier that reaches **65.2% test accuracy**, against a 55% baseline that simply picks the higher ELO fighter. 

## Tech Stack

- **Scraping and Automation:** Playwright, BeautifulSoup, Github Actions

- **Data Processing and Feature Engineering:** NumPy, pandas, PyArrow

- **Modeling:** XGBoost, scikit-learn

- **App:** Streamlit

## Data Acquisition and Scraping

- Scrapes historical fight-by-fight and round-by-round statistics from `ufcstats.com`, using Playwright for page rendering and BeautifulSoup for parsing and page traversal
- Each rerun only fetches new bouts and fighters
- Fighter career stats never reach the model to prevent data leakage (can't predict a fight by using data that occurred chronologically after!)
- Automated weekly through a Github Actions workflow

## Feature Engineering

Each feature is computed from a fighter's history prior to the fight it describes. For each bout, `shift(1)` is used on rolling calculations to prevent fights from seeing their own result.
Ex: Islam Makhachev vs Jack Della Maddalena. For this fight, Makhachev's fight history would end at his victory over Renato Moicano, while Della Maddalena's history would end at his victory over Belal Muhammad.

**Elo:** Elo was used to assess the relative quality of each fighter (K = 40, seed = 1500). 
**Peak Elo:** Career high pre-fight rating.

Differential features were used to assess relative advantage between two fighters, denoted as "A" and "B". Each feature enters the model as a difference between A and B since the gap between metrics is what matters rather than standalone values. Assignment to either A or B is randomized to prevent the model from learning a positional bias.

**Opponent-adjusted metrics:** Striking defense and takedown statistics weighted by opponent quality

**Trend features:** Career-level minus last-5-fight values per skill. Recency windows are fight-count based rather than time-based

**Durability:** Method-of-loss breakdown (KO/TKO, submission, decision) and fights since the last finish loss

**Mileage:** Cumulative damage absorbed &mdash; career and per-fight significant strikes, takedowns, knockdowns conceded

**Physical and activity:** Age at fight, height, reach, stance

**Things to note:** 
1. Reach history is missing for ~24% of fighters, imputed via linear regression on height.
2. No contests and draws are excluded from the training label but included for feature computation.

## Modeling

XGBoost, chronological 70/15/15 split, tuned with `RandomizedSearchCV`.

## Results

| Model | Test Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|
| Elo-only baseline | 0.5580 | — | — | — |
| **XGBoost** | **0.6517** | 0.6379 | 0.6957 | 0.2233 |

Training rows restricted to 2012+, older fights still contribute to every modern row's Elo and career features. 

## App

Fight predictor and ELO leaderboard. The fight predictor allows users to pit two fighters against each other filtered by weight class, displays the win probability split and SHAP waterfall contributing to the decision.

## Setup

```bash
git clone https://github.com/rabbilb/ufc-prediction-engine.git
cd ufc-prediction-engine
pip install -r requirements.txt
playwright install chromium
```

Run the pipeline in order:

```bash
python cleaner.py         
python ufc_modeling.py    
streamlit run app.py      
```
