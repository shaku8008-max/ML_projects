# ML_projects
I use machine learning to predict outcomes of matches. 

Disclaimer: The entirity of this project is vibe coded!

EPL Match Outcome Predictor (ML Pipeline)

An end-to-end machine learning pipeline for predicting English Premier League (EPL) match outcomes using historical match data, engineered features (form + Elo), and a calibrated Random Forest model.

The project supports:

Automated data updates

Feature rebuilding

Model training

Interactive CLI predictions

Current Scope

League: English Premier League (EPL)

Seasons: 2025/26 (auto-updated as matches are played)

Prediction type:
Binary classification

1 → Home win

0 → Away win or Draw

Odds-based evaluation is intentionally skipped (no odds data yet).

Data Source (Free)

football-data.co.uk (EPL CSV feed)
Used for match results (date, teams, goals).

The pipeline automatically downloads the latest data when run.

Features Used

Elo rating (with home advantage)

Recent form

Goals for / against (last 5 & last 10 matches)

Goal difference

Derived differences

Home vs Away comparisons

Model
Random Forest Classifier
Includes:
Threshold analysis
Confidence tiers (A / B / C)
Probability calibration (Platt scaling + Isotonic, evaluated)

ML_Projects/
│
├── data/
│   ├── raw/                # Auto-downloaded EPL match data
│   └── processed/          # Feature-engineered datasets
│
├── src/
│   ├── app/
│   │   ├── predict_cli.py  # Interactive predictor
│   │   └── run_update.py   # One-click update pipeline
│   │
│   ├── injest/             # (typo kept intentionally)
│   │   └── download_epl.py # Downloads latest EPL CSV
│   │
│   ├── features/           # Feature engineering scripts
│   └── model/              # Training & evaluation scripts
│
├── README.md
└── .gitignore

Running the predict_cli updates the csv for EPL and retrains the data. This is important for team winstreaks which is a big factor in predicting the outcome for matches. Prompts the user to input team neames and then gives the prediction.

Reproducibility & Transparency

This repository includes:
- Raw EPL match data
- Feature-engineered datasets
- Calibration outputs

All results shown can be reproduced by running the provided scripts.
