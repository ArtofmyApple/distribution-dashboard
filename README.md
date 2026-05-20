# Probability Distribution Dashboard

Interactive dashboard for exploring how probability distribution parameters change shape and sampled behavior.

## Features
- Select from multiple distributions (continuous and discrete)
- Dynamic sliders for distribution-specific parameters
- Live PDF/PMF visualization
- Optional sample overlay to compare theory vs simulation
- Summary statistics (mean, variance, skewness, kurtosis)

## Distributions included
- Normal
- Uniform
- Exponential
- Beta
- Gamma
- Lognormal
- Poisson
- Binomial

## Setup
```bash
cd "/Users/blau/OneDrive - RAND Corporation/Projects/distribution-dashboard"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).
