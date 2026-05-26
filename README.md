# Probability Distribution Dashboard

Interactive dashboard for exploring how probability distribution parameters change shape and sampled behavior.

## Features
- Select from multiple distributions (continuous and discrete)
- Dynamic sliders or type-in inputs for distribution-specific parameters
- Live PDF/PMF visualization with LaTeX formulas
- Optional sample overlay to compare theory vs simulation
- Summary statistics (mean, variance, skewness, kurtosis)
- Mixture (2-component) mode: combine any two continuous distributions with a mixing weight and see component overlays

## Distributions included
- Normal
- Uniform
- Exponential
- Beta
- Gamma
- Lognormal
- Poisson
- Binomial
- Triangular
- PERT
- Metalog (quantile-parameterized; unbounded, semi-bounded, or bounded)
- Mixture (2-component)

## Run
```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).
