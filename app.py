import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Probability Distribution Explorer", layout="wide")


def _continuous_x(dist):
    lo, hi = dist.ppf([0.001, 0.999])
    if not np.isfinite(lo):
        lo = dist.ppf(0.01)
    if not np.isfinite(hi):
        hi = dist.ppf(0.99)
    if not np.isfinite(lo):
        lo = -10.0
    if not np.isfinite(hi):
        hi = 10.0
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return np.linspace(lo, hi, 600)


def _discrete_x(dist):
    lo = int(np.floor(dist.ppf(0.001)))
    hi = int(np.ceil(dist.ppf(0.999)))
    if not np.isfinite(lo):
        lo = 0
    if not np.isfinite(hi):
        hi = 30
    lo = max(lo, 0)
    hi = max(hi, lo + 1)
    if hi - lo > 150:
        mean = float(dist.mean())
        lo = max(0, int(mean - 75))
        hi = int(mean + 75)
    return np.arange(lo, hi + 1)


DISTRIBUTIONS = {
    "Normal": {
        "kind": "continuous",
        "description": "Symmetric bell curve controlled by mean and standard deviation.",
        "params": [
            {"name": "mu", "label": "Mean (mu)", "min": -10.0, "max": 10.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "sigma", "label": "Std Dev (sigma)", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.norm(loc=p["mu"], scale=p["sigma"]),
    },
    "Uniform": {
        "kind": "continuous",
        "description": "Equal probability over [low, high].",
        "params": [
            {"name": "low", "label": "Low", "min": -10.0, "max": 10.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "high", "label": "High", "min": -9.0, "max": 20.0, "default": 2.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.uniform(loc=min(p["low"], p["high"] - 0.1), scale=max(p["high"] - p["low"], 0.1)),
    },
    "Exponential": {
        "kind": "continuous",
        "description": "Right-skewed waiting-time model with rate lambda.",
        "params": [
            {"name": "lambda", "label": "Rate (lambda)", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.expon(scale=1.0 / p["lambda"]),
    },
    "Beta": {
        "kind": "continuous",
        "description": "Flexible shape on [0, 1] using alpha and beta.",
        "params": [
            {"name": "alpha", "label": "Alpha", "min": 0.2, "max": 10.0, "default": 2.0, "step": 0.1, "type": "float"},
            {"name": "beta", "label": "Beta", "min": 0.2, "max": 10.0, "default": 5.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.beta(a=p["alpha"], b=p["beta"]),
    },
    "Gamma": {
        "kind": "continuous",
        "description": "Positive-valued skewed family with shape k and scale theta.",
        "params": [
            {"name": "k", "label": "Shape (k)", "min": 0.2, "max": 20.0, "default": 2.0, "step": 0.1, "type": "float"},
            {"name": "theta", "label": "Scale (theta)", "min": 0.1, "max": 10.0, "default": 1.5, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.gamma(a=p["k"], scale=p["theta"]),
    },
    "Lognormal": {
        "kind": "continuous",
        "description": "Positive distribution where log(X) is normal.",
        "params": [
            {"name": "mu", "label": "Log Mean (mu)", "min": -2.0, "max": 3.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "sigma", "label": "Log Std Dev (sigma)", "min": 0.1, "max": 2.5, "default": 0.5, "step": 0.05, "type": "float"},
        ],
        "build": lambda p: stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"])),
    },
    "Poisson": {
        "kind": "discrete",
        "description": "Count model with rate mu.",
        "params": [
            {"name": "mu", "label": "Rate (mu)", "min": 0.1, "max": 40.0, "default": 6.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.poisson(mu=p["mu"]),
    },
    "Binomial": {
        "kind": "discrete",
        "description": "Number of successes in n independent Bernoulli trials.",
        "params": [
            {"name": "n", "label": "Trials (n)", "min": 1, "max": 200, "default": 20, "step": 1, "type": "int"},
            {"name": "p", "label": "Success Prob (p)", "min": 0.01, "max": 0.99, "default": 0.5, "step": 0.01, "type": "float"},
        ],
        "build": lambda p: stats.binom(n=int(p["n"]), p=p["p"]),
    },
}


st.title("Interactive Probability Distribution Explorer")
st.write(
    "Select a distribution, tune its parameters with sliders, and see how the shape and sample behavior change."
)

with st.sidebar:
    st.header("Controls")
    dist_name = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
    show_samples = st.toggle("Show sampled data", value=True)
    sample_size = st.slider("Sample size", 200, 20000, 3000, 200)
    seed = st.number_input("Random seed", value=42, step=1)

config = DISTRIBUTIONS[dist_name]
st.caption(config["description"])

params = {}
for p in config["params"]:
    if p["type"] == "int":
        params[p["name"]] = st.sidebar.slider(
            p["label"],
            min_value=int(p["min"]),
            max_value=int(p["max"]),
            value=int(p["default"]),
            step=int(p["step"]),
        )
    else:
        params[p["name"]] = st.sidebar.slider(
            p["label"],
            min_value=float(p["min"]),
            max_value=float(p["max"]),
            value=float(p["default"]),
            step=float(p["step"]),
        )

dist = config["build"](params)
rng = np.random.default_rng(int(seed))

if config["kind"] == "continuous":
    x = _continuous_x(dist)
    y = dist.pdf(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PDF", line=dict(width=3, color="#005f73")))

    if show_samples:
        samples = dist.rvs(size=sample_size, random_state=rng)
        fig.add_trace(
            go.Histogram(
                x=samples,
                histnorm="probability density",
                nbinsx=60,
                opacity=0.45,
                name="Sample histogram",
                marker_color="#ee9b00",
            )
        )

    fig.update_layout(
        title=f"{dist_name} Distribution",
        xaxis_title="x",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        height=520,
    )
else:
    x = _discrete_x(dist)
    y = dist.pmf(x)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name="PMF", marker_color="#005f73", opacity=0.85))

    if show_samples:
        samples = dist.rvs(size=sample_size, random_state=rng)
        values, counts = np.unique(samples, return_counts=True)
        rel = counts / sample_size
        fig.add_trace(go.Bar(x=values, y=rel, name="Sample frequency", marker_color="#ee9b00", opacity=0.45))

    fig.update_layout(
        title=f"{dist_name} Distribution",
        xaxis_title="x",
        yaxis_title="Probability",
        barmode="overlay",
        template="plotly_white",
        height=520,
    )

mean, var, skew, kurt = dist.stats(moments="mvsk")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean", f"{float(mean):.4f}")
c2.metric("Variance", f"{float(var):.4f}")
c3.metric("Skewness", f"{float(skew):.4f}")
c4.metric("Excess Kurtosis", f"{float(kurt):.4f}")

st.plotly_chart(fig, use_container_width=True)

with st.expander("Current parameters"):
    st.json(params)
