import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.interpolate import interp1d

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


def _build_pert(a, mode, b, lam):
    """Build a PERT distribution as a scaled Beta."""
    a, b = min(a, b - 0.2), max(b, a + 0.2)
    mode = np.clip(mode, a + 0.01, b - 0.01)
    r = b - a
    alpha = 1 + lam * (mode - a) / r
    beta = 1 + lam * (b - mode) / r
    return stats.beta(a=alpha, b=beta, loc=a, scale=r)


class _MetalogDist:
    """Metalog distribution (bounded or unbounded) using a 3-term fit from quantile data."""

    def __init__(self, p10, p50, p90, bounded_lower=None, bounded_upper=None):
        self._p10 = p10
        self._p50 = p50
        self._p90 = p90
        self._bl = bounded_lower
        self._bu = bounded_upper

        ys = np.array([0.10, 0.50, 0.90])
        qs = np.array([p10, p50, p90])

        if bounded_lower is not None and bounded_upper is not None:
            qs = np.log((qs - bounded_lower) / (bounded_upper - qs))
        elif bounded_lower is not None:
            qs = np.log(qs - bounded_lower)

        logit_ys = np.log(ys / (1 - ys))
        A = np.column_stack([np.ones(3), logit_ys, (ys - 0.5) * logit_ys])
        self._a = np.linalg.solve(A, qs)

        self._y_grid = np.linspace(0.0005, 0.9995, 4000)
        self._x_grid = self._quantile(self._y_grid)
        self._pdf_grid = self._pdf_at_y(self._y_grid)
        self._cdf_interp = interp1d(self._x_grid, self._y_grid, bounds_error=False, fill_value=(0.0, 1.0))
        self._ppf_interp = interp1d(self._y_grid, self._x_grid, bounds_error=False, fill_value=(self._x_grid[0], self._x_grid[-1]))

    def _quantile(self, y):
        logit_y = np.log(y / (1 - y))
        m = self._a[0] + self._a[1] * logit_y + self._a[2] * (y - 0.5) * logit_y
        if self._bl is not None and self._bu is not None:
            return self._bl + (self._bu - self._bl) / (1 + np.exp(-m))
        elif self._bl is not None:
            return self._bl + np.exp(m)
        return m

    def _pdf_at_y(self, y):
        logit_y = np.log(y / (1 - y))
        dlogit = 1.0 / (y * (1 - y))
        dm_dy = self._a[1] * dlogit + self._a[2] * (logit_y + (y - 0.5) * dlogit)

        if self._bl is not None and self._bu is not None:
            exp_neg_m = np.exp(-(self._a[0] + self._a[1] * logit_y + self._a[2] * (y - 0.5) * logit_y))
            dx_dm = (self._bu - self._bl) * exp_neg_m / (1 + exp_neg_m) ** 2
            dx_dy = dx_dm * dm_dy
        elif self._bl is not None:
            m = self._a[0] + self._a[1] * logit_y + self._a[2] * (y - 0.5) * logit_y
            dx_dy = np.exp(m) * dm_dy
        else:
            dx_dy = dm_dy

        with np.errstate(divide="ignore", invalid="ignore"):
            pdf = np.where(dx_dy > 0, 1.0 / dx_dy, 0.0)
        return pdf

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self._x_grid, x)
        idx = np.clip(idx, 1, len(self._y_grid) - 1)
        y_vals = self._y_grid[idx]
        pdf_vals = self._pdf_at_y(y_vals)
        mask = (x >= self._x_grid[0]) & (x <= self._x_grid[-1])
        return np.where(mask, pdf_vals, 0.0)

    def cdf(self, x):
        return np.asarray(self._cdf_interp(x), dtype=float)

    def ppf(self, q):
        return np.asarray(self._ppf_interp(q), dtype=float)

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        u = random_state.uniform(0, 1, size=size)
        return self.ppf(u)

    def stats(self, moments="mvsk"):
        x = self._x_grid
        p = self._pdf_grid
        dx = np.diff(x)
        xm = 0.5 * (x[:-1] + x[1:])
        pm = 0.5 * (p[:-1] + p[1:])
        w = pm * dx

        mean = np.sum(xm * w)
        var = np.sum((xm - mean) ** 2 * w)
        std = np.sqrt(var) if var > 0 else 1e-10
        skew = np.sum(((xm - mean) / std) ** 3 * w)
        kurt = np.sum(((xm - mean) / std) ** 4 * w) - 3.0
        results = []
        for ch in moments:
            if ch == "m":
                results.append(mean)
            elif ch == "v":
                results.append(var)
            elif ch == "s":
                results.append(skew)
            elif ch == "k":
                results.append(kurt)
        return tuple(results)


def _build_metalog(p10, p50, p90, bound_type, lower, upper):
    if bound_type == "Unbounded":
        return _MetalogDist(p10, p50, p90)
    elif bound_type == "Lower-bounded":
        if lower >= p10:
            lower = p10 - 1.0
        return _MetalogDist(p10, p50, p90, bounded_lower=lower)
    else:
        if lower >= p10:
            lower = p10 - 1.0
        if upper <= p90:
            upper = p90 + 1.0
        return _MetalogDist(p10, p50, p90, bounded_lower=lower, bounded_upper=upper)


class _MixtureDist:
    """2-component mixture of continuous distributions."""

    def __init__(self, dist1, dist2, weight1):
        self._d1 = dist1
        self._d2 = dist2
        self._w1 = weight1
        self._w2 = 1.0 - weight1

    def pdf(self, x):
        return self._w1 * self._d1.pdf(x) + self._w2 * self._d2.pdf(x)

    def cdf(self, x):
        return self._w1 * self._d1.cdf(x) + self._w2 * self._d2.cdf(x)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        lo = min(self._d1.ppf(0.0001), self._d2.ppf(0.0001))
        hi = max(self._d1.ppf(0.9999), self._d2.ppf(0.9999))
        if not np.isfinite(lo):
            lo = -100.0
        if not np.isfinite(hi):
            hi = 100.0
        from scipy.optimize import brentq
        results = np.empty_like(q)
        for i, qi in enumerate(q.flat):
            try:
                results.flat[i] = brentq(lambda x: self.cdf(x) - qi, lo, hi, xtol=1e-8)
            except ValueError:
                results.flat[i] = lo if qi < 0.5 else hi
        return results

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        choices = random_state.random(size) < self._w1
        n1 = int(choices.sum())
        n2 = size - n1
        samples = np.empty(size)
        if n1 > 0:
            samples[choices] = self._d1.rvs(size=n1, random_state=random_state)
        if n2 > 0:
            samples[~choices] = self._d2.rvs(size=n2, random_state=random_state)
        return samples

    def stats(self, moments="mvsk"):
        m1, v1 = float(self._d1.stats(moments="mv")[0]), float(self._d1.stats(moments="mv")[1])
        m2, v2 = float(self._d2.stats(moments="mv")[0]), float(self._d2.stats(moments="mv")[1])
        mean = self._w1 * m1 + self._w2 * m2
        var = self._w1 * (v1 + m1**2) + self._w2 * (v2 + m2**2) - mean**2
        std = np.sqrt(var) if var > 0 else 1e-10
        x = np.linspace(self.ppf(np.array([0.001]))[0], self.ppf(np.array([0.999]))[0], 2000)
        pdf_vals = self.pdf(x)
        dx = x[1] - x[0]
        skew = np.sum(((x - mean) / std) ** 3 * pdf_vals) * dx
        kurt = np.sum(((x - mean) / std) ** 4 * pdf_vals) * dx - 3.0
        results = []
        for ch in moments:
            if ch == "m":
                results.append(mean)
            elif ch == "v":
                results.append(var)
            elif ch == "s":
                results.append(skew)
            elif ch == "k":
                results.append(kurt)
        return tuple(results)


DISTRIBUTIONS = {
    "Normal": {
        "kind": "continuous",
        "description": "Symmetric bell curve controlled by mean and standard deviation.",
        "formula": r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
        "params": [
            {"name": "mu", "label": "Mean (mu)", "min": -10.0, "max": 10.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "sigma", "label": "Std Dev (sigma)", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.norm(loc=p["mu"], scale=p["sigma"]),
    },
    "Uniform": {
        "kind": "continuous",
        "description": "Equal probability over [low, high].",
        "formula": r"f(x) = \frac{1}{b - a} \quad \text{for } a \le x \le b",
        "params": [
            {"name": "low", "label": "Low", "min": -10.0, "max": 10.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "high", "label": "High", "min": -9.0, "max": 20.0, "default": 2.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.uniform(loc=min(p["low"], p["high"] - 0.1), scale=max(p["high"] - p["low"], 0.1)),
    },
    "Exponential": {
        "kind": "continuous",
        "description": "Right-skewed waiting-time model with rate lambda.",
        "formula": r"f(x) = \lambda e^{-\lambda x} \quad \text{for } x \ge 0",
        "params": [
            {"name": "lambda", "label": "Rate (lambda)", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.expon(scale=1.0 / p["lambda"]),
    },
    "Beta": {
        "kind": "continuous",
        "description": "Flexible shape on [0, 1] using alpha and beta.",
        "formula": r"f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)} \quad \text{for } 0 \le x \le 1",
        "params": [
            {"name": "alpha", "label": "Alpha", "min": 0.2, "max": 10.0, "default": 2.0, "step": 0.1, "type": "float"},
            {"name": "beta", "label": "Beta", "min": 0.2, "max": 10.0, "default": 5.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.beta(a=p["alpha"], b=p["beta"]),
    },
    "Gamma": {
        "kind": "continuous",
        "description": "Positive-valued skewed family with shape k and scale theta.",
        "formula": r"f(x) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \, \Gamma(k)} \quad \text{for } x \ge 0",
        "params": [
            {"name": "k", "label": "Shape (k)", "min": 0.2, "max": 20.0, "default": 2.0, "step": 0.1, "type": "float"},
            {"name": "theta", "label": "Scale (theta)", "min": 0.1, "max": 10.0, "default": 1.5, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.gamma(a=p["k"], scale=p["theta"]),
    },
    "Lognormal": {
        "kind": "continuous",
        "description": "Positive distribution where log(X) is normal.",
        "formula": r"f(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right) \quad \text{for } x > 0",
        "params": [
            {"name": "mu", "label": "Log Mean (mu)", "min": -2.0, "max": 3.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "sigma", "label": "Log Std Dev (sigma)", "min": 0.1, "max": 2.5, "default": 0.5, "step": 0.05, "type": "float"},
        ],
        "build": lambda p: stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"])),
    },
    "Poisson": {
        "kind": "discrete",
        "description": "Count model with rate mu.",
        "formula": r"P(X=k) = \frac{\mu^k e^{-\mu}}{k!} \quad \text{for } k = 0, 1, 2, \ldots",
        "params": [
            {"name": "mu", "label": "Rate (mu)", "min": 0.1, "max": 40.0, "default": 6.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.poisson(mu=p["mu"]),
    },
    "Binomial": {
        "kind": "discrete",
        "description": "Number of successes in n independent Bernoulli trials.",
        "formula": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k = 0, 1, \ldots, n",
        "params": [
            {"name": "n", "label": "Trials (n)", "min": 1, "max": 200, "default": 20, "step": 1, "type": "int"},
            {"name": "p", "label": "Success Prob (p)", "min": 0.01, "max": 0.99, "default": 0.5, "step": 0.01, "type": "float"},
        ],
        "build": lambda p: stats.binom(n=int(p["n"]), p=p["p"]),
    },
    "Triangular": {
        "kind": "continuous",
        "description": "Simple three-point distribution defined by min, mode, and max.",
        "formula": r"f(x) = \begin{cases} \frac{2(x-a)}{(b-a)(c-a)} & a \le x \le c \\ \frac{2(b-x)}{(b-a)(b-c)} & c < x \le b \end{cases}",
        "params": [
            {"name": "low", "label": "Min (a)", "min": -100.0, "max": 100.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "mode", "label": "Mode (c)", "min": -100.0, "max": 200.0, "default": 5.0, "step": 0.1, "type": "float"},
            {"name": "high", "label": "Max (b)", "min": -100.0, "max": 500.0, "default": 10.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: stats.triang(
            c=np.clip((p["mode"] - min(p["low"], p["high"] - 0.1)) / max(max(p["high"], p["low"] + 0.2) - min(p["low"], p["high"] - 0.1), 0.1), 0, 1),
            loc=min(p["low"], p["high"] - 0.1),
            scale=max(max(p["high"], p["low"] + 0.2) - min(p["low"], p["high"] - 0.1), 0.1),
        ),
    },
    "PERT": {
        "kind": "continuous",
        "description": "Three-point estimate (min, mode, max) used in project planning. Built on a Beta distribution scaled to [min, max] with a shape parameter lambda that controls how strongly the mode dominates.",
        "formula": r"\alpha = 1 + \lambda \frac{m - a}{b - a}, \quad \beta = 1 + \lambda \frac{b - m}{b - a}, \quad f(x) = \frac{(x-a)^{\alpha-1}(b-x)^{\beta-1}}{(b-a)^{\alpha+\beta-1} B(\alpha,\beta)}",
        "params": [
            {"name": "min", "label": "Low", "min": -100.0, "max": 100.0, "default": 1.0, "step": 0.1, "type": "float"},
            {"name": "mode", "label": "Most Likely (mode)", "min": -100.0, "max": 200.0, "default": 4.0, "step": 0.1, "type": "float"},
            {"name": "max", "label": "High", "min": -100.0, "max": 500.0, "default": 7.0, "step": 0.1, "type": "float"},
            {"name": "lambda", "label": "Shape (lambda)", "min": 1.0, "max": 100.0, "default": 4.0, "step": 0.5, "type": "float"},
        ],
        "build": lambda p: _build_pert(p["min"], p["mode"], p["max"], p["lambda"]),
    },
    "Metalog": {
        "kind": "continuous",
        "description": "Flexible quantile-parameterized distribution. Specify your 10th, 50th, and 90th percentile estimates directly. Supports unbounded, semi-bounded, and bounded variants.",
        "formula": r"M(y) = a_1 + a_2 \ln\frac{y}{1-y} + a_3 (y - \tfrac{1}{2}) \ln\frac{y}{1-y}",
        "params": [
            {"name": "p10", "label": "10th Percentile", "min": -100.0, "max": 200.0, "default": 2.0, "step": 0.1, "type": "float"},
            {"name": "p50", "label": "50th Percentile (median)", "min": -100.0, "max": 500.0, "default": 5.0, "step": 0.1, "type": "float"},
            {"name": "p90", "label": "90th Percentile", "min": -100.0, "max": 1000.0, "default": 12.0, "step": 0.1, "type": "float"},
            {"name": "bound_type", "label": "Bound Type", "options": ["Unbounded", "Lower-bounded", "Bounded"], "default": "Unbounded", "type": "select"},
            {"name": "lower", "label": "Lower Bound", "min": -200.0, "max": 200.0, "default": 0.0, "step": 0.1, "type": "float"},
            {"name": "upper", "label": "Upper Bound", "min": -100.0, "max": 2000.0, "default": 20.0, "step": 0.1, "type": "float"},
        ],
        "build": lambda p: _build_metalog(p["p10"], p["p50"], p["p90"], p["bound_type"], p["lower"], p["upper"]),
    },
    "Mixture (2-component)": {
        "kind": "mixture",
        "description": "Weighted combination of two distributions. The PDF is w*f1(x) + (1-w)*f2(x).",
        "formula": r"f(x) = w \cdot f_1(x) + (1 - w) \cdot f_2(x)",
    },
}


COMPONENT_DISTS = {k: v for k, v in DISTRIBUTIONS.items() if v["kind"] != "mixture"}


def _render_params(param_defs, prefix, use_number_input):
    params = {}
    for p in param_defs:
        key = f"{prefix}_{p['name']}"
        if p["type"] == "select":
            params[p["name"]] = st.selectbox(p["label"], p["options"], index=p["options"].index(p["default"]), key=key)
        elif use_number_input:
            if p["type"] == "int":
                params[p["name"]] = st.number_input(p["label"], value=int(p["default"]), step=int(p["step"]), key=key)
            else:
                params[p["name"]] = st.number_input(p["label"], value=float(p["default"]), step=float(p["step"]), format="%.2f", key=key)
        else:
            if p["type"] == "int":
                params[p["name"]] = st.slider(p["label"], min_value=int(p["min"]), max_value=int(p["max"]), value=int(p["default"]), step=int(p["step"]), key=key)
            else:
                params[p["name"]] = st.slider(p["label"], min_value=float(p["min"]), max_value=float(p["max"]), value=float(p["default"]), step=float(p["step"]), key=key)
    return params


st.title("Interactive Probability Distribution Explorer")
st.write(
    "Select a distribution, tune its parameters with sliders, and see how the shape and sample behavior change."
)

with st.sidebar:
    st.header("Controls")
    dist_name = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
    use_number_input = st.toggle("Type-in parameter values", value=False)
    show_samples = st.toggle("Show sampled data", value=True)
    sample_size = st.slider("Sample size", 200, 20000, 3000, 200)
    seed = st.number_input("Random seed", value=42, step=1)

config = DISTRIBUTIONS[dist_name]
st.caption(config["description"])
if "formula" in config:
    st.latex(config["formula"])

if config["kind"] == "mixture":
    weight = st.sidebar.slider("Weight of Component 1", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    cont_names = [k for k, v in COMPONENT_DISTS.items() if v["kind"] == "continuous"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Component 1")
        name1 = st.selectbox("Distribution", cont_names, index=0, key="mix_dist1")
        cfg1 = COMPONENT_DISTS[name1]
        st.caption(cfg1["description"])
        params1 = _render_params(cfg1["params"], "c1", use_number_input)
        dist1 = cfg1["build"](params1)

    with col2:
        st.subheader("Component 2")
        name2 = st.selectbox("Distribution", cont_names, index=cont_names.index("Exponential") if "Exponential" in cont_names else 2, key="mix_dist2")
        cfg2 = COMPONENT_DISTS[name2]
        st.caption(cfg2["description"])
        params2 = _render_params(cfg2["params"], "c2", use_number_input)
        dist2 = cfg2["build"](params2)

    dist = _MixtureDist(dist1, dist2, weight)
    rng = np.random.default_rng(int(seed))

    x1 = _continuous_x(dist1)
    x2 = _continuous_x(dist2)
    lo = min(x1[0], x2[0])
    hi = max(x1[-1], x2[-1])
    x = np.linspace(lo, hi, 800)
    y = dist.pdf(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Mixture PDF", line=dict(width=3, color="#005f73")))
    fig.add_trace(go.Scatter(x=x, y=weight * dist1.pdf(x), mode="lines", name=f"w*{name1}", line=dict(width=1.5, dash="dash", color="#0a9396")))
    fig.add_trace(go.Scatter(x=x, y=(1 - weight) * dist2.pdf(x), mode="lines", name=f"(1-w)*{name2}", line=dict(width=1.5, dash="dash", color="#94d2bd")))

    if show_samples:
        samples = dist.rvs(size=sample_size, random_state=rng)
        fig.add_trace(
            go.Histogram(
                x=samples,
                histnorm="probability density",
                nbinsx=80,
                opacity=0.35,
                name="Sample histogram",
                marker_color="#ee9b00",
            )
        )

    fig.update_layout(
        title="Mixture Distribution",
        xaxis_title="x",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        height=520,
    )

else:
    params = {}
    for p in config["params"]:
        if p["type"] == "select":
            params[p["name"]] = st.sidebar.selectbox(p["label"], p["options"], index=p["options"].index(p["default"]))
        elif use_number_input:
            if p["type"] == "int":
                params[p["name"]] = st.sidebar.number_input(
                    p["label"],
                    value=int(p["default"]),
                    step=int(p["step"]),
                )
            else:
                params[p["name"]] = st.sidebar.number_input(
                    p["label"],
                    value=float(p["default"]),
                    step=float(p["step"]),
                    format="%.2f",
                )
        else:
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

st.plotly_chart(fig, width="stretch")

with st.expander("Current parameters"):
    if config["kind"] == "mixture":
        st.json({"weight": weight, "component_1": {name1: params1}, "component_2": {name2: params2}})
    else:
        st.json(params)
