# app.py — Streamlit Time Series Analysis & Dashboard
# --------------------------------------------------
# How to run locally:
#   1) Install deps:  pip install streamlit pandas numpy plotly statsmodels
#   2) Save this file as app.py
#   3) Run:           streamlit run app.py


import os
import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: try seasonal decomposition if statsmodels is present
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------

def try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    """Attempt to parse a series as datetimes. Returns parsed series or None."""
    try:
        parsed = pd.to_datetime(series, errors="raise", utc=False, infer_datetime_format=True)
        return parsed
    except Exception:
        return None


def infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "timestamp", "datetime", "date", "time", "Date", "Datetime", "DateTime",
        "Timestamp", "created_at", "recorded_at",
    ]
    # place obvious matches first
    for c in candidates:
        if c in df.columns:
            parsed = try_parse_datetime(df[c])
            if parsed is not None:
                df[c] = parsed
                return c
    # fallback: try any column that can parse as datetime
    for c in df.columns:
        if df[c].dtype == object:
            parsed = try_parse_datetime(df[c])
            if parsed is not None:
                df[c] = parsed
                return c
    return None


def infer_value_column(df: pd.DataFrame, datetime_col: str) -> Optional[str]:
    """Pick a likely numeric value column representing consumption/usage."""
    value_like = [
        "consumption", "kwh", "kw", "usage", "load", "power", "energy",
        "value", "reading", "consumed", "demand"
    ]
    lower_map = {c: c.lower() for c in df.columns if c != datetime_col}
    # priority by name
    for c in df.columns:
        if c == datetime_col:
            continue
        lc = c.lower()
        if any(tok in lc for tok in value_like) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # else first numeric column not datetime
    numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def load_dataset(uploaded: Optional[io.BytesIO]) -> Tuple[pd.DataFrame, str, str]:
    """Load dataset from uploaded file or local powerconsumption.csv.
    Returns (df, datetime_col, value_col). Raises st.error on failure.
    """
    # 1) try uploaded
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            try:
                df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                st.stop()
    else:
        # 2) try local file
        for candidate in ["powerconsumption.csv", "PowerConsumption.csv", "power_consumption.csv"]:
            if os.path.exists(candidate):
                try:
                    df = pd.read_csv(candidate)
                    break
                except Exception as e:
                    st.warning(f"Found {candidate} but failed to read as CSV: {e}")
        if df is None:
            st.info("Upload a CSV/XLSX using the sidebar to get started.")
            st.stop()

    # clean empty columns
    df = df.dropna(how="all", axis=1)

    # Infer datetime column
    dt_col = infer_datetime_column(df)
    if dt_col is None:
        # ask user to select
        st.warning("Couldn't detect a datetime column. Please select one in the sidebar.")
        with st.sidebar:
            dt_col = st.selectbox("Select datetime column", options=list(df.columns))
        parsed = try_parse_datetime(df[dt_col])
        if parsed is None:
            st.error("Selected column couldn't be parsed as datetime. Please upload a file with a valid datetime column.")
            st.stop()
        df[dt_col] = parsed

    # Ensure sorted by datetime
    df = df.sort_values(dt_col)

    # Infer value column
    val_col = infer_value_column(df, dt_col)
    if val_col is None:
        with st.sidebar:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != dt_col]
            if not numeric_cols:
                st.error("No numeric columns found. Please upload a file with a numeric consumption column.")
                st.stop()
            val_col = st.selectbox("Select value/consumption column", options=numeric_cols)

    return df, dt_col, val_col


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], utc=False)
    df["date"] = df[dt_col].dt.date
    df["year"] = df[dt_col].dt.year
    df["month"] = df[dt_col].dt.month
    df["day"] = df[dt_col].dt.day
    df["hour"] = df[dt_col].dt.hour
    df["dow"] = df[dt_col].dt.dayofweek  # 0=Mon
    df["week"] = df[dt_col].dt.isocalendar().week.astype(int)
    return df


def resample_series(df: pd.DataFrame, dt_col: str, val_col: str, rule: str, agg: str = "sum") -> pd.DataFrame:
    s = df.set_index(dt_col)[val_col]
    if agg == "sum":
        r = s.resample(rule).sum()
    elif agg == "mean":
        r = s.resample(rule).mean()
    elif agg == "max":
        r = s.resample(rule).max()
    elif agg == "min":
        r = s.resample(rule).min()
    else:
        r = s.resample(rule).sum()
    return r.reset_index().rename(columns={val_col: f"{val_col}_{agg}"})


def anomaly_zscore(x: pd.Series, window: int = 24, threshold: float = 3.0) -> pd.DataFrame:
    roll_mean = x.rolling(window, min_periods=max(3, window//3)).mean()
    roll_std = x.rolling(window, min_periods=max(3, window//3)).std(ddof=0)
    z = (x - roll_mean) / (roll_std.replace(0, np.nan))
    return pd.DataFrame({"value": x, "z": z, "is_anom": (np.abs(z) >= threshold)})


# -----------------------------
# Sidebar — data, filters, options
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])  # optional
    st.caption("If nothing is uploaded, the app will try 'powerconsumption.csv' in the working folder.")
    resample_rule = st.selectbox("Resample frequency", options=["H", "D", "W", "M"], index=0, help="Aggregate data for charts & KPIs")
    agg = st.selectbox("Aggregation", options=["sum", "mean", "max", "min"], index=0)
    roll_window = st.number_input("Rolling window (periods)", min_value=1, max_value=365, value=24)
    show_anoms = st.checkbox("Detect anomalies (z-score)", value=False)
    z_thresh = st.slider("Anomaly z-threshold", 1.5, 6.0, value=3.0, step=0.1)

# Load & prep
df_raw, DT, VAL = load_dataset(uploaded)
df = preprocess(df_raw, DT)

# Date range filter
min_date, max_date = df[DT].min().date(), df[DT].max().date()
with st.sidebar:
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:  # single date selected
        start_date, end_date = min_date, date_range

mask = (df[DT] >= pd.to_datetime(start_date)) & (df[DT] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
df = df.loc[mask].copy()

st.title("📊 Time Series Analysis — Energy/Power Consumption")
st.caption(f"Detected datetime column: **{DT}** · value column: **{VAL}** · Records: {len(df):,}")

# -----------------------------
# KPIs row
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)

# Resampled for KPIs
resampled = resample_series(df, DT, VAL, rule=resample_rule, agg=agg)
resampled = resampled.rename(columns={f"{VAL}_{agg}": "y"}).dropna()

if not resampled.empty:
    total = resampled["y"].sum()
    avg = resampled["y"].mean()
    peak = resampled["y"].max()
    trough = resampled["y"].min()
    delta = resampled["y"].iloc[-1] - resampled["y"].iloc[-2] if len(resampled) >= 2 else 0
else:
    total = avg = peak = trough = delta = 0

col1.metric("Total (resampled)", f"{total:,.2f}")
col2.metric("Average (resampled)", f"{avg:,.2f}")
col3.metric("Peak (resampled)", f"{peak:,.2f}")
col4.metric("Lowest (resampled)", f"{trough:,.2f}")
col5.metric("Last Δ vs Prev", f"{delta:,.2f}", delta=f"{delta:,.2f}")

# -----------------------------
# Tabs: Overview | Profiles | Decomposition | Data
# -----------------------------
t1, t2, t3, t4 = st.tabs(["Overview", "Profiles", "Decomposition", "Data"])

with t1:
    st.subheader("Trend & Rolling Average")
    df_line = df[[DT, VAL]].dropna().sort_values(DT)
    df_line["rolling"] = df_line[VAL].rolling(int(roll_window), min_periods=max(3, int(roll_window//3))).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_line[DT], y=df_line[VAL], mode="lines", name=VAL))
    fig.add_trace(go.Scatter(x=df_line[DT], y=df_line["rolling"], mode="lines", name=f"Rolling({int(roll_window)})"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, legend_title_text="Series")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resampled Aggregation")
    fig2 = px.bar(resampled, x=DT, y="y",
                  labels={DT: "Period", "y": f"{VAL} ({agg})"})
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
    st.plotly_chart(fig2, use_container_width=True)

    if show_anoms:
        st.subheader("Anomaly Detection (Z-score on original frequency)")
        s = df.set_index(DT)[VAL].astype(float)
        res = anomaly_zscore(s, window=int(roll_window), threshold=float(z_thresh))
        res = res.reset_index().rename(columns={"index": DT})
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=res[DT], y=res["value"], mode="lines", name=VAL))
        anom = res[res["is_anom"]]
        if not anom.empty:
            fig3.add_trace(go.Scatter(
                x=anom[DT], y=anom["value"], mode="markers", name="Anomaly", marker=dict(size=9, symbol="x")
            ))
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
        st.plotly_chart(fig3, use_container_width=True)

with t2:
    st.subheader("Daily & Weekly Profiles")
    # Day of week profile
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    by_dow = df.groupby("dow")[VAL].agg(["mean","sum"]).reset_index()
    by_dow["dow_name"] = by_dow["dow"].map(dow_map)

    fig4 = px.bar(by_dow.sort_values("dow"), x="dow_name", y="mean", labels={"dow_name":"Day of Week","mean":f"Avg {VAL}"})
    fig4.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig4, use_container_width=True)

    # Hour of day profile
    by_hour = df.groupby("hour")[VAL].agg(["mean","sum"]).reset_index()
    fig5 = px.line(by_hour, x="hour", y="mean", markers=True, labels={"hour":"Hour of Day","mean":f"Avg {VAL}"})
    fig5.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig5, use_container_width=True)

    # Heatmap: Day-of-week vs Hour
    st.subheader("Heatmap — Hour vs Day of Week (Avg)")
    pivot = df.pivot_table(index="dow", columns="hour", values=VAL, aggfunc="mean")
    fig6 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=[dow_map[d] for d in pivot.index]))
    fig6.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, xaxis_title="Hour", yaxis_title="Day of Week")
    st.plotly_chart(fig6, use_container_width=True)

with t3:
    st.subheader("Seasonal Decomposition (STL)")
    if not HAS_STATSMODELS:
        st.info("Install statsmodels to enable STL decomposition:  pip install statsmodels")
    else:
        # Resample to a regular frequency suitable for STL
        rule_map = {"H": 24, "D": 7, "W": 52, "M": 12}
        rule_for_stl = st.selectbox("Frequency for STL", options=["H","D","W","M"], index=1)
        y = df.set_index(DT)[VAL].astype(float).resample(rule_for_stl).mean().interpolate()
        if len(y) < rule_map[rule_for_stl] * 2:
            st.warning("Not enough data for STL at this frequency. Choose a lower frequency or provide longer series.")
        else:
            st.caption("STL = Seasonal-Trend decomposition using LOESS")
            stl = STL(y, robust=True)
            result = stl.fit()
            comp = pd.DataFrame({
                DT: y.index,
                "observed": y.values,
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
            })
            for col, title in [("observed","Observed"),("trend","Trend"),("seasonal","Seasonal"),("resid","Residual")]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=comp[DT], y=comp[col], mode="lines", name=col))
                fig.update_layout(title=title, height=280, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)

with t4:
    st.subheader("Preview & Download Cleaned Data")
    st.dataframe(df.head(100), use_container_width=True)
    # Provide a cleaned, filtered CSV download
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "Download filtered data as CSV",
        buf.getvalue().encode("utf-8"),
        file_name="timeseries_filtered.csv",
        mime="text/csv",
    )

# -----------------------------
# Extra: Quick Insights
# -----------------------------
with st.expander("🔎 Quick Insights (auto-generated)"):
    insights = []
    if not resampled.empty:
        # Trend direction using simple last vs median
        last = resampled["y"].iloc[-1]
        median = resampled["y"].median()
        if last > median * 1.1:
            insights.append("Recent period is above typical levels (last > 110% of median).")
        elif last < median * 0.9:
            insights.append("Recent period is below typical levels (last < 90% of median).")
        # Peak period
        peak_idx = resampled["y"].idxmax()
        peak_ts = resampled.loc[peak_idx, DT]
        insights.append(f"Peak {agg} occurred around {peak_ts}.")
        # Volatility via coefficient of variation
        cv = resampled["y"].std(ddof=0) / (resampled["y"].mean() + 1e-9)
        insights.append(f"Volatility (CV) ≈ {cv:.2f}.")
    if insights:
        for s in insights:
            st.write("• ", s)
    else:
        st.write("No insights available — try changing resample frequency or date range.")

# -----------------------------
# Footer
# -----------------------------
st.caption("Built with ❤️ in Streamlit · Supports CSV/XLSX · Works with hourly/daily/weekly/monthly data")
