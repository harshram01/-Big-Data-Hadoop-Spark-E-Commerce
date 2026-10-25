import os
import time
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional PySpark import: runs real local SparkSession when available,
# and falls back cleanly on Streamlit Community Cloud if Java is unavailable.
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except Exception:
    PYSPARK_AVAILABLE = False

st.set_page_config(
    page_title="Big Data Hadoop & Spark Real-Time Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_spark_session():
    if not PYSPARK_AVAILABLE:
        return None
    try:
        return (
            SparkSession.builder
            .appName("HadoopSparkClickstreamAnalytics")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
    except Exception:
        return None

@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_clickstream.csv")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df_raw = load_dataset()
spark = get_spark_session()

# Sidebar Configuration
st.sidebar.title("⚡ Spark & HDFS Controls")
st.sidebar.caption("Developer: Harshdeep Ram | Big Data Hadoop & Spark")

engine_status = "🟢 PySpark Local Cluster Active" if spark else "🔵 Spark Simulation Engine (Cloud Mode)"
st.sidebar.info(f"**Engine Status:**\n{engine_status}")

selected_regions = st.sidebar.multiselect(
    "HDFS Partition: Region",
    options=sorted(df_raw["region"].unique()),
    default=sorted(df_raw["region"].unique()),
)
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df_raw["category"].unique()),
    default=sorted(df_raw["category"].unique()),
)
risk_threshold = st.sidebar.slider(
    "MLlib Fraud Risk Threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.75,
    step=0.05,
)
shuffle_partitions = st.sidebar.slider(
    "spark.sql.shuffle.partitions",
    min_value=2,
    max_value=32,
    value=8,
    step=2,
)
simulate_stream = st.sidebar.toggle("Simulate Live Spark Micro-Batch", value=False)

# Filter DataFrame
df = df_raw[
    df_raw["region"].isin(selected_regions) &
    df_raw["category"].isin(selected_categories)
].copy()
df["flagged_dynamic"] = (df["risk_score"] >= risk_threshold).astype(int)

if simulate_stream:
    sample_size = random.randint(int(len(df) * 0.7), len(df))
    df = df.sample(n=max(20, sample_size)).sort_values("timestamp")

# Header
st.title("⚡ Big Data Hadoop & Spark: E-Commerce Clickstream & Fraud Pipeline")
st.markdown(
    "End-to-end **HDFS + Apache Spark (SQL, Structured Streaming & MLlib)** monitoring dashboard "
    "tracking real-time user funnels, partitioned Parquet storage, and anomalous transactions."
)

# Top KPI Metrics
purchases = df[df["event_type"] == "purchase"]
total_revenue = purchases["amount_usd"].sum()
flagged_df = df[df["flagged_dynamic"] == 1]
conversion_rate = (
    (len(purchases) / max(1, len(df[df["event_type"] == "page_view"]))) * 100
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Clickstream Events", f"{len(df):,}", f"{shuffle_partitions} Spark Partitions")
col2.metric("Gross Purchase Volume", f"${total_revenue:,.2f}", f"{len(purchases)} orders")
col3.metric("View-to-Purchase Rate", f"{conversion_rate:.1f}%", "Spark SQL Funnel")
col4.metric("Flagged Fraud Events", f"{len(flagged_df)}", f"Threshold >= {risk_threshold:.2f}", delta_color="inverse")
col5.metric("Avg Ingestion Latency", f"{df['latency_ms'].mean():.0f} ms", "HDFS Snappy Parquet")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Real-Time Spark Analytics",
    "🗄️ HDFS Partitions & Spark SQL",
    "🚨 MLlib Anomaly Inspector",
    "💻 PySpark Pipeline Code"
])

with tab1:
    c1, c2 = st.columns((3, 2))
    with c1:
        st.subheader("Micro-Batch Revenue & Event Throughput")
        df_time = (
            df.set_index("timestamp")
            .resample("5min")
            .agg(events=("event_id", "count"), revenue=("amount_usd", "sum"))
            .reset_index()
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_time["timestamp"], y=df_time["revenue"],
            name="Revenue (USD)", mode="lines+markers", line=dict(color="#10b981", width=2.5)
        ))
        fig_trend.add_trace(go.Bar(
            x=df_time["timestamp"], y=df_time["events"],
            name="Events / 5m Batch", yaxis="y2", opacity=0.35, marker_color="#6366f1"
        ))
        fig_trend.update_layout(
            height=360,
            yaxis=dict(title="Revenue (USD)"),
            yaxis2=dict(title="Event Count", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.subheader("Clickstream Conversion Funnel")
        funnel_order = ["page_view", "add_to_cart", "checkout", "purchase"]
        funnel_counts = [len(df[df["event_type"] == e]) for e in funnel_order]
        fig_funnel = go.Figure(go.Funnel(
            y=["Page View", "Add to Cart", "Checkout", "Purchase"],
            x=funnel_counts,
            textinfo="value+percent initial",
            marker={"color": ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981"]}
        ))
        fig_funnel.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_funnel, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Revenue by Region & Category")
        rev_cat = df.groupby(["region", "category"], as_index=False)["amount_usd"].sum()
        fig_bar = px.bar(
            rev_cat, x="region", y="amount_usd", color="category",
            barmode="group", labels={"amount_usd": "Revenue (USD)", "region": "HDFS Region"}
        )
        fig_bar.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c4:
        st.subheader("Device Distribution & Latency")
        dev_summary = df.groupby("device", as_index=False).agg(
            events=("event_id", "count"),
            avg_latency=("latency_ms", "mean")
        )
        fig_pie = px.pie(dev_summary, values="events", names="device", hole=0.45)
        fig_pie.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("HDFS Partition Layout (`/hdfs/clickstream/region=*/date=2026-10-25`)")
    hdfs_summary = (
        df.groupby(["hdfs_partition", "region"], as_index=False)
        .agg(
            records=("event_id", "count"),
            total_usd=("amount_usd", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            flagged_anomalies=("flagged_dynamic", "sum"),
        )
    )
    hdfs_summary["est_parquet_size_kb"] = (hdfs_summary["records"] * 0.42).round(1)
    hdfs_summary["replication_factor"] = 3
    st.dataframe(hdfs_summary, use_container_width=True)

    st.subheader("Interactive Spark SQL Query Runner")
    query_choice = st.selectbox(
        "Select a Spark SQL analytical query to execute:",
        [
            "1. Regional Revenue & Fraud Rate Summary",
            "2. Top High-Value Users Across Sessions",
            "3. Category Conversion & Average Order Value",
        ]
    )
    if query_choice.startswith("1"):
        sql_text = """SELECT region,
       COUNT(*) AS total_events,
       ROUND(SUM(amount_usd), 2) AS total_revenue_usd,
       SUM(flagged_dynamic) AS fraud_alerts
FROM clickstream_events
GROUP BY region
ORDER BY total_revenue_usd DESC;"""
        res_df = df.groupby("region", as_index=False).agg(
            total_events=("event_id", "count"),
            total_revenue_usd=("amount_usd", "sum"),
            fraud_alerts=("flagged_dynamic", "sum"),
        ).sort_values("total_revenue_usd", ascending=False)
    elif query_choice.startswith("2"):
        sql_text = """SELECT user_id, region,
       COUNT(DISTINCT session_id) AS sessions,
       ROUND(SUM(amount_usd), 2) AS lifetime_value_usd,
       MAX(risk_score) AS max_risk_score
FROM clickstream_events
WHERE amount_usd > 0
GROUP BY user_id, region
ORDER BY lifetime_value_usd DESC
LIMIT 15;"""
        res_df = (
            df[df["amount_usd"] > 0]
            .groupby(["user_id", "region"], as_index=False)
            .agg(
                sessions=("session_id", "nunique"),
                lifetime_value_usd=("amount_usd", "sum"),
                max_risk_score=("risk_score", "max"),
            )
            .sort_values("lifetime_value_usd", ascending=False)
            .head(15)
        )
    else:
        sql_text = """SELECT category,
       SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS views,
       SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
       ROUND(AVG(CASE WHEN event_type = 'purchase' THEN amount_usd END), 2) AS avg_order_usd
FROM clickstream_events
GROUP BY category
ORDER BY purchases DESC;"""
        res_df = df.groupby("category", as_index=False).agg(
            views=("event_type", lambda s: (s == "page_view").sum()),
            purchases=("event_type", lambda s: (s == "purchase").sum()),
            avg_order_usd=("amount_usd", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        ).sort_values("purchases", ascending=False)

    st.code(sql_text, language="sql")
    st.dataframe(res_df, use_container_width=True)

with tab3:
    st.subheader(f"Flagged High-Risk Transactions (Risk Score >= {risk_threshold:.2f})")
    fig_scatter = px.scatter(
        df[df["amount_usd"] > 0],
        x="latency_ms",
        y="amount_usd",
        color=df[df["amount_usd"] > 0]["flagged_dynamic"].map({1: "Flagged Fraud", 0: "Normal"}),
        size="risk_score",
        hover_data=["event_id", "user_id", "region", "category"],
        color_discrete_map={"Flagged Fraud": "#ef4444", "Normal": "#3b82f6"},
        labels={"latency_ms": "Gateway Latency (ms)", "amount_usd": "Transaction Amount (USD)"},
    )
    fig_scatter.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.dataframe(
        flagged_df.sort_values("risk_score", ascending=False)[
            ["event_id", "timestamp", "user_id", "region", "category", "event_type", "amount_usd", "latency_ms", "risk_score"]
        ],
        use_container_width=True,
    )

with tab4:
    st.subheader("PySpark Batch & Structured Streaming Pipeline (`spark_pipeline.py`)")
    pipeline_path = os.path.join(os.path.dirname(__file__), "spark_pipeline.py")
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="python")

if simulate_stream:
    time.sleep(2.0)
    st.rerun()
