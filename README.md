# ⚡ Big Data Hadoop & Spark: E-Commerce Clickstream & Fraud Pipeline

**Developer:** Harshdeep Ram  
**Certification:** Introduction to Big Data Hadoop and Spark Developer (Simplilearn SkillUp)

---

## 🎯 Project Overview
An end-to-end **Big Data Hadoop & Apache Spark** pipeline and interactive **Streamlit** dashboard that processes e-commerce clickstream logs, partitions data into **HDFS Snappy Parquet** format by region and date, executes **Spark RDD & Spark SQL** funnel aggregations, and scores anomalous transactions using **Spark MLlib**.

### Key Technical Features
1. **Hadoop HDFS Storage Layout**: Organizes raw clickstream events into partitioned Parquet directories (`/hdfs/clickstream/region=*/date=*`) with Snappy compression and 3x block replication metadata.
2. **Spark Core (RDDs) & Spark SQL**: Demonstrates low-level `map`, `filter`, and `reduceByKey` RDD transformations alongside DataFrame window functions (`Window.partitionBy`) and Spark SQL queries.
3. **Spark MLlib Anomaly Detection**: Standardizes transaction amount, gateway latency, and risk indicators using `VectorAssembler` and `StandardScaler`, clustering high-risk fraud anomalies via `KMeans`.
4. **Live Streamlit Web App (`app.py`)**: Interactive dashboard with real-time micro-batch simulation, HDFS partition explorer, Spark SQL runner, and MLlib scatter plots. Automatically runs on a local `SparkSession` when Java/Spark is installed and falls back seamlessly on **Streamlit Community Cloud**.

---

## 🚀 Quickstart (Run Locally in 60 Seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the live Streamlit dashboard
streamlit run app.py

# 3. (Optional) Run the standalone PySpark batch + MLlib job
python spark_pipeline.py
```

---

## ☁️ Deploy Live to Streamlit Community Cloud
1. Push this folder (`app.py`, `spark_pipeline.py`, `requirements.txt`, `data/sample_clickstream.csv`) to a public **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy!** — your live Streamlit URL will be ready to add to your resume and LinkedIn profile.
