"""
Big Data Hadoop & Spark Developer Project
Author: Harshdeep Ram
Pipeline: HDFS Clickstream Ingestion, RDD & Spark SQL Transformations, and MLlib Fraud Detection
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans


def create_spark_session(app_name: str = "HadoopSparkClickstreamPipeline") -> SparkSession:
    """Initializes SparkSession configured for local or YARN/HDFS execution."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def run_rdd_aggregations(spark: SparkSession, input_csv: str):
    """Demonstrates low-level Spark RDD map/filter/reduceByKey operations."""
    sc = spark.sparkContext
    lines = sc.textFile(input_csv)
    header = lines.first()
    records = lines.filter(lambda row: row != header).map(lambda row: row.split(","))

    # Pair RDD: (region, amount_usd) for purchase events
    region_sales_rdd = (
        records
        .filter(lambda cols: cols[6] == "purchase")
        .map(lambda cols: (cols[4], float(cols[8])))
        .reduceByKey(lambda a, b: a + b)
    )
    print("=== RDD Purchase Revenue by Region ===")
    for region, total in region_sales_rdd.collect():
        print(f"  {region:<18}: ${total:,.2f}")


def run_dataframe_and_sql_pipeline(spark: SparkSession, input_csv: str, output_parquet_dir: str):
    """Ingests raw CSV, computes session window metrics, writes partitioned HDFS Parquet, and runs Spark SQL."""
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input_csv)
        .withColumn("event_ts", F.to_timestamp("timestamp"))
        .withColumn("event_date", F.to_date("event_ts"))
    )

    # Window specification: cumulative spend per user session
    session_window = Window.partitionBy("session_id").orderBy("event_ts")
    df_enriched = (
        df.withColumn("session_running_usd", F.round(F.sum("amount_usd").over(session_window), 2))
          .withColumn("session_event_seq", F.row_number().over(session_window))
    )

    # Write partitioned Snappy Parquet (HDFS compatible path)
    (
        df_enriched.write
        .mode("overwrite")
        .partitionBy("region", "event_date")
        .parquet(output_parquet_dir)
    )
    print(f"Saved partitioned Parquet dataset to: {output_parquet_dir}")

    # Register Temporary View for Spark SQL
    df_enriched.createOrReplaceTempView("clickstream_events")

    funnel_sql = spark.sql("""
        SELECT region,
               COUNT(*) AS total_events,
               SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS page_views,
               SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS cart_adds,
               SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
               ROUND(SUM(amount_usd), 2) AS total_revenue_usd,
               SUM(is_fraud_flagged) AS flagged_anomalies
        FROM clickstream_events
        GROUP BY region
        ORDER BY total_revenue_usd DESC
    """)
    print("=== Spark SQL Regional Funnel & Fraud Summary ===")
    funnel_sql.show(truncate=False)
    return df_enriched


def run_mllib_anomaly_clustering(df):
    """Applies Spark MLlib KMeans clustering on transaction amount and latency to isolate anomalies."""
    tx_df = df.filter(F.col("amount_usd") > 0)
    assembler = VectorAssembler(
        inputCols=["amount_usd", "latency_ms", "risk_score"],
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    assembled = assembler.transform(tx_df)
    scaler_model = scaler.fit(assembled)
    scaled_df = scaler_model.transform(assembled)

    kmeans = KMeans(featuresCol="features", predictionCol="cluster_id", k=3, seed=42)
    model = kmeans.fit(scaled_df)
    clustered = model.transform(scaled_df)

    print("=== Spark MLlib Cluster Distribution ===")
    clustered.groupBy("cluster_id").agg(
        F.count("*").alias("tx_count"),
        F.round(F.avg("amount_usd"), 2).alias("avg_amount_usd"),
        F.round(F.avg("latency_ms"), 1).alias("avg_latency_ms"),
        F.round(F.avg("risk_score"), 3).alias("avg_risk_score"),
    ).orderBy("cluster_id").show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "data", "sample_clickstream.csv")
    parquet_out = os.path.join(base_dir, "output_hdfs_parquet")

    spark_session = create_spark_session()
    spark_session.sparkContext.setLogLevel("WARN")

    run_rdd_aggregations(spark_session, csv_file)
    enriched_df = run_dataframe_and_sql_pipeline(spark_session, csv_file, parquet_out)
    run_mllib_anomaly_clustering(enriched_df)

    spark_session.stop()
