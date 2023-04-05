# https://github.com/nicolearugay
# Uploaded 4/4/2023

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
import pandas as pd

# Initialize SparkSession
spark = SparkSession.builder.appName("CustomerChurnPrediction").config("spark.jars", "/path/to/jar/file").getOrCreate()

# Load data from CSV file and limit number of rows
churn_data = spark.read.csv("cust_churn.csv", header=True, inferSchema=True).limit(7000)

# Select columns to keep 
churn_data = churn_data.select(col("tenure"), col("MonthlyCharges"), col("TotalCharges"), col("Churn"))

# Convert the TotalCharges column to a numeric datatype
churn_data = churn_data.withColumn("TotalCharges", col("TotalCharges").cast("double"))

# Convert Churn column to numeric data type
indexer = StringIndexer(inputCol="Churn", outputCol="ChurnIndex")
indexed = indexer.fit(churn_data).transform(churn_data)

# Select relevant columns and create feature vector
assembler = VectorAssembler(inputCols=["tenure", "MonthlyCharges", "TotalCharges"], outputCol="features", handleInvalid="skip")
feature_vector = assembler.transform(indexed).select("features", "ChurnIndex")

# Split data into training and test sets
(training_data, test_data) = feature_vector.randomSplit([0.7, 0.3], seed=42)

# Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="ChurnIndex", maxIter=10)
model = lr.fit(training_data)

# Make predictions on test data
predictions = model.transform(test_data)

# Evaluate model performance
accuracy = predictions.filter(predictions.ChurnIndex == predictions.prediction).count() / float(test_data.count())

print("Model accuracy:", accuracy)

# Convert data types to match JDBC data types
pred_clean = predictions.withColumn("features", col("features").cast("string"))
pred_clean = pred_clean.withColumn("ChurnIndex", col("ChurnIndex").cast("integer"))
pred_clean = pred_clean.withColumn("rawPrediction", col("rawPrediction").cast("string"))
pred_clean = pred_clean.withColumn("probability", col("probability").cast("string"))
pred_clean = pred_clean.withColumn("prediction", col("prediction").cast("integer"))

# Save predictions to Postgres database
pred_clean.write.format("jdbc").options(
    url="jdbc:postgresql://localhost:5432/insert username",
    driver="org.postgresql.Driver",
    dbtable="predictions",
    user="insert username here",
    password="insert password here"
).mode("overwrite").save()

# Save predictions to CSV 
pandas_df = pred_clean.toPandas()
pandas_df.to_csv('predictions.csv', index=False)

# Stop SparkSession
spark.stop()