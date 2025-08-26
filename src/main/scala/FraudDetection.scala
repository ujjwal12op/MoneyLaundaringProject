import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

object FraudDetection {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Fraud Detection Project")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/raw/bank_transactions_data_2.csv")

    data.show(10, truncate = false)
    println(s"Total rows: ${data.count()}")
    data.printSchema()

    // Add isFraud column
    val dfWithLabel = data.withColumn("isFraud", when(col("TransactionAmount") > 10000, 1).otherwise(0))

    // Summary statistics
    dfWithLabel.describe().show()

    // Null check
    dfWithLabel.select(dfWithLabel.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)):_*).show()

    // Avg amount by fraud
    dfWithLabel.groupBy("isFraud").agg(avg("TransactionAmount")).show()

    // ========================
    // Fraud Analysis Section
    // ========================

    // 1. Count fraudulent vs non-fraudulent transactions
    val fraudCount = dfWithLabel.groupBy("isFraud").count()
    println("Fraudulent vs Non-Fraudulent Transactions:")
    fraudCount.show()

    // 2. Total amount in fraudulent transactions
    val fraudAmount = dfWithLabel.filter(col("isFraud") === 1)
      .agg(sum("TransactionAmount").alias("TotalFraudAmount"))
    println("Total Fraudulent Transaction Amount:")
    fraudAmount.show()

    // 3. Top 10 highest fraud transactions
    val topFrauds = dfWithLabel.filter(col("isFraud") === 1)
      .orderBy(desc("TransactionAmount"))
      .limit(10)
    println("Top 10 Highest Fraudulent Transactions:")
    topFrauds.show(false)

    // 4. Fraud percentage
    val totalTxns = dfWithLabel.count()
    val fraudTxns = dfWithLabel.filter(col("isFraud") === 1).count()
    val fraudPercentage = (fraudTxns.toDouble / totalTxns.toDouble) * 100
    println(s"Fraudulent Transactions: $fraudTxns / $totalTxns (~${fraudPercentage}%)")

    // ========================
    // ML Pipeline
    // ========================

    val indexer = new StringIndexer().setInputCol("TransactionType").setOutputCol("typeIndex")
    val encoder = new OneHotEncoder().setInputCol("typeIndex").setOutputCol("typeVec")

    val featureCols = Array("TransactionAmount", "AccountBalance", "typeVec")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val finalDf = dfWithLabel.withColumn("label", col("isFraud").cast("double"))

    val pipeline = new Pipeline().setStages(Array(indexer, encoder, assembler))
    val processedDf = pipeline.fit(finalDf).transform(finalDf)

    val Array(train, test) = processedDf.randomSplit(Array(0.7, 0.3), seed = 42)

    val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
    val lrModel = lr.fit(train)
    val lrPred = lrModel.transform(test)

    val rf = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setNumTrees(50)
    val rfModel = rf.fit(train)
    val rfPred = rfModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    println(s"Logistic Regression F1 = ${evaluator.evaluate(lrPred)}")
    println(s"Random Forest F1 = ${evaluator.evaluate(rfPred)}")

    spark.stop()
  }
}
