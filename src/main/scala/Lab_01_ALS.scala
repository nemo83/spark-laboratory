import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{DataFrame, SparkSession}

object Lab_01_ALS {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("Lab_01_ALS")
      .getOrCreate()

    case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

    val ratings: DataFrame = spark
      .read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)

      .csv("src/main/resources/ml-latest-small/ratings.csv")

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    ratings.columns.foreach(println)

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setRank(50)
      .setMaxIter(10)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    val predictions = model.transform(test)

    predictions.columns.foreach(println)
    //    userId
    //    movieId
    //    rating
    //    timestamp
    //    prediction

    import spark.implicits._

    predictions.foreach {
      row =>
        for(i <- 0 until row.size) {
          if (row.get(i) == null) {
            println(s"row($row) has a null value! index($i) ${row.get(i)}")
          }
        }
    }


    val MSE_10 = predictions.take(10).map {
      case row => {
        val rating = row.getAs[Double]("rating")
        val prediction = row.getAs[Float]("prediction")
        val bla = math.pow(rating - prediction, 2)

        println(s"bla: $bla")

        bla
      }
    }.reduce(_ + _) / 10

    val MSE = predictions.map {
      case row => {
        val rating = row.getAs[Double]("rating")
        val prediction = row.getAs[Float]("prediction")
        math.pow(rating - prediction, 2)
      }
    }.reduce(_ + _) / predictions.count()

    val RMSE = math.sqrt(MSE)

    println(s"MSE_10($MSE_10) MSE($MSE) and RMSE($RMSE)")

    //    val evaluator = new RegressionEvaluator()
    //      .setMetricName("rmse")
    //      .setLabelCol("rating")
    //      .setPredictionCol("prediction")


    //    val rmse = evaluator.evaluate(spark.createDataset(predictions.take(10)))

    //    println(s"Root-mean-square error = $rmse")

  }

}
