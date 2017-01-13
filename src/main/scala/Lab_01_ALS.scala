import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import readers.MovieLensLoader


/**
  * Trying to get to the bottom of the NaN in predictions.
  */
object Lab_01_ALS {

  private val movieLensLoader = MovieLensLoader()

  def main(args: Array[String]): Unit = {

    implicit val spark = SparkSession
      .builder()
      .master("local")
      .appName("Lab_01_ALS")
      .getOrCreate()

    import spark.implicits._

    val ratings: DataFrame = movieLensLoader.load()
    // userId
    // movieId
    // rating
    // timestamp

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))


    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setRank(50)
      .setMaxIter(10)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training)
    // userId
    // movieId
    // rating
    // timestamp
    // prediction

    // Evaluate the model by computing the RMSE on the test data
    val predictions = model.transform(test)


    val (userIds, movieIds) = test.map {
      case Row(userId: Int, movieId: Int, _, _) => {
        (userId, movieId)
      }
    }.collect().unzip

    val distinctUserIds = userIds.distinct
    val distinctMovieIds = movieIds.distinct

    val predictionAndErrors = predictions.map {
      case Row(userId: Int, movieId: Int, rating: Double, _, prediction: Float) => {
        val error = math.pow(rating - prediction, 2)
        (userId, movieId, rating, prediction, error)
      }
    }.collect()

    val numberOfNan = predictionAndErrors.count {
      case (userId, movieId, rating, prediction, error) => {
        val result = prediction.isNaN

        if (result) {
          if (!distinctMovieIds.contains(movieId) || !distinctUserIds.contains(userId)) {
            println(s"userId $userId or movieId $movieId NOT CONTAINED in training $prediction")
          } else {
            println(s"userId $userId and movieId $movieId contained in training $prediction")
          }
        }

        result
      }
    }

    println(s"${predictionAndErrors.size} Predictions and $numberOfNan NaN(s)")

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions.filter(row => !row.getAs[Float]("prediction").isNaN))

    println(s"Root-mean-square error = $rmse")

  }

}
