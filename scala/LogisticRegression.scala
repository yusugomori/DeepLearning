// $ scalac LogisticRegression.scala
// $ scala LogisticRegression

import scala.math

class LogisticRegression(val N: Int, val n_in: Int, val n_out: Int) {

  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in)
  val b: Array[Double] = new Array[Double](n_out)

  def train(x: Array[Int], y: Array[Int], lr: Double) {
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    val dy: Array[Double] = new Array[Double](n_out)

    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 until n_in) {
        p_y_given_x(i) += W(i)(j) * x(j)
      }
      p_y_given_x(i) += b(i)
    }
    softmax(p_y_given_x)

    for(i <- 0 until n_out) {
      dy(i) = y(i) - p_y_given_x(i)

      for(j <- 0 until n_in) {
        W(i)(j) += lr * dy(i) * x(j) / N
      }
      b(i) += lr * dy(i) / N
    }
  }


  def softmax(x: Array[Double]) {
    var max: Double = 0.0
    var sum: Double = 0.0

    var i: Int = 0
    for(i <- 0 until n_out) if(max < x(i)) max = x(i)

    for(i <- 0 until n_out) {
      x(i) = math.exp(x(i) - max)
      sum += x(i)
    }

    for(i <- 0 until n_out) x(i) /= sum
  }


  def predict(x: Array[Int], y: Array[Double]) {
    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_out) {
      y(i) = 0
      for(j <- 0 until n_in) {
        y(i) += W(i)(j) * x(j)
      }
      y(i) += b(i)
    }
    softmax(y)
  }

}


object LogisticRegression {
  def test_lr() {
    val learning_rate: Double = 0.1
    val n_epochs: Int = 500

    val train_N: Int = 6
    val test_N: Int = 2
    val n_in: Int = 6
    val n_out: Int = 2

    val train_X: Array[Array[Int]] = Array(
      Array(1, 1, 1, 0, 0, 0),
      Array(1, 0, 1, 0, 0, 0),
      Array(1, 1, 1, 0, 0, 0),
      Array(0, 0, 1, 1, 1, 0),
      Array(0, 0, 1, 0, 1, 0),
      Array(0, 0, 1, 1, 1, 0)
    )

    val train_Y: Array[Array[Int]] = Array(
      Array(1, 0),
      Array(1, 0),
      Array(1, 0),
      Array(0, 1),
      Array(0, 1),
      Array(0, 1)
    )

    // construct
    val classifier = new LogisticRegression(train_N, n_in, n_out)

    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      for(i <- 0 until train_N) {
        classifier.train(train_X(i), train_Y(i), learning_rate)
      }
      // learning_rate *= 0.95
    }

    // test data
    val test_X: Array[Array[Int]] = Array(
      Array(1, 0, 1, 0, 0, 0),
      Array(0, 0, 1, 1, 1, 0)
    )

    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_out)

    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      classifier.predict(test_X(i), test_Y(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
    }
  }

  def main(args: Array[String]) {
    test_lr()
  }

}
