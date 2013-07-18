// $ scalac RBM.scala
// $ scala RBM

import scala.util.Random
import scala.math

class RBM(val N: Int, val n_visible: Int, val n_hidden: Int,
          _W: Array[Array[Double]]=null, _hbias: Array[Double]=null, _vbias: Array[Double]=null,
          var rng: Random=null) {
  
  var W: Array[Array[Double]] = Array.ofDim[Double](n_hidden, n_visible)
  var hbias: Array[Double] = new Array[Double](n_hidden)
  var vbias: Array[Double] = new Array[Double](n_visible)


  if(rng == null) rng = new Random(1234)

  if(_W == null) {
    var i: Int = 0
    var j: Int = 0

    val a: Double = 1 / n_visible
    for(i <- 0 until n_hidden)
      for(j <- 0 until n_visible)
        W(i)(j) = uniform(-a, a)

  } else {
    W = _W
  }

  if(_hbias == null) {
    var i: Int = 0
    for(i <- 0 until n_hidden) hbias(i) = 0
  } else {
    hbias = _hbias
  }

  if(_vbias == null) {
    var i: Int = 0
    for(i <- 0 until n_visible) vbias(i) = 0
  } else {
    vbias = _vbias
  }


  def uniform(min: Double, max: Double): Double = rng.nextDouble() * (max - min) + min
  def binomial(n: Int, p: Double): Int = {
    if(p < 0 || p > 1) return 0
    
    var c: Int = 0
    var r: Double = 0

    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    c
  }

  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))

  
  def contrastive_divergence(input: Array[Int], lr: Double, k: Int) {
    val ph_mean: Array[Double] = new Array[Double](n_hidden)
    val ph_sample: Array[Int] = new Array[Int](n_hidden)
    val nv_means: Array[Double] = new Array[Double](n_visible)
    val nv_samples: Array[Int] = new Array[Int](n_visible)
    val nh_means: Array[Double] = new Array[Double](n_hidden)
    val nh_samples: Array[Int] = new Array[Int](n_hidden)

    /* CD-k */
    sample_h_given_v(input, ph_mean, ph_sample)

    var step: Int = 0
    for(step <- 0 until k) {
      if(step == 0) {
        gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples)
      } else {
        gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples)
      }
    }

    var i: Int = 0
    var j: Int = 0
    for(i <- 0 until n_hidden) {
      for(j <- 0 until n_visible) {
        // W(i)(j) += lr * (ph_sample(i) * input(j) - nh_means(i) * nv_samples(j)) / N
        W(i)(j) += lr * (ph_mean(i) * input(j) - nh_means(i) * nv_samples(j)) / N
      }
      hbias(i) += lr * (ph_sample(i) - nh_means(i)) / N
    }

    for(i <- 0 until n_visible) {
      vbias(i) += lr * (input(i) - nv_samples(i)) / N
    }
  }


  def sample_h_given_v(v0_sample: Array[Int], mean: Array[Double], sample: Array[Int]) {
    var i: Int = 0
    for(i <- 0 until n_hidden) {
      mean(i) = propup(v0_sample, W(i), hbias(i))
      sample(i) = binomial(1, mean(i))
    }
  }

  def sample_v_given_h(h0_sample: Array[Int], mean: Array[Double], sample: Array[Int]) {
    var i: Int = 0
    for(i <- 0 until n_visible) {
      mean(i) = propdown(h0_sample, i, vbias(i))
      sample(i) = binomial(1, mean(i))
    }
  }

  def propup(v: Array[Int], w: Array[Double], b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_visible) {
      pre_sigmoid_activation += w(j) * v(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  def propdown(h: Array[Int], i: Int, b: Double): Double = {
    var pre_sigmoid_activation: Double = 0
    var j: Int = 0
    for(j <- 0 until n_hidden) {
      pre_sigmoid_activation += W(j)(i) * h(j)
    }
    pre_sigmoid_activation += b
    sigmoid(pre_sigmoid_activation)
  }

  def gibbs_hvh(h0_sample: Array[Int], nv_means: Array[Double], nv_samples: Array[Int], nh_means: Array[Double], nh_samples: Array[Int]) {
    sample_v_given_h(h0_sample, nv_means, nv_samples)
    sample_h_given_v(nv_samples, nh_means, nh_samples)
  }


  def reconstruct(v: Array[Int], reconstructed_v: Array[Double]) {
    val h: Array[Double] = new Array[Double](n_hidden)
    var pre_sigmoid_activation: Double = 0
    
    var i: Int = 0
    var j: Int = 0
    
    for(i <- 0 until n_hidden) {
      h(i) = propup(v, W(i), hbias(i))
    }

    for(i <- 0 until n_visible) {
      pre_sigmoid_activation = 0
      for(j <- 0 until n_hidden) {
        pre_sigmoid_activation += W(j)(i) * h(j)
      }
      pre_sigmoid_activation += vbias(i)
      reconstructed_v(i) = sigmoid(pre_sigmoid_activation)
    }
  }
}


object RBM {
  def test_rbm() {
    val rng: Random = new Random(123)

    var learning_rate: Double = 0.1
    val training_epochs: Int = 1000
    val k: Int = 1

    val train_N: Int = 6;
    val test_N: Int = 2
    val n_visible: Int = 6
    val n_hidden: Int = 3

    val train_X: Array[Array[Int]] = Array(
      Array(1, 1, 1, 0, 0, 0),
      Array(1, 0, 1, 0, 0, 0),
      Array(1, 1, 1, 0, 0, 0),
      Array(0, 0, 1, 1, 1, 0),
      Array(0, 0, 1, 0, 1, 0),
      Array(0, 0, 1, 1, 1, 0)
    )


    val rbm: RBM = new RBM(train_N, n_visible, n_hidden, rng=rng)

    var i: Int = 0
    var j: Int = 0

    // train
    var epoch: Int = 0
    for(epoch <- 0 until training_epochs) {
      for(i <- 0 until train_N) {
        rbm.contrastive_divergence(train_X(i), learning_rate, k)
      }
    }

    // test data
    val test_X: Array[Array[Int]] = Array(
      Array(1, 1, 0, 0, 0, 0),
      Array(0, 0, 0, 1, 1, 0)
    )

    val reconstructed_X: Array[Array[Double]] = Array.ofDim[Double](test_N, n_visible)
    for(i <- 0 until test_N) {
      rbm.reconstruct(test_X(i), reconstructed_X(i))
      for(j <- 0 until n_visible) {
        printf("%.5f ", reconstructed_X(i)(j))
      }
      println()
    }

  }

  def main(args: Array[String]) {
    test_rbm()
  }

}
