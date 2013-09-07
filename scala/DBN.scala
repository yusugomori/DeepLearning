import scala.util.Random
import scala.math

class DBN(val N: Int, val n_ins: Int, hidden_layer_sizes: Array[Int], val n_outs: Int, val n_layers: Int, var rng: Random=null) {

  def sigmoid(x: Double): Double = {
    return 1.0 / (1.0 + math.pow(math.E, -x))
  }


  var input_size: Int = 0
  
  val sigmoid_layers: Array[HiddenLayer] = new Array[HiddenLayer](n_layers)
  val rbm_layers: Array[RBM] = new Array[RBM](n_layers)

  if(rng == null) rng = new Random(1234)

  var i: Int = 0
  // construct multi-layer
  for(i <- 0 until n_layers) {
    if(i == 0) {
      input_size = n_ins
    } else {
      input_size = hidden_layer_sizes(i-1)
    }

    // construct sigmoid_layer
    sigmoid_layers(i) = new HiddenLayer(N, input_size, hidden_layer_sizes(i), null, null, rng)

    // construct rbm_layer
    rbm_layers(i) = new RBM(N, input_size, hidden_layer_sizes(i), sigmoid_layers(i).W, sigmoid_layers(i).b, null, rng)

  }

  // layer for output using LogisticRegression
  val log_layer: LogisticRegression = new LogisticRegression(N, hidden_layer_sizes(n_layers-1), n_outs)


  def pretrain(train_X: Array[Array[Int]], lr: Double, k: Int, epochs: Int) {
    var layer_input: Array[Int] = new Array[Int](0)
    var prev_layer_input_size: Int = 0
    var prev_layer_input: Array[Int] = new Array[Int](0)
    
    var i: Int = 0
    var j: Int = 0
    var epoch: Int = 0
    var n: Int = 0
    var l: Int = 0

    for(i <- 0 until n_layers) {  // layer-wise
      for(epoch <- 0 until epochs) {  // training epochs
        for(n <- 0 until N) {  // input x1...xN
          // layer input
          for(l <- 0 to i) {
            if(l == 0) {
              layer_input = new Array[Int](n_ins)
              for(j <- 0 until n_ins) layer_input(j) = train_X(n)(j)

            } else {
              if(l == 1) prev_layer_input_size = n_ins
              else prev_layer_input_size = hidden_layer_sizes(l-2)

              prev_layer_input = new Array[Int](prev_layer_input_size)
              for(j <- 0 until prev_layer_input_size) prev_layer_input(j) = layer_input(j)

              layer_input = new Array[Int](hidden_layer_sizes(l-1))
              sigmoid_layers(l-1).sample_h_given_v(prev_layer_input, layer_input)
            }
          }

          rbm_layers(i).contrastive_divergence(layer_input, lr, k)
        }
      }
    }
  }


  def finetune(train_X: Array[Array[Int]], train_Y: Array[Array[Int]], lr: Double, epochs: Int) {
    var layer_input: Array[Int] = new Array[Int](0)
    var prev_layer_input: Array[Int] = new Array[Int](0)

    var epoch: Int = 0
    var n: Int = 0
    var i: Int = 0
    var j: Int = 0

    for(epoch <- 0 until epochs) {
      for(n <- 0 until N) {
        
        // layer input
        for(i <- 0 until n_layers) {
          if(i == 0) {
            prev_layer_input = new Array[Int](n_ins)
            for(j <- 0 until n_ins) prev_layer_input(j) = train_X(n)(j)
          } else {
            prev_layer_input = new Array[Int](hidden_layer_sizes(i-1))
            for(j <- 0 until hidden_layer_sizes(i-1)) prev_layer_input(j) = layer_input(j)
          }

          layer_input = new Array[Int](hidden_layer_sizes(i))
          sigmoid_layers(i).sample_h_given_v(prev_layer_input, layer_input)
        }

        log_layer.train(layer_input, train_Y(n), lr)
      }
      // lr *= 0.95
    }
  }

  def predict(x: Array[Int], y: Array[Double]) {
    var layer_input: Array[Double] = new Array[Double](0)
    var prev_layer_input: Array[Double] = new Array[Double](n_ins)

    var i: Int = 0
    var j: Int = 0
    var k: Int = 0

    for(j <- 0 until n_ins) prev_layer_input(j) = x(j)
    
    var linear_outoput: Double = 0

    // layer activation
    for(i <- 0 until n_layers) {
      layer_input = new Array[Double](sigmoid_layers(i).n_out)

      for(k <- 0 until sigmoid_layers(i).n_out) {
        linear_outoput = 0.0

        for(j <- 0 until sigmoid_layers(i).n_in) {
          linear_outoput += sigmoid_layers(i).W(k)(j) * prev_layer_input(j)
        }
        linear_outoput += sigmoid_layers(i).b(k)
        layer_input(k) = sigmoid(linear_outoput)
      }
      
      if(i < n_layers-1) {
        prev_layer_input = new Array[Double](sigmoid_layers(i).n_out)
        for(j <- 0 until sigmoid_layers(i).n_out) prev_layer_input(j) = layer_input(j)
      }
    }

    for(i <- 0 until log_layer.n_out) {
      y(i) = 0
      for(j <- 0 until log_layer.n_in) {
        y(i) += log_layer.W(i)(j) * layer_input(j)
      }
      y(i) += log_layer.b(i)
    }

    log_layer.softmax(y)
  }

}


object DBN {
  def test_dbn() {
    val rng: Random = new Random(123)

    val pretrain_lr: Double = 0.1
    val pretraining_epochs: Int = 1000
    val k: Int = 1
    val finetune_lr: Double = 0.1
    val finetune_epochs: Int = 500
    
    val train_N: Int = 6
    val test_N: Int = 4
    val n_ins: Int = 6
    val n_outs: Int = 2
    val hidden_layer_sizes: Array[Int] = Array(3, 3)
    val n_layers = hidden_layer_sizes.length


    // training data
    val train_X: Array[Array[Int]] = Array(
			Array(1, 1, 1, 0, 0, 0),
			Array(1, 0, 1, 0, 0, 0),
		  Array(1, 1, 1, 0, 0, 0),
			Array(0, 0, 1, 1, 1, 0),
			Array(0, 0, 1, 1, 0, 0),
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

    // construct DBN
    val dbn: DBN = new DBN(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng)

		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs);
		
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs);
		
		
		// test data
		val test_X: Array[Array[Int]] = Array(
			Array(1, 1, 0, 0, 0, 0),
			Array(1, 1, 1, 1, 0, 0),
			Array(0, 0, 0, 1, 1, 0),
			Array(0, 0, 1, 1, 1, 0)
		)

    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)

    var i: Int = 0
    var j: Int = 0

    // test
    for(i <- 0 until test_N) {
      dbn.predict(test_X(i), test_Y(i))
      for(j <- 0 until n_outs) {
        print(test_Y(i)(j) + " ")
      }
      println()
    }
    
  }
  

  def main(args: Array[String]) {
    test_dbn()
  }
}
