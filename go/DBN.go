package main

import (
	"fmt"
	"math/rand"
	u "./utils"
	H "./HiddenLayer"
	R "./RBM"
	L "./LogisticRegression"
)

type DBN struct {
	N int
	n_ins int
	hidden_layer_sizes []int
	n_outs int
	n_layers int
	sigmoid_layers []H.HiddenLayer
	rbm_layers []R.RBM
	log_layer L.LogisticRegression
}


func DBN__construct(this *DBN, N int, n_ins int, hidden_layer_sizes []int, n_outs int, n_layers int) {
	var input_size int
	
	this.N = N
	this.n_ins = n_ins
	this.hidden_layer_sizes = hidden_layer_sizes
	this.n_outs = n_outs
	this.n_layers = n_layers

	this.sigmoid_layers = make([]H.HiddenLayer, n_layers)
	this.rbm_layers = make([]R.RBM, n_layers)
	
	// construct multi-layer
	for i := 0; i < n_layers; i++ {
		if i == 0 {
			input_size = n_ins
		} else {
			input_size = hidden_layer_sizes[i-1]
		}

		// construct sigmoid_layer
		H.HiddenLayer__construct(&(this.sigmoid_layers[i]), N, input_size, hidden_layer_sizes[i], nil, nil)

		// construct rbm_layer
		R.RBM__construct(&(this.rbm_layers[i]), N, input_size, hidden_layer_sizes[i], this.sigmoid_layers[i].W, this.sigmoid_layers[i].B, nil)
	}

	// layer for output using LogisticRegression
	L.LogisticRegression__construct(&(this.log_layer), N, hidden_layer_sizes[n_layers-1], n_outs)
}

func DBN_pretrain(this *DBN, train_X [][]int, lr float64, k int, epochs int){
	var (
		layer_input []int
		prev_layer_input_size int
		prev_layer_input []int
	)


	for i := 0; i < this.n_layers; i++ {	// layer-wise
		for epoch := 0; epoch < epochs; epoch++ {	 // training epochs
			for n := 0; n < this.N; n++ {	 // input x1...xN

				// layer input
				for l := 0; l <= i; l++ {
					if l == 0 {
						layer_input = make([]int, this.n_ins)
						for j := 0; j < this.n_ins; j++ { layer_input[j] = train_X[n][j] }
					} else {
						if l == 1 {
							prev_layer_input_size = this.n_ins
						} else {
							prev_layer_input_size = this.hidden_layer_sizes[l-2]
						}

						prev_layer_input = make([]int, prev_layer_input_size)
						for j := 0; j < prev_layer_input_size; j++ { prev_layer_input[j] = layer_input[j] }

						layer_input = make([]int, this.hidden_layer_sizes[l-1])

						H.HiddenLayer_sample_h_given_v(&(this.sigmoid_layers[l-1]), prev_layer_input, layer_input)
					}
				}

				R.RBM_contrastive_divergence(&(this.rbm_layers[i]), layer_input, lr, k)
			}
		}
	}
}

func DBN_finetune(this *DBN, train_X [][]int, train_Y [][]int, lr float64, epochs int) {
	var (
		layer_input []int
		prev_layer_input []int
	)

	for epoch := 0; epoch < epochs; epoch++ {
		for n := 0; n < this.N; n++ {	 // input x1...xN

			// layer input
			for i := 0; i < this.n_layers; i++ {
				if i == 0 {
					prev_layer_input = make([]int, this.n_ins)
					for j := 0; j < this.n_ins; j++ { prev_layer_input[j] = train_X[n][j] }
				} else {
					prev_layer_input = make([]int, this.hidden_layer_sizes[i-1])
					for j:= 0; j < this.hidden_layer_sizes[i-1]; j++ { prev_layer_input[j] = layer_input[j] }
				}

				layer_input = make([]int, this.hidden_layer_sizes[i])
				H.HiddenLayer_sample_h_given_v(&(this.sigmoid_layers[i]), prev_layer_input, layer_input)
			}

			L.LogisticRegression_train(&(this.log_layer), layer_input, train_Y[n], lr)
		}
		// lr *= 0.95
	}
}

func DBN_predict(this *DBN, x []int, y []float64) {
	var (
		layer_input []float64
	)	
	prev_layer_input := make([]float64, this.n_ins)
	for j := 0; j < this.n_ins; j++ { prev_layer_input[j] = float64(x[j]) }

	
	// layer activation
	for i := 0; i < this.n_layers; i++ {
		layer_input = make([]float64, this.sigmoid_layers[i].N_out)

		for k := 0; k < this.sigmoid_layers[i].N_out; k++ {
			linear_outuput := 0.0

			for j := 0; j < this.sigmoid_layers[i].N_in; j++ {
				linear_outuput += this.sigmoid_layers[i].W[k][j] * prev_layer_input[j]
			}
			linear_outuput += this.sigmoid_layers[i].B[k]
			layer_input[k] = u.Sigmoid(linear_outuput)
		}

		if i < this.n_layers-1 {
			prev_layer_input = make([]float64, this.sigmoid_layers[i].N_out)

			for j := 0; j < this.sigmoid_layers[i].N_out; j++ {
				prev_layer_input[j] = layer_input[j]
			}
		}
	}

	for i := 0; i < this.log_layer.N_out; i++ {
		y[i] = 0
		for j := 0; j < this.log_layer.N_in; j++ {
			y[i] += this.log_layer.W[i][j] * layer_input[j]
		}
		y[i] += this.log_layer.B[i]
	}

	L.LogisticRegression_softmax(&(this.log_layer), y)
}


func test_dbn() {
	rand.Seed(0)

	pretrain_lr := 0.1
	pretraining_epochs := 1000
	k := 1
	finetune_lr := 0.1
	finetune_epochs := 500

	train_N := 6
	test_N := 4
	n_ins := 6
	n_outs := 2
	hidden_layer_sizes := []int {3, 3}
	n_layers := len(hidden_layer_sizes)


	// training data
	train_X := [][]int {
		{1, 1, 1, 0, 0, 0},
		{1, 0, 1, 0, 0, 0},
		{1, 1, 1, 0, 0, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 0, 1, 1, 0, 0},
		{0, 0, 1, 1, 1, 0},
	}

	train_Y := [][]int {
		{1, 0},
		{1, 0},
		{1, 0},
		{0, 1},
		{0, 1},
		{0, 1},
	}

	// construct DBN
	var dbn DBN
	DBN__construct(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers)

	// pretrain
	DBN_pretrain(&dbn, train_X, pretrain_lr, k, pretraining_epochs)

	// finetune
	DBN_finetune(&dbn, train_X, train_Y, finetune_lr, finetune_epochs)

	// test data
	test_X := [][]int {
		{1, 1, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0},
		{0, 0, 0, 1, 1, 0},
		{0, 0, 1, 1, 1, 0},
	}

	test_Y := make([][]float64, test_N)
	for i := 0; i < test_N; i++ { test_Y[i] = make([]float64, n_outs)}

	// test
	for i := 0; i < test_N; i++ {
		DBN_predict(&dbn, test_X[i], test_Y[i])
		for j := 0; j < n_outs; j++ {
			fmt.Printf("%.5f ", test_Y[i][j])
		}
		fmt.Printf("\n")
	}
}



func main() {
	test_dbn()
}
