package main

import (
	"fmt"
	"math/rand"
	u "./utils"
)

type dA struct {
	N int
	n_visible int
	n_hidden int
	W [][]float64
	hbias []float64
	vbias []float64
}


func dA__construct(this *dA, N int, n_visible int, n_hidden int, W [][]float64, hbias []float64, vbias []float64) {
	a := 1.0 / float64(n_visible)
	
	this.N = N
	this.n_visible = n_visible
	this.n_hidden = n_hidden

	if W == nil {
		this.W = make([][]float64, n_hidden)
		for i := 0; i < n_hidden; i++ { this.W[i] = make([]float64, n_visible) }

		for i := 0; i < n_hidden; i++ {
			for j := 0; j < n_visible; j++ {
				this.W[i][j] = u.Uniform(-a, a)
			}
		}
	} else {
		this.W = W
	}

	if hbias == nil {
		this.hbias = make([]float64, n_hidden)
	} else {
		this.hbias = hbias
	}

	if vbias == nil {
		this.vbias = make([]float64, n_visible)
	} else {
		this.vbias = vbias
	}
}

func dA_get_corrupted_input(this *dA, x []int, tilde_x []int, p float64) {
	for i := 0; i < this.n_visible; i++ {
		if x[i] == 0 {
			tilde_x[i] = 0
		} else {
			tilde_x[i] = u.Binomial(1, p)
		}
	}
}

// Encode
func dA_get_hidden_values(this *dA, x []int, y []float64) {
	for i := 0; i < this.n_hidden; i++ {
		y[i] = 0
		for j := 0; j < this.n_visible; j++ {
			y[i] += this.W[i][j] * float64(x[j])
		}
		y[i] += this.hbias[i]
		y[i] = u.Sigmoid(y[i])
	}
}

// Decode
func dA_get_reconstructed_input(this *dA, y []float64, z []float64) {
	for i := 0; i < this.n_visible; i++ {
		z[i] = 0
		for j := 0; j < this.n_hidden; j++ {
			z[i] += this.W[j][i] * y[j]
		}
		z[i] += this.vbias[i]
		z[i] = u.Sigmoid(z[i])
	}
}

func dA_train(this *dA, x []int, lr float64, corruption_level float64) {
	tilde_x := make([]int, this.n_visible)
	y := make([]float64, this.n_hidden)
	z := make([]float64, this.n_visible)

	L_vbias := make([]float64, this.n_visible)
	L_hbias := make([]float64, this.n_hidden)

	p := 1 - corruption_level

	dA_get_corrupted_input(this, x, tilde_x, p)
	dA_get_hidden_values(this, tilde_x, y)
	dA_get_reconstructed_input(this, y, z)

	// vbias
	for i := 0; i < this.n_visible; i++ {
		L_vbias[i] = float64(x[i]) - z[i]
		this.vbias[i] += lr * L_vbias[i] / float64(this.N)
	}

	// hbias
	for i := 0; i < this.n_hidden; i++ {
		L_hbias[i] = 0
		for j := 0; j < this.n_visible; j++ {
			L_hbias[i] += this.W[i][j] * L_vbias[j]
		}
		L_hbias[i] *= y[i] * (1- y[i])
		this.hbias[i] += lr * L_hbias[i] / float64(this.N)
	}

	// W
	for i := 0; i < this.n_hidden; i++ {
		for j := 0; j < this.n_visible; j++ {
			this.W[i][j] += lr * (L_hbias[i] * float64(tilde_x[j]) + L_vbias[j] * y[i]) / float64(this.N)
		}
	}
}

func dA_reconstruct(this *dA, x []int, z []float64) {
	y := make([]float64, this.n_hidden)

	dA_get_hidden_values(this, x, y)
	dA_get_reconstructed_input(this, y, z)
}




func test_dA() {
	rand.Seed(0)

	learning_rate := 0.1
	corruption_level := 0.3
	training_epochs := 1000

	train_N := 6
	test_N := 2
	n_visible := 20
	n_hidden := 5

	// training data
	train_X := [][]int {
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
	}

	// construct dA
	var da dA
	dA__construct(&da, train_N, n_visible, n_hidden, nil, nil, nil)

	// train
	for epoch := 0; epoch < training_epochs; epoch++ {
		for i := 0; i < train_N; i++ {
			dA_train(&da, train_X[i], learning_rate, corruption_level)
		}
	}

	// test data
	test_X := [][]int {
		{1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
	}
	reconstructed_X := make([][]float64, test_N)
	for i := 0; i < test_N; i++ { reconstructed_X[i] = make([]float64, n_visible)}

	
	// test
	for i := 0;  i < test_N; i++ {
		dA_reconstruct(&da, test_X[i], reconstructed_X[i])
		for j := 0; j < n_visible; j++ {
			fmt.Printf("%.5f ", reconstructed_X[i][j])
		}
		fmt.Printf("\n")
	}
}

func main() {
	test_dA()
}
