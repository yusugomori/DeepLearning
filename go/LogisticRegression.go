package main

import (
	"fmt"
	"math"
)

type LogisticRegression struct {
	N int
	n_in int
	n_out int
	W [][]float64
	b []float64
}


func LogisticRegression__construct(this *LogisticRegression, N int, n_in int, n_out int) {
	this.N = N
	this.n_in = n_in
	this.n_out = n_out

	this.W = make([][]float64, n_out)
	for i := 0; i < n_out; i++ { this.W[i] = make([]float64, n_in) }
	
	this.b = make([]float64, n_out)
}

func LogisticRegression_train(this *LogisticRegression, x []int, y []int, lr float64) {
	p_y_given_x := make([]float64, this.n_out)
	dy := make([]float64, this.n_out)
	
	for i := 0; i < this.n_out; i++ {
		p_y_given_x[i] = 0
		for j := 0; j < this.n_in; j++ {
			p_y_given_x[i] += this.W[i][j] * float64(x[j])
		}
		p_y_given_x[i] += this.b[i]
	}
	LogisticRegression_softmax(this, p_y_given_x)
	
	for i := 0; i < this.n_out; i++ {
		dy[i] = float64(y[i]) - p_y_given_x[i]
		
		for j := 0; j < this.n_in; j++ {
			this.W[i][j] += lr * dy[i] * float64(x[j]) / float64(this.N)
		}

		this.b[i] += lr * dy[i] / float64(this.N)
	}
	
}

func LogisticRegression_softmax(this *LogisticRegression, x []float64) {
	var (
		max float64
		sum float64
	)

	for i := 0; i < this.n_out; i++ { if max < x[i] {max = x[i]} }
	for i := 0; i < this.n_out; i++ {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}

	for i := 0; i < this.n_out; i++ { x[i] /= sum }
}

func LogisticRegression_predict(this *LogisticRegression, x []int, y []float64) {
	for i := 0; i < this.n_out; i++ {
		y[i] = 0
		for j := 0; j < this.n_in; j++ {
			y[i] += this.W[i][j] * float64(x[j])
		}
		y[i] += this.b[i]
	}

	LogisticRegression_softmax(this, y)
}



func test_lr() {
	
	learning_rate := 0.1
	n_epochs := 500

	train_N := 6
	test_N := 2
	n_in := 6
	n_out := 2

	
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

	
	// construct LogisticRegression
	var classifier LogisticRegression
	LogisticRegression__construct(&classifier, train_N, n_in, n_out)

	// train
	for epoch := 0; epoch < n_epochs; epoch++ {
		for i := 0; i < train_N; i++ {
			LogisticRegression_train(&classifier, train_X[i], train_Y[i], learning_rate)
		}
	}
	
	// test data
	test_X := [][]int {
		{1, 0, 1, 0, 0, 0},
		{0, 0, 1, 1, 1, 0},
	}
	
	test_Y := make([][]float64, test_N)
	for i := 0; i < test_N; i++ { test_Y[i] = make([]float64, n_out) }


	// test
	for i := 0; i < test_N; i++ {
		LogisticRegression_predict(&classifier, test_X[i], test_Y[i])
		for j := 0; j < n_out; j++ {
			fmt.Printf("%f ", test_Y[i][j])
		}
		fmt.Printf("\n")
	}
	
}


func main() {
	test_lr()
}

