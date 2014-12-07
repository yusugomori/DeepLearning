package HiddenLayer

import (
	u "../utils"
)


type HiddenLayer struct {
	N int
	N_in int
	N_out int
	W [][]float64
	B []float64
}


// HiddenLayer
func HiddenLayer__construct(this *HiddenLayer, N int, n_in int, n_out int, W [][]float64, b []float64) {
	a := 1.0 / float64(n_in)

	this.N = N
	this.N_in = n_in
	this.N_out = n_out

	if W == nil {
		this.W = make([][]float64, n_out)
		for i := 0; i < n_out; i++ { this.W[i] = make([]float64, n_in) }
		
		for i := 0; i < n_out; i++ {
			for j := 0; j < n_in; j++ {
				this.W[i][j] = u.Uniform(-a, a)
			}
		}
	} else {
		this.W = W
	}

	if b == nil {
		this.B = make([]float64, n_out)
	} else {
		this.B = b
	}
}

func HiddenLayer_output(this *HiddenLayer, input []int, w []float64, b float64) float64 {
	linear_output := 0.0

	for j := 0; j < this.N_in; j++ {
		linear_output += w[j] * float64(input[j])
	}
	linear_output += b

	return u.Sigmoid(linear_output)
}

func HiddenLayer_sample_h_given_v(this *HiddenLayer, input []int, sample []int) {
	for i := 0; i < this.N_out; i++ {
		sample[i] = u.Binomial(1, HiddenLayer_output(this, input, this.W[i], this.B[i]))
	}
}
