package RBM

import (
	u "../utils"
)

type RBM struct {
	N int
	n_visible int
	n_hidden int
	W [][]float64
	hbias []float64
	vbias []float64
}


func RBM__construct(this *RBM, N int, n_visible int, n_hidden int, W [][]float64, hbias []float64, vbias []float64) {
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

func RBM_contrastive_divergence(this *RBM, input []int, lr float64, k int) {
	ph_mean := make([]float64, this.n_hidden)
	ph_sample := make([]int, this.n_hidden)
	nv_means := make([]float64, this.n_visible)
	nv_samples := make([]int, this.n_visible)
	nh_means := make([]float64, this.n_hidden)
	nh_samples := make([]int, this.n_hidden)

	/* CD-k */
	RBM_sample_h_given_v(this, input, ph_mean, ph_sample)

	for step := 0; step < k; step++ {
		if step == 0 {
			RBM_gibbs_hvh(this, ph_sample, nv_means, nv_samples, nh_means, nh_samples)
		} else {
			RBM_gibbs_hvh(this, nh_samples, nv_means, nv_samples, nh_means, nh_samples)
		}
	}

	for i := 0; i < this.n_hidden; i++ {
		for j := 0; j < this.n_visible; j++ {
			this.W[i][j] += lr * (ph_mean[i] * float64(input[j]) - nh_means[i] * float64(nv_samples[j])) / float64(this.N)
		}
		this.hbias[i] += lr * (float64(ph_sample[i]) - nh_means[i]) / float64(this.N)
	}

	for i := 0; i < this.n_visible; i++ {
		this.vbias[i] += lr * float64(input[i] - nv_samples[i]) / float64(this.N)
	}
}

func RBM_sample_h_given_v(this *RBM, v0_sample []int, mean []float64, sample []int) {
	for i := 0; i < this.n_hidden; i++ {
		mean[i] = RBM_propup(this, v0_sample, this.W[i], this.hbias[i])
		sample[i] = u.Binomial(1, mean[i])
	}
}

func RBM_sample_v_given_h(this *RBM, h0_sample []int, mean []float64, sample []int) {
	for i := 0; i < this.n_visible; i++ {
		mean[i] = RBM_propdown(this, h0_sample, i, this.vbias[i])
		sample[i] = u.Binomial(1, mean[i])
	}
}

func RBM_propup(this *RBM, v []int, w []float64, b float64) float64 {
	pre_sigmoid_activation := 0.0
	
	for j := 0; j < this.n_visible; j++ {
		pre_sigmoid_activation += w[j] * float64(v[j])
	}
	pre_sigmoid_activation += b
	
	return u.Sigmoid(pre_sigmoid_activation)
}

func RBM_propdown(this *RBM,	h []int, i int, b float64) float64 {
	pre_sigmoid_activation := 0.0
	
	for j := 0; j < this.n_hidden; j++ {
		pre_sigmoid_activation += this.W[j][i] * float64(h[j])
	}
	pre_sigmoid_activation += b

	return u.Sigmoid(pre_sigmoid_activation)
}

func RBM_gibbs_hvh(this *RBM, h0_sample []int, nv_means []float64, nv_samples []int, nh_means []float64, nh_samples []int) {
	RBM_sample_v_given_h(this, h0_sample, nv_means, nv_samples)
	RBM_sample_h_given_v(this, nv_samples, nh_means, nh_samples)
}

func RBM_reconstruct(this *RBM, v []int, reconstructed_v []float64) {
	h := make([]float64, this.n_hidden)
	var pre_sigmoid_activation float64

	for i := 0; i < this.n_hidden; i++ {
		h[i] = RBM_propup(this, v, this.W[i], this.hbias[i])
	}

	for i := 0; i < this.n_visible; i++ {
		pre_sigmoid_activation = 0.0
		for j := 0; j < this.n_hidden; j++ {
			pre_sigmoid_activation += this.W[j][i] * h[j]
		}
		pre_sigmoid_activation += this.vbias[i]

		reconstructed_v[i] = u.Sigmoid(pre_sigmoid_activation)
	}
}
