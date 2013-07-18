#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "RBM.h"
#include "utils.h"


void test_rbm(void);


double uniform(double min, double max) {
  return rand() / (RAND_MAX + 1.0) * (max - min) + min;  
}

int binomial(int n, double p) {
  if(p < 0 || p > 1) return 0;

  int i;
  int c = 0;
  double r;

  for(i=0; i<n; i++) {
    r = rand() / (RAND_MAX + 1.0);
    if (r < p) c++;
  }

  return c;
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}


void RBM__construct(RBM* this, int N, int n_visible, int n_hidden, \
                    double **W, double *hbias, double *vbias) {
  int i, j;
  double a = 1.0 / n_visible;

  this->N = N;
  this->n_visible = n_visible;
  this->n_hidden = n_hidden;

  if(W == NULL) {
    this->W = (double **)malloc(sizeof(double*) * n_hidden);
    this->W[0] = (double *)malloc(sizeof(double) * n_visible * n_hidden);
    for(i=0; i<n_hidden; i++) this->W[i] = this->W[0] + i * n_visible;

    for(i=0; i<n_hidden; i++) {
      for(j=0; j<n_visible; j++) {
        this->W[i][j] = uniform(-a, a);
      }
    }
  } else {
    this->W = W;
  }

  if(hbias == NULL) {
    this->hbias = (double *)malloc(sizeof(double) * n_hidden);
    for(i=0; i<n_hidden; i++) this->hbias[i] = 0;
  } else {
    this->hbias = hbias;
  }

  if(vbias == NULL) {
    this->vbias = (double *)malloc(sizeof(double) * n_visible);
    for(i=0; i<n_visible; i++) this->vbias[i] = 0;
  } else {
    this->vbias = vbias;
  }
}

void RBM__destruct(RBM* this) {
  free(this->W[0]);
  free(this->W);
  free(this->hbias);
  free(this->vbias);
}

void RBM_contrastive_divergence(RBM* this, int *input, double lr, int k) {
  int i, j, step;
  
  double *ph_mean = (double *)malloc(sizeof(double) * this->n_hidden);
  int *ph_sample = (int *)malloc(sizeof(int) * this->n_hidden);
  double *nv_means = (double *)malloc(sizeof(double) * this->n_visible);
  int *nv_samples = (int *)malloc(sizeof(int) * this->n_visible);
  double *nh_means = (double *)malloc(sizeof(double) * this->n_hidden);
  int *nh_samples = (int *)malloc(sizeof(int) * this->n_hidden);

  /* CD-k */
  RBM_sample_h_given_v(this, input, ph_mean, ph_sample);

  for(step=0; step<k; step++) {
    if(step == 0) {
      RBM_gibbs_hvh(this, ph_sample, nv_means, nv_samples, nh_means, nh_samples);
    } else {
      RBM_gibbs_hvh(this, nh_samples, nv_means, nv_samples, nh_means, nh_samples);
    }
  }

  for(i=0; i<this->n_hidden; i++) {
    for(j=0; j<this->n_visible; j++) {
      // this->W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / this->N;
      this->W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / this->N;
    }
    this->hbias[i] += lr * (ph_sample[i] - nh_means[i]) / this->N;
  }

  for(i=0; i<this->n_visible; i++) {
    this->vbias[i] += lr * (input[i] - nv_samples[i]) / this->N;
  }
  

  free(ph_mean);
  free(ph_sample);
  free(nv_means);
  free(nv_samples);
  free(nh_means);
  free(nh_samples);
}


void RBM_sample_h_given_v(RBM* this, int *v0_sample, double *mean, int *sample) {
  int i;
  for(i=0; i<this->n_hidden; i++) {
    mean[i] = RBM_propup(this, v0_sample, this->W[i], this->hbias[i]);
    sample[i] = binomial(1, mean[i]);
  }
}

void RBM_sample_v_given_h(RBM* this, int *h0_sample, double *mean, int *sample) {
  int i;
  for(i=0; i<this->n_visible; i++) {
    mean[i] = RBM_propdown(this, h0_sample, i, this->vbias[i]);
    sample[i] = binomial(1, mean[i]);
  }
}

double RBM_propup(RBM* this, int *v, double *w, double b) {
  int j;
  double pre_sigmoid_activation = 0.0;
  for(j=0; j<this->n_visible; j++) {
    pre_sigmoid_activation += w[j] * v[j];
  }
  pre_sigmoid_activation += b;
  return sigmoid(pre_sigmoid_activation);
}

double RBM_propdown(RBM* this, int *h, int i, double b) {
  int j;
  double pre_sigmoid_activation = 0.0;

  for(j=0; j<this->n_hidden; j++) {
    pre_sigmoid_activation += this->W[j][i] * h[j];
  }
  pre_sigmoid_activation += b;
  return sigmoid(pre_sigmoid_activation);
}

void RBM_gibbs_hvh(RBM* this, int *h0_sample, double *nv_means, int *nv_samples, \
                   double *nh_means, int *nh_samples) {
  RBM_sample_v_given_h(this, h0_sample, nv_means, nv_samples);
  RBM_sample_h_given_v(this, nv_samples, nh_means, nh_samples);
}

void RBM_reconstruct(RBM* this, int *v, double *reconstructed_v) {
  int i, j;
  double *h = (double *)malloc(sizeof(double) * this->n_hidden);
  double pre_sigmoid_activation;

  for(i=0; i<this->n_hidden; i++) {
    h[i] = RBM_propup(this, v, this->W[i], this->hbias[i]);
  }

  for(i=0; i<this->n_visible; i++) {
    pre_sigmoid_activation = 0.0;
    for(j=0; j<this->n_hidden; j++) {
      pre_sigmoid_activation += this->W[j][i] * h[j];
    }
    pre_sigmoid_activation += this->vbias[i];

    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
  }

  free(h);
}




void test_rbm(void) {
  srand(0);

  int i, j, epoch;

  double learning_rate = 0.1;
  int training_epochs = 1000;
  int k = 1;
  
  int train_N = 6;
  int test_N = 2;
  int n_visible = 6;
  int n_hidden = 3;

  // training data
  int train_X[6][6] = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 1, 1, 1, 0}
  };

  // construct RBM
  RBM rbm;
  RBM__construct(&rbm, train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      RBM_contrastive_divergence(&rbm, train_X[i], learning_rate, k);
    }
  }


  // test data
  int test_X[2][6] = {
    {1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0}
  };
  double reconstructed_X[2][6];

  // test
  for(i=0; i<test_N; i++) {
    RBM_reconstruct(&rbm, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }


  // destruct RBM
  RBM__destruct(&rbm);
}



int main(void) {
  test_rbm();
  
  return 0;
}
