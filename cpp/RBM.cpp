#include <iostream>
#include <math.h>
#include "utils.h"
#include "RBM.h"
using namespace std;
using namespace utils;


RBM::RBM(int size, int n_v, int n_h, double **w, double *hb, double *vb) {
  N = size;
  n_visible = n_v;
  n_hidden = n_h;

  if(w == NULL) {
    W = new double*[n_hidden];
    for(int i=0; i<n_hidden; i++) W[i] = new double[n_visible];
    double a = 1.0 / n_visible;

    for(int i=0; i<n_hidden; i++) {
      for(int j=0; j<n_visible; j++) {
        W[i][j] = uniform(-a, a);
      }
    }
  } else {
    W = w;
  }

  if(hb == NULL) {
    hbias = new double[n_hidden];
    for(int i=0; i<n_hidden; i++) hbias[i] = 0;
  } else {
    hbias = hb;
  }

  if(vb == NULL) {
    vbias = new double[n_visible];
    for(int i=0; i<n_visible; i++) vbias[i] = 0;
  } else {
    vbias = vb;
  }
}

RBM::~RBM() {
  for(int i=0; i<n_hidden; i++) delete[] W[i];
  delete[] W;
  delete[] hbias;
  delete[] vbias;
}


void RBM::contrastive_divergence(int *input, double lr, int k) {
  double *ph_mean = new double[n_hidden];
  int *ph_sample = new int[n_hidden];
  double *nv_means = new double[n_visible];
  int *nv_samples = new int[n_visible];
  double *nh_means = new double[n_hidden];
  int *nh_samples = new int[n_hidden];

  /* CD-k */
  sample_h_given_v(input, ph_mean, ph_sample);

  for(int step=0; step<k; step++) {
    if(step == 0) {
      gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
    } else {
      gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
    }
  }

  for(int i=0; i<n_hidden; i++) {
    for(int j=0; j<n_visible; j++) {
      // W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
      W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
    }
    hbias[i] += lr * (ph_sample[i] - nh_means[i]) / N;
  }

  for(int i=0; i<n_visible; i++) {
    vbias[i] += lr * (input[i] - nv_samples[i]) / N;
  }

  delete[] ph_mean;
  delete[] ph_sample;
  delete[] nv_means;
  delete[] nv_samples;
  delete[] nh_means;
  delete[] nh_samples;
}

void RBM::sample_h_given_v(int *v0_sample, double *mean, int *sample) {
  for(int i=0; i<n_hidden; i++) {
    mean[i] = propup(v0_sample, W[i], hbias[i]);
    sample[i] = binomial(1, mean[i]);
  }
}

void RBM::sample_v_given_h(int *h0_sample, double *mean, int *sample) {
  for(int i=0; i<n_visible; i++) {
    mean[i] = propdown(h0_sample, i, vbias[i]);
    sample[i] = binomial(1, mean[i]);
  }
}

double RBM::propup(int *v, double *w, double b) {
  double pre_sigmoid_activation = 0.0;
  for(int j=0; j<n_visible; j++) {
    pre_sigmoid_activation += w[j] * v[j];
  }
  pre_sigmoid_activation += b;
  return sigmoid(pre_sigmoid_activation);
}

double RBM::propdown(int *h, int i, double b) {
  double pre_sigmoid_activation = 0.0;
  for(int j=0; j<n_hidden; j++) {
    pre_sigmoid_activation += W[j][i] * h[j];
  }
  pre_sigmoid_activation += b;
  return sigmoid(pre_sigmoid_activation);
}

void RBM::gibbs_hvh(int *h0_sample, double *nv_means, int *nv_samples, \
                    double *nh_means, int *nh_samples) {
  sample_v_given_h(h0_sample, nv_means, nv_samples);
  sample_h_given_v(nv_samples, nh_means, nh_samples);
}

void RBM::reconstruct(int *v, double *reconstructed_v) {
  double *h = new double[n_hidden];
  double pre_sigmoid_activation;

  for(int i=0; i<n_hidden; i++) {
    h[i] = propup(v, W[i], hbias[i]);
  }

  for(int i=0; i<n_visible; i++) {
    pre_sigmoid_activation = 0.0;
    for(int j=0; j<n_hidden; j++) {
      pre_sigmoid_activation += W[j][i] * h[j];
    }
    pre_sigmoid_activation += vbias[i];

    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
  }

  delete[] h;
}


void test_rbm() {
  srand(0);

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
  RBM rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train
  for(int epoch=0; epoch<training_epochs; epoch++) {
    for(int i=0; i<train_N; i++) {
      rbm.contrastive_divergence(train_X[i], learning_rate, k);
    }
  }

  // test data
  int test_X[2][6] = {
    {1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0}
  };
  double reconstructed_X[2][6];


  // test
  for(int i=0; i<test_N; i++) {
    rbm.reconstruct(test_X[i], reconstructed_X[i]);
    for(int j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    cout << endl;
  }

}



int main() {
  test_rbm();
  return 0;
}
