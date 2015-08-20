#include <iostream>
#include <math.h>
#include "utils.h"

#include "dA.h"
using namespace std;
using namespace utils;


dA::dA(int size, int n_v, int n_h, double **w, double *hb, double *vb) {
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

dA::~dA() {
  for(int i=0; i<n_hidden; i++) delete[] W[i];
  delete[] W;
  delete[] hbias;
  delete[] vbias;
}

void dA::get_corrupted_input(int *x, int *tilde_x, double p) {
  for(int i=0; i<n_visible; i++) {
    if(x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial(1, p);
    }
  }
}

// Encode
void dA::get_hidden_values(int *x, double *y) {
  for(int i=0; i<n_hidden; i++) {
    y[i] = 0;
    for(int j=0; j<n_visible; j++) {
      y[i] += W[i][j] * x[j];
    }
    y[i] += hbias[i];
    y[i] = sigmoid(y[i]);
  }
}

// Decode
void dA::get_reconstructed_input(double *y, double *z) {
  for(int i=0; i<n_visible; i++) {
    z[i] = 0;
    for(int j=0; j<n_hidden; j++) {
      z[i] += W[j][i] * y[j];
    }
    z[i] += vbias[i];
    z[i] = sigmoid(z[i]);
  }
}

void dA::train(int *x, double lr, double corruption_level) {
  int *tilde_x = new int[n_visible];
  double *y = new double[n_hidden];
  double *z = new double[n_visible];

  double *L_vbias = new double[n_visible];
  double *L_hbias = new double[n_hidden];

  double p = 1 - corruption_level;

  get_corrupted_input(x, tilde_x, p);
  get_hidden_values(tilde_x, y);
  get_reconstructed_input(y, z);
  
  // vbias
  for(int i=0; i<n_visible; i++) {
    L_vbias[i] = x[i] - z[i];
    vbias[i] += lr * L_vbias[i] / N;
  }

  // hbias
  for(int i=0; i<n_hidden; i++) {
    L_hbias[i] = 0;
    for(int j=0; j<n_visible; j++) {
      L_hbias[i] += W[i][j] * L_vbias[j];
    }
    L_hbias[i] *= y[i] * (1 - y[i]);

    hbias[i] += lr * L_hbias[i] / N;
  }
  
  // W
  for(int i=0; i<n_hidden; i++) {
    for(int j=0; j<n_visible; j++) {
      W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
    }
  }

  delete[] L_hbias;
  delete[] L_vbias;
  delete[] z;
  delete[] y;
  delete[] tilde_x;
}

void dA::reconstruct(int *x, double *z) {
  double *y = new double[n_hidden];

  get_hidden_values(x, y);
  get_reconstructed_input(y, z);

  delete[] y;
}



void test_dA() {
  srand(0);
  
  double learning_rate = 0.1;
  double corruption_level = 0.3;
  int training_epochs = 100;

  int train_N = 10;
  int test_N = 2;
  int n_visible = 20;
  int n_hidden = 5;

  // training data
  int train_X[10][20] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
  };

  // construct dA
  dA da(train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train
  for(int epoch=0; epoch<training_epochs; epoch++) {
    for(int i=0; i<train_N; i++) {
      da.train(train_X[i], learning_rate, corruption_level);
    }
  }

  // test data
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  double reconstructed_X[2][20];


  // test
  for(int i=0; i<test_N; i++) {
    da.reconstruct(test_X[i], reconstructed_X[i]);
    for(int j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    cout << endl;
  }

  cout << endl;
}



int main() {
  test_dA();
  return 0;
}
