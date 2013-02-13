#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dA.h"
#include "utils.h"


void test_dbn(void);


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


void dA__construct(dA* this, int N, int n_visible, int n_hidden, \
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

void dA__destruct(dA* this) {
  free(this->W[0]);
  free(this->W);
  free(this->hbias);
  free(this->vbias);
}

void dA_get_corrupted_input(dA* this, int *x, int *tilde_x, double p) {
  int i;
  for(i=0; i<this->n_visible; i++) {
    if(x[i] == 0) {
      tilde_x[i] = 0;
    } else {
      tilde_x[i] = binomial(1, p);
    }
  }
}

// Encode
void dA_get_hidden_values(dA* this, int *x, double *y) {
  int i,j;
  for(i=0; i<this->n_hidden; i++) {
    y[i] = 0;
    for(j=0; j<this->n_visible; j++) {
      y[i] += this->W[i][j] * x[j];
    }
    y[i] += this->hbias[i];
    y[i] = sigmoid(y[i]);
  }
}

// Decode
void dA_get_reconstructed_input(dA* this, double *y, double *z) {
  int i, j;
  for(i=0; i<this->n_visible; i++) {
    z[i] = 0;
    for(j=0; j<this->n_hidden; j++) {
      z[i] += this->W[j][i] * y[j];
    }
    z[i] += this->vbias[i];
    z[i] = sigmoid(z[i]);
  }
}


void dA_train(dA* this, int *x, double lr, double corruption_level) {
  int i, j;
  
  int *tilde_x = (int *)malloc(sizeof(int) * this->n_visible);
  double *y = (double *)malloc(sizeof(double) * this->n_hidden);
  double *z = (double *)malloc(sizeof(double) * this->n_visible);

  double *L_vbias = (double *)malloc(sizeof(double) * this->n_visible);
  double *L_hbias = (double *)malloc(sizeof(double) * this->n_hidden);

  double p = 1 - corruption_level;

  dA_get_corrupted_input(this, x, tilde_x, p);
  dA_get_hidden_values(this, tilde_x, y);
  dA_get_reconstructed_input(this, y, z);

  // vbias
  for(i=0; i<this->n_visible; i++) {
    L_vbias[i] = x[i] - z[i];
    this->vbias[i] += lr * L_vbias[i] / this->N;
  }

  // hbias
  for(i=0; i<this->n_hidden; i++) {
    L_hbias[i] = 0;
    for(j=0; j<this->n_visible; j++) {
      L_hbias[i] += this->W[i][j] * L_vbias[j];
    }
    L_hbias[i] *= y[i] * (1 - y[i]);

    this->hbias[i] += lr * L_hbias[i] / this->N;
  }

  // W
  for(i=0; i<this->n_hidden; i++) {
    for(j=0; j<this->n_visible; j++) {
      this->W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / this->N;
    }
  }

  free(L_hbias);
  free(L_vbias);
  free(z);
  free(y);
  free(tilde_x);
}

void dA_reconstruct(dA* this, int *x, double *z) {
  int i;
  double *y = (double *)malloc(sizeof(double) * this->n_hidden);

  dA_get_hidden_values(this, x, y);
  dA_get_reconstructed_input(this, y, z);

  free(y);
}

void test_dbn(void) {
  srand(0);
  int i, j, epoch;
  
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
  dA da;
  dA__construct(&da, train_N, n_visible, n_hidden, NULL, NULL, NULL);

  // train
  for(epoch=0; epoch<training_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      dA_train(&da, train_X[i], learning_rate, corruption_level);
    }
  }

  // test data
  int test_X[2][20] = {
    {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
  };
  double reconstructed_X[2][20];

  
  // test
  for(i=0; i<test_N; i++) {
    dA_reconstruct(&da, test_X[i], reconstructed_X[i]);
    for(j=0; j<n_visible; j++) {
      printf("%.5f ", reconstructed_X[i][j]);
    }
    printf("\n");
  }


  // destruct dA
  dA__destruct(&da);
}


int main(void) {
  test_dbn();
  return 0;
}
