#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "LogisticRegression.h"

void test_lr(void);


void LogisticRegression__construct(LogisticRegression *this, int N, int n_in, int n_out) {
  int i, j;
  this->N = N;
  this->n_in = n_in;
  this->n_out = n_out;

  this->W = (double **)malloc(sizeof(double*) * n_out);
  this->W[0] = (double *)malloc(sizeof(double) * n_in * n_out);
  for(i=0; i<n_out; i++) this->W[i] = this->W[0] + i * n_in;
  this->b = (double *)malloc(sizeof(double) * n_out);

  for(i=0; i<n_out; i++) {
    for(j=0; j<n_in; j++) {
      this->W[i][j] = 0;
    }
    this->b[i] = 0;
  }
}

void LogisticRegression__destruct(LogisticRegression *this) {
  free(this->W[0]);
  free(this->W);
  free(this->b);
}

void LogisticRegression_train(LogisticRegression *this, int *x, int *y, double lr) {
  int i,j;
  double *p_y_given_x = (double *)malloc(sizeof(double) * this->n_out);
  double *dy = (double *)malloc(sizeof(double) * this->n_out);

  for(i=0; i<this->n_out; i++) {
    p_y_given_x[i] = 0;
    for(j=0; j<this->n_in; j++) {
      p_y_given_x[i] += this->W[i][j] * x[j];
    }
    p_y_given_x[i] += this->b[i];
  }
  LogisticRegression_softmax(this, p_y_given_x);

  for(i=0; i<this->n_out; i++) {
    dy[i] = y[i] - p_y_given_x[i];

    for(j=0; j<this->n_in; j++) {
      this->W[i][j] += lr * dy[i] * x[j] / this->N;
    }

    this->b[i] += lr * dy[i] / this->N;
  }

  free(p_y_given_x);
  free(dy);
}

void LogisticRegression_softmax(LogisticRegression *this, double *x) {
  int i;
  double max = 0.0;
  double sum = 0.0;

  for(i=0; i<this->n_out; i++) if(max < x[i]) max = x[i];
  for(i=0; i<this->n_out; i++) {
    x[i] = exp(x[i] - max);
    sum += x[i];
  }

  for(i=0; i<this->n_out; i++) x[i] /= sum;
}

void LogisticRegression_predict(LogisticRegression *this, int *x, double *y) {
  int i,j;

  for(i=0; i<this->n_out; i++) {
    y[i] = 0;
    for(j=0; j<this->n_in; j++) {
      y[i] += this->W[i][j] * x[j];
    }
    y[i] += this->b[i];
  }

  LogisticRegression_softmax(this, y);
}




void test_lr(void) {
  int i, j, epoch;

  double learning_rate = 0.1;
  int n_epochs = 500;

  int train_N = 6;
  int test_N = 2;
  int n_in = 6;
  int n_out = 2;


  // training data
  int train_X[6][6] = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 0}
  };

  int train_Y[6][2] = {
    {1, 0},
    {1, 0},
    {1, 0},
    {0, 1},
    {0, 1},
    {0, 1}
  };


  // construct LogisticRegression
  LogisticRegression classifier;
  LogisticRegression__construct(&classifier, train_N, n_in, n_out);


  // train
  for(epoch=0; epoch<n_epochs; epoch++) {
    for(i=0; i<train_N; i++) {
      LogisticRegression_train(&classifier, train_X[i], train_Y[i], learning_rate);
    }
    // learning_rate *= 0.95;
  }


  // test data
  int test_X[2][6] = {
    {1, 0, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0}
  };

  double test_Y[2][2];


  // test
  for(i=0; i<test_N; i++) {
    LogisticRegression_predict(&classifier, test_X[i], test_Y[i]);
    for(j=0; j<n_out; j++) {
      printf("%f ", test_Y[i][j]);
    }
    printf("\n");
  }



  // destruct LogisticRegression
  LogisticRegression__destruct(&classifier);
}




int main(void) {
  test_lr();
  
  return 0;
}
