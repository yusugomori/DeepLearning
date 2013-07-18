#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "HiddenLayer.h"
#include "RBM.h"
#include "LogisticRegression.h"
#include "DBN.h"
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



// DBN
void DBN__construct(DBN* this, int N, \
                    int n_ins, int *hidden_layer_sizes, int n_outs, int n_layers) {
  int i, input_size;

  this->N = N;
  this->n_ins = n_ins;
  this->hidden_layer_sizes = hidden_layer_sizes;
  this->n_outs = n_outs;
  this->n_layers = n_layers;

  this->sigmoid_layers = (HiddenLayer *)malloc(sizeof(HiddenLayer) * n_layers);
  this->rbm_layers = (RBM *)malloc(sizeof(RBM) * n_layers);

  // construct multi-layer
  for(i=0; i<n_layers; i++) {
    if(i == 0) {
      input_size = n_ins;
    } else {
      input_size = hidden_layer_sizes[i-1];
    }

    // construct sigmoid_layer
    HiddenLayer__construct(&(this->sigmoid_layers[i]), \
                           N, input_size, hidden_layer_sizes[i], NULL, NULL);

    // construct rbm_layer
    RBM__construct(&(this->rbm_layers[i]), N, input_size, hidden_layer_sizes[i], \
                   this->sigmoid_layers[i].W, this->sigmoid_layers[i].b, NULL);
    
  }

  // layer for output using LogisticRegression
  LogisticRegression__construct(&(this->log_layer), \
                                N, hidden_layer_sizes[n_layers-1], n_outs);
  
}

void DBN__destruct(DBN* this) {
  int i;
  for(i=0; i<this->n_layers; i++) {
    HiddenLayer__destruct(&(this->sigmoid_layers[i]));
    RBM__destruct(&(this->rbm_layers[i]));
  }
  free(this->sigmoid_layers);
  free(this->rbm_layers);
}

void DBN_pretrain(DBN* this, int *input, double lr, int k, int epochs) {
  int i, j, l, m, n, epoch;
  
  int *layer_input;
  int prev_layer_input_size;
  int *prev_layer_input;

  int *train_X = (int *)malloc(sizeof(int) * this->n_ins);

  for(i=0; i<this->n_layers; i++) { // layer-wise

    for(epoch=0; epoch<epochs; epoch++) { // training epochs

      for(n=0; n<this->N; n++) { // input x1...xN
        // initial input
        for(m=0; m<this->n_ins; m++) train_X[m] = input[n * this->n_ins + m];

        // layer input
        for(l=0; l<=i; l++) {
          if(l == 0) {
            layer_input = (int *)malloc(sizeof(int) * this->n_ins);
            for(j=0; j<this->n_ins; j++) layer_input[j] = train_X[j];
          } else {
            if(l == 1) prev_layer_input_size = this->n_ins;
            else prev_layer_input_size = this->hidden_layer_sizes[l-2];

            prev_layer_input = (int *)malloc(sizeof(int) * prev_layer_input_size);
            for(j=0; j<prev_layer_input_size; j++) prev_layer_input[j] = layer_input[j];
            free(layer_input);

            layer_input = (int *)malloc(sizeof(int) * this->hidden_layer_sizes[l-1]);

            HiddenLayer_sample_h_given_v(&(this->sigmoid_layers[l-1]), \
                                         prev_layer_input, layer_input);
            free(prev_layer_input);
          }
        }

        RBM_contrastive_divergence(&(this->rbm_layers[i]), layer_input, lr, k);
      }
      
    }
  }
  
  free(train_X);
  free(layer_input);
}

void DBN_finetune(DBN* this, int *input, int *label, double lr, int epochs) {
  int i, j, m, n, epoch;
  
  int *layer_input;
  // int prev_layer_input_size;
  int *prev_layer_input;

  int *train_X = (int *)malloc(sizeof(int) * this->n_ins);
  int *train_Y = (int *)malloc(sizeof(int) * this->n_outs);

  for(epoch=0; epoch<epochs; epoch++) {
    for(n=0; n<this->N; n++) { // input x1...xN
      // initial input
      for(m=0; m<this->n_ins; m++)  train_X[m] = input[n * this->n_ins + m];
      for(m=0; m<this->n_outs; m++) train_Y[m] = label[n * this->n_outs + m];

      // layer input
      for(i=0; i<this->n_layers; i++) {
        if(i == 0) {
          prev_layer_input = (int *)malloc(sizeof(int) * this->n_ins);
          for(j=0; j<this->n_ins; j++) prev_layer_input[j] = train_X[j];
        } else {
          prev_layer_input = (int *)malloc(sizeof(int) * this->hidden_layer_sizes[i-1]);
          for(j=0; j<this->hidden_layer_sizes[i-1]; j++) prev_layer_input[j] = layer_input[j];
          free(layer_input);
        }


        layer_input = (int *)malloc(sizeof(int) * this->hidden_layer_sizes[i]);
        HiddenLayer_sample_h_given_v(&(this->sigmoid_layers[i]), \
                                     prev_layer_input, layer_input);
        free(prev_layer_input);
      }

      LogisticRegression_train(&(this->log_layer), layer_input, train_Y, lr);
    }
    // lr *= 0.95;
  }

  free(layer_input);
  free(train_X);
  free(train_Y);
}

void DBN_predict(DBN* this, int *x, double *y) {
  int i, j, k;
  double *layer_input;
  // int prev_layer_input_size;
  double *prev_layer_input;

  double linear_output;

  prev_layer_input = (double *)malloc(sizeof(double) * this->n_ins);
  for(j=0; j<this->n_ins; j++) prev_layer_input[j] = x[j];

  // layer activation
  for(i=0; i<this->n_layers; i++) {
    layer_input = (double *)malloc(sizeof(double) * this->sigmoid_layers[i].n_out);

    for(k=0; k<this->sigmoid_layers[i].n_out; k++) {
      linear_output = 0.0;

      for(j=0; j<this->sigmoid_layers[i].n_in; j++) {
        linear_output += this->sigmoid_layers[i].W[k][j] * prev_layer_input[j];
      }
      linear_output += this->sigmoid_layers[i].b[k];
      layer_input[k] = sigmoid(linear_output);
    }
    free(prev_layer_input);

    if(i < this->n_layers-1) {
      prev_layer_input = (double *)malloc(sizeof(double) * this->sigmoid_layers[i].n_out);
      for(j=0; j<this->sigmoid_layers[i].n_out; j++) prev_layer_input[j] = layer_input[j];
      free(layer_input);
    }
  }

  for(i=0; i<this->log_layer.n_out; i++) {
    y[i] = 0;
    for(j=0; j<this->log_layer.n_in; j++) {
      y[i] += this->log_layer.W[i][j] * layer_input[j];
    }
    y[i] += this->log_layer.b[i];
  }

  LogisticRegression_softmax(&(this->log_layer), y);

  free(layer_input);
}



// HiddenLayer
void HiddenLayer__construct(HiddenLayer* this, int N, int n_in, int n_out, \
                            double **W, double *b) {
  int i, j;
  double a = 1.0 / n_in;

  this->N = N;
  this->n_in = n_in;
  this->n_out = n_out;
  
  if(W == NULL) {
    this->W = (double **)malloc(sizeof(double*) * n_out);
    this->W[0] = (double *)malloc(sizeof(double) * n_in * n_out);
    for(i=0; i<n_out; i++) this->W[i] = this->W[0] + i * n_in;

    for(i=0; i<n_out; i++) {
      for(j=0; j<n_in; j++) {
        this->W[i][j] = uniform(-a, a);
      }
    }
  } else {
    this->W = W;
  }

  if(b == NULL) {
    this->b = (double *)malloc(sizeof(double) * n_out);
  } else {
    this->b = b;
  }
}

void HiddenLayer__destruct(HiddenLayer* this) {
  free(this->W[0]);
  free(this->W);
  free(this->b);
}

double HiddenLayer_output(HiddenLayer* this, int *input, double *w, double b) {
  int j;
  double linear_output = 0.0;
  for(j=0; j<this->n_in; j++) {
    linear_output += w[j] * input[j];
  }
  linear_output += b;
  return sigmoid(linear_output);
}

void HiddenLayer_sample_h_given_v(HiddenLayer* this, int *input, int *sample) {
  int i;
  for(i=0; i<this->n_out; i++) {
    sample[i] = binomial(1, HiddenLayer_output(this, input, this->W[i], this->b[i]));
  }
}


// RBM
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
  // free(this->W[0]);
  // free(this->W);
  // free(this->hbias);
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


// LogisticRegression
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


void test_dbn(void) {
  srand(0);

  int i, j;
  
  double pretrain_lr = 0.1;
  int pretraining_epochs = 1000;
  int k = 1;
  double finetune_lr = 0.1;
  int finetune_epochs = 500;

  int train_N = 6;
  int test_N = 4;
  int n_ins = 6;
  int n_outs = 2;
  int hidden_layer_sizes[] = {3, 3};
  int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);


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

  // construct DBN
  DBN dbn;
  DBN__construct(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

  // pretrain
  DBN_pretrain(&dbn, *train_X, pretrain_lr, k, pretraining_epochs);

  // finetune
  DBN_finetune(&dbn, *train_X, *train_Y, finetune_lr, finetune_epochs);

  // test data
  int test_X[4][6] = {
    {1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 0, 0},
    {0, 0, 0, 1, 1, 0},
    {0, 0, 1, 1, 1, 0}
  };

  double test_Y[4][2];

  // test
  for(i=0; i<test_N; i++) {
    DBN_predict(&dbn, test_X[i], test_Y[i]);
    for(j=0; j<n_outs; j++) {
      printf("%.5f ", test_Y[i][j]);
    }
    printf("\n");
  }

  // destruct DBN
  DBN__destruct(&dbn);
  
}



int main(void) {
  test_dbn();
  return 0;
}
