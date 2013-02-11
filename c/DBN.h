#ifndef DBN_H
#define DBN_H

typedef struct {
  int N;
  int n_ins;
  int *hidden_layer_sizes;
  int n_outs;
  int n_layers;
  HiddenLayer *sigmoid_layers;
  RBM *rbm_layers;
  LogisticRegression log_layer;
} DBN;

void DBN__construct(DBN*, int, int, int*, int, int);
void DBN__destruct(DBN*);
void DBN_pretrain(DBN*, int*, double, int, int);
void DBN_finetune(DBN*, int*, int*, double, int);
void DBN_predict(DBN*, int*, double*);

#endif
