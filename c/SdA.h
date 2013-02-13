#ifndef SDA_H
#define SDA_H

typedef struct {
  int N;
  int n_ins;
  int *hidden_layer_sizes;
  int n_outs;
  int n_layers;
  HiddenLayer *sigmoid_layers;
  dA *dA_layers;
  LogisticRegression log_layer;
} SdA;

void SdA__construct(SdA*, int, int, int*, int, int);
void SdA__destruct(SdA*);
void SdA_pretrain(SdA*, int*, double, double, int);
void SdA_finetune(SdA*, int*, int*, double, int);
void SdA_predict(SdA*, int*, double*);

#endif
