#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

typedef struct {
  int N;
  int n_in;
  int n_out;
  double **W;
  double *b;
} HiddenLayer;

void HiddenLayer__construct(HiddenLayer*, int, int, int, double**, double*);
void HiddenLayer__destruct(HiddenLayer*);
double HiddenLayer_output(HiddenLayer*, int*, double*, double);
void HiddenLayer_sample_h_given_v(HiddenLayer*, int*, int*);

#endif
