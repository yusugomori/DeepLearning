#ifndef DA_H
#define DA_H

typedef struct {
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
} dA;

void dA__construct(dA*, int, int, int, double**, double*, double*);
void dA__destruct(dA*);
void dA_get_corrupted_input(dA*, int*, int*, double);
void dA_get_hidden_values(dA*, int*, double*);
void dA_get_reconstructed_input(dA*, double*, double*);
void dA_train(dA*, int*, double, double);
void dA_reconstruct(dA*, int*, double*);

#endif
