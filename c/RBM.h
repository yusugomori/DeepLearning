#ifndef RBM_H
#define RBM_H

typedef struct {
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
} RBM;

void RBM__construct(RBM*, int, int, int, double**, double*, double*);
void RBM__destruct(RBM*);
void RBM_contrastive_divergence(RBM*, int*, double, int);
void RBM_sample_h_given_v(RBM*, int*, double*, int*);
void RBM_sample_v_given_h(RBM*, int*, double*, int*);
double RBM_propup(RBM*, int*, double*, double);
double RBM_propdown(RBM*, int*, int, double);
void RBM_gibbs_hvh(RBM*, int*, double*, int*, double*, int*);
void RBM_reconstruct(RBM*, int*, double*);

#endif
