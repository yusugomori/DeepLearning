#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

typedef struct {
  int N;
  int n_in;
  int n_out;
  double **W;
  double *b;
} LogisticRegression;

void LogisticRegression__construct(LogisticRegression*, int, int, int);
void LogisticRegression__destruct(LogisticRegression*);
void LogisticRegression_train(LogisticRegression*, int*, int*, double);
void LogisticRegression_softmax(LogisticRegression*, double*);
void LogisticRegression_predict(LogisticRegression*, int*, double*);

#endif
