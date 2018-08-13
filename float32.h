#ifndef FLOAT32_H_
#define FLOAT32_H_

#include <iostream>
#include <string>
#include <math.h>
#include "cblas.h"

inline float sigmoid(const float in) { return 1.0 / (1.0 + exp(-in)); }
inline void gemm(float* X, float* W, const int M, const int N, const int K,
          float* buffer, bool bias) {
  const float bias_f = bias ? 1.0 : 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K, W,
              N, bias_f, buffer, N);
}

#endif