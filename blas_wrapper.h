#ifndef BLAS_WRAPPER_H_
#define BLAS_WRAPPER_H_

#include "cblas.h"

namespace blas_wrapper {

void gemm(float* X, float* W, const int M, const int N, const int K,
          float* buffer, bool bias) {
  const float bias_f = bias ? 1.0 : 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K, W,
              N, bias_f, buffer, N);
}

}

#endif