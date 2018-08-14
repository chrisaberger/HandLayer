#ifndef LINEAR_H_
#define LINEAR_H_

#include <math.h>
#include <vector>
#include "tensor.h"

template <class T>
struct Linear {
  const size_t in_features;
  const size_t out_features;
  const bool bias;

  Tensor<T> weight_l;
  Tensor<T> bias_l;

  Linear(const size_t _in_features, const size_t _out_features,
         const bool _bias = true);

  void set_weights(Tensor<T> _weight_l, Tensor<T> _bias_l = Tensor<T>({0}));
  Tensor<T> forward(const Tensor<T>& X);
};

#endif