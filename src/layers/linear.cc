#include "layers/linear.h"

template <class T>
Linear<T>::Linear(const size_t _in_features, const size_t _out_features,
                  const bool _bias)
    : in_features(_in_features), out_features(_out_features), bias(_bias) {
  weight_l = Tensor<T>({in_features, out_features});
  bias_l = Tensor<T>({out_features});
}

template <class T>
void Linear<T>::set_weights(Tensor<T> _weight_l, Tensor<T> _bias_l) {
  for (size_t i = 0; i < in_features; ++i) {
    for (size_t j = 0; j < out_features; ++j) {
      weight_l(i, j) = _weight_l(j, i);
    }
  }

  bias_l = _bias_l;
}

template <class T>
Tensor<T> Linear<T>::forward(const Tensor<T>& X) {
  Tensor<T> output({X.shape[0], out_features});

  if (bias) {
    for (size_t i = 0; i < X.shape[0]; ++i) {
      for (size_t j = 0; j < out_features; ++j) {
        output(i, j) = bias_l[j];
      }
    }
  }

  gemm(X.data_ptr(), weight_l.data_ptr(), X.shape[0], out_features, in_features,
       output.data_ptr(), bias);
  return output;
}

template class Linear<bfloat16>;
template class Linear<float>;
