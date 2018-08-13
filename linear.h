
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
         const bool _bias = true)
      : in_features(_in_features), out_features(_out_features), bias(_bias) {
    weight_l = Tensor<T>({in_features, out_features});
    bias_l = Tensor<T>({out_features});
  }

  void set_weights(Tensor<T> _weight_l, Tensor<T> _bias_l = Tensor<T>({0})) {
    for (size_t i = 0; i < in_features; ++i) {
      for (size_t j = 0; j < out_features; ++j) {
        weight_l(i, j) = _weight_l(j, i);
      }
    }

    bias_l = _bias_l;
  }

  Tensor<T> forward(const Tensor<T>& X) {
    Tensor<T> output({X.shape[0], out_features});

    if(bias){
      for(size_t i = 0; i < X.shape[0]; ++i){
        for(size_t j = 0; j < out_features; ++j){
          output(i, j) = bias_l[j];
          std::cout << bias_l[j] << std::endl;
        }
      }
    }

    X.print();
    weight_l.print();
    bias_l.print();
    std::cout << "OUT" << std::endl;
    output.print();
    std::cout << X.shape[0] << " " << out_features << " " << in_features << std::endl;
    gemm(X.data_ptr(), weight_l.data_ptr(), X.shape[0],
                       out_features, in_features, output.data_ptr(), bias);
    return output;
  }
};

#endif