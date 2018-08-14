#ifndef LSTM_H_
#define LSTM_H_

#include <math.h>
#include <vector>
#include "tensor.h"

template<class T>
struct LSTM {
  /*
  Model Parameters.

  Weights look like:

                   hidden_size   hidden_size   hidden_size   hidden_size
                 ----------------------------------------------------------
                |             |              |             |              |
    inp size    |             |              |             |              |
                |             |              |             |              |
                --------------|--------------------------------------------
                |             |              |             |              |
    hidden size |             |              |             |              |
                |             |              |             |              |
                -----------------------------------------------------------

  Inputs look like:

              inp size  hidden_size
              ----------------------
             |        |            |
  batch_size |        |            |
             |        |            |
              ----------------------
  */
  Tensor<T> weights_l;
  Tensor<T> bias_ih_l;
  Tensor<T> bias_hh_l;

  /*
  Intermediate Buffers.
  */
  Tensor<T> x_h_in_cat;
  Tensor<T> buffers;

  const size_t input_size;
  const size_t hidden_size;
  const size_t num_layers;
  const bool bias;
  const bool batch_first;
  const int dropout;
  const bool bidirectional;

  LSTM(const size_t input_size, const size_t hidden_size,
       const size_t batch_size, const size_t num_layers = 1,
       const bool bias = true, const bool batch_first = false,
       const int dropout = 0, const bool bidirectional = false);

  std::tuple<Tensor<T>, Tensor<T>> forward(const Tensor<T>& X,
                                     const Tensor<T>& H,
                                     const Tensor<T>& C);

  void set_weights(const Tensor<T>& weight_ih_l,
                   const Tensor<T>& weight_hh_l,
                   const Tensor<T>& bias_ih_ll,
                   const Tensor<T>& bias_hh_ll);

};

#endif