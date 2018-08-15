#include "layers/lstm.h"

template <class T>
LSTM<T>::LSTM(const size_t input_size, const size_t hidden_size,
              const size_t batch_size, const size_t num_layers,
              const bool bias, const bool batch_first,
              const int dropout, const bool bidirectional)
    : input_size(input_size),
      hidden_size(hidden_size),
      num_layers(num_layers),
      bias(bias),
      batch_first(batch_first),
      dropout(dropout),
      bidirectional(bidirectional) {
  // Allocate parameters.
  // std::vector<size_t> a = ;
  weights_l = Tensor<T>({input_size + hidden_size, 4 * hidden_size});
  weights_l.zero();
  bias_ih_l = Tensor<T>({4, hidden_size});
  bias_hh_l = Tensor<T>({4, hidden_size});

  // Allocate intermediate buffers.
  buffers = Tensor<T>({batch_size, 4 * hidden_size});
  x_h_in_cat = Tensor<T>({batch_size, hidden_size + input_size});
}

template <class T>
std::tuple<Tensor<T>, Tensor<T>> LSTM<T>::forward(const Tensor<T>& X,
                                                  const Tensor<T>& H,
                                                  const Tensor<T>& C) {
  const size_t batch_size = X.shape[0];
  /*
  Output buffers.
  */
  Tensor<T> h_t({batch_size, hidden_size});
  Tensor<T> c_t({batch_size, hidden_size});

  // Copy into x_h.
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < input_size; ++j) {
      x_h_in_cat(i, j) = X(i, j);
    }
  }
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < hidden_size; ++j) {
      x_h_in_cat(i, j + input_size) = H(i, j);
    }
  }

  // Copy bias term into place.
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < hidden_size; ++j) {
      buffers(i, j) = bias_ih_l(0, j) + bias_hh_l(0, j);
      buffers(i, j + hidden_size) = bias_ih_l(1, j) + bias_hh_l(1, j);
      buffers(i, j + 2 * hidden_size) = bias_ih_l(2, j) + bias_hh_l(2, j);
      buffers(i, j + 3 * hidden_size) = bias_ih_l(3, j) + bias_hh_l(3, j);
    }
  }

  // weights_l.print();
  gemm(x_h_in_cat.data_ptr(), weights_l.data_ptr(), batch_size, 4 * hidden_size,
       input_size + hidden_size, buffers.data_ptr(), bias);

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < hidden_size; ++j) {
      const T i_t = sigmoid(buffers(i, j));
      const T f_t = sigmoid(buffers(i, j + hidden_size));
      const T g_t = tanh(buffers(i, j + 2 * hidden_size));
      const T o_t = sigmoid(buffers(i, j + 3 * hidden_size));

      const T tmp1 = f_t * C(i, j);
      const T tmp2 = i_t * g_t;
      c_t(i, j) = tmp1 + tmp2;
      h_t(i, j) = tanh(c_t(i, j)) * o_t;
    }
  }
  return std::make_tuple(std::move(h_t), std::move(c_t));
}

template <class T>
void LSTM<T>::set_weights(const Tensor<T>& weight_ih_l,
                          const Tensor<T>& weight_hh_l,
                          const Tensor<T>& bias_ih_ll,
                          const Tensor<T>& bias_hh_ll) {
  // Take matrix and transpose it.
  for (size_t i = 0; i < input_size; ++i) {
    for (size_t j = 0; j < hidden_size; ++j) {
      weights_l(i, j) = weight_ih_l(j, i);
      weights_l(i, j + hidden_size) = weight_ih_l(j + hidden_size, i);
      weights_l(i, j + 2 * hidden_size) = weight_ih_l(j + hidden_size * 2, i);
      weights_l(i, j + 3 * hidden_size) = weight_ih_l(j + hidden_size * 3, i);
    }
  }
  for (size_t i = 0; i < hidden_size; ++i) {
    for (size_t j = 0; j < hidden_size; ++j) {
      weights_l(i + input_size, j) = weight_hh_l(j, i);
      weights_l(i + input_size, hidden_size + j) =
          weight_hh_l(hidden_size + j, i);
      weights_l(i + input_size, 2 * hidden_size + j) =
          weight_hh_l(hidden_size * 2 + j, i);
      weights_l(i + input_size, 3 * hidden_size + j) =
          weight_hh_l(hidden_size * 3 + j, i);
    }
  }

  bias_ih_l = Tensor<T>::copy(bias_ih_ll);
  bias_ih_l.shape = {4, hidden_size};
  bias_hh_l = Tensor<T>::copy(bias_hh_ll);
  bias_hh_l.shape = {4, hidden_size};
}

template class LSTM<float>;
