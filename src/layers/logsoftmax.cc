#include "layers/logsoftmax.h"

template <class T>
Tensor<T> LogSoftmax<T>::forward(const Tensor<T>& inp) {
  Tensor<T> out = Tensor<T>::copy(inp);
  assert(inp.shape.size() == 2);
  for (size_t i = 0; i < inp.shape[0]; ++i) {
    T value = (T)0;

    T max_value = (T)0;

    // Find max value.
    for (size_t j = 0; j < inp.shape[1]; ++j) {
      if (inp(i, j) > max_value) {
        max_value = inp(i, j);
      }
    }

    // Stable softmax
    Tensor<T> new_t = out.view(i, i + 1);
    T sum = (T)0;
    for (size_t j = 0; j < inp.shape[1]; ++j) {
      new_t[j] -= max_value;
      new_t[j] = exp(new_t[j]);
      sum += new_t[j];
    }
    for (size_t j = 0; j < inp.shape[1]; ++j) {
      new_t[j] = log(new_t[j] / sum);
    }
  }
  return out;
}

template class LogSoftmax<float>;