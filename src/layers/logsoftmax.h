#ifndef LOGSOFTMAX_H_
#define LOGSOFTMAX_H_

#include <math.h>
#include <vector>
#include "tensor.h"

template <class T>
struct LogSoftmax {

  static Tensor<T> forward(const Tensor<T>& inp);
};

#endif