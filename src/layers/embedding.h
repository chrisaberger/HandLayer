#ifndef EMBEDDING_H_
#define EMBEDDING_H_

#include <math.h>
#include <vector>
#include "tensor.h"

template <class T>
struct Embedding {
  Tensor<T> weight;
  const size_t num_embeddings;
  const size_t embedding_dim;

  Embedding(Tensor<T> _weight);

  Tensor<T> forward(const Tensor<size_t>& indexes);
};

#endif