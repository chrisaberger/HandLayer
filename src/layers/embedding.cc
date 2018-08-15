#include "layers/embedding.h"

template <class T>
Embedding<T>::Embedding(Tensor<T> _weight)
    : num_embeddings(_weight.shape[0]), embedding_dim(_weight.shape[1]) {
  weight = _weight;
}

template <class T>
Tensor<T> Embedding<T>::forward(const Tensor<size_t>& indexes) {
  Tensor<T> result({indexes.numel * embedding_dim});
  result.shape = {indexes.numel, embedding_dim};
  result.numel = indexes.numel * embedding_dim;

  for(size_t i = 0; i < indexes.numel; ++i){
    const size_t idx = indexes[i];
    Tensor<T> dst = result.view(i, i+1);
    Tensor<T> e = weight.view(idx, idx+1);
    dst.copy_data_from(e);
  }

  result.shape = indexes.shape;
  result.shape.push_back(embedding_dim);
  return result;
}

template class Embedding<float>;