#include <iostream>
#include "lstm.h"
#include "tensor.h"
#include "linear.h"
#include "embedding.h"
#include "logsoftmax.h"

int main() {
  const size_t embedding_dim = 10;
  const size_t num_words = 9;
  const size_t hidden_dim = 6;
  const size_t num_out_features = 3;

  Tensor<size_t> input = Tensor<size_t>::from_vec({0, 1, 2, 3, 4}, {5});

  Embedding<float> emb =
      Embedding<float>(Tensor<float>::from_npy("../pos_tagging/embedding.npy"));
  // Embedding<float>(Tensor<float>({num_words,embedding_dim}));

  emb.weight.print();

  LSTM lstm = LSTM(embedding_dim, hidden_dim, 1, 1, true);
  std::cout << "INIT LSTM" << std::endl;
  lstm.set_weights(Tensor<float>::from_npy("../pos_tagging/weight_ih_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/weight_hh_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/bias_ih_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/bias_hh_l0.npy"));
  std::cout << "INIT LSTM" << std::endl;

  Linear<float> linear(hidden_dim, num_out_features, true);
  linear.set_weights(
      Tensor<float>::from_npy("../pos_tagging/linear_weight.npy"),
      Tensor<float>::from_npy("../pos_tagging/linear_bias.npy"));

  // out.print();
  // exit(0);

  Tensor<float> h({1, hidden_dim});
  Tensor<float> c({1, hidden_dim});
  h.zero();
  c.zero();

  Tensor<float> dec = emb.forward(input);
  dec.print();

  std::cout << "LSTM FORWARD" << std::endl;
  auto h_c = lstm.forward(dec.view(0, 1), h, c);
  std::cout << "h_t" << std::endl;
  std::get<0>(h_c).print();
  std::cout << "c_t" << std::endl;
  std::get<1>(h_c).print();

  Tensor<float> lin_out = linear.forward(std::get<0>(h_c));
  lin_out.print();

  Tensor<float> out = LogSoftmax<float>::forward(lin_out);

  out.print();
  return 0;
}