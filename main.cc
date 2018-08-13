#include <iostream>
#include "embedding.h"
#include "linear.h"
#include "logsoftmax.h"
#include "lstm.h"
#include "tensor.h"

int main() {
  /*
  auto bf = bfloat16(82.3);
  std::cout << sizeof(bfloat16) << std::endl;
  std::cout << bf._idata << " " << bf._fdata << std::endl;
  bf.print_fields();
  bf.clamp();
  bf.print_fields();

  auto bf2 = bf + bfloat16(1.0);
  std::cout << bf2 << std::endl;
  bf2.print_fields();

  bf2 = bf * bfloat16(2.0);
  std::cout << bf2 << std::endl;
  bf2.print_fields();

  Tensor<bfloat16> myt = Tensor<bfloat16>(
      Tensor<float>::from_vec({82.3, 83.3, -3.0, -162.4}, {2, 2}));
  myt.print();

  Tensor<float> myt2 = Tensor<float>(
      Tensor<float>::from_vec({82.3, 83.3, -3.0, -162.4}, {2, 2}));
  myt2.print();

  Embedding<bfloat16> emb2 = Embedding<bfloat16>(Tensor<bfloat16>(
      Tensor<float>::from_npy("../pos_tagging/embedding.npy")));
  */

  const size_t embedding_dim = 10;
  const size_t num_words = 9;
  const size_t hidden_dim = 6;
  const size_t num_out_features = 3;

  /*
  LSTM<bfloat16> lstm2 = LSTM<bfloat16>(embedding_dim, hidden_dim, 1, 1, true);
  std::cout << "INIT LSTM" << std::endl;
  lstm2.set_weights(Tensor<float>::from_npy("../pos_tagging/weight_ih_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/weight_hh_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/bias_ih_l0.npy"),
                   Tensor<float>::from_npy("../pos_tagging/bias_hh_l0.npy"));


  exit(0);
  */



  Tensor<size_t> input = Tensor<size_t>::from_vec({0, 1, 2, 3, 4}, {5});

  Embedding<float> emb =
      Embedding<float>(Tensor<float>::from_npy("../pos_tagging/embedding.npy"));
  // Embedding<float>(Tensor<float>({num_words,embedding_dim}));

  emb.weight.print();

  LSTM<float> lstm = LSTM<float>(embedding_dim, hidden_dim, 1, 1, true);
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