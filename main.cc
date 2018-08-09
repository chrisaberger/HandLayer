#include <iostream>
#include "lstm.h"
#include "tensor.h"

int main() {
  std::vector<float> input_data = {
      0.8310,  -0.2477, -0.8029, 
      0.2366,  0.2857,  0.6898,  
      -0.6331, 0.8795,  -0.6842, 
      0.4533,  0.2912,  -0.8317, 
      -0.5525, 0.6355,  -0.3968, 
      -0.6571, -1.6428, 0.9803,  
      -0.0421, -0.8206, 0.3133,  
      -1.1352, 0.3773,  -0.2824,
      -2.5667, -1.4303, 0.5009,  
      0.5438,  -0.4057, 1.1341};

  size_t input_size = 3;
  size_t batch_size = 2;
  size_t n_hidden = 3;

  Tensor<float> input = Tensor<float>::from_vec(
      input_data, {input_data.size() / input_size, input_size});

  input.print();

  Tensor<float> h = Tensor<float>::from_npy("../h.npy");
  h.shape = {batch_size, n_hidden};
  h.print();

  Tensor<float> c = Tensor<float>::from_npy("../c.npy");
  c.shape = {batch_size, n_hidden};
  c.print();

  //std::vector<float> h = {1, 1, 1, 1, 1, 1};
  //std::vector<float> c = {0, 0, 0, 0, 0, 0};


  LSTM lstm = LSTM(input_size, n_hidden, batch_size, 1, true);
  lstm.set_weights(Tensor<float>::from_npy("../weight_ih.npy"),
                   Tensor<float>::from_npy("../weight_hh.npy"),
                   Tensor<float>::from_npy("../bias_ih.npy"),
                   Tensor<float>::from_npy("../bias_hh.npy"));
  std::tuple< Tensor<float>, Tensor<float> > h_c =
      lstm.forward(input, h, c, batch_size, 0);

  std::cout << "h_t" << std::endl;
  std::get<0>(h_c).print();
  std::cout << "c_t" << std::endl;
  std::get<1>(h_c).print();

  return 0;
}