#include <iostream>
#include "lstm.h"

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

  /*

  std::vector<float> h = {-0.6331, 0.8795, -0.6842, 0.4533, 0.2912, -0.8317};

  */
  
  std::vector<float> h = {-0.6331, 0.8795, -0.6842, -1.1352,  0.3773, -0.2824};
  std::vector<float> c = {-0.5525, 0.6355, -0.3968, -0.6571, -1.6428, 0.9803};

  //std::vector<float> h = {1, 1, 1, 1, 1, 1};
  //std::vector<float> c = {0, 0, 0, 0, 0, 0};

  int input_size = 3;
  int batch_size = 2;
  int n_hidden = 3;
  LSTM lstm = LSTM(input_size, n_hidden, batch_size, 1, true);
  std::tuple<float*, float*> h_c =
      lstm.forward(input_data.data(), h.data(), c.data(), batch_size);

  std::cout << "h_t" << std::endl;
  lstm.print_matrix(std::get<0>(h_c), batch_size, lstm.hidden_size);
  std::cout << "c_t" << std::endl;
  lstm.print_matrix(std::get<1>(h_c), batch_size, lstm.hidden_size);
  return 0;
}