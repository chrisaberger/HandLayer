#include <iostream>
#include "lstm.h"

int main()
{
  std::vector<float> input_data = { 
    0.8310, -0.2477, -0.8029,
    0.2366,  0.2857,  0.6898,
    -0.6331,  0.8795, -0.6842,
    0.4533,  0.2912, -0.8317,
    -0.5525,  0.6355, -0.3968,
    -0.6571, -1.6428,  0.9803, 
    -0.0421, -0.8206,  0.3133,
    -1.1352,  0.3773, -0.2824,
    -2.5667, -1.4303,  0.5009,
    0.5438, -0.4057,  1.1341 };

  int input_size = 3;
  int batch_size = 2;
  int n_hidden = 3;
  LSTM lstm = LSTM(input_size, n_hidden);
  lstm.forward(input_data.data(), batch_size);
  //lstm.print_matrix(lstm.weight_ih_l, 4 * n_hidden, input_size);
  return 0;
}