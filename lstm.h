#include <vector>
#include "cblas.h"
#include <math.h>

namespace {
  float sigmoid(float in){
    return 1.0 / (1.0 + exp(-in));
  }

}

struct LSTM {
  float* weight_ih_l;
  float* weight_hh_l;
  float* bias_ih_l;
  float* bias_hh_l;

  const int input_size;
  const int hidden_size;
  const int num_layers;
  const bool bias;
  const bool batch_first;
  const int dropout;
  const bool bidirectional;

  LSTM(const int input_size, const int hidden_size, const int num_layers = 1,
       const bool bias = true, const bool batch_first = false,
       const int dropout = 0, const bool bidirectional = false)
      : input_size(input_size),
        hidden_size(hidden_size),
        num_layers(num_layers),
        bias(bias),
        batch_first(batch_first),
        dropout(dropout),
        bidirectional(bidirectional) {
    weight_ih_l = (float*)malloc(sizeof(float) * 4 * hidden_size * input_size);
    weight_hh_l = (float*)malloc(sizeof(float) * 4 * hidden_size * hidden_size);

    bias_ih_l = (float*)malloc(sizeof(float) * 4 * hidden_size);
    bias_hh_l = (float*)malloc(sizeof(float) * 4 * hidden_size);

    if (input_size == 3 && hidden_size == 3) {
      std::vector<float> weight_ih_l_data = {
          0.2975,  -0.2548, -0.1119, 0.2710, -0.5435, 0.3462,  -0.1188, 0.2937,
          0.0803,  -0.0707, 0.1601,  0.0285, 0.2109,  -0.2250, -0.0421, -0.0520,
          0.0837,  -0.0023, 0.5047,  0.1797, -0.2150, -0.3487, -0.0968, -0.2490,
          -0.1850, 0.0276,  0.3442,  0.3138, -0.5644, 0.3579,  0.1613,  0.5476,
          0.3811,  -0.5260, -0.5489, -0.2785};

      std::vector<float> weight_hh_l_data = {
          0.5070,  -0.0962, 0.2471,  -0.2683, 0.5665,  -0.2443,
          0.4330,  0.0068,  -0.3042, 0.2968,  -0.3065, 0.1698,
          -0.1667, -0.0633, -0.5551, -0.2753, 0.3133,  -0.1403,
          0.5751,  0.4628,  -0.0270, -0.3854, 0.3516,  0.1792,
          -0.3732, 0.3750,  0.3505,  0.5120,  -0.3236, -0.0950,
          -0.0112, 0.0843,  -0.4382, -0.4097, 0.3141,  -0.1354};

      std::vector<float> bias_ih_l_data = {0.2820,  0.0329,  0.1896,  0.1270,
                                           0.2099,  0.2862,  -0.5347, 0.2906,
                                           -0.4059, -0.4356, 0.0351,  -0.0984};

      std::vector<float> bias_hh_l_data = {0.3391,  -0.3344, -0.5133, 0.4202,
                                           -0.0856, 0.3247,  0.1856,  -0.4329,
                                           0.1160,  0.1387,  -0.3866, -0.2739};

      memcpy(weight_ih_l, weight_ih_l_data.data(),
             sizeof(float) * 4 * hidden_size * input_size);
      memcpy(weight_hh_l, weight_hh_l_data.data(),
             sizeof(float) * 4 * hidden_size * hidden_size);
      memcpy(bias_ih_l, bias_ih_l_data.data(), sizeof(float) * 4 * hidden_size);
      memcpy(bias_hh_l, bias_hh_l_data.data(), sizeof(float) * 4 * hidden_size);
    }
  }

  void layer(float* X, float* W, float* bias, const int M, const int N,
                     const int K, float* buffer, std::function<float(float)> f) {
    //memcpy(buffer, bias, sizeof(float) * 4 * hidden_size);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, X, K,
                W, N, 0.0, buffer, N);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << buffer[i * input_size + j] << " " << f(buffer[i * input_size + j]) << std::endl;
        buffer[i * input_size + j] = f(buffer[i * input_size + j]);
      }
    }
  }

  void forward(float* X, int batch_size){
    float* state_g = (float*)malloc(sizeof(float) * batch_size * hidden_size);

    print_matrix(bias_ih_l, 1, hidden_size);

    std::cout << " " << std::endl;
    print_matrix(X, batch_size, input_size);

    std::cout << " " << std::endl;
    print_matrix(weight_ih_l, hidden_size, input_size);

    layer(weight_ih_l, X, bias_ih_l, batch_size, hidden_size,
                  input_size, state_g, sigmoid);

    print_matrix(state_g, batch_size, hidden_size);
    std::cout << "End forward" << std::endl;
  }


  void print_matrix(float* data, int dim1, int dim2) {
    std::cout << " [ ";
    for (int i = 0; i < dim1; ++i) {
      std::cout << " [ ";
      for (int j = 0; j < dim2; ++j) {
        std::cout << data[i * dim2 + j] << ",";
      }
      std::cout << " ] " << std::endl;
    }
    std::cout << " ] " << std::endl;
  }
};