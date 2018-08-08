#include <math.h>
#include <vector>
#include "cblas.h"

namespace {
float fp_sigmoid(const float in) { return 1.0 / (1.0 + exp(-in)); }
float fp_tanh(const float in) { return tanh(in); }
}

struct LSTM {
  /*
  Model Parameters.
  */
  float* weights_l;
  float* bias_ih_l;
  float* bias_hh_l;

  /*
  Intermediate Buffers.
  */
  float* f_t;
  float* i_t;
  float* g_t;
  float* o_t;
  float* x_h_in_cat;

  const int input_size;
  const int hidden_size;
  const int num_layers;
  const bool bias;
  const bool batch_first;
  const int dropout;
  const bool bidirectional;

  LSTM(const int input_size, const int hidden_size, const int batch_size,
       const int num_layers = 1, const bool bias = true,
       const bool batch_first = false, const int dropout = 0,
       const bool bidirectional = false)
      : input_size(input_size),
        hidden_size(hidden_size),
        num_layers(num_layers),
        bias(bias),
        batch_first(batch_first),
        dropout(dropout),
        bidirectional(bidirectional) {
    // Allocate parameters.
    weights_l = (float*)malloc(sizeof(float) * 4 * hidden_size *
                               (input_size + hidden_size));
    bias_ih_l = (float*)malloc(sizeof(float) * 4 * hidden_size);
    bias_hh_l = (float*)malloc(sizeof(float) * 4 * hidden_size);

    // Allocate intermediate buffers.
    f_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    i_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    g_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    o_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    x_h_in_cat =
        (float*)malloc(sizeof(float) * batch_size * (hidden_size + input_size));

    if (input_size == 3 && hidden_size == 3) {
      std::vector<float> weight_ih_l_data = {
          0.2975,  -0.2548, -0.1119, 0.2710, -0.5435, 0.3462,  -0.1188, 0.2937,
          0.0803,  -0.0707, 0.1601,  0.0285, 0.2109,  -0.2250, -0.0421, -0.0520,
          0.0837,  -0.0023, 0.5047,  0.1797, -0.2150, -0.3487, -0.0968, -0.2490,
          -0.1850, 0.0276,  0.3442,  0.3138, -0.5644, 0.3579,  0.1613,  0.5476,
          0.3811,  -0.5260, -0.5489, -0.2785};

      /*
      std::vector<float> weight_ih_l_data = {
         0.2975, -0.2548, -0.1119,
         0.2710, -0.5435,  0.3462,
         -0.1188,  0.2937,  0.0803,
         1,  1, 1,
          1,  1, 1,
          1,  1, 1,
          0.5047,  0.1797, -0.2150,
          -0.3487, -0.0968, -0.2490,
          -0.1850,  0.0276,  0.3442,
          1,  1, 1,
          1,  1, 1,
          1,  1, 1,
        };
        */

      /*
  std::vector<float> weight_ih_l_data = {
      1,  1, 1, 1, 1, 1,  1, 1,
      1,  1, 1,  1, 1,  1, 1, 1,
      1,  1, 1,  1, 1, 1, 1, 1,
      1, 1,  1,  1, 1, 1,  1,  1,
      1,  1, 1, 1};
*/

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

      size_t input_copy_size = hidden_size * input_size;
      size_t input_offset = (input_size + hidden_size) * hidden_size;
      size_t hidden_copy_size = hidden_size * hidden_size;

      memcpy(weights_l, weight_ih_l_data.data(),
             sizeof(float) * input_copy_size);
      memcpy(&weights_l[input_offset],
             &weight_ih_l_data.data()[input_copy_size],
             sizeof(float) * input_copy_size);
      memcpy(&weights_l[2 * input_offset],
             &weight_ih_l_data.data()[2 * input_copy_size],
             sizeof(float) * input_copy_size);
      memcpy(&weights_l[3 * input_offset],
             &weight_ih_l_data.data()[3 * input_copy_size],
             sizeof(float) * input_copy_size);

      memcpy(&weights_l[input_copy_size], weight_hh_l_data.data(),
             sizeof(float) * hidden_copy_size);
      memcpy(&weights_l[input_copy_size + input_offset],
             &weight_hh_l_data.data()[hidden_copy_size],
             sizeof(float) * hidden_copy_size);
      memcpy(&weights_l[input_copy_size + 2 * input_offset],
             &weight_hh_l_data.data()[2 * hidden_copy_size],
             sizeof(float) * hidden_copy_size);
      memcpy(&weights_l[input_copy_size + 3 * input_offset],
             &weight_hh_l_data.data()[3 * hidden_copy_size],
             sizeof(float) * hidden_copy_size);

      // Take matrix and transpose it.

      for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
          weights_l[i * hidden_size + j] =
              weight_ih_l_data.data()[j * input_size + i];
          weights_l[input_offset + i * hidden_size + j] =
              weight_ih_l_data.data()[input_copy_size + j * input_size + i];
          weights_l[2 * input_offset + i * hidden_size + j] =
              weight_ih_l_data.data()[2 * input_copy_size + j * input_size + i];
          weights_l[3 * input_offset + i * hidden_size + j] =
              weight_ih_l_data.data()[3 * input_copy_size + j * input_size + i];
        }
      }
      for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
          weights_l[input_copy_size + i * hidden_size + j] =
              weight_hh_l_data.data()[j * hidden_size + i];
          weights_l[input_copy_size + input_offset + i * hidden_size + j] =
              weight_hh_l_data.data()[hidden_copy_size + j * hidden_size + i];
          weights_l[input_copy_size + 2 * input_offset + i * hidden_size + j] =
              weight_hh_l_data
                  .data()[2 * hidden_copy_size + j * hidden_size + i];
          weights_l[input_copy_size + 3 * input_offset + i * hidden_size + j] =
              weight_hh_l_data
                  .data()[3 * hidden_copy_size + j * hidden_size + i];
        }
      }

      /*
      memcpy(bias_ih_l, bias_ih_l_data.data(), sizeof(float) * 4 * hidden_size);
      memcpy(bias_hh_l, bias_hh_l_data.data(), sizeof(float) * 4 * hidden_size);
      */
    }
  }

  void layer(float* X, float* W, float* bias, const int M, const int N,
             const int K, float* buffer, std::function<float(float)> f) {
    // memcpy(buffer, bias, sizeof(float) * 4 * hidden_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K,
                W, N, 0.0, buffer, N);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        buffer[i * N + j] = f(buffer[i * N + j]);
      }
    }
  }

  std::tuple<float*, float*> forward(const float* X, const float* H,
                                     const float* C, const int batch_size) {
    /*
    Output buffers.
    */
    float* h_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    float* c_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);

    // Copy into x_h.
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < input_size; ++j) {
        x_h_in_cat[i * (input_size + hidden_size) + j] = X[i * input_size + j];
      }
    }
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        x_h_in_cat[i * (input_size + hidden_size) + j + input_size] =
            H[i * hidden_size + j];
      }
    }

    // We might not always need to do this if the input comes in concatenated.
    const size_t weights_index = (input_size + hidden_size) * hidden_size;

    layer(x_h_in_cat, &weights_l[weights_index * 0], bias_ih_l, batch_size,
          hidden_size, (input_size + hidden_size), i_t, fp_sigmoid);

    layer(x_h_in_cat, &weights_l[weights_index * 1], bias_ih_l, batch_size,
          hidden_size, (input_size + hidden_size), f_t, fp_sigmoid);

    layer(x_h_in_cat, &weights_l[weights_index * 2], bias_ih_l, batch_size,
          hidden_size, (input_size + hidden_size), g_t, fp_tanh);

    layer(x_h_in_cat, &weights_l[weights_index * 3], bias_ih_l, batch_size,
          hidden_size, (input_size + hidden_size), o_t, fp_sigmoid);

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < hidden_size; ++j) {
        const size_t index = i * hidden_size + j;

        const float tmp1 = f_t[index] * C[index];
        const float tmp2 = i_t[i * hidden_size + j] * g_t[i * hidden_size + j];
        c_t[index] = tmp1 + tmp2;
        h_t[index] = tanh(c_t[index]) * o_t[index];
      }
    }
    return std::make_tuple(h_t, c_t);
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