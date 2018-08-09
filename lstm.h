#include <math.h>
#include <vector>
#include "cblas.h"
#include "tensor.h"

namespace {
float fp_sigmoid(const float in) { return 1.0 / (1.0 + exp(-in)); }
float fp_tanh(const float in) { return tanh(in); }
}

struct LSTM {
  /*
  Model Parameters.

  Weights look like:
                   
                   hidden_size   hidden_size   hidden_size   hidden_size
                 ----------------------------------------------------------
                |             |              |             |              |
    inp size    |             |              |             |              |
                |             |              |             |              |
                --------------|--------------------------------------------  
                |             |              |             |              |
    hidden size |             |              |             |              |
                |             |              |             |              |
                -----------------------------------------------------------

  Inputs look like:
                
              inp size  hidden_size   i   h    i    h   i   h   i   h
              ---------------------------------------------------------
             |        |            |    |    |    |   |   |   |   |   |
  batch_size |        |            |    |    |    |   |   |   |   |   |
             |        |            |    |    |    |   |   |   |   |   |
              ---------------------------------------------------------     
  */    
  Tensor<float> weights_l;
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
  float* buffers;

  const size_t input_size;
  const size_t hidden_size;
  const size_t num_layers;
  const bool bias;
  const bool batch_first;
  const int dropout;
  const bool bidirectional;

  LSTM(const size_t input_size, const size_t hidden_size, const size_t batch_size,
       const size_t num_layers = 1, const bool bias = true,
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
    //std::vector<size_t> a = ;
    weights_l = Tensor<float>({input_size + hidden_size, 4 * hidden_size});
    weights_l.zero();

    bias_ih_l = (float*)malloc(sizeof(float) * 4 * hidden_size);
    bias_hh_l = (float*)malloc(sizeof(float) * 4 * hidden_size);

    // Allocate intermediate buffers.
    buffers = (float*)malloc(4 * sizeof(float) * batch_size * hidden_size);
    i_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    g_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    o_t = (float*)malloc(sizeof(float) * batch_size * hidden_size);
    x_h_in_cat =
        (float*)malloc(sizeof(float) * batch_size * (hidden_size + input_size));
  }

  void gemm(float* X, float* W, const int M, const int N,
             const int K, float* buffer) {
    // memcpy(buffer, bias, sizeof(float) * 4 * hidden_size);
    const float bias_f = bias ? 1.0 : 0.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K,
                W, N, bias_f, buffer, N);
  }

  void layer(float* X, float* W, float* bias, const int M, const int N,
             const int K, float* buffer, std::function<float(float)> f) {
    // memcpy(buffer, bias, sizeof(float) * 4 * hidden_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K,
                W, N, 1.0, buffer, N);
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

    // Copy bias term into place.
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        buffers[i * hidden_size * 4 + j] =
            bias_ih_l[0*hidden_size + j] + bias_hh_l[0* hidden_size + j];
        buffers[hidden_size + i * hidden_size * 4 + j] =
            bias_ih_l[1*hidden_size + j] + bias_hh_l[1* hidden_size + j];

        buffers[2 * hidden_size + i * hidden_size * 4 + j] =
            bias_ih_l[2*hidden_size + j] + bias_hh_l[2* hidden_size + j];

        buffers[3 * hidden_size + i * hidden_size * 4 + j] =
            bias_ih_l[3*hidden_size + j] + bias_hh_l[3* hidden_size + j];
      }
    }

    print_matrix(buffers, batch_size, 4 * hidden_size);

    print_matrix(x_h_in_cat, batch_size, input_size + hidden_size);

    weights_l.print();

    gemm(x_h_in_cat, weights_l.data_ptr(), batch_size, 4 * hidden_size,
         input_size + hidden_size, buffers);

    print_matrix(buffers, batch_size, 4* hidden_size);


    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < hidden_size; ++j) {
        const float i_t = fp_sigmoid(buffers[i * hidden_size * 4 + j]);
        const float f_t =
            fp_sigmoid(buffers[hidden_size + i * hidden_size * 4 + j]);
        const float g_t =
            fp_tanh(buffers[2 * hidden_size + i * hidden_size * 4 + j]);
        const float o_t =
            fp_sigmoid(buffers[3 * hidden_size + i * hidden_size * 4 + j]);

        const float tmp1 = f_t * C[i * hidden_size + j];
        const float tmp2 = i_t * g_t;
        c_t[i * hidden_size + j] = tmp1 + tmp2;
        h_t[i * hidden_size + j] =
            fp_tanh(c_t[i * hidden_size + j]) * o_t;
      }
    }
    return std::make_tuple(h_t, c_t);
  }


  void set_weights(const Tensor<float>& weight_ih_l,
                   const Tensor<float>& weight_hh_l,
                   const Tensor<float>& bias_ih_ll,
                   const Tensor<float>& bias_hh_ll) {
    // Take matrix and transpose it.
    for (size_t i = 0; i < input_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        weights_l(i, j) = weight_ih_l(j, i);
        weights_l(i, j + hidden_size) = weight_ih_l(j + input_size, i);
        weights_l(i, j + 2 * hidden_size) = weight_ih_l(j + input_size * 2, i);
        weights_l(i, j + 3 * hidden_size) = weight_ih_l(j + input_size * 3, i);
      }
    }
    for (size_t i = 0; i < hidden_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        weights_l(i + input_size, j) = weight_hh_l(j, i);
        weights_l(i + input_size, hidden_size + j) =
            weight_hh_l(hidden_size + j, i);
        weights_l(i + input_size, 2 * hidden_size + j) =
            weight_hh_l(hidden_size * 2 + j, i);
        weights_l(i + input_size, 3 * hidden_size + j) =
            weight_hh_l(hidden_size * 3 + j, i);
      }
    }

    memcpy(bias_ih_l, bias_ih_ll.data.get(), sizeof(float) * 4 * hidden_size);
    memcpy(bias_hh_l, bias_hh_ll.data.get(), sizeof(float) * 4 * hidden_size);
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