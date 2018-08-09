#include <math.h>
#include <vector>
#include "cblas.h"
#include "tensor.h"

typedef  Tensor<float> t_type;

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
                
              inp size  hidden_size 
              ----------------------
             |        |            |
  batch_size |        |            |
             |        |            |
              ----------------------   
  */    
  Tensor<float> weights_l;
  Tensor<float> bias_ih_l;
  Tensor<float> bias_hh_l;

  /*
  Intermediate Buffers.
  */
  Tensor<float> x_h_in_cat;
  Tensor<float> buffers;

  const size_t input_size;
  const size_t hidden_size;
  const size_t num_layers;
  const bool bias;
  const bool batch_first;
  const int dropout;
  const bool bidirectional;

  LSTM(const size_t input_size, const size_t hidden_size,
       const size_t batch_size, const size_t num_layers = 1,
       const bool bias = true, const bool batch_first = false,
       const int dropout = 0, const bool bidirectional = false)
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
    // Allocate intermediate buffers.
    buffers = Tensor<float>({batch_size, 4 * hidden_size});
    x_h_in_cat = Tensor<float>({batch_size, hidden_size + input_size});
  }

  void gemm(float* X, float* W, const int M, const int N,
             const int K, float* buffer) {
    // memcpy(buffer, bias, sizeof(float) * 4 * hidden_size);
    const float bias_f = bias ? 1.0 : 0.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, X, K,
                W, N, bias_f, buffer, N);
  }

  std::tuple<t_type, t_type> forward(const Tensor<float>& X,
                                     const Tensor<float>& H,
                                     const Tensor<float>& C) {
    const size_t batch_size = X.shape[0];
    /*
    Output buffers.
    */
    Tensor<float> h_t({batch_size, hidden_size});
    Tensor<float> c_t({batch_size, hidden_size});

    // Copy into x_h.
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < input_size; ++j) {
        x_h_in_cat(i, j) = X(i, j);
      }
    }
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        x_h_in_cat(i, j + input_size) = H(i, j);
      }
    }

    // Copy bias term into place.
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < hidden_size; ++j) {
        buffers(i, j) = bias_ih_l(0, j) + bias_hh_l(0, j);
        buffers(i, j + hidden_size) = bias_ih_l(1, j) + bias_hh_l(1, j);
        buffers(i, j + 2 * hidden_size) = bias_ih_l(2, j) + bias_hh_l(2, j);
        buffers(i, j + 3 * hidden_size) = bias_ih_l(3, j) + bias_hh_l(3, j);
      }
    }

    weights_l.print();

    gemm(x_h_in_cat.data_ptr(), weights_l.data_ptr(), batch_size,
         4 * hidden_size, input_size + hidden_size, buffers.data_ptr());

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < hidden_size; ++j) {
        const float i_t = fp_sigmoid(buffers(i, j));
        const float f_t = fp_sigmoid(buffers(i, j + hidden_size));
        const float g_t = fp_tanh(buffers(i, j + 2 * hidden_size));
        const float o_t = fp_sigmoid(buffers(i, j + 3 * hidden_size));

        const float tmp1 = f_t * C(i, j);
        const float tmp2 = i_t * g_t;
        c_t(i, j) = tmp1 + tmp2;
        h_t(i, j) = fp_tanh(c_t(i, j)) * o_t;
      }
    }
    return std::make_tuple(std::move(h_t), std::move(c_t));
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

    bias_ih_l = Tensor<float>::copy(bias_ih_ll);
    bias_ih_l.shape = {4, hidden_size};
    bias_hh_l = Tensor<float>::copy(bias_hh_ll);
    bias_hh_l.shape = {4, hidden_size};
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