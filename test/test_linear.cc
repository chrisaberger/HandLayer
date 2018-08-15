#include "third_party/catch/catch.hpp"
#include "tensor.h"
#include "layers/linear.h"
#include <limits>

TEST_CASE("Testing linear", "[float_linear]") {
  const size_t hidden_dim = 6;
  const size_t num_out_features = 3;
  const size_t batch_size = 5;

  Tensor<float> h_t = Tensor<float>::from_vec(
      {
          0.024514690041542053,  -0.16978850960731506,  -0.052973240613937378,
          -0.11851642280817032,  -0.070385850965976715, -0.039756070822477341,
          -0.023068031296133995, -0.075698599219322205, 0.21819275617599487,
          -0.081390604376792908, -0.21726676821708679,  -0.076678812503814697,
          0.064265936613082886,  -0.010811485350131989, -0.30380198359489441,
          -0.032774690538644791, -0.088030479848384857, -0.158418208360672,
          -0.15570321679115295,  -0.11451181024312973,  0.0054684164933860302,
          0.047292158007621765,  -0.1657407283782959,   -0.087142571806907654,
          -0.22385551035404205,  -0.047219909727573395, 0.048143111169338226,
          0.074343286454677582,  0.26692992448806763,   0.028252197429537773
      },
      {batch_size, hidden_dim});

  Linear<float> linear(hidden_dim, num_out_features, true);
  linear.set_weights(Tensor<float>::from_npy("../test/data/linear_weight.npy"),
                     Tensor<float>::from_npy("../test/data/linear_bias.npy"));

  Tensor<float> lin_out = linear.forward(h_t);
  Tensor<float> lin_out_baseline = Tensor<float>::from_vec(
      {0.30326727032661438, -0.24030521512031555, -0.18203482031822205,
       0.4341903030872345, -0.2601546049118042, -0.10730992257595062,
       0.28393235802650452, -0.13939939439296722, -0.3071112334728241,
       0.35775145888328552, -0.13673341274261475, -0.21190550923347473,
       0.29672056436538696, -0.1552102118730545, -0.30953633785247803},
      {batch_size, num_out_features});

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_out_features; ++j) {
      REQUIRE(lin_out(i, j) == lin_out_baseline(i, j));
    }
  }
}
