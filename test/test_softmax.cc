#include "third_party/catch/catch.hpp"
#include "tensor.h"
#include "layers/logsoftmax.h"
#include <limits>

TEST_CASE("Testing logsoftmax", "[float_logsoftmax]") {
  const size_t num_out_features = 3;
  const size_t batch_size = 5;

  Tensor<float> lin_out = Tensor<float>::from_vec(
      {0.30326727032661438, -0.24030521512031555, -0.18203482031822205,
       0.4341903030872345, -0.2601546049118042, -0.10730992257595062,
       0.28393235802650452, -0.13939939439296722, -0.3071112334728241,
       0.35775145888328552, -0.13673341274261475, -0.21190550923347473,
       0.29672056436538696, -0.1552102118730545, -0.30953633785247803},
      {batch_size, num_out_features});

  Tensor<float> out = LogSoftmax<float>::forward(lin_out);

  Tensor<float> out_baseline = Tensor<float>::from_vec(
      {-0.78672003746032715, -1.3302924633026123, -1.2720221281051636,
       -0.73298126459121704, -1.4273260831832886, -1.2744814157485962,
       -0.79236358404159546, -1.2156953811645508, -1.3834072351455688,
       -0.77730643749237061, -1.2717914581298828, -1.3469634056091309,
       -0.78014415502548218, -1.2320748567581177, -1.3864010572433472},
      {batch_size, num_out_features});

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_out_features; ++j) {
      REQUIRE(out(i, j) == out_baseline(i, j));
    }
  }
}
