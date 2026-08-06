#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "MNNTestSuite.h"

using namespace MNN::Express;

class HexagonMulSiluPWLTest : public MNNTestCase {
public:
  virtual bool run(int precision) override {
    if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_HEXAGON) {
      MNN_PRINT(
          "Skip HexagonMulSiluPWLTest: Hexagon backend is not selected\n");
      return true;
    }

    constexpr int kSize = 8193;
    std::vector<float> upData(kSize);
    std::vector<float> gateData(kSize);
    for (int i = 0; i < kSize; ++i) {
      gateData[i] = -12.0f + 24.0f * static_cast<float>(i) /
                                 static_cast<float>(kSize - 1);
      upData[i] = -8.0f + 16.0f * static_cast<float>((i * 37) % kSize) /
                              static_cast<float>(kSize - 1);
    }

    int cursor = 0;
    for (int edge = -32; edge <= 32; ++edge) {
      const float x = 0.25f * static_cast<float>(edge);
      gateData[cursor++] = x - 0.01f;
      gateData[cursor++] = x;
      gateData[cursor++] = x + 0.01f;
    }

    auto up = _Input({kSize}, NCHW, halide_type_of<float>());
    auto gate = _Input({kSize}, NCHW, halide_type_of<float>());
    std::copy(upData.begin(), upData.end(), up->writeMap<float>());
    std::copy(gateData.begin(), gateData.end(), gate->writeMap<float>());
    auto result = _MulSilu(up, gate);
    const float *output = result->readMap<float>();
    if (output == nullptr) {
      MNN_ERROR("HexagonMulSiluPWLTest: failed to read output\n");
      return false;
    }

    float maxError = 0.0f;
    int maxIndex = 0;
    for (int i = 0; i < kSize; ++i) {
      const float expected =
          upData[i] * gateData[i] / (1.0f + std::exp(-gateData[i]));
      const float error = std::fabs(output[i] - expected);
      if (error > maxError) {
        maxError = error;
        maxIndex = i;
      }
    }
    MNN_PRINT("Hexagon MUL_SILU PWL max abs error: %.8f at up=%.8f gate=%.8f\n",
              maxError, upData[maxIndex], gateData[maxIndex]);
    if (maxError > 0.08f) {
      MNN_ERROR("HexagonMulSiluPWLTest: error %.8f exceeds 0.08\n", maxError);
      return false;
    }
    return true;
  }
};

MNNTestSuiteRegister(HexagonMulSiluPWLTest, "op/hexagon/mul-silu-pwl");
