#include <MNN/AutoTime.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "MNNTestSuite.h"

using namespace MNN::Express;

class MulSiluPWLSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_HEXAGON) {
            MNN_PRINT("Skip MulSiluPWLSpeed: Hexagon backend is not selected\n");
            return true;
        }

        constexpr int kElementCount = 262144;
        constexpr int kWarmup = 10;
        constexpr int kIterations = 50;
        auto up = _Input({kElementCount}, NCHW, halide_type_of<float>());
        auto gate = _Input({kElementCount}, NCHW, halide_type_of<float>());
        float* upPtr = up->writeMap<float>();
        float* gatePtr = gate->writeMap<float>();
        for (int i = 0; i < kElementCount; ++i) {
            upPtr[i] = -8.0f + 16.0f * static_cast<float>((i * 37) % 4096) / 4095.0f;
            gatePtr[i] = -8.0f + 16.0f * static_cast<float>(i % 4096) / 4095.0f;
        }
        auto output = _MulSilu(up, gate);

        for (int i = 0; i < kWarmup; ++i) {
            up->writeMap<float>()[0] = static_cast<float>(i) * 0.001f;
            output->readMap<float>();
        }

        MNN::Timer timer;
        for (int i = 0; i < kIterations; ++i) {
            up->writeMap<float>()[0] = static_cast<float>(i) * 0.001f;
            output->readMap<float>();
        }
        const float averageMs = timer.durationInUs() / 1000.0f / static_cast<float>(kIterations);
        const float nsPerElement = averageMs * 1.0e6f / static_cast<float>(kElementCount);
        MNN_PRINT("Hexagon MUL_SILU speed: %.4f ms, %.4f ns/element (%d elements)\n", averageMs, nsPerElement,
                  kElementCount);
        return true;
    }
};

MNNTestSuiteRegister(MulSiluPWLSpeed, "speed/MulSiluPWL");
