#ifdef MNN_KLEIDIAI_ENABLED
#ifdef MNN_LOW_MEMORY

#include <cmath>
#include <cstring>
#include <vector>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "CommonOpCreator.hpp"
#include "core/Backend.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace {

struct QuantCase {
    int ic;
    int oc;
    int area;
    int blockSize;
    bool asymmetric;
    const char* tag;
};

void quantizeDequantize(std::vector<float>& weight, std::vector<float>& alpha, int ic, int oc, int blockSize,
                        bool asymmetric) {
    const int blockNum = ic / blockSize;
    const float threshold = 7.0f;
    const float clampMin = asymmetric ? -8.0f : -7.0f;

    alpha.resize(asymmetric ? 2 * oc * blockNum : oc * blockNum);
    for (int o = 0; o < oc; ++o) {
        for (int b = 0; b < blockNum; ++b) {
            const int begin = o * ic + b * blockSize;
            if (asymmetric) {
                float minValue = weight[begin];
                float maxValue = weight[begin];
                for (int i = 1; i < blockSize; ++i) {
                    minValue = std::min(minValue, weight[begin + i]);
                    maxValue = std::max(maxValue, weight[begin + i]);
                }
                float scale = 0.0f;
                const float range = maxValue - minValue;
                if (range >= 1e-6f) {
                    scale = range / (threshold - clampMin);
                }
                alpha[2 * (o * blockNum + b)] = minValue;
                alpha[2 * (o * blockNum + b) + 1] = scale;
                const float inv = scale >= 1e-6f ? (1.0f / scale) : 0.0f;
                for (int i = 0; i < blockSize; ++i) {
                    int code = (int)std::round((weight[begin + i] - minValue) * inv + clampMin);
                    code = (int)std::max(clampMin, std::min(threshold, (float)code));
                    weight[begin + i] = ((float)code - clampMin) * scale + minValue;
                }
            } else {
                float absMax = 1e-8f;
                for (int i = 0; i < blockSize; ++i) {
                    absMax = std::max(absMax, std::fabs(weight[begin + i]));
                }
                const float scale = absMax / threshold;
                alpha[o * blockNum + b] = scale;
                const float inv = scale >= 1e-6f ? (1.0f / scale) : 0.0f;
                for (int i = 0; i < blockSize; ++i) {
                    int code = (int)std::round(weight[begin + i] * inv);
                    code = (int)std::max(clampMin, std::min(threshold, (float)code));
                    weight[begin + i] = (float)code * scale;
                }
            }
        }
    }
}

bool runHybridInt4Case(const QuantCase& tc, BackendConfig::PrecisionMode precision) {
    std::vector<float> weight((size_t)tc.oc * tc.ic);
    std::vector<float> bias(tc.oc);
    std::vector<float> input((size_t)tc.ic * tc.area);

    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = ((float)((i * 1103515245u + 12345u) % 65536) / 65536.0f) - 0.5f;
    }
    for (int i = 0; i < tc.oc; ++i) {
        bias[i] = ((float)((i * 2654435761u) % 65536) / 65536.0f) - 0.5f;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = ((float)((i * 40503u) % 65536) / 65536.0f) - 0.5f;
    }

    std::vector<float> alpha;
    quantizeDequantize(weight, alpha, tc.ic, tc.oc, tc.blockSize, tc.asymmetric);

    auto runOne = [&](bool enableKleidiAI, std::vector<float>& out) -> bool {
        BackendConfig config;
        config.precision = precision;
        config.memory = BackendConfig::Memory_Low;

        auto exe = Executor::newExecutor(MNN_FORWARD_CPU, config, 1);
        ExecutorScope scope(exe);

        RuntimeHint hint;
        hint.enableKleidiAI = enableKleidiAI;
        hint.dynamicQuantOption = 1;
        scope.Current()->getRuntime().second->setRuntimeHint(hint);

        auto x = _Input({1, tc.ic, 1, tc.area}, NCHW, halide_type_of<float>());
        ::memcpy(x->writeMap<float>(), input.data(), input.size() * sizeof(float));
        x->unMap();

        auto y = _HybridConv(weight, bias, alpha, x, {tc.ic, tc.oc}, {1, 1}, CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false,
                             false, 4, tc.asymmetric);
        y = _Convert(y, NCHW);
        const float* outPtr = y->readMap<float>();
        if (outPtr == nullptr) {
            MNN_ERROR("KleidiAIInt4 readMap null for %s (enableKleidiAI=%d)\n", tc.tag, (int)enableKleidiAI);
            return false;
        }
        out.assign(outPtr, outPtr + (size_t)tc.oc * tc.area);
        return true;
    };

    std::vector<float> outRef;
    std::vector<float> outKleidiAI;
    if (!runOne(false, outRef) || !runOne(true, outKleidiAI)) {
        return false;
    }

    const float tol = (precision == BackendConfig::Precision_Low) ? 0.05f : 0.01f;
    if (!checkVectorByRelativeError<float>(outKleidiAI.data(), outRef.data(), (int)outRef.size(), tol)) {
        MNN_ERROR("KleidiAIInt4 divergence for %s (precision=%d)\n", tc.tag, (int)precision);
        return false;
    }
    return true;
}

} // namespace

class KleidiAIConvInt4E2ETest : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;

        std::vector<BackendConfig::PrecisionMode> precisions = {
            BackendConfig::Precision_High,
            BackendConfig::Precision_Low,
        };

        std::vector<QuantCase> baseCases = {
            // Symmetric per-channel (regression focus: IC=16/32/48 behavior).
            {16, 24, 1, 16, false, "sym-per-channel-f32-gemv-ic16"},
            {32, 24, 1, 32, false, "sym-per-channel-f32-gemv-ic32"},
            {48, 24, 1, 48, false, "sym-per-channel-f32-gemv-ic48"},
            {48, 24, 8, 48, false, "sym-per-channel-f32-gemm-ic48"},

            // Asymmetric per-channel.
            {32, 24, 1, 32, true, "asym-per-channel-gemv"},
            {32, 24, 8, 32, true, "asym-per-channel-gemm"},

            // Per-block paths.
            {64, 24, 1, 32, false, "sym-per-block-gemv"},
            {64, 24, 8, 32, false, "sym-per-block-gemm"},
            {64, 24, 1, 32, true, "asym-per-block-gemv"},
            {64, 24, 8, 32, true, "asym-per-block-gemm"},
        };

        for (auto p : precisions) {
            for (const auto& tc : baseCases) {
                if (!runHybridInt4Case(tc, p)) {
                    MNN_ERROR("KleidiAI int4 e2e failed for %s\n", tc.tag);
                    return false;
                }
            }
        }

        // This suite is intended to catch ISA dispatch regressions (for example SIGILL on
        // unsupported devices). Reaching here means all selected int4 routes executed safely.
        return true;
    }
};

MNNTestSuiteRegister(KleidiAIConvInt4E2ETest, "kleidiai/int4_conv_e2e");

#endif // MNN_LOW_MEMORY
#endif // MNN_KLEIDIAI_ENABLED
