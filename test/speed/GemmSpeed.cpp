//
//  VulkanCoopMatSpeed.cpp
//  MNNTests
//
//  Generic GEMM/GEMV performance benchmark via Conv1x1.
//  Supports CPU / OpenCL / Vulkan backends.
//  Supports float / int8(block0) / int4(block64) weight quantization.
//
//  Usage:
//    # CPU backend (default)
//    ./run_test.out speed/GemmSpeedAll
//
//    # Vulkan backend (type=7), precision=Low(2)
//    ./run_test.out speed/GemmSpeedAll 7 2
//
//    # OpenCL buffer mode: type=3, precision=2, numthread=68
//    #   (68 = MNN_GPU_MEMORY_BUFFER(64) | MNN_GPU_TUNING_FAST(4))
//    ./run_test.out speed/GemmSpeedAll 3 2 68
//
//    # Individual tests:
//    ./run_test.out speed/GemmSpeedFloat 7 2
//    ./run_test.out speed/GemmSpeedInt8 7 2
//    ./run_test.out speed/GemmSpeedInt4 7 2
//

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "CommonOpCreator.hpp"
#include <MNN/MNNForwardType.h>

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::Express;

// ---------------------------------------------------------------------------
// Helper: print current backend configuration
// ---------------------------------------------------------------------------
static void printBackendConfig(int forwardType, int precision, int thread) {
    const char* backendName = "CPU";
    switch (forwardType) {
        case MNN_FORWARD_OPENCL:
            backendName = "OpenCL";
            break;
        case MNN_FORWARD_VULKAN:
            backendName = "Vulkan";
            break;
        case MNN_FORWARD_METAL:
            backendName = "Metal";
            break;
        case MNN_FORWARD_CUDA:
            backendName = "CUDA";
            break;
        default:
            break;
    }
    MNN_PRINT("Backend: %s (type=%d), precision=%d, numthread=%d\n", backendName, forwardType, precision, thread);
    if (forwardType == MNN_FORWARD_OPENCL && thread > 0) {
        bool isBuffer = (thread & MNN_GPU_MEMORY_BUFFER) != 0;
        MNN_PRINT("  OpenCL memory mode: %s\n", isBuffer ? "BUFFER" : "IMAGE");
    }
}

// ---------------------------------------------------------------------------
// Mode 0: Float Conv1x1
// ---------------------------------------------------------------------------
static VARP buildFloatConv1x1(VARP x, int ic, int oc) {
    std::vector<float> weight(oc * ic);
    for (int i = 0; i < (int)weight.size(); ++i) {
        weight[i] = ((float)(i % 127) - 63.0f) / 1000.0f;
    }
    std::vector<float> bias(oc, 0.0f);
    return _Conv(std::move(weight), std::move(bias), x, {ic, oc}, {1, 1}, PaddingMode::VALID, {1, 1}, {1, 1}, 1, {0, 0},
                 false, false);
}

// ---------------------------------------------------------------------------
// Mode 1: Int4-block64 quantized Conv1x1 (asymmetric, via _HybridConv)
// Mode 2: Int8-block0 quantized Conv1x1 (asymmetric, blockSize=K, via _HybridConv)
// ---------------------------------------------------------------------------
static VARP buildHybridConv1x1(VARP x, int ic, int oc, int nbit, int blockSize) {
    MNN_ASSERT(ic % blockSize == 0);
    int blockNum = ic / blockSize;

    // Generate float weights
    std::vector<float> weightFp32(oc * ic);
    float fac = 0.23f;
    for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < ic; ++j) {
            weightFp32[i * ic + j] = ((i * ic + j) % nbit) * fac;
        }
    }

    // Generate asymmetric scale: alpha = [offset0, scale0, offset1, scale1, ...]
    // Layout: oc * blockNum pairs
    std::vector<float> wScale(2 * oc * blockNum);
    for (int k = 0; k < oc; ++k) {
        for (int b = 0; b < blockNum; ++b) {
            wScale[2 * (k * blockNum + b)] = -0.5f;     // offset
            wScale[2 * (k * blockNum + b) + 1] = 0.01f; // scale
        }
    }

    std::vector<float> bias(oc, 0.0f);

    return _HybridConv(weightFp32, std::move(bias), wScale, x, {ic, oc}, {1, 1}, PaddingMode::CAFFE, {1, 1}, {1, 1}, 1,
                       {0, 0}, false, false, nbit, true);
}

// ---------------------------------------------------------------------------
// Benchmark runner
//   mode: 0=float, 1=int4-block64, 2=int8-block0
// ---------------------------------------------------------------------------
static void benchConv1x1(const char* tag, int M, int K, int N, int mode, int forwardType, int precision, int thread) {
    // Create executor with Memory_Low for quantized modes
    BackendConfig bnConfig;
    bnConfig.precision = (BackendConfig::PrecisionMode)precision;
    bnConfig.memory = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor((MNNForwardType)forwardType, bnConfig, thread);
    ExecutorScope scope(exe);

    // Input: {batch=1, channel=K, height=1, width=M}
    auto x = _Input({1, K, 1, M}, NC4HW4, halide_type_of<float>());

    VARP y;
    switch (mode) {
        case 0:
            y = buildFloatConv1x1(x, K, N);
            break;
        case 1:
            y = buildHybridConv1x1(x, K, N, /*nbit=*/4, /*blockSize=*/64);
            break;
        case 2:
            y = buildHybridConv1x1(x, K, N, /*nbit=*/8, /*blockSize=*/K);
            break;
        default:
            MNN_ASSERT(false);
            return;
    }

    x.fix(VARP::INPUT);

    // Warm up (multiple rounds to stabilize GPU clocks and caches)
    for (int w = 0; w < 3; ++w) {
        auto xPtr = x->writeMap<float>();
        ::memset(xPtr, 0, x->getInfo()->size * sizeof(float));
        y->readMap<float>();
    }

    // Benchmark
    const int LOOP = 10;
    auto executor = ExecutorScope::Current();
    {
        MNN::Timer _t;
        for (int i = 0; i < LOOP; ++i) {
            x->writeMap<float>();
            y->readMap<float>();
        }
        float totalMs = (float)_t.durationInUs() / 1000.0f;
        float avgMs = totalMs / (float)LOOP;

        // Compute GFLOPS: GEMM is 2*M*K*N FLOPs
        double flops = 2.0 * (double)M * (double)K * (double)N;
        double gflops = flops / (avgMs * 1e6);

        // Try to get GPU timestamp-based time from the last execution
        float gpuMs = executor->getLastGpuTimeMs();
        if (gpuMs > 0.0f) {
            double gpuGflops = flops / ((double)gpuMs * 1e6);
            MNN_PRINT("  %-28s  M=%-5d K=%-5d N=%-5d  total=%.3f ms (%.2f GFLOPS)  gpu=%.3f ms (%.2f GFLOPS)\n", tag, M,
                      K, N, avgMs, gflops, gpuMs, gpuGflops);
        } else {
            MNN_PRINT("  %-28s  M=%-5d K=%-5d N=%-5d  avg=%.3f ms  %.2f GFLOPS\n", tag, M, K, N, avgMs, gflops);
        }
    }
}

// ---------------------------------------------------------------------------
// Common shape configurations (typical LLM dimensions)
//
// Each entry represents a (K, N) pair for Conv1x1 = GEMM [M, K] × [K, N].
// maxM: maximum M value to test (0 = use all default M values).
//       Set to 1 for very large shapes where only GEMV (decode) is practical.
// ---------------------------------------------------------------------------
struct ShapeConfig {
    int K;
    int N;
    int maxM;          // 0 = all M values, >0 = only test M <= maxM
    const char* label; // annotation for output
};

static const std::vector<ShapeConfig>& defaultConfigs() {
    static std::vector<ShapeConfig> configs = {
        {2560, 4096, 0, "K=2560 N=4096"}, {2560, 1024, 0, "K=2560 N=1024"}, {4096, 2560, 0, "K=4096 N=2560"},
        {2560, 9728, 0, "K=2560 N=9728"}, {9728, 2560, 0, "K=9728 N=2560"},
    };
    return configs;
}

// M values: GEMM(prefill) only, no GEMV(decode)
static const std::vector<int>& defaultMValues() {
    static std::vector<int> mValues = {8, 32, 128, 512};
    return mValues;
}

// ---------------------------------------------------------------------------
// Test: Float GEMM
// ---------------------------------------------------------------------------
class GemmSpeedFloat : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== Float Conv1x1 GEMM Speed =====\n");
        printBackendConfig(st.forwardType, st.precision, st.thread);

        for (auto& cfg : defaultConfigs()) {
            MNN_PRINT("--- %s ---\n", cfg.label);
            for (int m : defaultMValues()) {
                if (cfg.maxM > 0 && m > cfg.maxM)
                    continue;
                benchConv1x1("float-gemm", m, cfg.K, cfg.N, 0, st.forwardType, st.precision, st.thread);
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Test: Int8-block0 GEMM
// ---------------------------------------------------------------------------
class GemmSpeedInt8 : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== Int8-Block0 Conv1x1 GEMM Speed =====\n");
        printBackendConfig(st.forwardType, st.precision, st.thread);

        for (auto& cfg : defaultConfigs()) {
            MNN_PRINT("--- %s ---\n", cfg.label);
            for (int m : defaultMValues()) {
                if (cfg.maxM > 0 && m > cfg.maxM)
                    continue;
                benchConv1x1("int8b0-gemm", m, cfg.K, cfg.N, 2, st.forwardType, st.precision, st.thread);
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Test: Int4-block64 GEMM
// ---------------------------------------------------------------------------
class GemmSpeedInt4 : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== Int4-Block64 Conv1x1 GEMM Speed =====\n");
        printBackendConfig(st.forwardType, st.precision, st.thread);

        for (auto& cfg : defaultConfigs()) {
            if (cfg.K % 64 != 0)
                continue;
            MNN_PRINT("--- %s ---\n", cfg.label);
            for (int m : defaultMValues()) {
                if (cfg.maxM > 0 && m > cfg.maxM)
                    continue;
                benchConv1x1("int4b64-gemm", m, cfg.K, cfg.N, 1, st.forwardType, st.precision, st.thread);
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Combined test: run all modes in one shot
// ---------------------------------------------------------------------------
class GemmSpeedAll : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        auto& st = MNNTestSuite::get()->pStaus;
        MNN_PRINT("\n===== All GEMM Speed Benchmark =====\n");
        printBackendConfig(st.forwardType, st.precision, st.thread);

        std::vector<int> mValues = {8, 32, 128, 512};

        // mode: 0=float, 1=int4b64, 2=int8b0
        const char* modeNames[] = {"float", "int8b0", "int4b64"};
        int modes[] = {0, 2, 1};
        const int numModes = 3;

        for (auto& cfg : defaultConfigs()) {
            MNN_PRINT("\n--- %s ---\n", cfg.label);
            for (int mi = 0; mi < numModes; ++mi) {
                // Skip int4 modes if K is not divisible by 64
                if (modes[mi] == 1 && cfg.K % 64 != 0)
                    continue;
                for (int m : mValues) {
                    if (cfg.maxM > 0 && m > cfg.maxM)
                        continue;
                    char tag[64];
                    snprintf(tag, sizeof(tag), "%s-gemm", modeNames[mi]);
                    benchConv1x1(tag, m, cfg.K, cfg.N, modes[mi], st.forwardType, st.precision, st.thread);
                }
            }
        }
        return true;
    }
};

// Register all test cases
MNNTestSuiteRegister(GemmSpeedFloat, "speed/GemmSpeedFloat");
MNNTestSuiteRegister(GemmSpeedInt8, "speed/GemmSpeedInt8");
MNNTestSuiteRegister(GemmSpeedInt4, "speed/GemmSpeedInt4");
MNNTestSuiteRegister(GemmSpeedAll, "speed/GemmSpeedAll");