//
//  GemvBWTest.cpp
//  MNNTests
//
//  Standalone GEMV bandwidth microbenchmark for the MNN CPU backend.
//
//  Mirrors llama.cpp's gemv_roofline.cpp layout: pick a single (M, K) shape,
//  sweep thread counts, measure decode-batch (= 1) latency for w8 / w4 / w3 / w2,
//  and report effective bandwidth vs. the 4-thread / sweeping memcpy ceiling.
//
//  Default shape: M = oc = 4096, K = ic = 14336 (Llama-3-8B FFN-ish).
//
//  Usage:
//    ./run_test.out speed/GemvBW 0 2
//        # default: M=4096 K=14336, threads=4
//    ./run_test.out speed/GemvBW 0 2 8
//        # override threads to 8
//

#include <math.h>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "MNNTestSuite.h"
#include "CommonOpCreator.hpp"

using namespace MNN::Express;
using namespace MNN;

namespace {

using clk = std::chrono::high_resolution_clock;

static double seconds_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// Empirical peak DRAM bandwidth via parallel memcpy on a buffer larger than L3.
// Counts both read and write traffic (memcpy moves 2x bytes).
static double measurePeakBwGBs(size_t bytes, int threads, int repeats) {
    std::vector<uint8_t> src(bytes), dst(bytes);
    std::memset(src.data(), 0xa5, bytes);
    std::memset(dst.data(), 0x00, bytes);
    std::memcpy(dst.data(), src.data(), bytes); // warmup

    double best = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = clk::now();
        std::vector<std::thread> ts;
        size_t chunk = bytes / threads;
        for (int t = 0; t < threads; ++t) {
            size_t off = t * chunk;
            size_t len = (t == threads - 1) ? (bytes - off) : chunk;
            ts.emplace_back([&, off, len] { std::memcpy(dst.data() + off, src.data() + off, len); });
        }
        for (auto& th : ts)
            th.join();
        double dt = seconds_since(t0);
        double gbs = (2.0 * bytes) / dt / 1e9;
        if (gbs > best)
            best = gbs;
    }
    if (dst[0] == 0x12 && src[bytes - 1] == 0x34)
        std::printf("?"); // prevent DCE
    return best;
}

struct GemvResult {
    int nbit;
    int threads;
    int M, K;
    double avgUs;       // best avg us/iter (over 3 outer reps)
    double weightBytes; // weight + scale + zp bytes (storage we actually move)
    double effBwGBs;
    double gflops;
};

// One GEMV measurement: 1x1 hybrid conv with batch=1 input, oc=M, ic=K.
// Returns best avg us/iter over 3 outer reps of `iters` cold-cache runs.
static GemvResult benchGemv(int M, int K, int nbit, int blocksize, int precision, int threads, int iters) {
    BackendConfig bnConfig;
    bnConfig.precision = (BackendConfig::PrecisionMode)precision;
    bnConfig.memory = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, threads);
    ExecutorScope scope(exe);

    INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, kernel = {1, 1};

    int oc = M, ic = K;
    int blockNum = 1;
    int bs = blocksize;
    if (bs == 0 || ic % bs != 0) {
        bs = ic;
        blockNum = 1;
    } else {
        blockNum = ic / bs;
    }

    std::vector<float> weightFp32(oc * ic);
    std::vector<float> wScale(2 * oc * blockNum);
    std::vector<float> bias(oc, 0);

    float fac = 0.23f;
    for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < ic; ++j) {
            weightFp32[i * ic + j] = ((i * ic + j) % nbit) * fac;
        }
    }
    for (int k = 0; k < oc; ++k) {
        for (int b = 0; b < blockNum; ++b) {
            wScale[2 * (k * blockNum + b)] = -0.5f;
            wScale[2 * (k * blockNum + b) + 1] = 0.01f;
        }
    }

    auto x = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < ic; ++i)
        xPtr[i] = (float)((i % 17) - 8);
    x = _Convert(x, NC4HW4);
    x->writeScaleMap(1.0f, 0.f);

    auto y = _HybridConv(weightFp32, std::move(bias), wScale, x, {ic, oc}, kernel, PaddingMode::CAFFE, strides, dilate,
                         1, pad, false, false, nbit, true);
    x.fix(VARP::INPUT);

    // Warmup
    x->writeMap<float>();
    y->readMap<float>();

    // Cold-cache: flush a 64 MiB buffer before each iter to force weight reload from DRAM.
    std::vector<uint8_t> flushBuf(64 * 1024 * 1024, 1);
    auto flushCache = [&]() {
        volatile uint64_t sink = 0;
        for (size_t i = 0; i < flushBuf.size(); i += 64) {
            sink += flushBuf[i];
        }
        (void)sink;
    };

    int outerReps = 3;
    double bestUs = 1e18;
    for (int r = 0; r < outerReps; ++r) {
        double total = 0;
        for (int i = 0; i < iters; ++i) {
            flushCache();
            auto t0 = clk::now();
            x->writeMap<float>();
            y->readMap<float>();
            total += seconds_since(t0);
        }
        double avgUs = (total / iters) * 1e6;
        if (avgUs < bestUs)
            bestUs = avgUs;
    }

    GemvResult r;
    r.nbit = nbit;
    r.threads = threads;
    r.M = M;
    r.K = K;
    r.avgUs = bestUs;
    // Weight buffer storage we actually pull from DRAM each decode.
    // Counts packed weight + per-block scale/zp (fp16 each), but not the input vector
    // (small) and not the output (1 row).  Matches llama.cpp's W bytes accounting.
    double pureWeight = (double)oc * ic * nbit / 8.0;
    double scaleZp = (double)oc * blockNum * 2.0 * 2.0; // alpha + bias as fp16
    r.weightBytes = pureWeight + scaleZp;
    double secs = bestUs / 1e6;
    r.effBwGBs = r.weightBytes / secs / 1e9;
    r.gflops = (2.0 * oc * ic) / secs / 1e9;
    return r;
}

} // namespace

class GemvBWTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Defaults match llama.cpp's gemv_roofline.cpp.
        int M = 4096;
        int K = 14336;

        int threads = MNNTestSuite::get()->pStaus.thread > 0 ? MNNTestSuite::get()->pStaus.thread : 4;

        const int blocksize = 64;
        const int iters = 200;

        std::printf("\n## GemvBW (precision=%d, blocksize=%d)\n", precision, blocksize);

        // Streaming bandwidth roofline for the selected thread count.
        std::printf("\n## Peak streaming bandwidth (memcpy, 256 MiB buffer)\n");
        std::printf("threads | GB/s\n");
        std::printf("-------:|-----:\n");
        double peakBw = measurePeakBwGBs((size_t)256 << 20, threads, 5);
        std::printf("%7d | %5.1f\n", threads, peakBw);
        std::printf("-> peak %.1f GB/s @ %d threads (used as roofline)\n", peakBw, threads);

        std::printf("\n## GEMV: y = W(%dx%d) * x(%d), block=%d\n", M, K, K, blocksize);
        std::printf("type | thr |   us/iter |  W MiB | bytes/elem | eff GB/s | %%peak |  GFLOPS |  AI (op/B)\n");
        std::printf("-----|----:|----------:|-------:|-----------:|---------:|------:|--------:|----------:\n");

        std::vector<int> bitsList = {8, 4, 3, 2};
        for (int nbit : bitsList) {
            GemvResult r = benchGemv(M, K, nbit, blocksize, precision, threads, iters);
            double bpe = r.weightBytes / ((double)M * K);
            double pct = 100.0 * r.effBwGBs / peakBw;
            double ai = 2.0 / bpe;
            std::printf("w%-3d | %3d | %9.1f | %6.1f | %10.4f | %8.1f | %5.1f | %7.1f | %9.2f\n", nbit, threads,
                        r.avgUs, r.weightBytes / (1024.0 * 1024.0), bpe, r.effBwGBs, pct, r.gflops, ai);
        }

        std::printf("\nNotes:\n");
        std::printf(" * us/iter is best-of-3 outer reps, each averaged over %d cold-cache iters.\n", iters);
        std::printf(" * W MiB / bytes/elem include weight + per-block (alpha + zp) fp16 metadata.\n");
        std::printf(" * AI = 2/bpe (1 mul + 1 add per weight, weight bytes drive the ratio).\n");
        std::printf(" * %%peak compares against the best (sweep-max) memcpy bandwidth.\n");
        return true;
    }
};

MNNTestSuiteRegister(GemvBWTest, "speed/GemvBW");
