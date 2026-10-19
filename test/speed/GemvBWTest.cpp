//
//  GemvBWTest.cpp
//  MNNTests
//
//  Standalone GEMV bandwidth microbenchmark for the MNN CPU backend.
//
//  Layout: pick a single (M, K) shape,
//  measure decode-batch (= 1) latency for w8 / w4 / w3 / w2 at the selected thread count,
//  and report logical-byte throughput alongside memcpy read+write throughput.
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

// Parallel memcpy throughput, counting requested read + write bytes, not measured DRAM traffic.
static double measureMemcpyReadWriteGBs(size_t bytes, int threads, int repeats) {
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

constexpr int kOuterReps = 3;

struct TimingStats {
    int count = 0;
    double meanUs = 0.0;
    double m2Us = 0.0;

    void add(double us) {
        double delta = us - meanUs;
        meanUs += delta / ++count;
        m2Us += delta * (us - meanUs);
    }

    double sdUs() const {
        return count > 1 ? sqrt(m2Us / (count - 1)) : 0.0;
    }
};

struct GemvResult {
    int nbit;
    int threads;
    int M, K;
    TimingStats groups[kOuterReps];
    TimingStats overall;
    double bestAvgUs;
    double logicalWeightBytes; // Unpadded payload + scale/bias estimate, not measured traffic.
    double logicalBwGBs;
    double gflops;
};

// One GEMV measurement: 1x1 hybrid conv with batch=1 input, oc=M, ic=K.
// Retains each group's statistics and the best group average separately.
static GemvResult benchGemv(int M, int K, int nbit, int blocksize, int precision, int threads, int iters,
                            MNNForwardType forwardType) {
    BackendConfig bnConfig;
    bnConfig.precision = (BackendConfig::PrecisionMode)precision;
    bnConfig.memory = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor(forwardType, bnConfig, threads);
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

    // Cache conditioning only: this scan does not guarantee cold weights or DRAM reads.
    std::vector<uint8_t> flushBuf(64 * 1024 * 1024, 1);
    auto flushCache = [&]() {
        volatile uint64_t sink = 0;
        for (size_t i = 0; i < flushBuf.size(); i += 64) {
            sink += flushBuf[i];
        }
        (void)sink;
    };

    GemvResult r;
    r.bestAvgUs = 1e18;
    for (int rep = 0; rep < kOuterReps; ++rep) {
        for (int i = 0; i < iters; ++i) {
            flushCache();
            auto t0 = clk::now();
            x->writeMap<float>();
            y->readMap<float>();
            double us = seconds_since(t0) * 1e6;
            r.groups[rep].add(us);
            r.overall.add(us);
        }
        if (r.groups[rep].meanUs < r.bestAvgUs)
            r.bestAvgUs = r.groups[rep].meanUs;
    }

    r.nbit = nbit;
    r.threads = threads;
    r.M = M;
    r.K = K;
    // CPU hybrid int8 packing uses FP32 scale + bias even with FP16 output.
    double pureWeight = ceil((double)oc * ic * nbit / 8.0);
    double metadataBytes = forwardType == MNN_FORWARD_CPU ? sizeof(float) : 2.0;
    double scaleBias = (double)oc * blockNum * 2.0 * metadataBytes;
    r.logicalWeightBytes = pureWeight + scaleBias;
    double secs = r.bestAvgUs / 1e6;
    r.logicalBwGBs = r.logicalWeightBytes / secs / 1e9;
    r.gflops = (2.0 * oc * ic) / secs / 1e9;
    return r;
}

} // namespace

class GemvBWTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        // Default shape is a Llama-3-8B-style FFN projection.
        int M = 4096;
        int K = 14336;
        // Optional shape overrides for decode-attribution runs, e.g.
        // Qwen3-0.6B plain convs: o_proj (1024,2048), down (1024,3072),
        // lm_head (151936,1024).
        if (const char* e = getenv("MNN_GEMVBW_M")) {
            if (atoi(e) > 0) M = atoi(e);
        }
        if (const char* e = getenv("MNN_GEMVBW_K")) {
            if (atoi(e) > 0) K = atoi(e);
        }

        int threads = MNNTestSuite::get()->pStaus.thread > 0 ? MNNTestSuite::get()->pStaus.thread : 4;
        MNNForwardType forwardType = (MNNForwardType)MNNTestSuite::get()->pStaus.forwardType;
        const char* backendName = forwardType == MNN_FORWARD_METAL ? "Metal"
                                  : forwardType == MNN_FORWARD_CPU ? "CPU"
                                                                   : "Other";

        const int blocksize = 64;
        const int iters = 200;

        std::printf("\n## GemvBW (backend=%s, precision=%d, blocksize=%d)\n", backendName, precision, blocksize);

        std::printf("\n## memcpy read+write throughput (256 MiB each for src/dst, best of 5)\n");
        std::printf("threads | read+write GB/s\n");
        std::printf("-------:|----------------:\n");
        double memcpyBw = measureMemcpyReadWriteGBs((size_t)256 << 20, threads, 5);
        std::printf("%7d | %15.1f\n", threads, memcpyBw);

        std::printf("\n## GEMV: y = W(%dx%d) * x(%d), block=%d\n", M, K, K, blocksize);
        std::printf("Group statistics (64 MiB cache conditioning before each timed iter):\n");
        std::printf("type | thr | group | iters | mean us/iter | sample SD us\n");
        std::printf("-----|----:|------:|------:|-------------:|-------------:\n");

        // Metal supports w8 / w4 / w3 / w2 hybrid quant GEMV (decode, area==1) via
        // the 2sg kernel (see MetalConvolution1x1.mm conv1x1_gemv_g4m1_2sg_wquant_sg).
        std::vector<int> bitsList = {8, 4, 3, 2};
        if (const char* e = getenv("MNN_GEMVBW_BITS")) {
            if (atoi(e) > 0) bitsList = {atoi(e)};
        }
        std::vector<GemvResult> results;
        for (int nbit : bitsList) {
            GemvResult r = benchGemv(M, K, nbit, blocksize, precision, threads, iters, forwardType);
            for (int rep = 0; rep < kOuterReps; ++rep) {
                const auto& stats = r.groups[rep];
                std::printf("w%-3d | %3d | %5d | %5d | %12.3f | %12.3f\n", nbit, threads, rep + 1,
                            stats.count, stats.meanUs, stats.sdUs());
            }
            results.push_back(r);
        }

        std::printf("\n## Final summary (latencies in us/iter; byte counts are unpadded logical estimates)\n");
        std::printf("type | thr | best avg | overall mean | overall SD | logical B | MiB | bytes/elem | "
                    "logical GB/s | GFLOPS | AI (op/B)\n");
        std::printf("-----|----:|---------:|-------------:|-----------:|----------:|----:|-----------:|"
                    "-------------:|-------:|----------:\n");
        for (const auto& r : results) {
            double bpe = r.logicalWeightBytes / ((double)r.M * r.K);
            double ai = 2.0 / bpe;
            std::printf("w%-3d | %3d | %8.3f | %12.3f | %10.3f | %9.0f | %5.1f | %10.4f | %12.1f | %6.1f | %9.2f\n",
                        r.nbit, r.threads, r.bestAvgUs, r.overall.meanUs, r.overall.sdUs(), r.logicalWeightBytes,
                        r.logicalWeightBytes / (1024.0 * 1024.0), bpe, r.logicalBwGBs, r.gflops, ai);
        }

        std::printf("\nNotes:\n");
        std::printf(" * Timing includes writeMap/readMap and expression execution, not just the raw kernel.\n");
        std::printf(" * Best avg is the minimum of %d group means (%d iters each); overall mean/SD use all %d iters.\n",
                    kOuterReps, iters, kOuterReps * iters);
        std::printf(" * SD is sample SD of individual latencies, not uncertainty of the best avg.\n");
        std::printf(" * Logical bytes = ceil(M*K*bits/8) + M*blocks*2*metadata bytes: %s.\n",
                    forwardType == MNN_FORWARD_CPU ? "CPU FP32 scale + bias (4 bytes each)"
                                                  : "nominal FP16 scale + bias (2 bytes each), unverified");
        std::printf(" * Excludes kernel OC/K padding, repacking and auxiliary/input/output traffic; not actual DRAM bytes.\n");
        std::printf(" * Logical GB/s and GFLOPS use best avg; AI = 2/bytes-per-element using the same logical estimate.\n");
        std::printf(" * memcpy counts requested read+write bytes (2*buffer size), includes thread startup/join,\n");
        std::printf("   and is a separate throughput reference, not a GEMV saturation measure.\n");
        std::printf(" * The 64 MiB scan is untimed cache conditioning, not guaranteed eviction or forced DRAM access\n");
        std::printf("   on any backend; weights may remain in CPU/GPU/unified caches.\n");
        return true;
    }
};

MNNTestSuiteRegister(GemvBWTest, "speed/GemvBW");
