//
//  gemvbw_roofline.cpp
//  MNN
//
//  Standalone GEMV bandwidth roofline microbenchmark, extracted from
//  test/speed/GemvBWTest.cpp so it can run inside the HarmonyOS app process
//  (the device kernel XPM forbids exec of standalone /data ELF). Driven by the
//  native-bench "@gemvbw" mode. Measures peak DRAM bandwidth (parallel memcpy)
//  and decode-batch (=1) w4/w8 GEMV effective bandwidth for given (M=oc, K=ic)
//  shapes, reporting eff GB/s and %peak so we can see how far the small per-layer
//  projections sit from the memory roofline.
//

#include <math.h>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <thread>
#include <atomic>
#include <dirent.h>
#include <sys/syscall.h>
#include <map>
#include <cstring>
#include <vector>
#include <arm_neon.h>
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>

// Pin calling thread to the big cores (Kirin 9030: cpu12-13, 2.75GHz) so the
// microbench runs on a performance core at max freq under sustained load —
// independent of libqos.so (which fails to load on this device). Without this the
// scheduler may place the thread on a little/idle core and the roofline is junk.
static void pinCore(int c) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
static int g_pinCore = 12;
static void pinBigCore() { pinCore(g_pinCore); }
static long curFreqKHz(int c) {
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_cur_freq", c);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}
extern "C" void MNNSetOHOSUclamp(int on);
extern "C" void MNNSetOHOSPinMode(int m);
static void applyUclampMaster() {
    struct sched_attr {
        uint32_t size; uint32_t sched_policy; uint64_t sched_flags;
        int32_t sched_nice; uint32_t sched_priority;
        uint64_t sched_runtime; uint64_t sched_deadline; uint64_t sched_period;
        uint32_t sched_util_min; uint32_t sched_util_max;
    } attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.sched_flags = 0x20;
    attr.sched_util_min = 1024;
    int ret = syscall(SYS_sched_setattr, 0, &attr, 0);
    std::printf("[uclamp] master sched_setattr ret=%d errno=%d\n", ret, errno);
}
// PMU hardware counters to read stall reasons directly (backend/frontend stall,
// L1D miss, branch miss) instead of black-box guessing. perf_event_open may be
// restricted on OHOS; print errno if so.
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
struct PmuCounters {
    int fds[6] = {-1,-1,-1,-1,-1,-1};
    long long vals[6] = {0};
    static int openEv(unsigned type, unsigned long long config, int group) {
        struct perf_event_attr a;
        memset(&a, 0, sizeof(a));
        a.size = sizeof(a);
        a.type = type;
        a.config = config;
        a.disabled = (group == -1) ? 1 : 0;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        int fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, group, 0);
        return fd;
    }
    void start() {
        unsigned long long cfg[6] = {
            PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
            PERF_COUNT_HW_STALLED_CYCLES_BACKEND, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND,
            PERF_COUNT_HW_BRANCH_MISSES,
            (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)
        };
        for (int i = 0; i < 6; ++i) {
            fds[i] = openEv(i == 5 ? PERF_TYPE_HW_CACHE : PERF_TYPE_HARDWARE, cfg[i], i == 0 ? -1 : fds[0]);
        }
        for (int i = 0; i < 6; ++i) if (fds[i] >= 0) { ioctl(fds[i], PERF_EVENT_IOC_RESET, 0); ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0); }
    }
    void stop(const char* tag) {
        for (int i = 0; i < 6; ++i) if (fds[i] >= 0) ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
        for (int i = 0; i < 6; ++i) {
            if (fds[i] < 0) { vals[i] = -1; continue; }
            struct { long long value; long long enabled; long long running; } r;
            if (read(fds[i], &r, sizeof(r)) == sizeof(r)) vals[i] = r.value; else vals[i] = -2;
            close(fds[i]);
        }
        std::printf("[PMU %s] cyc=%lld ins=%lld stallBE=%lld stallFE=%lld brMiss=%lld l1dMiss=%lld\n",
                    tag, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
        if (vals[0] > 0 && vals[1] > 0) {
            std::printf("[PMU %s] IPC=%.2f stallBE%%=%.1f stallFE%%=%.1f\n", tag,
                        (double)vals[1] / vals[0], 100.0 * vals[2] / vals[0], 100.0 * vals[3] / vals[0]);
        }
        std::fflush(stdout);
    }
};

// Sample the pinned core's actual freq during the benchmark (post-hoc reads see
// idle freq and are useless). Returns max sample seen while running.
struct FreqSampler {
    std::thread th;
    std::atomic<bool> stop{false};
    std::atomic<long> maxKHz{0};
    void start(int core) {
        th = std::thread([this, core] {
            while (!stop.load()) {
                long v = curFreqKHz(core);
                long prev = maxKHz.load();
                while (v > prev && !maxKHz.compare_exchange_weak(prev, v)) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    long finish() { stop.store(true); if (th.joinable()) th.join(); return maxKHz.load(); }
};

// Raise calling thread to QOS_USER_INTERACTIVE(4) so mid/big clusters boost to
// max frequency — matches the production inference QoS fix. Without this the
// default QoS keeps cores at ~60-80% freq and the roofline is understated.
static void setQoS4() {
    typedef int (*QoSFunc)(int);
    static QoSFunc setQoS = []() -> QoSFunc {
        void* h = dlopen("libqos.so", RTLD_NOW);
        if (!h) { std::printf("[QoS] dlopen libqos.so FAILED: %s\n", dlerror()); return nullptr; }
        auto f = reinterpret_cast<QoSFunc>(dlsym(h, "OH_QoS_SetThreadQoS"));
        std::printf("[QoS] libqos.so loaded, OH_QoS_SetThreadQoS=%p\n", (void*)f);
        const char* probes[] = {"OH_QoS_SetProcessQoS", "OH_QoS_SetThreadQoSWithLevel",
                                "OH_QoS_ApplyQosForProcess", "OH_QoS_SetQosForThread",
                                "OH_QoS_RegisterQos", "OH_QoS_SetThreadQosLevel"};
        for (auto p : probes) {
            void* s = dlsym(h, p);
            if (s) std::printf("[QoS] found symbol %s=%p\n", p, s);
        }
        return f;
    }();
    if (setQoS) {
        int ret = setQoS(5); // QOS_USER_INTERACTIVE (official=5)
        static bool logged = false;
        if (!logged) { std::printf("[QoS] setQoS(5) ret=%d\n", ret); logged = true; }
    }
}
static void raiseQoS() {
    pinBigCore();
    setQoS4();
}
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/AutoTime.hpp>
#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"

using namespace MNN::Express;
using namespace MNN;

namespace {
using clk = std::chrono::high_resolution_clock;
static double seconds_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

static PadMode _convertPadMode(Express::PaddingMode mode) {
    switch (mode) {
        case Express::PaddingMode::CAFFE: return PadMode_CAFFE;
        case Express::PaddingMode::VALID: return PadMode_VALID;
        case Express::PaddingMode::SAME:  return PadMode_SAME;
        default: return PadMode_CAFFE;
    }
}

static Express::VARP _HybridConv(const std::vector<float>& weight, const std::vector<float>& bias,
                                 const std::vector<float>& alpha, Express::VARP x, std::vector<int> channel,
                                 std::vector<int> kernelSize, Express::PaddingMode pad, std::vector<int> stride,
                                 std::vector<int> dilate, int group, std::vector<int> pads, bool relu, bool relu6,
                                 int nbits, bool async) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    int kSize = kernelSize[0] * kernelSize[1] * channel[0] / group;
    int kNum = channel[1];
    int clampMin = -(1 << (nbits - 1));
    auto alphasize = alpha.size();
    if (async) alphasize /= 2;
    int blocknum = alphasize / channel[1];
    int blocksize = kSize / blocknum;
    conv2D->quanParameter = std::move(IDSTEncoder::encode(weight.data(), alpha, blocksize, kNum * blocknum, async,
                                                          nullptr, clampMin, {nbits, false}));
    conv2D->common->padMode = _convertPadMode(pad);
    if (pads.size() == 2) {
        conv2D->common->padX = pads[0];
        conv2D->common->padY = pads[1];
    } else {
        conv2D->common->pads = std::move(pads);
    }
    conv2D->common->strideX = stride[0];
    conv2D->common->strideY = stride[1];
    conv2D->common->group = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount = channel[0];
    conv2D->common->dilateX = dilate[0];
    conv2D->common->dilateY = dilate[1];
    conv2D->common->kernelX = kernelSize[0];
    conv2D->common->kernelY = kernelSize[1];
    conv2D->common->relu6 = relu6;
    conv2D->common->relu = relu;
    conv2D->weight.clear();
    conv2D->bias = bias;
    return (Express::Variable::create(Express::Expr::create(convOp.get(), {x})));
}

static double measurePeakBwGBs(size_t bytes, int threads, int repeats) {
    std::vector<uint8_t> src(bytes), dst(bytes);
    std::memset(src.data(), 0xa5, bytes);
    std::memset(dst.data(), 0x00, bytes);
    std::memcpy(dst.data(), src.data(), bytes);
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
        for (auto& th : ts) th.join();
        double dt = seconds_since(t0);
        double gbs = (2.0 * bytes) / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    if (dst[0] == 0x12 && src[bytes - 1] == 0x34) std::printf("?");
    return best;
}

struct GemvResult {
    int nbit, threads, M, K;
    double avgUs, weightBytes, effBwGBs, gflops;
};

static GemvResult benchGemv(int M, int K, int nbit, int blocksize, int precision, int threads, int iters) {
    BackendConfig bnConfig;
    bnConfig.precision = (BackendConfig::PrecisionMode)precision;
    bnConfig.memory = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, threads);
    ExecutorScope scope(exe);
    INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, kernel = {1, 1};
    int oc = M, ic = K, blockNum = 1, bs = blocksize;
    if (bs == 0 || ic % bs != 0) { bs = ic; blockNum = 1; } else { blockNum = ic / bs; }
    std::vector<float> weightFp32(oc * ic), wScale(2 * oc * blockNum), bias(oc, 0);
    float fac = 0.23f;
    for (int i = 0; i < oc; ++i)
        for (int j = 0; j < ic; ++j)
            weightFp32[i * ic + j] = ((i * ic + j) % nbit) * fac;
    for (int k = 0; k < oc; ++k)
        for (int b = 0; b < blockNum; ++b) {
            wScale[2 * (k * blockNum + b)] = -0.5f;
            wScale[2 * (k * blockNum + b) + 1] = 0.01f;
        }
    auto x = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < ic; ++i) xPtr[i] = (float)((i % 17) - 8);
    x = _Convert(x, NC4HW4);
    x->writeScaleMap(1.0f, 0.f);
    auto y = _HybridConv(weightFp32, std::move(bias), wScale, x, {ic, oc}, kernel, PaddingMode::CAFFE, strides, dilate,
                         1, pad, false, false, nbit, true);
    x.fix(VARP::INPUT);
    x->writeMap<float>();
    y->readMap<float>();
    std::vector<uint8_t> flushBuf(64 * 1024 * 1024, 1);
    auto flushCache = [&]() {
        volatile uint64_t sink = 0;
        for (size_t i = 0; i < flushBuf.size(); i += 64) sink += flushBuf[i];
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
        if (avgUs < bestUs) bestUs = avgUs;
    }
    GemvResult r;
    r.nbit = nbit; r.threads = threads; r.M = M; r.K = K; r.avgUs = bestUs;
    double pureWeight = (double)oc * ic * nbit / 8.0;
    double scaleZp = (double)oc * blockNum * 2.0 * 2.0;
    r.weightBytes = pureWeight + scaleZp;
    double secs = bestUs / 1e6;
    r.effBwGBs = r.weightBytes / secs / 1e9;
    r.gflops = (2.0 * oc * ic) / secs / 1e9;
    return r;
}
} // namespace

// Warm GEMM (batch = M spatial positions through a 1x1 conv), NO cold-cache
// flush — matches the model's hot streaming execution. Reports GFLOPS so we can
// gauge compute-bound prefill GEMM against the core's empirical compute ceiling.
static GemvResult benchGemmWarm(int M, int oc, int ic, int nbit, int blocksize, int precision, int threads, int iters, int pinMode = 0) {
    BackendConfig bnConfig;
    bnConfig.precision = (BackendConfig::PrecisionMode)precision;
    bnConfig.memory = BackendConfig::Memory_Low;
    auto exe = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, threads);
    ExecutorScope scope(exe);
    INTS strides = {1, 1}, dilate = {1, 1}, pad = {0, 0}, kernel = {1, 1};
    int blockNum = 1, bs = blocksize;
    if (bs == 0 || ic % bs != 0) { bs = ic; blockNum = 1; } else { blockNum = ic / bs; }
    std::vector<float> weightFp32(oc * ic), wScale(2 * oc * blockNum), bias(oc, 0);
    float fac = 0.23f;
    for (int i = 0; i < oc; ++i)
        for (int j = 0; j < ic; ++j)
            weightFp32[i * ic + j] = ((i * ic + j) % nbit) * fac;
    for (int k = 0; k < oc; ++k)
        for (int b = 0; b < blockNum; ++b) {
            wScale[2 * (k * blockNum + b)] = -0.5f;
            wScale[2 * (k * blockNum + b) + 1] = 0.01f;
        }
    // batch M as spatial height: input {1, ic, M, 1} -> output {1, oc, M, 1}.
    auto x = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < ic * M; ++i) xPtr[i] = (float)((i % 17) - 8);
    x = _Convert(x, NC4HW4);
    x->writeScaleMap(1.0f, 0.f);
    auto y = _HybridConv(weightFp32, std::move(bias), wScale, x, {ic, oc}, kernel, PaddingMode::CAFFE, strides, dilate,
                         1, pad, false, false, nbit, true);
    x.fix(VARP::INPUT);
    // Warmup (weight reorder + cache warm), then time hot iterations back-to-back.
    for (int i = 0; i < 5; ++i) { x->writeMap<float>(); y->readMap<float>(); }
    int outerReps = 3;
    double bestUs = 1e18;
    for (int r = 0; r < outerReps; ++r) {
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) { x->writeMap<float>(); y->readMap<float>(); }
        double avgUs = seconds_since(t0) / iters * 1e6;
        if (avgUs < bestUs) bestUs = avgUs;
    }
    GemvResult res;
    res.nbit = nbit; res.threads = threads; res.M = oc; res.K = ic; res.avgUs = bestUs;
    double pureWeight = (double)oc * ic * nbit / 8.0;
    res.weightBytes = pureWeight + (double)oc * blockNum * 2.0 * 2.0;
    double secs = bestUs / 1e6;
    res.effBwGBs = res.weightBytes / secs / 1e9;
    res.gflops = (2.0 * M * oc * ic) / secs / 1e9;
    return res;
}

// Pure int8 sdot compute microbench: 8 independent accumulator chains in
// registers (no memory traffic, no quantization), asm volatile so the compiler
// cannot eliminate them. Measures the core's real sdot throughput ceiling —
// the hardware compute roofline that the w4a8 GEMM kernel should be compared to.
static double sdotCoreGmacs(int iters) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0), a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
    int8x16_t x = vdupq_n_s8(3), y = vdupq_n_s8(5);
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        asm volatile(
            "sdot %0.4s, %8.16b, %9.16b\n\t"
            "sdot %1.4s, %8.16b, %9.16b\n\t"
            "sdot %2.4s, %8.16b, %9.16b\n\t"
            "sdot %3.4s, %8.16b, %9.16b\n\t"
            "sdot %4.4s, %8.16b, %9.16b\n\t"
            "sdot %5.4s, %8.16b, %9.16b\n\t"
            "sdot %6.4s, %8.16b, %9.16b\n\t"
            "sdot %7.4s, %8.16b, %9.16b\n\t"
            : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7)
            : "w"(x), "w"(y));
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(a0, 0) + vgetq_lane_s32(a7, 0);
    (void)sink;
    // 8 sdot/iter, each sdot = 4 lanes x 4 int8 = 16 MACs.
    return (double)iters * 8.0 * 16.0 / secs / 1e9; // GMAC/s
}

// Lane-broadcast sdot variant (the form the w4 GEMM kernel actually uses:
// sdot vD.4s, vN.16b, vM.4b[lane]). May have different throughput than the
// full-vector sdot above on some cores — measures the GEMM-relevant ceiling.
static double sdotCoreGmacsLane(int iters) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0), a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
    register int8x16_t x asm("v8") = vdupq_n_s8(3);
    register int8x16_t y asm("v9") = vdupq_n_s8(5);
    register int32x4_t r0 asm("v0") = a0, r1 asm("v1") = a1, r2 asm("v2") = a2, r3 asm("v3") = a3;
    register int32x4_t r4 asm("v4") = a4, r5 asm("v5") = a5, r6 asm("v6") = a6, r7 asm("v7") = a7;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        asm volatile(
            "sdot v0.4s, v8.16b, v9.4b[0]\n\t"
            "sdot v1.4s, v8.16b, v9.4b[1]\n\t"
            "sdot v2.4s, v8.16b, v9.4b[2]\n\t"
            "sdot v3.4s, v8.16b, v9.4b[3]\n\t"
            "sdot v4.4s, v8.16b, v9.4b[0]\n\t"
            "sdot v5.4s, v8.16b, v9.4b[1]\n\t"
            "sdot v6.4s, v8.16b, v9.4b[2]\n\t"
            "sdot v7.4s, v8.16b, v9.4b[3]\n\t"
            : "+w"(r0), "+w"(r1), "+w"(r2), "+w"(r3), "+w"(r4), "+w"(r5), "+w"(r6), "+w"(r7)
            : "w"(x), "w"(y));
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(r0, 0) + vgetq_lane_s32(r7, 0);
    (void)sink;
    return (double)iters * 8.0 * 16.0 / secs / 1e9;
}

// Block-structure replica: inner 512-iter accumulate + periodic epilogue + reset.
// withEpi=1 mode3, =0 mode4. Pointers stream via mod.
static double sdotCoreGmacsBlock(int iters, int withEpi) {
    static std::vector<int8_t> wbuf(8*1024*1024,3), sbuf(2*1024*1024,5);
    static std::vector<float> obuf(1024,0.f);
    const int8_t* wp=wbuf.data(); const int8_t* sp=sbuf.data(); float* op=obuf.data();
    register int32x4_t r8 asm("v8") = vdupq_n_s32(0);
    register int32x4_t r9 asm("v9") = vdupq_n_s32(0);
    register int32x4_t r10 asm("v10") = vdupq_n_s32(0);
    register int32x4_t r11 asm("v11") = vdupq_n_s32(0);
    register int32x4_t r12 asm("v12") = vdupq_n_s32(0);
    register int32x4_t r13 asm("v13") = vdupq_n_s32(0);
    register int32x4_t r14 asm("v14") = vdupq_n_s32(0);
    register int32x4_t r15 asm("v15") = vdupq_n_s32(0);
    register int32x4_t r16 asm("v16") = vdupq_n_s32(0);
    register int32x4_t r17 asm("v17") = vdupq_n_s32(0);
    register int32x4_t r18 asm("v18") = vdupq_n_s32(0);
    register int32x4_t r19 asm("v19") = vdupq_n_s32(0);
    register int32x4_t r20 asm("v20") = vdupq_n_s32(0);
    register int32x4_t r21 asm("v21") = vdupq_n_s32(0);
    register int32x4_t r22 asm("v22") = vdupq_n_s32(0);
    register int32x4_t r23 asm("v23") = vdupq_n_s32(0);
    register int32x4_t r24 asm("v24") = vdupq_n_s32(0);
    register int32x4_t r25 asm("v25") = vdupq_n_s32(0);
    register int32x4_t r26 asm("v26") = vdupq_n_s32(0);
    register int32x4_t r27 asm("v27") = vdupq_n_s32(0);
    register int32x4_t r28 asm("v28") = vdupq_n_s32(0);
    register int32x4_t r29 asm("v29") = vdupq_n_s32(0);
    register int32x4_t r30 asm("v30") = vdupq_n_s32(0);
    register int32x4_t r31 asm("v31") = vdupq_n_s32(0);
    register int8x16_t x asm("v0") = vdupq_n_s8(3);
    register int8x16_t y asm("v1") = vdupq_n_s8(5);
    auto t0 = clk::now();
    const int BLK = 512;
    for (int base = 0; base < iters; base += BLK) {
        for (int i = 0; i < BLK; ++i) {
            const int8_t* w = wp + ((base+i)*16) % (8*1024*1024-16);
            const int8_t* sr = sp + ((base+i)*48) % (2*1024*1024-48);
            asm volatile(
                "ld1 {v5.16b}, [%[w]]\n\t"
                "ld1 {v0.16b, v1.16b, v2.16b}, [%[s]]\n\t"
                "ushr v3.16b, v5.16b, #4\n\t"
                "and v4.16b, v5.16b, v7.16b\n\t"
                "sdot v8.4s, v3.16b, v0.4b[0]\n\t"
                "sdot v9.4s, v3.16b, v0.4b[1]\n\t"
                "sdot v10.4s, v3.16b, v0.4b[2]\n\t"
                "sdot v11.4s, v3.16b, v0.4b[3]\n\t"
                "sdot v12.4s, v3.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v3.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v3.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v3.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v3.16b, v2.4b[0]\n\t"
                "sdot v17.4s, v3.16b, v2.4b[1]\n\t"
                "sdot v18.4s, v3.16b, v2.4b[2]\n\t"
                "sdot v19.4s, v3.16b, v2.4b[3]\n\t"
                "sdot v20.4s, v4.16b, v0.4b[0]\n\t"
                "sdot v21.4s, v4.16b, v0.4b[1]\n\t"
                "sdot v22.4s, v4.16b, v0.4b[2]\n\t"
                "sdot v23.4s, v4.16b, v0.4b[3]\n\t"
                "sdot v24.4s, v4.16b, v1.4b[0]\n\t"
                "sdot v25.4s, v4.16b, v1.4b[1]\n\t"
                "sdot v26.4s, v4.16b, v1.4b[2]\n\t"
                "sdot v27.4s, v4.16b, v1.4b[3]\n\t"
                "sdot v28.4s, v4.16b, v2.4b[0]\n\t"
                "sdot v29.4s, v4.16b, v2.4b[1]\n\t"
                "sdot v30.4s, v4.16b, v2.4b[2]\n\t"
                "sdot v31.4s, v4.16b, v2.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : [w] "r"(w), [s] "r"(sr), "w"(x), "w"(y)
                : "v3","v4","v5","v7","memory");
        }
        if (withEpi) {
            asm volatile(
                "scvtf v8.4s, v8.4s\n\t"
                "scvtf v9.4s, v9.4s\n\t"
                "scvtf v10.4s, v10.4s\n\t"
                "scvtf v11.4s, v11.4s\n\t"
                "scvtf v12.4s, v12.4s\n\t"
                "scvtf v13.4s, v13.4s\n\t"
                "scvtf v14.4s, v14.4s\n\t"
                "scvtf v15.4s, v15.4s\n\t"
                "scvtf v16.4s, v16.4s\n\t"
                "scvtf v17.4s, v17.4s\n\t"
                "scvtf v18.4s, v18.4s\n\t"
                "scvtf v19.4s, v19.4s\n\t"
                "scvtf v20.4s, v20.4s\n\t"
                "scvtf v21.4s, v21.4s\n\t"
                "scvtf v22.4s, v22.4s\n\t"
                "scvtf v23.4s, v23.4s\n\t"
                "scvtf v24.4s, v24.4s\n\t"
                "scvtf v25.4s, v25.4s\n\t"
                "scvtf v26.4s, v26.4s\n\t"
                "scvtf v27.4s, v27.4s\n\t"
                "scvtf v28.4s, v28.4s\n\t"
                "scvtf v29.4s, v29.4s\n\t"
                "scvtf v30.4s, v30.4s\n\t"
                "scvtf v31.4s, v31.4s\n\t"
                "fmul v8.4s, v8.4s, v2.4s\n\t"
                "fmul v9.4s, v9.4s, v2.4s\n\t"
                "fmul v10.4s, v10.4s, v2.4s\n\t"
                "fmul v11.4s, v11.4s, v2.4s\n\t"
                "fmul v12.4s, v12.4s, v2.4s\n\t"
                "fmul v13.4s, v13.4s, v2.4s\n\t"
                "fmul v14.4s, v14.4s, v2.4s\n\t"
                "fmul v15.4s, v15.4s, v2.4s\n\t"
                "fmul v16.4s, v16.4s, v2.4s\n\t"
                "fmul v17.4s, v17.4s, v2.4s\n\t"
                "fmul v18.4s, v18.4s, v2.4s\n\t"
                "fmul v19.4s, v19.4s, v2.4s\n\t"
                "fmul v20.4s, v20.4s, v2.4s\n\t"
                "fmul v21.4s, v21.4s, v2.4s\n\t"
                "fmul v22.4s, v22.4s, v2.4s\n\t"
                "fmul v23.4s, v23.4s, v2.4s\n\t"
                "fmul v24.4s, v24.4s, v2.4s\n\t"
                "fmul v25.4s, v25.4s, v2.4s\n\t"
                "fmul v26.4s, v26.4s, v2.4s\n\t"
                "fmul v27.4s, v27.4s, v2.4s\n\t"
                "fmul v28.4s, v28.4s, v2.4s\n\t"
                "fmul v29.4s, v29.4s, v2.4s\n\t"
                "fmul v30.4s, v30.4s, v2.4s\n\t"
                "fmul v31.4s, v31.4s, v2.4s\n\t"
                "fadd v8.4s, v8.4s, v3.4s\n\t"
                "fadd v9.4s, v9.4s, v3.4s\n\t"
                "fadd v10.4s, v10.4s, v3.4s\n\t"
                "fadd v11.4s, v11.4s, v3.4s\n\t"
                "fadd v12.4s, v12.4s, v3.4s\n\t"
                "fadd v13.4s, v13.4s, v3.4s\n\t"
                "fadd v14.4s, v14.4s, v3.4s\n\t"
                "fadd v15.4s, v15.4s, v3.4s\n\t"
                "fadd v16.4s, v16.4s, v3.4s\n\t"
                "fadd v17.4s, v17.4s, v3.4s\n\t"
                "fadd v18.4s, v18.4s, v3.4s\n\t"
                "fadd v19.4s, v19.4s, v3.4s\n\t"
                "fadd v20.4s, v20.4s, v3.4s\n\t"
                "fadd v21.4s, v21.4s, v3.4s\n\t"
                "fadd v22.4s, v22.4s, v3.4s\n\t"
                "fadd v23.4s, v23.4s, v3.4s\n\t"
                "fadd v24.4s, v24.4s, v3.4s\n\t"
                "fadd v25.4s, v25.4s, v3.4s\n\t"
                "fadd v26.4s, v26.4s, v3.4s\n\t"
                "fadd v27.4s, v27.4s, v3.4s\n\t"
                "fadd v28.4s, v28.4s, v3.4s\n\t"
                "fadd v29.4s, v29.4s, v3.4s\n\t"
                "fadd v30.4s, v30.4s, v3.4s\n\t"
                "fadd v31.4s, v31.4s, v3.4s\n\t"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[o1]]\n\t"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[o2]]\n\t"
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[o3]]\n\t"
                "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[o4]]\n\t"
                "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[o5]]\n\t"
                "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[o6]]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : [o1] "r"(op), [o2] "r"(op+16), [o3] "r"(op+32), [o4] "r"(op+48), [o5] "r"(op+64), [o6] "r"(op+80)
                : "v2","v3","memory");
        }
        asm volatile(
            "movi v8.16b, #0\n\t"
            "movi v9.16b, #0\n\t"
            "movi v10.16b, #0\n\t"
            "movi v11.16b, #0\n\t"
            "movi v12.16b, #0\n\t"
            "movi v13.16b, #0\n\t"
            "movi v14.16b, #0\n\t"
            "movi v15.16b, #0\n\t"
            "movi v16.16b, #0\n\t"
            "movi v17.16b, #0\n\t"
            "movi v18.16b, #0\n\t"
            "movi v19.16b, #0\n\t"
            "movi v20.16b, #0\n\t"
            "movi v21.16b, #0\n\t"
            "movi v22.16b, #0\n\t"
            "movi v23.16b, #0\n\t"
            "movi v24.16b, #0\n\t"
            "movi v25.16b, #0\n\t"
            "movi v26.16b, #0\n\t"
            "movi v27.16b, #0\n\t"
            "movi v28.16b, #0\n\t"
            "movi v29.16b, #0\n\t"
            "movi v30.16b, #0\n\t"
            "movi v31.16b, #0\n\t"
            : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31));
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(r8,0)+vgetq_lane_s32(r31,0); (void)sink;
    return (double)iters * 24.0 * 16.0 / secs / 1e9;
}

// GEMM inner-loop replica: 24 lane-bcast sdot + loads + int4 unpack over
// malloc streams (mode 0), or pure 24 sdot in same environment (mode 1).
static double sdotCoreGmacsMem(int iters, int mode) {
    if (mode == 3) return sdotCoreGmacsBlock(iters, 1);
    if (mode == 4) return sdotCoreGmacsBlock(iters, 0);
    static std::vector<int8_t> wbuf(8 * 1024 * 1024, 3), sbuf(2 * 1024 * 1024, 5);
    const int8_t* wp = wbuf.data();
    const int8_t* sp = sbuf.data();
    static std::vector<float> obuf(1024, 0.f);
    float* op = obuf.data();
    register int8x16_t x asm("v0") = vdupq_n_s8(3);
    register int8x16_t y asm("v1") = vdupq_n_s8(5);
    register int32x4_t r8 asm("v8") = vdupq_n_s32(0);
    register int32x4_t r9 asm("v9") = vdupq_n_s32(0);
    register int32x4_t r10 asm("v10") = vdupq_n_s32(0);
    register int32x4_t r11 asm("v11") = vdupq_n_s32(0);
    register int32x4_t r12 asm("v12") = vdupq_n_s32(0);
    register int32x4_t r13 asm("v13") = vdupq_n_s32(0);
    register int32x4_t r14 asm("v14") = vdupq_n_s32(0);
    register int32x4_t r15 asm("v15") = vdupq_n_s32(0);
    register int32x4_t r16 asm("v16") = vdupq_n_s32(0);
    register int32x4_t r17 asm("v17") = vdupq_n_s32(0);
    register int32x4_t r18 asm("v18") = vdupq_n_s32(0);
    register int32x4_t r19 asm("v19") = vdupq_n_s32(0);
    register int32x4_t r20 asm("v20") = vdupq_n_s32(0);
    register int32x4_t r21 asm("v21") = vdupq_n_s32(0);
    register int32x4_t r22 asm("v22") = vdupq_n_s32(0);
    register int32x4_t r23 asm("v23") = vdupq_n_s32(0);
    register int32x4_t r24 asm("v24") = vdupq_n_s32(0);
    register int32x4_t r25 asm("v25") = vdupq_n_s32(0);
    register int32x4_t r26 asm("v26") = vdupq_n_s32(0);
    register int32x4_t r27 asm("v27") = vdupq_n_s32(0);
    register int32x4_t r28 asm("v28") = vdupq_n_s32(0);
    register int32x4_t r29 asm("v29") = vdupq_n_s32(0);
    register int32x4_t r30 asm("v30") = vdupq_n_s32(0);
    register int32x4_t r31 asm("v31") = vdupq_n_s32(0);
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        const int8_t* w = wp + (i * 16) % (8 * 1024 * 1024 - 16);
        const int8_t* sr = (mode == 6) ? (sp + (i * 4096) % (2 * 1024 * 1024 - 48))
                                       : (sp + (i * 48) % (2 * 1024 * 1024 - 48));
        if (mode == 0 || mode == 6) {
            asm volatile(
                "ld1 {v5.16b}, [%[w]]\n\t"
                "ld1 {v0.16b, v1.16b, v2.16b}, [%[s]]\n\t"
                "ushr v3.16b, v5.16b, #4\n\t"
                "and v4.16b, v5.16b, v7.16b\n\t"
                "sdot v8.4s, v3.16b, v0.4b[0]\n\t"
                "sdot v9.4s, v3.16b, v0.4b[1]\n\t"
                "sdot v10.4s, v3.16b, v0.4b[2]\n\t"
                "sdot v11.4s, v3.16b, v0.4b[3]\n\t"
                "sdot v12.4s, v3.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v3.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v3.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v3.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v3.16b, v2.4b[0]\n\t"
                "sdot v17.4s, v3.16b, v2.4b[1]\n\t"
                "sdot v18.4s, v3.16b, v2.4b[2]\n\t"
                "sdot v19.4s, v3.16b, v2.4b[3]\n\t"
                "sdot v20.4s, v4.16b, v0.4b[0]\n\t"
                "sdot v21.4s, v4.16b, v0.4b[1]\n\t"
                "sdot v22.4s, v4.16b, v0.4b[2]\n\t"
                "sdot v23.4s, v4.16b, v0.4b[3]\n\t"
                "sdot v24.4s, v4.16b, v1.4b[0]\n\t"
                "sdot v25.4s, v4.16b, v1.4b[1]\n\t"
                "sdot v26.4s, v4.16b, v1.4b[2]\n\t"
                "sdot v27.4s, v4.16b, v1.4b[3]\n\t"
                "sdot v28.4s, v4.16b, v2.4b[0]\n\t"
                "sdot v29.4s, v4.16b, v2.4b[1]\n\t"
                "sdot v30.4s, v4.16b, v2.4b[2]\n\t"
                "sdot v31.4s, v4.16b, v2.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : [w] "r"(w), [s] "r"(sr), "w"(x), "w"(y)
                : "v3", "v4", "v5", "v7", "memory");
        } else if (mode == 2) {
            asm volatile(
                "ld1 {v5.16b}, [%[w]]\n\t"
                "ld1 {v0.16b, v1.16b, v2.16b}, [%[s]]\n\t"
                "ushr v3.16b, v5.16b, #4\n\t"
                "and v4.16b, v5.16b, v7.16b\n\t"
                "sdot v8.4s, v3.16b, v0.4b[0]\n\t"
                "sdot v9.4s, v3.16b, v0.4b[1]\n\t"
                "sdot v10.4s, v3.16b, v0.4b[2]\n\t"
                "sdot v11.4s, v3.16b, v0.4b[3]\n\t"
                "sdot v12.4s, v3.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v3.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v3.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v3.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v3.16b, v2.4b[0]\n\t"
                "sdot v17.4s, v3.16b, v2.4b[1]\n\t"
                "sdot v18.4s, v3.16b, v2.4b[2]\n\t"
                "sdot v19.4s, v3.16b, v2.4b[3]\n\t"
                "sdot v20.4s, v4.16b, v0.4b[0]\n\t"
                "sdot v21.4s, v4.16b, v0.4b[1]\n\t"
                "sdot v22.4s, v4.16b, v0.4b[2]\n\t"
                "sdot v23.4s, v4.16b, v0.4b[3]\n\t"
                "sdot v24.4s, v4.16b, v1.4b[0]\n\t"
                "sdot v25.4s, v4.16b, v1.4b[1]\n\t"
                "sdot v26.4s, v4.16b, v1.4b[2]\n\t"
                "sdot v27.4s, v4.16b, v1.4b[3]\n\t"
                "sdot v28.4s, v4.16b, v2.4b[0]\n\t"
                "sdot v29.4s, v4.16b, v2.4b[1]\n\t"
                "sdot v30.4s, v4.16b, v2.4b[2]\n\t"
                "sdot v31.4s, v4.16b, v2.4b[3]\n\t"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[o1]]\n\t"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[o2]]\n\t"
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[o3]]\n\t"
                "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[o4]]\n\t"
                "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[o5]]\n\t"
                "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[o6]]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : [w] "r"(w), [s] "r"(sr), [o1] "r"(op), [o2] "r"(op+16), [o3] "r"(op+32),
                  [o4] "r"(op+48), [o5] "r"(op+64), [o6] "r"(op+80), "w"(x), "w"(y)
                : "v3", "v4", "v5", "v7", "memory");
        } else {
            asm volatile(
                "sdot v8.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v9.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v10.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v11.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v12.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v17.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v18.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v19.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v20.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v21.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v22.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v23.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v24.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v25.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v26.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v27.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v28.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v29.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v30.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v31.4s, v0.16b, v1.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : "w"(x), "w"(y));
        }
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(r8, 0) + vgetq_lane_s32(r31, 0);
    (void)sink;
    return (double)iters * 24.0 * 16.0 / secs / 1e9;
}

// Chain-count scan: n independent loop-carried sdot accumulators per iteration
// (n=8/16/24), loop-invariant sources. Isolates loop-body size effects.
static double sdotCoreGmacsN(int iters, int n) {
    register int8x16_t x asm("v0") = vdupq_n_s8(3);
    register int8x16_t y asm("v1") = vdupq_n_s8(5);
    register int32x4_t r8 asm("v8") = vdupq_n_s32(0);
    register int32x4_t r9 asm("v9") = vdupq_n_s32(0);
    register int32x4_t r10 asm("v10") = vdupq_n_s32(0);
    register int32x4_t r11 asm("v11") = vdupq_n_s32(0);
    register int32x4_t r12 asm("v12") = vdupq_n_s32(0);
    register int32x4_t r13 asm("v13") = vdupq_n_s32(0);
    register int32x4_t r14 asm("v14") = vdupq_n_s32(0);
    register int32x4_t r15 asm("v15") = vdupq_n_s32(0);
    register int32x4_t r16 asm("v16") = vdupq_n_s32(0);
    register int32x4_t r17 asm("v17") = vdupq_n_s32(0);
    register int32x4_t r18 asm("v18") = vdupq_n_s32(0);
    register int32x4_t r19 asm("v19") = vdupq_n_s32(0);
    register int32x4_t r20 asm("v20") = vdupq_n_s32(0);
    register int32x4_t r21 asm("v21") = vdupq_n_s32(0);
    register int32x4_t r22 asm("v22") = vdupq_n_s32(0);
    register int32x4_t r23 asm("v23") = vdupq_n_s32(0);
    register int32x4_t r24 asm("v24") = vdupq_n_s32(0);
    register int32x4_t r25 asm("v25") = vdupq_n_s32(0);
    register int32x4_t r26 asm("v26") = vdupq_n_s32(0);
    register int32x4_t r27 asm("v27") = vdupq_n_s32(0);
    register int32x4_t r28 asm("v28") = vdupq_n_s32(0);
    register int32x4_t r29 asm("v29") = vdupq_n_s32(0);
    register int32x4_t r30 asm("v30") = vdupq_n_s32(0);
    register int32x4_t r31 asm("v31") = vdupq_n_s32(0);
    auto t0 = clk::now();
    if (n == 8) {
        for (int i = 0; i < iters; ++i) {
            asm volatile(
                "sdot v8.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v9.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v10.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v11.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v12.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v0.16b, v1.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15)
                : "w"(x), "w"(y));
        }
    } else if (n == 16) {
        for (int i = 0; i < iters; ++i) {
            asm volatile(
                "sdot v8.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v9.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v10.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v11.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v12.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v17.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v18.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v19.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v20.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v21.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v22.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v23.4s, v0.16b, v1.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23)
                : "w"(x), "w"(y));
        }
    } else {
        for (int i = 0; i < iters; ++i) {
            asm volatile(
                "sdot v8.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v9.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v10.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v11.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v12.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v13.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v14.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v15.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v16.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v17.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v18.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v19.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v20.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v21.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v22.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v23.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v24.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v25.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v26.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v27.4s, v0.16b, v1.4b[3]\n\t"
                "sdot v28.4s, v0.16b, v1.4b[0]\n\t"
                "sdot v29.4s, v0.16b, v1.4b[1]\n\t"
                "sdot v30.4s, v0.16b, v1.4b[2]\n\t"
                "sdot v31.4s, v0.16b, v1.4b[3]\n\t"
                : "+w"(r8), "+w"(r9), "+w"(r10), "+w"(r11), "+w"(r12), "+w"(r13), "+w"(r14), "+w"(r15), "+w"(r16), "+w"(r17), "+w"(r18), "+w"(r19), "+w"(r20), "+w"(r21), "+w"(r22), "+w"(r23), "+w"(r24), "+w"(r25), "+w"(r26), "+w"(r27), "+w"(r28), "+w"(r29), "+w"(r30), "+w"(r31)
                : "w"(x), "w"(y));
        }
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(r8, 0) + vgetq_lane_s32(r31, 0);
    (void)sink;
    return (double)iters * (double)n * 16.0 / secs / 1e9;
}

// unpack + sdot mix (the w4 GEMM inner-loop shape): 2 NEON int unpack ops
// (ushr high nibble + and low nibble) feeding 8 lane-bcast sdot. Isolates whether
// the NEON integer unpack throughput is the GEMM efficiency bottleneck.
static double sdotCoreGmacsUnpack(int iters) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0), a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
    register int8x16_t w asm("v8") = vdupq_n_s8(0x35);
    register int8x16_t y asm("v9") = vdupq_n_s8(5);
    register int8x16_t mask asm("v12") = vdupq_n_s8(15);
    register int32x4_t r0 asm("v0") = a0, r1 asm("v1") = a1, r2 asm("v2") = a2, r3 asm("v3") = a3;
    register int32x4_t r4 asm("v4") = a4, r5 asm("v5") = a5, r6 asm("v6") = a6, r7 asm("v7") = a7;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        asm volatile(
            "ushr v10.16b, v8.16b, #4\n\t"
            "and v11.16b, v8.16b, v12.16b\n\t"
            "sdot v0.4s, v10.16b, v9.4b[0]\n\t"
            "sdot v1.4s, v10.16b, v9.4b[1]\n\t"
            "sdot v2.4s, v10.16b, v9.4b[2]\n\t"
            "sdot v3.4s, v10.16b, v9.4b[3]\n\t"
            "sdot v4.4s, v11.16b, v9.4b[0]\n\t"
            "sdot v5.4s, v11.16b, v9.4b[1]\n\t"
            "sdot v6.4s, v11.16b, v9.4b[2]\n\t"
            "sdot v7.4s, v11.16b, v9.4b[3]\n\t"
            : "+w"(r0), "+w"(r1), "+w"(r2), "+w"(r3), "+w"(r4), "+w"(r5), "+w"(r6), "+w"(r7)
            : "w"(w), "w"(y), "w"(mask)
            : "v10", "v11");
    }
    double secs = seconds_since(t0);
    volatile int sink = vgetq_lane_s32(r0, 0) + vgetq_lane_s32(r7, 0);
    (void)sink;
    return (double)iters * 8.0 * 16.0 / secs / 1e9;
}

// SMT-aware: one thread per PHYSICAL core (mid phys=4,6,8,10; big phys=12; little 0-3).
extern "C" int runSdotPhys(int threads, int iters, int n) {
    const int phys[14] = {4, 6, 8, 10, 12, 0, 1, 2, 3, 5, 7, 9, 11, 13};
    if (threads <= 0) threads = 5;
    if (threads > 14) threads = 14;
    std::vector<double> per(threads, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] {
            pinCore(phys[t]); setQoS4();
            per[t] = (n == 24) ? sdotCoreGmacsN(iters, 24) : sdotCoreGmacsLane(iters);
            std::printf("phys-worker t%d cpu=%d rate=%.1f\n", t, sched_getcpu(), per[t]);
        });
    for (auto& th : ts) th.join();
    double sum = 0;
    for (int t = 0; t < threads; ++t) sum += per[t];
    std::printf("SMT-phys %d-thread aggregate: %.1f GMAC/s (%.1f GFLOPS)\n", threads, sum, sum * 2);
    std::fflush(stdout);
    return 0;
}

extern "C" int runCpuset() {
    auto cat = [](const char* path) -> std::string {
        FILE* f = fopen(path, "r");
        if (!f) return "ERR";
        char buf[128]; std::string r;
        if (fgets(buf, sizeof(buf), f)) r = buf;
        fclose(f);
        while (!r.empty() && (r.back()=='\n'||r.back()=='\r')) r.pop_back();
        return r;
    };
    std::printf("self cpuset=%s\n", cat("/proc/self/cpuset").c_str());
    int tid = (int)syscall(SYS_gettid);
    const char* groups[] = {"top-app", "foreground", "system"};
    for (auto g : groups) {
        for (auto file : {"cgroup.procs", "tasks"}) {
            char path[128];
            std::snprintf(path, sizeof(path), "/dev/cpuset/%s/%s", g, file);
            FILE* f = fopen(path, "w");
            if (!f) { std::printf("write %s: fopen ERR errno=%d\n", path, errno); continue; }
            int n = fprintf(f, "%d\n", tid);
            int rc = fclose(f);
            std::printf("write %s tid=%d: n=%d fclose=%d errno=%d -> self=%s\n",
                        path, tid, n, rc, errno, cat("/proc/self/cpuset").c_str());
        }
    }
    std::fflush(stdout);
    return 0;
}

extern "C" int runTopo() {
    auto cat = [](const char* path) -> std::string {
        FILE* f = fopen(path, "r");
        if (!f) return std::string("ERR");
        char buf[128]; std::string r;
        if (fgets(buf, sizeof(buf), f)) r = buf;
        fclose(f);
        while (!r.empty() && (r.back()=='\n' || r.back()=='\r')) r.pop_back();
        return r;
    };
    std::printf("smt/active=%s\n", cat("/sys/devices/system/cpu/smt/active").c_str());
    for (int c = 0; c < 14; ++c) {
        char p[128];
        std::snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        std::string sib = cat(p);
        std::snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        std::string cid = cat(p);
        std::snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        std::string pkg = cat(p);
        std::printf("cpu%d: siblings=%s core_id=%s pkg=%s\n", c, sib.c_str(), cid.c_str(), pkg.c_str());
    }
    std::fflush(stdout);
    return 0;
}

extern "C" int runSdotMem(int mode, int iters, int core, int threads) {
    if (core < 0 || core > 13) core = 6;
    g_pinCore = core;
    pinCore(core);
    setQoS4();
    {
        auto warmEnd = clk::now() + std::chrono::milliseconds(300);
        while (clk::now() < warmEnd) { sdotCoreGmacsMem(100000, mode); }
    }
    FreqSampler sampler;
    sampler.start(core);
    double best = 0;
    PmuCounters pmu;
    pmu.start();
    for (int r = 0; r < 3; ++r) {
        double g = sdotCoreGmacsMem(iters, mode);
        if (g > best) best = g;
    }
    pmu.stop("sdotmem");
    long f = sampler.finish();
    std::printf("\n## sdot mem-replica mode=%d core=%d freq=%ld -> %.1f GMAC/s (%.2f sdot/cy)\n",
                mode, core, f, best, best * 1e9 / 16.0 / (double)f / 1000.0);
    if (threads > 1) {
        std::vector<double> per(threads, 0);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                pinCore(4 + t); setQoS4();
                per[t] = sdotCoreGmacsMem(iters, mode);
            });
        for (auto& th : ts) th.join();
        double sum = 0;
        for (int t = 0; t < threads; ++t) sum += per[t];
        std::printf("mem-replica %d-thread aggregate: %.1f GMAC/s (%.1f/thread)\n", threads, sum, sum / threads);
    }
    std::fflush(stdout);
    return 0;
}

extern "C" int runSdotN(int n, int iters, int core, int threads) {
    if (core < 0 || core > 13) core = 6;
    g_pinCore = core;
    pinCore(core);
    setQoS4();
    {
        auto warmEnd = clk::now() + std::chrono::milliseconds(300);
        while (clk::now() < warmEnd) { sdotCoreGmacsN(100000, n); }
    }
    FreqSampler sampler;
    sampler.start(core);
    double best = 0;
    for (int r = 0; r < 3; ++r) {
        double g = sdotCoreGmacsN(iters, n);
        if (g > best) best = g;
    }
    long f = sampler.finish();
    std::printf("\n## sdot chain-count scan: n=%d core=%d freq=%ld KHz -> %.1f GMAC/s (%.2f sdot/cycle)\n",
                n, core, f, best, best * 1e9 / 16.0 / (double)f / 1000.0);
    if (threads > 1) {
        std::vector<double> per(threads, 0);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                pinCore(4 + t); setQoS4();
                per[t] = sdotCoreGmacsN(iters, n);
                std::printf("N-worker t%d: cpu=%d rate=%.1f GMAC/s\n", t, sched_getcpu(), per[t]);
            });
        for (auto& th : ts) th.join();
        double sum = 0;
        for (int t = 0; t < threads; ++t) sum += per[t];
        std::printf("N %d-thread aggregate: %.1f GMAC/s (%.2f sdot/cycle/thread)\n", threads, sum,
                    sum / threads * 1e9 / 16.0 / (double)f / 1000.0);
    }
    std::fflush(stdout);
    return 0;
}

// Entry called from native-bench "@sdot threads iters".
// Reports per-core and aggregate int8 sdot GMAC/s (x2 = GOPS) — the compute roofline.
extern "C" int runSdotPeak(int threads, int iters, int pinCoreIdx) {
    if (threads <= 0) threads = 1;
    if (iters <= 0) iters = 20000000;
    if (pinCoreIdx < 0 || pinCoreIdx > 13) pinCoreIdx = 12;
    g_pinCore = pinCoreIdx;
    setQoS4();
    // Sustained warmup so the cpufreq governor boosts the pinned core to max before
    // timing — a short burst finishes before the governor responds and measures at
    // idle freq (the Kirin big core stuck at 1200MHz instead of 2750MHz otherwise).
    pinCore(pinCoreIdx);
    {
        auto warmEnd = clk::now() + std::chrono::milliseconds(300);
        int32x4_t z = vdupq_n_s32(0);
        int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(5);
        while (clk::now() < warmEnd) {
            for (int k = 0; k < 100000; ++k) {
                z = vdotq_s32(z, a, b);
                asm volatile("" : "+w"(z));
            }
        }
        volatile int sink = vgetq_lane_s32(z, 0); (void)sink;
    }
    FreqSampler sampler;
    sampler.start(pinCoreIdx);
    // single-core peak (best of 3)
    double core = 0;
    for (int r = 0; r < 3; ++r) {
        double g = sdotCoreGmacs(iters);
        if (g > core) core = g;
    }
    double lane = 0;
    for (int r = 0; r < 3; ++r) {
        double g = sdotCoreGmacsLane(iters);
        if (g > lane) lane = g;
    }
    long sampledKHz = sampler.finish();
    std::printf("\n## int8 sdot compute roofline (iters=%d, core=%d, freq=%ld KHz sampled-max)\n", iters, pinCoreIdx, sampledKHz);
    double unpack = 0;
    for (int r = 0; r < 3; ++r) {
        double g = sdotCoreGmacsUnpack(iters);
        if (g > unpack) unpack = g;
    }
    std::printf("single-core full-vector: %.1f GMAC/s (%.1f GOPS)\n", core, core * 2);
    std::printf("single-core lane-bcast : %.1f GMAC/s (%.1f GOPS)  [GEMM uses this form]\n", lane, lane * 2);
    std::printf("single-core unpack+sdot: %.1f GMAC/s (%.1f GOPS)  [2 unpack + 8 sdot mix]\n", unpack, unpack * 2);
    if (threads > 1) {
        std::vector<double> per(threads, 0);
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                pinCore(4 + t); setQoS4();
                per[t] = sdotCoreGmacsLane(iters);
                std::printf("worker t%d: cpu=%d rate=%.1f GMAC/s freq=%ld KHz\n", t, sched_getcpu(), per[t], curFreqKHz(4 + t));
            });
        for (auto& th : ts) th.join();
        double sum = 0;
        for (int t = 0; t < threads; ++t) sum += per[t];
        std::printf("%d-thread aggregate: %.1f GMAC/s (%.1f GOPS)\n", threads, sum, sum * 2);
    }
    std::fflush(stdout);
    return 0;
}

// Entry called from native-bench "@gemm M oc ic threads precision nbit".
// Warm compute-bound GEMM roofline: GFLOPS for batch=M through oc x ic w<nbit>.
extern "C" int runGemmRoofline(int M, int oc, int ic, int threads, int precision, int nbit, int pinMode) {
    if (M <= 0) M = 512;
    if (oc <= 0) oc = 1024;
    if (ic <= 0) ic = 1024;
    if (threads <= 0) threads = 6;
    if (nbit <= 0) nbit = 4;
    const int iters = 50;
    std::printf("\n## Warm GEMM roofline (CPU, precision=%d, M=%d oc=%d ic=%d w%d, threads=%d)\n",
                precision, M, oc, ic, nbit, threads);
    // Thread-placement diagnostic: sample /proc/self/task/*/stat during the GEMM
    // to see which CPUs the worker threads actually run on and their CPU time.
    struct ThreadStat { long ticks = 0; int lastCpu = -1; std::vector<int> cpus; };
    std::map<int, ThreadStat> tstats;
    std::atomic<bool> tstop{false};
    std::thread tsampler([&] {
        while (!tstop.load()) {
            DIR* d = opendir("/proc/self/task");
            if (d) {
                struct dirent* e;
                while ((e = readdir(d)) != nullptr) {
                    int tid = atoi(e->d_name);
                    if (tid <= 0) continue;
                    char path[64];
                    std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
                    FILE* f = fopen(path, "r");
                    if (!f) continue;
                    char buf[512];
                    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
                    fclose(f);
                    buf[n] = 0;
                    char* close = strrchr(buf, ')');
                    if (!close) continue;
                    std::vector<std::string> fields;
                    char* tok = std::strtok(close + 2, " ");
                    while (tok) { fields.push_back(tok); tok = std::strtok(nullptr, " "); }
                    if (fields.size() < 37) continue;
                    long ticks = atol(fields[11].c_str()) + atol(fields[12].c_str());
                    int cpu = atoi(fields[36].c_str());
                    auto& ts = tstats[tid];
                    ts.ticks = ticks;
                    ts.lastCpu = cpu;
                    if (ts.cpus.empty() || ts.cpus.back() != cpu) ts.cpus.push_back(cpu);
                }
                closedir(d);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    });
    std::map<int, long> t0ticks;
    for (auto& kv : tstats) t0ticks[kv.first] = kv.second.ticks;
    // pinMode 3: after workers exist (~600ms), pin each hot thread to its own mid
    // core directly via sched_setaffinity (bypasses MNN config propagation).
    if (pinMode == 3) {
        pinCore(4);
        setQoS4();
        std::printf("master tid=%d pin cpu4\n", (int)syscall(SYS_gettid));
    }
    MNNSetOHOSUclamp(pinMode == 4 ? 1 : 0);
    MNNSetOHOSPinMode(pinMode == 5 ? 1 : 0);
    if (pinMode == 5) {
        pinCore(4);
        setQoS4();
    }
    if (pinMode == 4) {
        setQoS4();
        applyUclampMaster();
    }
    std::thread pinner;
    if (pinMode == 4) {
        pinner = std::thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        std::vector<std::pair<long, int>> hot;
        for (auto& kv : tstats) {
            long d = kv.second.ticks - t0ticks[kv.first];
            if (d > 20) hot.push_back({d, kv.first});
        }
        std::sort(hot.begin(), hot.end(), std::greater<std::pair<long, int>>());
        int nc = (int)hot.size();
        if (nc > threads + 1) nc = threads + 1;
        for (int i = 0; i < nc; ++i) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(4 + i, &set);
            int ret = syscall(SYS_sched_setaffinity, hot[i].second, sizeof(set), &set);
            std::printf("pin tid=%d -> cpu%d ret=%d\n", hot[i].second, 4 + i, ret);
        }
        std::fflush(stdout);
        });
    }
    PmuCounters pmu;
    pmu.start();
    GemvResult r = benchGemmWarm(M, oc, ic, nbit, 64, precision, threads, iters, pinMode);
    pmu.stop("gemm");
    if (pinner.joinable()) pinner.join();
    tstop.store(true);
    tsampler.join();
    std::vector<std::pair<long, int>> rank;
    for (auto& kv : tstats) {
        long d = kv.second.ticks - t0ticks[kv.first];
        if (d > 0) rank.push_back({d, kv.first});
    }
    std::sort(rank.begin(), rank.end(), std::greater<std::pair<long, int>>());
    std::printf("thread placement (top %zu by CPU ticks, 100 ticks/s):\n", rank.size() < 10 ? rank.size() : 10);
    for (size_t i = 0; i < rank.size() && i < 10; ++i) {
        auto& ts = tstats[rank[i].second];
        std::printf("  tid=%d ticks=%ld lastCpu=%d cpus=", rank[i].second, rank[i].first, ts.lastCpu);
        for (size_t c = 0; c < ts.cpus.size() && c < 12; ++c) std::printf("%d,", ts.cpus[c]);
        std::printf("\n");
    }
    double flop = 2.0 * M * oc * ic;
    std::printf("us/iter=%.1f  GFLOPS=%.1f  (work=%.2f GFLOP)\n", r.avgUs, r.gflops, flop / 1e9);
    std::fflush(stdout);
    return 0;
}

// Entry called from native-bench "@gemvbw M K threads precision".
// Prints peak DRAM BW then w8/w4/w3/w2 GEMV eff GB/s / %peak for the shape.
extern "C" int runGemvBWRoofline(int M, int K, int threads, int precision) {
    if (M <= 0) M = 1024;
    if (K <= 0) K = 1024;
    if (threads <= 0) threads = 4;
    const int blocksize = 64;
    const int iters = 100;
    std::printf("\n## GemvBW roofline (CPU, precision=%d, block=%d, M=oc=%d K=ic=%d, threads=%d)\n",
                precision, blocksize, M, K, threads);
    double peakBw = measurePeakBwGBs((size_t)256 << 20, threads, 5);
    std::printf("peak streaming BW = %.1f GB/s @ %d threads (roofline)\n", peakBw, threads);
    std::printf("type | thr |  us/iter |  W MiB | eff GB/s | %%peak | GFLOPS\n");
    std::printf("-----|----:|---------:|-------:|---------:|------:|-------:\n");
    for (int nbit : {8, 4, 3, 2}) {
        GemvResult r = benchGemv(M, K, nbit, blocksize, precision, threads, iters);
        double pct = 100.0 * r.effBwGBs / peakBw;
        std::printf("w%-3d | %3d | %8.1f | %6.2f | %8.1f | %5.1f | %6.1f\n",
                    nbit, threads, r.avgUs, r.weightBytes / (1024.0 * 1024.0), r.effBwGBs, pct, r.gflops);
    }
    std::fflush(stdout);
    return 0;
}