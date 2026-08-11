//
//  MetalBackend.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalEnv.hpp"
#import "backend/metal/MetalReplay.hpp"
#define MNN_METAL
#import <MNN/MNNSharedContext.h>
#define METAL_CONST_BUFFER_LIMIT 128
#define METAL_SEPERATE_MAX_COUNT 2
// overload of MTLGPUFamilyMetal3/MTLGPUFamilyMetal4 (not available in some environments)
#define MTLGPUFamilyMetal3_MNN 5001
#define MTLGPUFamilyMetal4_MNN 5002

#define CHECK_IOS_UI_STATUS
#if MNN_METAL_ENABLED
#include <mutex>
#include <chrono>
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/TensorUtils.hpp"
#include "MetalCache_generated.h"
#include "core/MNNFileUtils.h"
#import "backend/metal/MetalConvolution1x1.hpp"
#import "backend/metal/MetalRaster.hpp"
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor) {
    if (nullptr == content || nullptr == tensor) {
        return 0;
    }
    auto t = (MNN::Tensor*)tensor;
    auto des = MNN::TensorUtils::getDescribeOrigin(t);
    content->buffer = ((MNN::MetalRuntimeAllocator::MetalBufferAlloc*)t->deviceId())->getBuffer();
    content->texture = nil;
    content->offset = des->offset;
    return 0;
}

#if MNN_METAL_OP_PROFILE
#include <algorithm>
namespace {
// Thread-safe aggregator of per-op GPU time (accumulated from command buffer
// completion handlers). Prints a sorted summary at process exit.
class MetalOpProfiler {
public:
    void add(const std::string& name, double ms) {
        std::lock_guard<std::mutex> _l(mMutex);
        auto& e = mStat[name];
        e.first  += ms;
        e.second += 1;
        mTotal   += ms;
    }
    // Record one (start, end) sample in nanoseconds (GPU clock, tick-scaled).
    // Only accumulated when MNN_METAL_OP_PROFILE_TIMELINE=<path> is set; the
    // aggregate table is unaffected either way. The timeline is dumped to the
    // given path (CSV) at process exit so a gantt chart can be built off it.
    void addSample(const std::string& name, double t0_ns, double t1_ns) {
        if (!timelineEnabled()) return;
        std::lock_guard<std::mutex> _l(mMutex);
        mTimeline.push_back({t0_ns, t1_ns, name});
    }
    // Global (backend-instance-independent) op name registry: create backend and
    // execute backend may differ, so names must be keyed by the execution pointer.
    void registerName(const void* exe, const std::string& name) {
        std::lock_guard<std::mutex> _l(mNameMutex);
        mNames[exe] = name;
    }
    std::string lookupName(const void* exe) {
        std::lock_guard<std::mutex> _l(mNameMutex);
        auto it = mNames.find(exe);
        return it != mNames.end() ? it->second : std::string("Unknown");
    }
    void print() {
        std::lock_guard<std::mutex> _l(mMutex);
        if (!mStat.empty()) {
            std::vector<std::pair<std::string, std::pair<double, int>>> items(mStat.begin(), mStat.end());
            std::sort(items.begin(), items.end(),
                      [](const std::pair<std::string, std::pair<double, int>>& a,
                         const std::pair<std::string, std::pair<double, int>>& b) {
                          return a.second.first > b.second.first;
                      });
            printf("\n===== Metal Per-Op GPU Time Profile =====\n");
            printf("%-22s %12s %10s %12s %8s\n", "OpType", "GPU(ms)", "Calls", "Avg(us)", "Ratio");
            for (auto& it : items) {
                double t = it.second.first;
                int    c = it.second.second;
                printf("%-22s %12.3f %10d %12.3f %7.2f%%\n", it.first.c_str(), t, c,
                       c > 0 ? t * 1000.0 / c : 0.0, mTotal > 0 ? t / mTotal * 100.0 : 0.0);
            }
            printf("%-22s %12.3f\n", "TOTAL", mTotal);
            printf("=========================================\n");
            mStat.clear();
            mTotal = 0;
        }
        // Dump timeline to CSV if requested. Format: start_ns,end_ns,dur_us,name
        // start_ns is rebased to the earliest sample so numbers are readable;
        // downstream tools can subtract further if they want a rel-first frame.
        const char* csvPath = MetalEnv::get().opProfileTimeline;
        if (csvPath != nullptr && csvPath[0] != '\0' && !mTimeline.empty()) {
            std::sort(mTimeline.begin(), mTimeline.end(),
                      [](const TimelineEntry& a, const TimelineEntry& b){ return a.t0_ns < b.t0_ns; });
            double base = mTimeline.front().t0_ns;
            FILE* fp = fopen(csvPath, "w");
            if (fp != nullptr) {
                fprintf(fp, "start_ns,end_ns,dur_us,name\n");
                for (const auto& e : mTimeline) {
                    fprintf(fp, "%.0f,%.0f,%.3f,%s\n",
                            e.t0_ns - base, e.t1_ns - base,
                            (e.t1_ns - e.t0_ns) / 1000.0, e.name.c_str());
                }
                fclose(fp);
                printf("[MetalOpProfiler] timeline dumped: %s (%zu samples)\n",
                       csvPath, mTimeline.size());
            } else {
                printf("[MetalOpProfiler] failed to open timeline path: %s\n", csvPath);
            }
            mTimeline.clear();
        }
    }
    ~MetalOpProfiler() {
        print();
    }
private:
    static bool timelineEnabled() {
        return MetalEnv::get().opProfileTimeline != nullptr;
    }
    struct TimelineEntry {
        double t0_ns;
        double t1_ns;
        std::string name;
    };
    std::mutex mMutex;
    std::map<std::string, std::pair<double, int>> mStat;
    double mTotal = 0;
    std::vector<TimelineEntry> mTimeline;
    std::mutex mNameMutex;
    std::map<const void*, std::string> mNames;
};
static MetalOpProfiler gMetalOpProfiler;

// GPU-tick → nanosecond calibration for MTLCounterSampleBuffer timestamps.
// Two correlated (cpu, gpu) samples give ns-per-tick; cpu timestamps are ns.
struct MetalGpuTickScale {
    std::mutex mMutex;
    MTLTimestamp mCpu0 = 0, mGpu0 = 0;
    bool mHasBase = false;
    double mNsPerTick = 0.0;
    void begin(id<MTLDevice> device) {
        std::lock_guard<std::mutex> _l(mMutex);
        if (!mHasBase) {
            if (@available(iOS 14.0, macOS 11.0, *)) {
                [device sampleTimestamps:&mCpu0 gpuTimestamp:&mGpu0];
                mHasBase = (mGpu0 != 0);
            }
        }
    }
    double nsPerTick(id<MTLDevice> device) {
        std::lock_guard<std::mutex> _l(mMutex);
        if (mNsPerTick > 0.0) {
            return mNsPerTick;
        }
        if (@available(iOS 14.0, macOS 11.0, *)) {
            MTLTimestamp cpu1 = 0, gpu1 = 0;
            [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
            // require >= 2ms of elapsed cpu time for a stable ratio
            if (mHasBase && gpu1 > mGpu0 && cpu1 > mCpu0 && (cpu1 - mCpu0) > 2000000ULL) {
                mNsPerTick = double(cpu1 - mCpu0) / double(gpu1 - mGpu0);
                return mNsPerTick;
            }
        }
        return 1.0;  // assume ns ticks until calibrated
    }
};
static MetalGpuTickScale gMetalGpuTickScale;

// Pool of counter sample buffers, recycled by command buffer completion
// handlers. Global (not per-backend) so a handler outliving its backend
// never touches freed state.
static std::mutex gProfileSampleBufferPoolMutex;
static std::vector<id<MTLCounterSampleBuffer>> gProfileSampleBufferPool;
static constexpr int kProfileSampleBufferCapacity = 1024;  // 512 encoders per command buffer
} // namespace
#endif

namespace MNN {

static void _MetalApplyTensor(uint8_t* host, size_t offset, Tensor* t) {
    // ptr of MetalBufferAlloc
    t->buffer().device = (uint64_t)host;
    auto des = TensorUtils::getDescribeOrigin(t);
    des->offset = offset;
}
BufferAllocator* MetalRuntime::createDynamicAllocator(int index, bool secondResize) const {
    if (hint().memoryAllocatorType == Runtime::Allocator_Defer && secondResize) {
        return new DeferBufferAllocator(buffer(index), 1024, _MetalApplyTensor);
    }
    if (mStaticAllocatorRaw.get() != nullptr) {
        return new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocatorRaw.get()), 1024);
    }
    return new EagerBufferAllocator(BufferAllocator::Allocator::createRecurse(mStaticAllocator.get()), 1024);
}

struct TunedInfo {
    std::vector<std::unique_ptr<MetalCache::OpInfoT>> mInfos;
};

void registerMetalOps();
#ifdef MNN_SUPPORT_RENDER
extern void registerMetalRenderOps();
#endif

static inline std::map<OpType, MetalBackend::Creator *> *getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, MetalBackend::Creator *> *ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, MetalBackend::Creator *>; });
    return ret;
}

void MetalBackend::addCreator(OpType t, Creator *c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
    }
    map->insert(std::make_pair(t, c));
}

MetalBackend::MetalBackend(const MetalRuntime* runtime, bool usefp16AsFp32, BackendConfig::MemoryMode mode) : Backend(MNN_FORWARD_METAL),
    mEmptyMem(nil)
    {
    mRuntime = runtime;
    auto ctx = (__bridge MNNMetalContext *)runtime->context();
    mBufferPool.reset(runtime->createDynamicAllocator(0, false));
    mExecutionBufferPool.reset(new EagerBufferAllocator(runtime->buffer(0)->root, 1024));
    mCurrentAllocator = mBufferPool.get();
    mUseFloatAsFp16 = usefp16AsFp32;
    mMemoryMode = mode;
    mIsIphone = ctx.isIphone;
    if (runtime->getCommandQueue() == nil) {
        // one command queue can create only a few command buffer, so let each backend own a command queue
        _commandQueue = [[ctx device] newCommandQueue];
    } else {
        // otherwise forbid defer encode optimize
        _commandQueue = runtime->getCommandQueue();
    }
#if MNN_METAL_OP_PROFILE
    {
        const bool sLegacy = MetalEnv::get().opProfileLegacy;
        mProfileCounterMode = false;
        if (!sLegacy) {
            if (@available(iOS 14.0, macOS 11.0, *)) {
                id<MTLDevice> device = [ctx device];
                bool stageBoundary = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
                bool hasTimestamp = false;
                for (id<MTLCounterSet> cs in device.counterSets) {
                    if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                        hasTimestamp = true;
                        break;
                    }
                }
                mProfileCounterMode = stageBoundary && hasTimestamp;
                if (mProfileCounterMode) {
                    gMetalGpuTickScale.begin(device);
                }
            }
        }
        static bool sLogged = false;
        if (!sLogged) {
            sLogged = true;
            MNN_PRINT("[MetalProfile] mode: %s\n", mProfileCounterMode ?
                      "counter-sample (per-encoder GPU timestamps, accurate absolute times)" :
                      "legacy (per-op command buffer, relative ordering only)");
        }
    }
#endif
    if(((MetalRuntime *)mRuntime)->supportTensorOps()) {
        mSupportTensorApi = true;
        // Probe every matmul2d descriptor shape actually used by MNN kernels
        // (attention 32x32x32, conv 32x64x64 / 32x64x32 / 64x64x32, plus the
        // dynamic-K device-tensor form). If a future MPP header rejects any of
        // them, the probe fails and the tensor api is disabled as a whole,
        // instead of passing a toy shape and then failing kernel compilation
        // at runtime on every dispatch.
        const char * src_tensor_f16 = "\n"
            "#include <metal_stdlib> \n"
            "#include <metal_tensor> \n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
            " \n"
            "using namespace metal; \n"
            "using namespace mpp::tensor_ops; \n"
            " \n"
            "template <int M, int N, int K> \n"
            "static void probe_static_shape(threadgroup half* buf) { \n"
            "    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(buf, dextents<int32_t, 2>(K, M)); \n"
            "    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(buf + M * K, dextents<int32_t, 2>(K, N)); \n"
            "    matmul2d< \n"
            "        matmul2d_descriptor(M, N, K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), \n"
            "        execution_simdgroups<4>> mm; \n"
            "    auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
            "    auto sA = tA.slice(0, 0); \n"
            "    auto sB = tB.slice(0, 0); \n"
            "    mm.run(sA, sB, cT); \n"
            "    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)buf, dextents<int32_t, 2>(N, M)); \n"
            "    cT.store(tC); \n"
            "} \n"
            " \n"
            "kernel void dummy_kernel( \n"
            "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
            "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
            "    device float * C [[buffer(2)]], \n"
            "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
            "{ \n"
            "    auto tA = A.slice(0, (int)tgid.y); \n"
            "    auto tB = B.slice((int)tgid.x, 0); \n"
            " \n"
            "    matmul2d< \n"
            "        matmul2d_descriptor(16, 8, dynamic_extent), \n"
            "        execution_simdgroups<4>> mm; \n"
            " \n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
            " \n"
            "    auto sA = tA.slice(0, 0); \n"
            "    auto sB = tB.slice(0, 0); \n"
            "    mm.run(sB, sA, cT); \n"
            " \n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
            " \n"
            "    cT.store(tC); \n"
            " \n"
            "    threadgroup half sdata[6144]; \n"
            "    probe_static_shape<32, 32, 32>(sdata); \n"
            "    probe_static_shape<32, 64, 64>(sdata); \n"
            "    probe_static_shape<32, 64, 32>(sdata); \n"
            "    probe_static_shape<64, 64, 32>(sdata); \n"
            "}";
        
        auto pipeline = makeComputePipelineWithSourceOption(src_tensor_f16, "dummy_kernel", nullptr);
        if(pipeline == nullptr) {
            MNN_PRINT("Metal4 Tensor api compile err, disable tensor api.\n");
            mSupportTensorApi = false;
        }
    }
    if (mSupportTensorApi) {
        // Separate probe for matmul2d INPUT cooperative tensors. Fused attention
        // needs the QK destination to become the PV left operand in registers,
        // which requires input cooperative tensors -- and those are only allowed
        // at single-simdgroup scope (MPPTensorOpsMatMul2dImpl.h: "Input
        // cooperative tensors require a single SIMD group"). None of MNN's other
        // tensor kernels use this, so it gets its own capability flag.
        // Covers all three shapes the fused kernel needs: QK (B transposed),
        // PV (neither transposed), and PV with an fp32 left operand (the
        // softmax output is kept in fp32).
        const char* src_coop_input = "\n"
            "#include <metal_stdlib> \n"
            "#include <metal_tensor> \n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
            "using namespace metal; \n"
            "using namespace mpp::tensor_ops; \n"
            " \n"
            "template <bool TB, typename AT> \n"
            "static float probe_coop_input(float seed) { \n"
            "    matmul2d<matmul2d_descriptor(16, 32, 16, false, TB, true, \n"
            "             matmul2d_descriptor::mode::multiply_accumulate), \n"
            "             metal::execution_simdgroup> mm; \n"
            "    auto ct_a = mm.template get_left_input_cooperative_tensor<AT, half, float>(); \n"
            "    auto ct_b = mm.template get_right_input_cooperative_tensor<AT, half, float>(); \n"
            "    auto ct_c = mm.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>(); \n"
            "    for (ushort i = 0; i < 8; i++) { ct_a[i] = AT(seed); } \n"
            "    for (ushort i = 0; i < 16; i++) { ct_b[i] = half(seed); ct_c[i] = 0.0f; } \n"
            "    mm.run(ct_a, ct_b, ct_c); \n"
            "    float acc = 0.0f; \n"
            "    for (ushort i = 0; i < 16; i++) { acc += float(ct_c[i]); } \n"
            "    return acc; \n"
            "} \n"
            " \n"
            "kernel void probe_kernel(device float* out [[buffer(0)]], \n"
            "                        uint tid [[thread_position_in_grid]]) { \n"
            "    float acc = probe_coop_input<true,  half >(1.0f) \n"
            "              + probe_coop_input<false, half >(1.0f) \n"
            "              + probe_coop_input<false, float>(1.0f); \n"
            "    out[tid] = acc; \n"
            "}";
        auto coopPipeline = makeComputePipelineWithSourceOption(src_coop_input, "probe_kernel", nullptr);
        mSupportTensorCoopInput = (coopPipeline != nullptr);
        if (!mSupportTensorCoopInput) {
            MNN_PRINT("Metal4 tensor input-cooperative-tensor unsupported, fused attention disabled.\n");
        }
    }
    _commandBuffer = nil;
    setUpGPUEnabledSwitch();
}
MetalBackend::~MetalBackend() {
    flushEncoder();
    removeNotificationsObservers();
}



void MetalBackend::setUpGPUEnabledSwitch() {
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    __block UIApplicationState state;
    if ([NSThread isMainThread]) {
        state = [UIApplication sharedApplication].applicationState;
    } else {
        dispatch_semaphore_t latch = dispatch_semaphore_create(0);
        dispatch_async(dispatch_get_main_queue(), ^{
            state = [UIApplication sharedApplication].applicationState;
            dispatch_semaphore_signal(latch);
        });
        dispatch_semaphore_wait(latch, DISPATCH_TIME_FOREVER);
    }
    mGPUEnabledSwitch.store(state == UIApplicationStateActive);
    // Use DidBecomeActive instead of WillEnterForeground: a backend created while the app
    // is Inactive (launch transition / screen locked) would otherwise never re-enable GPU.
    mForegroundObserver = [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationDidBecomeActiveNotification object:nil queue:nil usingBlock:^(NSNotification * _Nonnull notification) {
        mGPUEnabledSwitch.store(true);
    }];
    mBackgroundObserver = [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil queue:nil usingBlock:^(NSNotification * _Nonnull notification) {
        mGPUEnabledSwitch.store(false);
    }];
#endif
}

void MetalBackend::removeNotificationsObservers() {
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    [[NSNotificationCenter defaultCenter] removeObserver:mForegroundObserver];
    [[NSNotificationCenter defaultCenter] removeObserver:mBackgroundObserver];
#endif
}

id<MTLComputeCommandEncoder> MetalBackend::encoder_net() const {
#if MNN_METAL_OP_PROFILE
    if (mProfileCounterMode) {
        if (@available(iOS 14.0, macOS 11.0, *)) {
            auto cmdBuffer = getCommandBufferForNet();
            if (nil != mProfileSampleBuffer && mProfileSampleCursor + 2 > kProfileSampleBufferCapacity) {
                // current sample buffer exhausted mid-command-buffer: seal it and continue
                mProfileSealedBuffers.push_back({mProfileSampleBuffer, mProfileSampleCursor, std::move(mProfilePendingSamples)});
                mProfilePendingSamples.clear();
                mProfileSampleBuffer = nil;
            }
            if (nil == mProfileSampleBuffer) {
                mProfileSampleBuffer = profileAcquireSampleBuffer();
                mProfileSampleCursor = 0;
            }
            if (nil != mProfileSampleBuffer && mProfileSampleCursor + 2 <= kProfileSampleBufferCapacity) {
                MTLComputePassDescriptor* passDesc = [MTLComputePassDescriptor computePassDescriptor];
                MTLComputePassSampleBufferAttachmentDescriptor* att = passDesc.sampleBufferAttachments[0];
                att.sampleBuffer = mProfileSampleBuffer;
                att.startOfEncoderSampleIndex = mProfileSampleCursor;
                att.endOfEncoderSampleIndex = mProfileSampleCursor + 1;
                id<MTLComputeCommandEncoder> result = [cmdBuffer computeCommandEncoderWithDescriptor:passDesc];
                if (nil != result) {
                    mProfileCurSampleIndex = mProfileSampleCursor;
                    mProfileSampleCursor += 2;
                    return result;
                }
            }
            // sample buffer exhausted or unavailable — untimed encoder
            mProfileCurSampleIndex = -1;
            return [cmdBuffer computeCommandEncoder];
        }
    }
#endif
    id<MTLComputeCommandEncoder> result = [getCommandBufferForNet() computeCommandEncoder];
#if MNN_METAL_DEBUG || MNN_METAL_BENCHMARK
    result.label = nil;
#endif
    return result;
}

void *MetalBackend::context() const {
    return mRuntime->context();
}

class MetalMemRelease : public Backend::MemObj {
public:
    MetalMemRelease(MemChunk buffer, BufferAllocator* allocator) {
        mBuffer = buffer;
        mAllocator = allocator;
    }
    virtual ~ MetalMemRelease() {
        mAllocator->free(mBuffer);
    }
    MemChunk chunk() override {
        return mBuffer;
    }
private:
    MemChunk mBuffer;
    BufferAllocator* mAllocator;
};
size_t MetalBackend::getTensorSizeInBytes(const Tensor* tensor) const {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    size_t size;
    if (MNN_DATA_FORMAT_NC4HW4 == format && tensor->dimensions() >= 2) {
        int width = 1;
        int height = 1;
        int batch    = tensor->length(0);
        int channel  = tensor->length(1);
        if (tensor->dimensions() >= 3) {
            height = tensor->length(2);
        }
        for (int i=3; i<tensor->dimensions(); ++i) {
            width *= tensor->length(i);
        }
        int alignC = ROUND_UP(channel, 4);
        int hR = ROUND_UP(height, 4) - height;
        // width parallel 4, may exceed 3 elements
        int wR = ROUND_UP(width + 3, 4) - width;
        int bhw = batch * width * height;
        int bhwR = UP_DIV(bhw, 16) * 16 - bhw;
        int extraPadding = ALIMAX(bhwR, (hR * width + wR));
        size = batch * alignC * width * height;
        size = size + extraPadding * 4;
    } else {
        size = 1;
        for (int i=0; i<tensor->dimensions(); ++i) {
            size *= tensor->length(i);
        }
        size = ROUND_UP(size, 4);
    }
    if (0 == size) {
        return 0;
    }
    // use metal_float when meets float
    if (halide_type_float == tensor->buffer().type.code && tensor->buffer().type.bits == 32 && mUseFloatAsFp16) {
        size *= 2;
    } else {
        size *= tensor->getType().bytes();
    }
    size_t align = 4 * sizeof(int);
    size = ROUND_UP(size, align);
    return size;
}

Backend::MemObj* MetalBackend::onAcquire(const Tensor *_tensor, StorageType storageType) {
    auto tensor  = const_cast<Tensor *>(_tensor);
    size_t size = getTensorSizeInBytes(_tensor);
    if (0 == size) {
        return nullptr;
    }
    // reuse if possible
    MemChunk buffer;
    BufferAllocator* allocator = nullptr;
    switch (storageType) {
        case Backend::STATIC: {
            buffer = mRuntime->mStaticAllocator->alloc(size, false);
            allocator = mRuntime->mStaticAllocator.get();
            if (nullptr == buffer.first && nullptr != mRuntime->mStaticAllocatorRaw.get()) {
                buffer = mRuntime->mStaticAllocatorRaw->alloc(size, false);
                allocator = mRuntime->mStaticAllocatorRaw.get();
            }
        } break;
        case Backend::DYNAMIC: {
            buffer = mCurrentAllocator->alloc(size, false);
            allocator = mCurrentAllocator;
        } break;
        case Backend::DYNAMIC_SEPERATE: {
            buffer = mCurrentAllocator->alloc(size, true);
            allocator = mCurrentAllocator;
        } break;
        case Backend::DYNAMIC_IN_EXECUTION: {
            buffer = mExecutionBufferPool->alloc(size, false);
            allocator = mExecutionBufferPool.get();
        } break;
        default:{
            break;
        }
    }
    if (storageType == Backend::STATIC) {
        if(nullptr == buffer.first) {
            MNN_ERROR("onAcquireBuffer error!\n");
            return nullptr;
        }
    } else {
        buffer.attach(tensor);
    }
    if (nullptr == buffer.first) {
        _MetalApplyTensor((uint8_t*)(&mEmptyMem), 0, (Tensor*)_tensor);
    } else {
        _MetalApplyTensor((uint8_t*)buffer.first, buffer.second, (Tensor*)_tensor);
    }
    return new MetalMemRelease(buffer, allocator);
}

bool MetalBackend::onClearBuffer() {
    mCurrentAllocator->release(true);
    if (mExecutionBufferPool.get() != nullptr) {
        mExecutionBufferPool->release(true);
    }
    if (nullptr != mRuntime->mStaticAllocatorRaw.get()) {
        mRuntime->mStaticAllocator->sync();
        mRuntime->mStaticAllocator = mRuntime->mStaticAllocatorRaw;
        mRuntime->mStaticAllocatorRaw = nullptr;
    }
    return true;
}

Execution *MetalBackend::onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                  const Op *op) {
    auto map  = getCreatorMap();

    auto iter = map->find(op->type());
    if (iter == map->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type [%s], %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type [%s]\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
    //MNN_PRINT("support type [%s]\n", EnumNameOpType(op->type()));

    auto exe = iter->second->onCreate(inputs, op, this, outputs);
    if (NULL == exe) {
        MNN_PRINT("The Creator Don't support type [%s], %s\n", MNN::EnumNameOpType(op->type()), op->name() ? op->name()->c_str() : "");
        return NULL;
    }
#if MNN_METAL_OP_PROFILE
    profileRegisterOp(exe, EnumNameOpType(op->type()));
#endif
    return exe;
}
void MetalBackend::flushEncoder() const {
    if (nil != mComputeEncoder) {
        [mComputeEncoder endEncoding];
        mComputeEncoder = nil;
#if MNN_METAL_OP_PROFILE
        if (mProfileCounterMode && mProfileCurSampleIndex >= 0) {
            std::string name = mCurProfileName.empty() ? std::string("Other") : mCurProfileName;
            mProfilePendingSamples.push_back({mProfileCurSampleIndex, std::move(name)});
            mProfileCurSampleIndex = -1;
        }
#endif
    }
}
#if MNN_METAL_OP_PROFILE
id<MTLCounterSampleBuffer> MetalBackend::profileAcquireSampleBuffer() const {
    {
        std::lock_guard<std::mutex> _l(gProfileSampleBufferPoolMutex);
        if (!gProfileSampleBufferPool.empty()) {
            auto buffer = gProfileSampleBufferPool.back();
            gProfileSampleBufferPool.pop_back();
            return buffer;
        }
    }
    if (@available(iOS 14.0, macOS 11.0, *)) {
        auto ctx = (__bridge MNNMetalContext *)context();
        id<MTLDevice> device = [ctx device];
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestampSet = cs;
                break;
            }
        }
        if (nil == timestampSet) {
            return nil;
        }
        MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = timestampSet;
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = kProfileSampleBufferCapacity;
        NSError* error = nil;
        return [device newCounterSampleBufferWithDescriptor:desc error:&error];
    }
    return nil;
}
void MetalBackend::profileOpEncoded() const {
    if (mProfileCounterMode) {
        // one encoder per op: end it so the op gets its own timestamp pair
        flushEncoder();
    }
}
id<MTLComputeCommandEncoder> MetalBackend::profileNextSubpass(const std::string& subtag) const {
    flushEncoder();
    if (!mProfileCounterMode) {
        // legacy mode times whole command buffers — commit one per sub-pass
        commit_net();
    }
    setProfileSubtag(subtag);
    return encoder_for_net();
}
#endif
void MetalBackend::_resetDynamicMemory() const {
    mRuntime->pCurrentStatus = mCurrentAllocator->apply();
    if (NO_ERROR != mRuntime->pCurrentStatus) {
        return;
    }
    if (nullptr != mBufferPoolShapeImmutable.get()) {
        mRuntime->pCurrentStatus = mBufferPoolShapeImmutable->apply();
    }
}

void MetalBackend::onExecuteBegin() const {
    _resetDynamicMemory();
    mEncoderCount = 0;
}
void MetalBackend::onExecuteEnd() const {
    flushEncoder();
    commit_net();
}
    
BufferAllocator* MetalBackend::getBufferPool() const {
    return mCurrentAllocator;
}

bool MetalBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2) {
        return false;
    }
    if (maxIndex == 2 && mBufferPoolShapeImmutable.get() == nullptr) {
        mBufferPoolShapeImmutable.reset(mRuntime->createDynamicAllocator(1, true));
        mBufferPool.reset(mRuntime->createDynamicAllocator(0, true));
    }
    if (1 == index) {
        mCurrentAllocator = mBufferPoolShapeImmutable.get();
    } else {
        mCurrentAllocator = mBufferPool.get();
    }
    return true;
}

bool MetalBackend::onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
    if (nullptr == dstInfo) {
        return true;
    }
    auto dst = (MNNMetalTensorContent*)dstInfo;
    dst->type.code = halide_type_float;
    if (mUseFloatAsFp16) {
        dst->type.bits = 16;
    } else {
        dst->type.bits = 32;
    }
    MNNMetalGetTensorContent(dst, (void*)tensor);
    return true;
}

bool MetalBackend::isCmdBufferCommit() {
#if MNN_METAL_OP_PROFILE
    // Legacy profiling: commit one command buffer per op so that each command
    // buffer's GPUEndTime-GPUStartTime measures a single op's GPU time.
    // Counter mode times per-encoder via MTLCounterSampleBuffer and keeps the
    // normal commit cadence (accurate absolute numbers, low overhead).
    if (!mProfileCounterMode) {
        return true;
    }
#endif
    auto ctx = (__bridge MNNMetalContext *)context();
    
    //TODO: set magic number
    // Experiment: MNN_METAL_COMMIT_NUM overrides ops-per-commit cadence
    const int sEnvCommitNum = MetalEnv::get().commitNum;
    const int magicNum = sEnvCommitNum > 0 ? sEnvCommitNum : mRuntime->hint().encorderNumForCommit;
    mEncoderCount++;
    if(mEncoderCount != 0 && mEncoderCount % magicNum == 0) {
        return true;
    }
    return false;
}

#if MNN_METAL_OP_PROFILE
void MetalBackend::profileRegisterOp(const Execution* exe, const std::string& name) const {
    gMetalOpProfiler.registerName(exe, name);
}
void MetalBackend::profileMarkOp(const Execution* exe) const {
    mCurProfileName = gMetalOpProfiler.lookupName(exe);
}
void MetalBackend::setProfileSubtag(const std::string& subtag) const {
    if (subtag.empty()) return;
    // Preserve OpType prefix (before first '/') and rewrite everything after.
    // This lets an op split its work into sub-passes with independent profile tags
    // (e.g. outer-dequant weight dequant vs gemm) by calling this before each commit.
    auto pos = mCurProfileName.find('/');
    std::string base = (pos == std::string::npos) ? mCurProfileName : mCurProfileName.substr(0, pos);
    mCurProfileName = base.empty() ? subtag : (base + "/" + subtag);
}
#endif

id<MTLBuffer> MetalBackend::getHostBuffer(size_t size) const {
    size = UP_DIV(size, METAL_CONST_BUFFER_LIMIT) * METAL_CONST_BUFFER_LIMIT;
    // reuse
    if (nullptr != mHostBuffer && mHostBuffer.length >= size) {
        return mHostBuffer;
    }

    // create larger
    auto context = (__bridge MNNMetalContext *)this->context();
    mHostBuffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return mHostBuffer;
}

id<MTLBuffer> MetalBackend::acquireUploadStaging(size_t size) const {
    size = UP_DIV(size, METAL_CONST_BUFFER_LIMIT) * METAL_CONST_BUFFER_LIMIT;
    for (auto& slot : mUploadStagingRing) {
        bool free = (nil == slot.lastUse) || (slot.lastUse.status >= MTLCommandBufferStatusCompleted);
        if (free && slot.buffer.length >= size) {
            slot.lastUse = nil;
            return slot.buffer;
        }
    }
    auto context = (__bridge MNNMetalContext *)this->context();
    UploadStagingSlot slot;
    slot.buffer = [context newDeviceBuffer:size access:CPUReadWrite];
    mUploadStagingRing.push_back(slot);
    return slot.buffer;
}
void MetalBackend::markUploadStagingUse(id<MTLBuffer> staging, id<MTLCommandBuffer> cmd) const {
    for (auto& slot : mUploadStagingRing) {
        if (slot.buffer == staging) {
            slot.lastUse = cmd;
            return;
        }
    }
}

id<MTLBuffer> MetalBackend::getConstBuffer(size_t size) const {
    if (size < METAL_CONST_BUFFER_LIMIT) {
        if (!mHoldBuffers.empty()) {
            auto res = mHoldBuffers.front();
            mHoldBuffers.pop();
            return res;
        }
        size = METAL_CONST_BUFFER_LIMIT;
    }
    auto context = (__bridge MNNMetalContext *)this->context();
    auto buffer  = [context newDeviceBuffer:size access:CPUReadWrite];
    return buffer;
}
void MetalBackend::returnConstBuffer(id<MTLBuffer> buffer) const {
    mHoldBuffers.push(buffer);
}
static inline void _getNCPlane(const Tensor* tensor, int& s, int& c, int& b) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    s = 1, c = 1, b = 1;
    b = tensor->length(0);
    if (format == MNN_DATA_FORMAT_NHWC) {
        c = tensor->length(tensor->dimensions()-1);
        for (int i=1; i<tensor->dimensions()-1; ++i) {
            s *= tensor->length(i);
        }
    } else {
        c = tensor->length(1);
        for (int i=2; i<tensor->dimensions(); ++i) {
            s *= tensor->length(i);
        }
    }
}
MTLSize getTensorShape(id<MTLBuffer> shape, const Tensor *tensor) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    int s, b, c;
    _getNCPlane(tensor, s, c, b);
    int z = UP_DIV(c, 4);

    // shape
    ((int *)shape.contents)[0] = b;
    ((int *)shape.contents)[1] = c;
    ((int *)shape.contents)[2] = s;
    ((int *)shape.contents)[3] = 1;
    
    // stride
    if (format == MNN_DATA_FORMAT_NHWC) {
        ((int *)shape.contents)[4] = s * c;
        ((int *)shape.contents)[5] = 1;
        ((int *)shape.contents)[6] = c;
        ((int *)shape.contents)[7] = 1;
    } else {
        ((int *)shape.contents)[4] = s * c;
        ((int *)shape.contents)[5] = s;
        ((int *)shape.contents)[6] = 1;
        ((int *)shape.contents)[7] = 1;
    }
    // threads
    MTLSize threads = {(NSUInteger)s * b * z, 1, 1};
    return threads;
}
static const char* gTranspose = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct tensor_shape {
    uint4 size; // n, c, plane, 1
    uint4 stride;
};
kernel void main0(const device IType* in [[buffer(0)]], device OType* out [[buffer(1)]], constant tensor_shape &uConstant [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    int channel = uConstant.size.y;
    if (gid < channel * uConstant.size.x * uConstant.size.z) {
        int tmp = gid % (channel * uConstant.size.x);
        int x = gid / (channel * uConstant.size.x);
        int b = tmp / channel;
        int c = tmp % channel;
        int outPos = b * uConstant.size.y * uConstant.size.z + c * uConstant.size.z + x;
        int inPos = b * uConstant.size.y * uConstant.size.z + c + x * uConstant.size.y;
        out[outPos] = (OType)(in[inPos]);
    }
})metal";

static const char* gNC4HW4Convert = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct tensor_shape {
    uint4 size; // n, c, plane, 1
    uint4 stride;
};
kernel void main0(const device IType* in [[buffer(0)]], device OType* out [[buffer(1)]], constant tensor_shape &uConstant [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    int channelC4 = (uConstant.size.y + 3) / 4;
    if (gid < channelC4 * uConstant.size.x * uConstant.size.z)
    {
        int3 pos;
        pos.z = gid % (channelC4 * uConstant.size.x);
        pos.y = gid / (channelC4 * uConstant.size.x);
        pos.x = 0;
        int batchIndex = pos.z / channelC4;
        int zDiv4 = pos.z % channelC4;

        int lastZ = uConstant.size.y / 4;
        int cIndex = uConstant.size.y % 4;

        int z = zDiv4*4;
        int basicOffset = 0
            + batchIndex*uConstant.stride.x
            + z * uConstant.stride.y
            + pos.y * uConstant.stride.z
            ;
#ifdef MNN_OUTPUT_C4
        OType color = OType(0);
        if(zDiv4 == lastZ)
        {
            if(cIndex == 1)
            {
                color.r = in[basicOffset+0];
                color.g = 0.0;
                color.b = 0.0;
                color.a = 0.0;
            }
            else if(cIndex == 2)
            {
                color.r = in[basicOffset+0];
                color.g = in[basicOffset+1*uConstant.stride.y];
                color.b = 0.0;
                color.a = 0.0;
            }
            else
            {
                color.r = in[basicOffset+0];
                color.g = in[basicOffset+1*uConstant.stride.y];
                color.b = in[basicOffset+2*uConstant.stride.y];
                color.a = 0.0;
            }
        }
        else
        {
            color.r = in[basicOffset+0];
            color.g = in[basicOffset+1*uConstant.stride.y];
            color.b = in[basicOffset+2*uConstant.stride.y];
            color.a = in[basicOffset+3*uConstant.stride.y];
        }

        out[0
            + pos.y
            + uConstant.size.x * uConstant.size.z*zDiv4
            + batchIndex*uConstant.size.z
            ] = color;
#else
        IType color = in[0
            + pos.y
            + uConstant.size.x * uConstant.size.z*zDiv4
            + batchIndex*uConstant.size.z
            ];
        if(zDiv4 == lastZ)
        {
            if(cIndex == 1)
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
            }
            else if(cIndex == 2)
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
                out[basicOffset+1*uConstant.stride.y] = color.g;
            }
            else
            {
                out[basicOffset+0*uConstant.stride.y] = color.r;
                out[basicOffset+1*uConstant.stride.y] = color.g;
                out[basicOffset+2*uConstant.stride.y] = color.b;
            }
        }
        else
        {
            out[basicOffset+0*uConstant.stride.y] = color.r;
            out[basicOffset+1*uConstant.stride.y] = color.g;
            out[basicOffset+2*uConstant.stride.y] = color.b;
            out[basicOffset+3*uConstant.stride.y] = color.a;
        }
#endif
    }
}
)metal";

static const char* gCopy = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
kernel void main0(const device IType *in [[buffer(0)]], device OType *out [[buffer(1)]], constant uint4& limit [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < limit.x) {
        out[int(gid)] = (OType)in[int(gid)];
    }
})metal";

void MetalBackend::onResizeBegin() {    
    // Abort last inference task if needed
    flushEncoder();
    _commandBuffer = nil;
    // Per-backend fence (default): before resetting OUR allocator we only need
    // OUR OWN in-flight GPU work to finish. The legacy wait() drained the
    // runtime's last commit — which for LLM decode belongs to another module's
    // backend (the per-token logits-slice submodule resize was draining the
    // whole main graph, serializing CPU resize against GPU and costing ~14%
    // decode on Qwen3-0.6B/M4 Pro). Note the legacy wait was not a true
    // cross-queue drain either: backends own separate queues and _waiting only
    // tracks the latest commit.
    //   MNN_METAL_RESIZE_WAIT=global  -> legacy behavior (rollback/A-B)
    //   MNN_METAL_RESIZE_WAIT=none    -> skip both fences (experiment only)
    const int sResizeWaitMode = MetalEnv::get().resizeWaitMode;
    if (sResizeWaitMode == 1) {
        wait(0);
    } else if (sResizeWaitMode == 0) {
        waitOwnInflight();
    }
    mCurrentAllocator->reset();
    // Clear Gate/Up fusion mappings from previous resize
    clearConv1x1Map();
}

ErrorCode MetalBackend::onResizeEnd() {
    auto ctx = (__bridge MNNMetalContext *)context();
    auto err = mCurrentAllocator->compute();
    if (err != NO_ERROR) {
        return err;
    }
    // Export-time fused projections wire up their own leader/follower dispatch
    // from the exported member order. Must run after compute(): the setup
    // re-homes follower outputs to STATIC, which only sticks once the dynamic
    // allocator has assigned addresses.
    for (auto* host : mFusedProjs) {
        host->setupFusion();
    }
    return applyLinearAttnGateFolds();
}

ErrorCode MetalBackend::applyLinearAttnGateFolds() {
    // The gate/beta fold is declared by the exported LinearAttentionParam
    // (gate_fold): rawA/rawB are the op's own inputs and the per-head constants
    // come from the param, so there is nothing to match here. All that remains
    // is the STATIC re-home, and it must happen here rather than in the op's
    // onResize: the pipeline's resize sweep nulls a consumer's input memory when
    // its useCount exhausts (Pipeline.cpp _releaseTensor), which would free a
    // STATIC home acquired earlier in the sweep before encode runs.
    //
    // Failure here is fatal rather than a fallback: the exporter already removed
    // the gate chain from the graph, so an unfolded dispatch would consume the
    // raw `a` projection as the decay gate.
    // Re-home on EVERY resize, like MetalFusedProj::setupFusion does. The resize
    // sweep drops the previous home unconditionally: _releaseTensor nulls `mem`
    // whatever its storage type was (Pipeline.cpp), and _allocTensor then
    // re-acquires DYNAMIC. Latching on a "already folded" flag would leave the
    // raw a/b inputs back in the reusable pool from the second resize onwards,
    // while the fold stays active -- and LLM decode re-resizes every token.
    for (auto* req : mLinearAttnFolds) {
        if (!req->exportFold) {
            continue;
        }
        if (req->rawA == nullptr || req->rawB == nullptr || req->numHeads <= 0) {
            MNN_ERROR("MetalBackend: incomplete LinearAttention gate fold request\n");
            return NOT_SUPPORT;
        }
        if (!onAcquireBuffer(req->rawA, Backend::STATIC)) {
            MNN_ERROR("MetalBackend: cannot re-home LinearAttention gate fold input\n");
            return OUT_OF_MEMORY;
        }
        if (!onAcquireBuffer(req->rawB, Backend::STATIC)) {
            onReleaseBuffer(req->rawA, Backend::STATIC);
            MNN_ERROR("MetalBackend: cannot re-home LinearAttention beta fold input\n");
            return OUT_OF_MEMORY;
        }
        req->gateFolded = true;
        req->betaFolded = true;
    }
    return NO_ERROR;
}

// Byte span of a tensor inside its backing MTLBuffer. The span must be the
// ALLOCATED size, not elementSize() * type.bytes(): getTensorSizeInBytes pads
// NC4HW4 channels to 4 and adds extraPadding, and halves float32 under fp16
// mode. Under-reporting here silently turns a real alias into "no overlap".
static bool _tensorSpan(const MetalBackend* backend, const Tensor* t, void*& buf, size_t& begin, size_t& end) {
    if (t == nullptr || t->deviceId() == 0) return false;
    auto alloc = (MetalRuntimeAllocator::MetalBufferAlloc *)t->deviceId();
    begin = (size_t)TensorUtils::getDescribeOrigin(t)->offset;
    end = begin + backend->getTensorSizeInBytes(t);
    buf = (__bridge void*)alloc->getBuffer();
    return true;
}
// True when two tensors share backing memory. Used by fusions that must not let
// one kernel write a buffer another part of the same dispatch still reads.
bool MetalBackend::tensorsOverlap(const Tensor* a, const Tensor* b) const {
    void* ba; void* bb; size_t a0, a1, b0, b1;
    if (!_tensorSpan(this, a, ba, a0, a1) || !_tensorSpan(this, b, bb, b0, b1)) return false;
    return ba == bb && a0 < b1 && b0 < a1;
}

static std::string _getType(const halide_type_t& type, MNN_DATA_FORMAT format, bool useFp16AsFp32) {
    std::string res;
    if (type.code == halide_type_float) {
        if (useFp16AsFp32) {
            res = "half";
        } else {
            res = "float";
        }
    } else {
        switch (type.bytes()) {
            case 1:
                res = "char";
                break;
            case 2:
                res = "short";
                break;
            case 4:
                res = "int";
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    }
    if (format == MNN_DATA_FORMAT_NC4HW4) {
        return res + "4";
    }
    return res;
}
MetalBackend::CopyPipeline MetalBackend::_makeCopyInfo(const Tensor *src, const Tensor *dst, id<MTLBuffer> shape, int castType) const {
    auto ctx = (__bridge MNNMetalContext *)context();
    MetalBackend::CopyPipeline res;
    auto sfmt = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt = TensorUtils::getDescribe(dst)->dimensionFormat;
    if (shape == nil) {
        shape = getConstBuffer(8 * sizeof(int));
    }
    res.shape = shape;
    if (sfmt == dfmt || src->dimensions() <= 1) {
        auto srcType = _getType(src->getType(), MNN_DATA_FORMAT_NC4HW4, mUseFloatAsFp16 && castType != 1);
        auto dstType = _getType(dst->getType(), MNN_DATA_FORMAT_NC4HW4, mUseFloatAsFp16 && castType != 2);
        auto size      = dst->elementSize();
        size = UP_DIV(size, 4);
        std::vector<std::string> keys = {
            "copyC4",
            srcType,
            dstType
        };
        ((uint32_t*)[shape contents])[0] = size;
        id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
            [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
            option.preprocessorMacros = dic;
            pipeline = makeComputePipelineWithSourceOption(gCopy, "main0", option);
            mRuntime->insertPipeline(keys, pipeline);
        }
        res.groupSize = MTLSizeMake(UP_DIV(size, 256), 1, 1);
        res.localSize = MTLSizeMake(256, 1, 1);
        res.pipeline = pipeline;
        return res;
    }
    auto srcType = _getType(src->getType(), sfmt, mUseFloatAsFp16 && castType != 1);
    auto dstType = _getType(dst->getType(), dfmt, mUseFloatAsFp16 && castType != 2);
    if (sfmt == MNN_DATA_FORMAT_NC4HW4 || dfmt == MNN_DATA_FORMAT_NC4HW4) {
        auto normalTensor = dst;
        if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
            normalTensor = src;
        }
        // convert C4 / NCHW
        std::vector<std::string> keys = {
            "c4convert",
            srcType,
            dstType
        };
        if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
            keys.emplace_back("outputc4");
        }
        id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
            [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
            if (dfmt == MNN_DATA_FORMAT_NC4HW4) {
                [dic setValue:@"1" forKey:@"MNN_OUTPUT_C4"];
            }
            option.preprocessorMacros = dic;
            pipeline = makeComputePipelineWithSourceOption(gNC4HW4Convert, "main0", option);
            mRuntime->insertPipeline(keys, pipeline);
        }
        res.pipeline = pipeline;
        auto size = getTensorShape(shape, normalTensor);
        auto gl = [ctx computeBestGroupAndLocal:pipeline threads:size];
        res.groupSize = gl.first;
        res.localSize = gl.second;
        return res;
    }
    // NCHW <-> NHWC
    std::vector<std::string> keys = {
        "transpose",
        srcType,
        dstType
    };
    id<MTLComputePipelineState> pipeline = mRuntime->findPipeline(keys);
    if (nil == pipeline) {
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(keys[1].c_str()) forKey:@"IType"];
        [dic setValue:@(keys[2].c_str()) forKey:@"OType"];
        option.preprocessorMacros = dic;
        pipeline = makeComputePipelineWithSourceOption(gTranspose, "main0", option);
        mRuntime->insertPipeline(keys, pipeline);
    }
    res.pipeline = pipeline;
    int n, c, plane;
    _getNCPlane(dst, plane, c, n);
    auto shapePtr = (uint32_t*)shape.contents;
    shapePtr[0] = n;
    shapePtr[3] = 1;
    if (MNN_DATA_FORMAT_NHWC == dfmt) {
        shapePtr[1] = plane;
        shapePtr[2] = c;
    } else {
        shapePtr[1] = c;
        shapePtr[2] = plane;
    }
    auto size = plane * n * c;
    res.localSize = MTLSizeMake(256, 1, 1);
    res.groupSize = MTLSizeMake(UP_DIV(size, 256), 1, 1);
    return res;
}

static void _execute(id<MTLComputeCommandEncoder> encoder, const MetalBackend::CopyPipeline& info, std::pair<id<MTLBuffer>, int> src, std::pair<id<MTLBuffer>, int> dst) {
    [encoder setComputePipelineState:info.pipeline];
    [encoder setBuffer:src.first offset:src.second atIndex:0];
    [encoder setBuffer:dst.first offset:dst.second atIndex:1];
    [encoder setBuffer:info.shape offset:0 atIndex:2];
    [encoder dispatchThreadgroups:info.groupSize threadsPerThreadgroup:info.localSize];
}
void MetalBackend::onCopyDeviceToDevice(const Tensor *src, const Tensor *dst,
                                        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape, int castType) const {
    auto ctx    = (__bridge MNNMetalContext *)context();
    auto info = _makeCopyInfo(src, dst, shape, castType);
    auto standalone = encoder == nil;
    encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
    _execute(encoder, info, MetalBackend::getBuffer(src), MetalBackend::getBuffer(dst));
    if (standalone) {
        [encoder endEncoding];
        MNN_PRINT_ENCODER(ctx, encoder);
    }
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst) const {
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    commit_net();
    
    _resetDynamicMemory();
    onCopyBuffer(src, dst, nil, nil);
}

id<MTLComputeCommandEncoder> MetalBackend::encoder_for_net() const {
    if (nil == mComputeEncoder) {
        mComputeEncoder = encoder_net();//TO DO :: use which cmdBuffer
    }
    return mComputeEncoder;
}

void MetalBackend::onCopyBuffer(const Tensor *src, const Tensor *dst, id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> shape) const {
    MNN_ASSERT(src->buffer().dimensions == dst->buffer().dimensions);
    
    if (!src->buffer().host && !dst->buffer().host) {
        onCopyDeviceToDevice(src, dst, encoder, shape);
        return;
    }
    auto sfmt = TensorUtils::getDescribe(src)->dimensionFormat;
    auto dfmt = TensorUtils::getDescribe(dst)->dimensionFormat;
    bool formatDiff = sfmt != dfmt && src->dimensions() > 1;
    auto floats  = src->getType().code == halide_type_float;
    bool dataTypeDiff = floats && mUseFloatAsFp16;
    bool needConvert = formatDiff || dataTypeDiff;

    if (!src->buffer().host && dst->buffer().host) {
        auto device = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)src->deviceId())->getBuffer();
        auto devicePtr = (uint8_t*)device.contents + TensorUtils::getDescribeOrigin(src)->offset;
        if (needConvert) {
            auto tDst = const_cast<Tensor*>(dst);
            auto tmpBuffer = getHostBuffer(dst->usize());
            auto info = _makeCopyInfo(src, dst, shape, 2);
            auto standalone = encoder == nil;
            encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
            _execute(encoder, info, MetalBackend::getBuffer(src), std::make_pair(tmpBuffer, 0));
            if (standalone) {
                [encoder endEncoding];
            }
#if MNN_METAL_OP_PROFILE
            mCurProfileName = "ConvertCopy";
#endif
            commit();
            devicePtr = (uint8_t*)tmpBuffer.contents;
        }
        wait(1);
        ::memcpy(dst->host<void>(), devicePtr, dst->usize());
        return;
    }
    if (src->buffer().host && !dst->buffer().host) {
        // Queued upload (default): stage the host bytes into a ring slot and
        // encode the staging->dst copy on the command queue. Queue order makes
        // it safe against in-flight GPU readers of dst from the previous
        // forward, so the full pre-upload drain (wait) is no longer needed —
        // that drain serialized CPU against GPU once per decode token.
        // MNN_METAL_H2D_QUEUED=0 restores the legacy drain+direct-write path.
        const bool sH2DQueued = MetalEnv::get().h2dQueued;
        auto srcSize = src->usize();
        if (sH2DQueued && encoder == nil) {
            flushEncoder();
            auto staging = acquireUploadStaging(srcSize);
            ::memcpy(staging.contents, src->host<void>(), srcSize);
            auto cmd = getCommandBufferForBufferCopy();
            if (needConvert) {
                auto info = _makeCopyInfo(src, dst, shape, 1);
                auto convertEncoder = [cmd computeCommandEncoder];
                _execute(convertEncoder, info, std::make_pair(staging, 0), MetalBackend::getBuffer(dst));
                [convertEncoder endEncoding];
            } else {
                auto dstBuffer = MetalBackend::getBuffer(dst);
                auto blit = [cmd blitCommandEncoder];
                [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dstBuffer.first destinationOffset:dstBuffer.second size:srcSize];
                [blit endEncoding];
            }
            markUploadStagingUse(staging, cmd);
#if MNN_METAL_OP_PROFILE
            mCurProfileName = "ConvertCopy";
#endif
            commit();
            return;
        }
        // For command queue from user, need user to make sure last frame's gpu work is ready
        bool needWait = !mRuntime->userSync();
        if (needWait) {
            wait(2);
        }
        if (needConvert) {
            auto tmpBuffer = getHostBuffer(srcSize);
            ::memcpy(tmpBuffer.contents, src->host<void>(), srcSize);
            auto info = _makeCopyInfo(src, dst, shape, 1);
            auto standalone = encoder == nil;
            encoder = encoder ?: [getCommandBufferForBufferCopy() computeCommandEncoder];
            _execute(encoder, info, std::make_pair(tmpBuffer, 0), MetalBackend::getBuffer(dst));
            if (standalone) {
                [encoder endEncoding];
            }
#if MNN_METAL_OP_PROFILE
            mCurProfileName = "ConvertCopy";
#endif
            commit();
        } else {
            auto device = (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)dst->deviceId())->getBuffer();
            auto devicePtr = (uint8_t*)device.contents + TensorUtils::getDescribeOrigin(dst)->offset;
            ::memcpy(devicePtr, src->host<void>(), srcSize);
        }
        return;
    }
    MNN_ASSERT(false); // should not be handled here
}
int MetalBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    if (mRuntime->pExecutionStatus == NO_EXECUTION) {
#ifdef CHECK_IOS_UI_STATUS
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
        if (!mGPUEnabledSwitch) {
            return NO_EXECUTION;
        }
        mRuntime->pExecutionStatus = NO_ERROR;
#else
        return NO_EXECUTION;
#endif
#else
        return NO_EXECUTION;
#endif
    }
    flushEncoder();
    auto ctx = (__bridge MNNMetalContext *)context();
    commit_net();
    
    if (toCpu) {
        wait(3);
    }
    return 0;
}
id<MTLCommandBuffer> MetalBackend::getCommandBufferForBufferCopy() const {
    if (nil == _commandBuffer) {
        _commandBuffer = [_commandQueue commandBuffer];
    }
    return _commandBuffer;
}
id<MTLCommandBuffer> MetalBackend::getCommandBufferForNet() const {
    return getCommandBufferForBufferCopy();
}

void MetalBackend::setTensor(const MNN::Tensor* tensor, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:((MetalRuntimeAllocator::MetalBufferAlloc *)tensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(tensor)->offset atIndex:index];
    // Encode-replay annotation: while a recording proxy is active, tag the
    // binding just recorded with its source tensor for replay-time validation.
    if (nil != gMetalReplayProxy) {
        [gMetalReplayProxy annotateTensor:tensor atIndex:index];
    }
}
void MetalBackend::setMem(const MemChunk& chunk, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:((MetalRuntimeAllocator::MetalBufferAlloc *)chunk.first)->getBuffer() offset:chunk.second atIndex:index];
}
uint8_t* MetalBackend::getMemPtr(const MemChunk& chunk) {
    return (uint8_t*)((MetalRuntimeAllocator::MetalBufferAlloc *)chunk.first)->getBuffer().contents + chunk.second;
}
void MetalBackend::setBuffer(id<MTLBuffer> buffer, int offset, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:buffer offset:offset atIndex:index];
}
std::pair<id<MTLBuffer>, int> MetalBackend::getBuffer(const MNN::Tensor* tensor) {
    return std::make_pair(((MetalRuntimeAllocator::MetalBufferAlloc *)tensor->deviceId())->getBuffer(), TensorUtils::getDescribeOrigin(tensor)->offset);
}


void MetalBackend::commit() const {
#ifdef CHECK_IOS_UI_STATUS
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
    if (!mGPUEnabledSwitch) {
        mRuntime->pExecutionStatus = NO_EXECUTION;
        _commandBuffer = nil;
        return;
    }
#endif
#endif
    mRuntime->pExecutionStatus = NO_ERROR;
    if (nil != _commandBuffer &&  _commandBuffer.status < MTLCommandBufferStatusCommitted) {
#if MNN_METAL_OP_PROFILE
        if (mProfileCounterMode) {
            if (@available(iOS 14.0, macOS 11.0, *)) {
                // collect current + sealed sample buffers for this command buffer
                auto groups = std::make_shared<std::vector<ProfileSealedBuffer>>();
                for (auto& sealed : mProfileSealedBuffers) {
                    if (!sealed.samples.empty()) {
                        groups->push_back(std::move(sealed));
                    }
                }
                mProfileSealedBuffers.clear();
                if (!mProfilePendingSamples.empty() && nil != mProfileSampleBuffer) {
                    groups->push_back({mProfileSampleBuffer, mProfileSampleCursor, std::move(mProfilePendingSamples)});
                    mProfilePendingSamples.clear();
                }
                if (!groups->empty()) {
                    auto ctx = (__bridge MNNMetalContext *)context();
                    id<MTLDevice> device = [ctx device];
                    [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        double nsPerTick = gMetalGpuTickScale.nsPerTick(device);
                        for (const auto& group : *groups) {
                            NSData* data = [group.buffer resolveCounterRange:NSMakeRange(0, (NSUInteger)group.usedCount)];
                            if (nil != data && data.length >= group.usedCount * sizeof(MTLCounterResultTimestamp)) {
                                const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)data.bytes;
                                for (const auto& p : group.samples) {
                                    uint64_t t0 = ts[p.index].timestamp;
                                    uint64_t t1 = ts[p.index + 1].timestamp;
                                    if (t0 != MTLCounterErrorValue && t1 != MTLCounterErrorValue && t1 > t0) {
                                        gMetalOpProfiler.add(p.name, double(t1 - t0) * nsPerTick / 1.0e6);
                                        // Preserve absolute timestamps (tick-scaled to ns) for the
                                        // timeline dump — the aggregate table above loses ordering.
                                        gMetalOpProfiler.addSample(p.name,
                                                                   double(t0) * nsPerTick,
                                                                   double(t1) * nsPerTick);
                                    }
                                }
                            }
                            std::lock_guard<std::mutex> _l(gProfileSampleBufferPoolMutex);
                            gProfileSampleBufferPool.push_back(group.buffer);
                        }
                    }];
                } else {
                    // no per-encoder samples (copy/sync buffers) — whole-buffer attribution
                    std::string profName = mCurProfileName.empty() ? std::string("CopyBuffer/Sync") : mCurProfileName;
                    [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                        double ms = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
                        gMetalOpProfiler.add(profName, ms);
                    }];
                }
            }
            mProfileSampleBuffer = nil;
            mProfileSampleCursor = 0;
            mProfileCurSampleIndex = -1;
            mCurProfileName.clear();
        } else {
            std::string profName = mCurProfileName.empty() ? std::string("CopyBuffer/Sync") : mCurProfileName;
            [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                if (@available(iOS 10.3, macOS 10.15, *)) {
                    double ms = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
                    gMetalOpProfiler.add(profName, ms);
                }
            }];
            mCurProfileName.clear();
        }
#endif
#ifdef MNN_SESSION_CPU_TRACE
        [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            if (@available(iOS 10.3, macOS 10.15, *)) {
                uint64_t busyNs = (uint64_t)((buffer.GPUEndTime - buffer.GPUStartTime) * 1e9);
                metalCpuTrace().gpuBusyNs += busyNs;
                metalCpuTrace().gpuBuffers += 1;
                double prevEnd = metalCpuTrace().gpuPrevEnd.exchange(buffer.GPUEndTime);
                if (prevEnd > 0.0 && buffer.GPUStartTime > prevEnd) {
                    metalCpuTrace().gpuGapNs += (uint64_t)((buffer.GPUStartTime - prevEnd) * 1e9);
                }
            }
        }];
#endif
        [_commandBuffer commit];
        mRuntime->_waiting = _commandBuffer;
        mLastOwnCommandBuffer = _commandBuffer;
        _commandBuffer = nil;
    }
}

void MetalBackend::waitOwnInflight() const {
    if (nil != mLastOwnCommandBuffer) {
        if (mLastOwnCommandBuffer.status < MTLCommandBufferStatusCompleted) {
            [mLastOwnCommandBuffer waitUntilCompleted];
        }
        mLastOwnCommandBuffer = nil;
    }
}

void MetalBackend::commit_net() const {
    commit();
}

void MetalBackend::wait(int traceSite) const {
    if (nil != mRuntime->_waiting) {
        auto buffer = mRuntime->_waiting;
        if (buffer.status >= MTLCommandBufferStatusCompleted) {
            if (buffer.error) {
                MNN_ERROR("[METAL] command buffer error: %s\n", buffer.error.localizedDescription.UTF8String);
            }
            mRuntime->_waiting = nil;
            return;
        }
#ifdef MNN_SESSION_CPU_TRACE
        {
            auto t0 = std::chrono::steady_clock::now();
            [buffer waitUntilCompleted];
            auto t1 = std::chrono::steady_clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            metalCpuTrace().waitNs += ns;
            metalCpuTrace().waitCalls += 1;
            if (traceSite >= 0 && traceSite < 4) {
                metalCpuTrace().waitSiteNs[traceSite] += ns;
                metalCpuTrace().waitSiteCalls[traceSite] += 1;
            }
        }
#endif

#if MNN_METAL_BENCHMARK
        NSTimeInterval begin = [NSDate timeIntervalSinceReferenceDate];
        [buffer waitUntilCompleted];
        NSTimeInterval end = [NSDate timeIntervalSinceReferenceDate];
        if (@available(iOS 10.3, *)) {
            printf("[METAL] commit costs: %.3fms\t(kernel: %.3fms, GPU: %.3fms)\n", (end - begin) * 1000.f,
                   (buffer.kernelEndTime - buffer.kernelStartTime) * 1000.f,
                   (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f);
        } else {
            printf("[METAL] commit costs: %.3fms\n", (end - begin) * 1000.f);
        }
#else
        [buffer waitUntilCompleted];
#endif

        if (buffer.error) {
            MNN_ERROR("[METAL] command buffer error: %s\n", buffer.error.localizedDescription.UTF8String);
        }
    }
    mRuntime->_waiting = nil;
}

id<MTLComputePipelineState> MetalBackend::makeComputePipelineWithSourceOption(const char* csource, const char* cname, MTLCompileOptions *options) const{
    auto ctx = (__bridge MNNMetalContext *)context();
    auto source = [[NSString alloc] initWithUTF8String:csource];
    auto name = [[NSString alloc] initWithUTF8String:cname];
    auto pipeline = [ctx pipelineWithSourceOption:source name:name options:options];
    if (nil == pipeline) {
        mRuntime->pCurrentStatus = NOT_SUPPORT;
        MNN_ERROR("pipelineWithSourceOption error.\n");
    }
    return pipeline;
}
void MetalRuntime::setCommandQueue(id<MTLCommandQueue> queue, bool userSync) {
    mQueue = queue;
    mUserSync = userSync;
}
id<MTLComputePipelineState> MetalRuntime::findPipeline(const std::vector<std::string>& keys) const {
    auto iter = mCachePipeine.find(keys);
    if (iter == mCachePipeine.end()) {
        return nil;
    }
    return iter->second;
}
void MetalRuntime::insertPipeline(const std::vector<std::string>& keys, id<MTLComputePipelineState> pipeline) const {
    if (nil != pipeline) {
        mCachePipeine.insert(std::make_pair(keys, pipeline));
    } else {
        mFailedPipeline.insert(keys);
    }
}
bool MetalRuntime::pipelineCompileFailed(const std::vector<std::string>& keys) const {
    return mFailedPipeline.find(keys) != mFailedPipeline.end();
}

void MetalRuntime::setGpuMode(const int mode_num) {
    int totalSet = 0;
    bool isSet = (mode_num & MNN_GPU_MEMORY_BUFFER);
    if(isSet) {
        totalSet++;
    }
    isSet = (mode_num & MNN_GPU_MEMORY_IMAGE);
    if(isSet) {
        totalSet++;
    }
    if(totalSet > 0) {
        MNN_PRINT("warning: set BUFFER and IMAGE mode is not useful for metal, it doesn't matter, cl_mode:%x！\n", mode_num);
    }
    
    totalSet = 0;
    isSet = (mode_num & MNN_GPU_TUNING_NONE);
    if(isSet) {
        mTuneLevel = Never;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_FAST);
    if(isSet) {
        mTuneLevel = Fast;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_NORMAL);
    if(isSet) {
        mTuneLevel = Normal;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_HEAVY);
    if(isSet) {
        mTuneLevel = Heavy;
        totalSet++;
    }
    
    isSet = (mode_num & MNN_GPU_TUNING_WIDE);
    if(isSet) {
        mTuneLevel = Wide;
        totalSet++;
    }

    if(totalSet != 1) {
        MNN_PRINT("set multi tuning mode is not permitted, please check cl_mode:%x！\n", mode_num);
    }
}

struct MetalContext {
    std::mutex pLock;
    MNNMetalContext* pContext;
    id<MTLDevice> pDevice;
};
static MetalContext* gContext = nullptr;
MetalRuntime* MetalRuntime::create(const Backend::Info& info) {
    std::unique_lock<std::mutex> _l(gContext->pLock);
    MNNMetalSharedContext sharedContext;
    sharedContext.device = nil;
    sharedContext.queue = nil;
    if (info.user != nullptr) {
        if (info.user->sharedContext != nullptr) {
            sharedContext.device = ((MNNMetalSharedContext*)info.user->sharedContext)->device;
            sharedContext.queue = ((MNNMetalSharedContext*)info.user->sharedContext)->queue;
        }
    }
    if (nil == sharedContext.device) {
        sharedContext.device = MTLCreateSystemDefaultDevice();
    }
    if (nil == gContext->pContext || gContext->pDevice != sharedContext.device) {
        gContext->pContext = [[MNNMetalContext alloc] init];
        gContext->pDevice = sharedContext.device;
        BOOL res = [gContext->pContext initWithSharedContext:&sharedContext dev:sharedContext.device];
        if (!res) {
            gContext->pContext = nil;
            return nullptr;
        }
    }
    auto mContext = (__bridge_retained void *)(gContext->pContext);
    auto rt = new MetalRuntime(mContext);
    rt->setGpuMode(info.gpuMode);
    if (nil != sharedContext.queue) {
        rt->setCommandQueue(sharedContext.queue, true);
    }
    bool supportDefer = info.numThread & MNN_GPU_RECORD_BATCH;
    if ((!supportDefer) && nil == sharedContext.queue) {
        id<MTLCommandQueue> queue = [sharedContext.device newCommandQueue];
        rt->setCommandQueue(queue, false);
    }
    if (nullptr != info.user) {
        rt->mDefaultConfig = *info.user;
    }
    return rt;
}

MetalRuntime::MetalRuntime(void* context) {
    mContext = context;
    auto ctx = (__bridge MNNMetalContext *)mContext;
    std::shared_ptr<EagerBufferAllocator::Allocator> allocator(new MetalRuntimeAllocator([ctx device]));
    // supportsFamily: is available since iOS 13.0 / macOS 10.15, must check before calling
    if (@available(iOS 13.0, macOS 10.15, *)) {
        mSimdGroupReduce = [[ctx device] supportsFamily:MTLGPUFamilyApple7];
        mSimdGroupReduce |= [[ctx device] supportsFamily:(MTLGPUFamily)MTLGPUFamilyMetal3_MNN];
        mSimdGroupMatrix = [[ctx device] supportsFamily:MTLGPUFamilyApple7];
    } else {
        mSimdGroupReduce = false;
        mSimdGroupMatrix = false;
    }
    mMaxThreadSize = [[ctx device] maxThreadsPerThreadgroup].width;
    // Metal4 Support M1/A14 and later chips
#ifdef MNN_METAL_TENSOR
    if (@available(iOS 13.0, macOS 10.15, *)) {
        mTensorOps = [[ctx device] supportsFamily:(MTLGPUFamily)MTLGPUFamilyMetal4_MNN];
    } else {
        mTensorOps = false;
    }

    // AI TensorCore device support from M5/A19
    bool noAICoreDevice = [[[ctx device] name] containsString:@"M1"] || \
                        [[[ctx device] name] containsString:@"M2"] || \
                        [[[ctx device] name] containsString:@"M3"] || \
                        [[[ctx device] name] containsString:@"M4"] || \
                        [[[ctx device] name] containsString:@"A14"] || \
                        [[[ctx device] name] containsString:@"A15"] || \
                        [[[ctx device] name] containsString:@"A16"] || \
                        [[[ctx device] name] containsString:@"A17"] || \
                        [[[ctx device] name] containsString:@"A18"];
    mTensorOps = mTensorOps && !noAICoreDevice;
#else
    mTensorOps = false;
#endif
    // M4-class capability gate (device-name based: M3 and M4 share MTLGPUFamilyApple9).
    // Used for heuristics only calibrated on M4/A-series; M1/M2/M3 keep legacy routes.
    bool isOldMacGpu = [[[ctx device] name] containsString:@"M1"] || \
                       [[[ctx device] name] containsString:@"M2"] || \
                       [[[ctx device] name] containsString:@"M3"];
    mPreferInShaderPrefillDequant = mSimdGroupMatrix && !isOldMacGpu;
    // M64 outer-dequant GEMM tile tier, MLX-style arch parse (family API can't
    // tell M3 from M4 -- both MTLGPUFamilyApple9). architecture.name is
    // "applegpu_g<gen><size>" (g13=M1 .. g16=M4/A18, size: p=phone, g=base/pro,
    // s=max, d=ultra). M4-class Macs (gen >= 16, non-phone) take the 64x64 tile:
    // M4 Pro paired rep5x2 pp2048 +1.1~2.4%, pp512 neutral. M3 Pro pp512 -1.4%
    // keeps gen <= 15 off; phones ('p') stay off pending calibration. Older OS
    // exposes no architecture -> off (conservative).
    if (@available(iOS 17.0, macOS 14.0, *)) {
        const char* archName = [[[[ctx device] architecture] name] UTF8String];
        const char* kPrefix = "applegpu_g";
        if (archName != nullptr && strncmp(archName, kPrefix, strlen(kPrefix)) == 0) {
            const char* p = archName + strlen(kPrefix);
            int gen = atoi(p);
            char size = archName[strlen(archName) - 1];
            mPreferM64Gemm = mSimdGroupMatrix && gen >= 16 && size != 'p';
        }
    }
//    MNN_PRINT("Metal device name %s, open tensor: %d\n\n", [[[ctx device] name] UTF8String], mTensorOps);
    mStaticAllocator.reset(new EagerBufferAllocator(allocator));
    mDynamic.resize(METAL_SEPERATE_MAX_COUNT);
    for (auto& buf : mDynamic) {
        buf.root = allocator;
    }
    mTunedInfo = new TunedInfo;
}

MetalRuntime::~ MetalRuntime() {
    if(mContext) {
        CFRelease(mContext);
    }
    delete mTunedInfo;
}

bool MetalRuntime::setCache(std::pair<const void*, size_t> cache) {//Get Cache
    auto buffer = cache.first;
    auto size   = cache.second;
    if (nullptr == buffer) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return false;//actually get nothing
    }
    mCacheOutsideSize = size;
    mCacheOutside = buffer;
    auto cacheBuffer = GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)cache.first, cache.second);
    if (false == VerifyCacheBuffer(verify)) {
        return false;
    }
    if (nullptr == cacheBuffer->tunings()) {
        return false;
    }

    // Load Auto Tuning Info
    if (nullptr != cacheBuffer->tunings()) {
        auto tuningInfo = cacheBuffer->tunings();
        for (int i=0; i<tuningInfo->size(); ++i) {
            auto tun = tuningInfo->GetAs<Autotuning>(i);
            if (nullptr == tun->threadSize() || nullptr == tun->groupSize() || nullptr == tun->key()) {
                MNN_ERROR("Error tunning info\n");
                continue;
            }
            std::vector<uint32_t> glo(tun->threadSize()->size());
            for (int v=0; v<glo.size(); ++v) {
                glo[v] = tun->threadSize()->data()[v];
            }
            std::vector<uint32_t> grop(tun->groupNum()->size());
            for (int v=0; v<grop.size(); ++v) {
                grop[v] = tun->groupNum()->data()[v];
            }
            std::vector<uint32_t> loc(tun->groupSize()->size());
            for (int v=0; v<loc.size(); ++v) {
                loc[v] = tun->groupSize()->data()[v];
            }
            uint32_t cost = tun->timeCost();
            mTunedThreadGroup.insert(std::make_pair(std::make_pair(tun->key()->str(), glo), std::make_tuple(grop, loc, cost)));
            mTunedThreadGroupVec[tun->key()->str()].emplace_back(std::make_pair(glo, std::make_tuple(grop, loc, cost)));
        }
    }
    return true;
}

std::pair<const void*, size_t> MetalRuntime::makeCache(TunedInfo* info) {//make Cache
    std::unique_ptr<CacheT> cache(new CacheT);
    // Get All Autotuning cache
    for (auto& iter : mTunedThreadGroup) {
        std::unique_ptr<AutotuningT> tuning(new AutotuningT);
        tuning->key = iter.first.first;
        tuning->threadSize = iter.first.second;
        
        tuning->groupNum = std::get<0>(iter.second);
        tuning->groupSize = std::get<1>(iter.second);
        tuning->timeCost = std::get<2>(iter.second);

        cache->tunings.emplace_back(std::move(tuning));
    }
    cache->tuned = std::move(info->mInfos);

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

int MetalRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    if (STATUS_SUPPORT_SIMD_GROUP_REDUCE == statusEnum) {
        return mSimdGroupReduce ? 1 : 0;
    }
    return 0;
}

float MetalRuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mStaticAllocator->totalSize() / 1024.0f / 1024.0f;
    float dynamicMemoryInMB = 0.0f;
    for (auto& buf : mDynamic) {
        dynamicMemoryInMB += buf.currentSize / 1024.0f / 1024.0f;
    }
    return staticMemoryInMB + dynamicMemoryInMB;
}

void MetalRuntime::onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           const MNN::Op* op) {
    if (nullptr != op->name()) {
        auto dstInfo = mTunedInfo;
        std::unique_ptr<MetalCache::OpInfoT> opInfo(new MetalCache::OpInfoT);;
        opInfo->type = op->type();
        opInfo->name = op->name()->str();
        opInfo->inputs.resize(inputs.size());
        for (int v=0; v<opInfo->inputs.size(); ++v) {
            opInfo->inputs[v].reset(new MetalCache::TensorInfoT);
            opInfo->inputs[v]->shape.resize(inputs[v]->dimensions());
            for (int u=0; u<opInfo->inputs[v]->shape.size(); ++u) {
                opInfo->inputs[v]->shape[u] = inputs[v]->length(u);
            }
        }
        opInfo->outputs.resize(outputs.size());
        for (int v=0; v<opInfo->outputs.size(); ++v) {
            opInfo->outputs[v].reset(new MetalCache::TensorInfoT);
            opInfo->outputs[v]->shape.resize(outputs[v]->dimensions());
            for (int u=0; u<opInfo->outputs[v]->shape.size(); ++u) {
                opInfo->outputs[v]->shape[u] = outputs[v]->length(u);
            }
        }
        dstInfo->mInfos.emplace_back(std::move(opInfo));
    }
}
static bool _checkTensorInfo(const MetalCache::TensorInfoT* dst, const Tensor* src) {
    if (dst->shape.size() != src->dimensions()) {
        return false;
    }
    for (int j=0; j<dst->shape.size(); ++j) {
        if (dst->shape[j] != src->length(j)) {
            return false;
        }
    }
    return true;
}
bool MetalRuntime::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const MNN::Op* op, Runtime::OpInfo& dstInfo) const {
    dstInfo.initCostLong = true;
    if (nullptr == op->name()) {
        dstInfo.initCostLong = false;
        return true;
    }
    for(auto& info : mTunedInfo->mInfos) {
        if (info->type != op->type()) {
            continue;
        }
        if (info->name != op->name()->str()) {
            continue;
        }
        if (info->inputs.size() != inputs.size() || info->outputs.size() != outputs.size()) {
            continue;
        }
        bool match = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto& dst = info->inputs[i];
            auto src = inputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto& dst = info->outputs[i];
            auto src = outputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (match) {
            // All Info is match
            dstInfo.initCostLong = false;
            break;
        }
    }
    return true;
}

class MetalWrapAllocator : public BufferAllocator::Allocator {
private:
    std::shared_ptr<BufferAllocator::Allocator> mOrigin;
    id<MTLDevice> mDevice;
public:
    MetalWrapAllocator(std::shared_ptr<BufferAllocator::Allocator> origin, id<MTLDevice> device) : mOrigin(origin), mDevice(device) {}
    virtual ~ MetalWrapAllocator() {
        // Do nothing
    }
    virtual void sync() override {
        mOrigin->sync();
    };
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        auto mem = mOrigin->onAlloc(size, align);
        MNN_ASSERT(mem.second == 0);
        if (mem.first == nullptr) {
            return MemChunk(nullptr, 0);
        }
        MTLResourceOptions opts = MTLResourceStorageModeShared;
        id<MTLBuffer> buffer = [mDevice newBufferWithBytesNoCopy:mem.first length:size options:opts deallocator:nil];
        if (buffer == nil) {
            mOrigin->onRelease(mem);
            return MemChunk(nullptr, 0);
        }
        auto wrap = new MetalRuntimeAllocator::MetalBufferAlloc(buffer);
        return MemChunk((void *)wrap, 0);
    }
    virtual void onRelease(MemChunk chunk) override {
        auto mem = (MetalRuntimeAllocator::MetalBufferAlloc *)chunk.first;
        mOrigin->onRelease(MemChunk(mem->getBuffer().contents));
        delete mem;
    }
};
Backend* MetalRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    if (hint().weightMemoryPath.size() > 0 && mStaticAllocatorMMap.get() == nullptr) {
        // Only support set weightmap dir once
        mStaticAllocatorRaw = mStaticAllocator;
        // forward_type, precision_type, memory_type, power_type
        std::string prefix = "1_0_0_0_";
        std::string posfix = "metal.weight";
        auto syncPath = prefix + "sync." + posfix;
        bool autoRemove = true;
        if (hint().useCachedMmap) {
            autoRemove = false;
            std::string fileName = MNNFilePathConcat(hint().weightMemoryPath, syncPath);
            const_cast<RuntimeHint&>(hint()).useCachedMmap += MNNFileExist(fileName.c_str());
        }
        auto ctx = (__bridge MNNMetalContext *)mContext;
        auto mmap = BufferAllocator::Allocator::createMmap(hint().weightMemoryPath.c_str(), prefix.c_str(), posfix.c_str(), autoRemove);
        std::shared_ptr<BufferAllocator::Allocator> mmapMem(new MetalWrapAllocator(mmap, [ctx device]));
        mStaticAllocator.reset(new EagerBufferAllocator(mmapMem, 32, 1024 * 1024 * 1024));
        mStaticAllocatorMMap = mStaticAllocator;
    }
    BackendConfig::PrecisionMode precision = mDefaultConfig.precision;
    BackendConfig::MemoryMode memory = mDefaultConfig.memory;
    if (nullptr != config) {
        precision = config->precision;
        memory = config->memory;
    }
    bool useFp16AsFp32 = precision != BackendConfig::Precision_High;
    auto backend = new MetalBackend(this, useFp16AsFp32, memory);
    backend->setMetaPtr(pMeta);
    return backend;
}

void MetalRuntime::onGabageCollect(int level) {
    mStaticAllocator->release(false);
    if (nullptr != mStaticAllocatorMMap) {
        mStaticAllocatorMMap->release(false);
    }
    if (level >= 100) {
        for (auto& buf : mDynamic) {
            buf.release();
        }
    }
}

std::pair<const void*, size_t> MetalRuntime::onGetCache() {//make Cache
    return makeCache(mTunedInfo);
}

bool MetalRuntime::onSetCache(const void* buffer, size_t size) {//set Cache
    if (nullptr == buffer) {
        return false;
    }
    auto cacheBuffer = MetalCache::GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)buffer, size);
    if (false == VerifyCacheBuffer(verify)) {
        return false;
    }
    if(nullptr != cacheBuffer->tuned()) {
        for (int i=0; i<cacheBuffer->tuned()->size(); ++i) {
            auto srcInfo = cacheBuffer->tuned()->GetAs<MetalCache::OpInfo>(i);
            std::unique_ptr<MetalCache::OpInfoT> dst(srcInfo->UnPack());
            mTunedInfo->mInfos.emplace_back(std::move(dst));
        }
    }
    return setCache(std::make_pair(buffer, size));
}

MemChunk MetalRuntimeAllocator::onAlloc(size_t size, size_t align) {
    auto buffer = [mDevice newBufferWithLength:size options:MTLCPUCacheModeDefaultCache];
    auto mMetalBufferAlloc = new MetalBufferAlloc(buffer);
    return MemChunk((void *)mMetalBufferAlloc, 0);
}
void MetalRuntimeAllocator::onRelease(MemChunk ptr) {
    delete (MetalBufferAlloc *)ptr.first;
}

class MetalRuntimeCreator : public RuntimeCreator {
public:
    MetalRuntimeCreator() {
        // Do nothing
    }
    virtual ~ MetalRuntimeCreator() {
        // Do nothing
    }
    virtual Runtime *onCreate(const Backend::Info &info) const {
        auto rt = MetalRuntime::create(info);
        return rt;
    }
private:
    id<MTLDevice> mDevice;
};

void registerMetalRuntimeCreator() {
    // according to
    // https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/HardwareGPUInformation/HardwareGPUInformation.html
    // not all device with iOS 8+ supports metal.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (nil != device) {
        gContext = new MetalContext;
        gContext->pContext = nil;
        gContext->pDevice = nil;
        registerMetalOps();
#ifdef MNN_SUPPORT_RENDER
        registerMetalRenderOps();
#endif
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_METAL, new MetalRuntimeCreator, false);
    } else {
        MNN_ERROR("Init Metal Error\n");
    }
}
} // namespace MNN
#else
namespace MNN {
void registerMetalRuntimeCreator() {
}
};
int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor) {
    return -1;
}

#endif