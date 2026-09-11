//
//  OpenCLRuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp

#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "Type_generated.h"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include "MNN/MNNForwardType.h"
#include "core/TensorUtils.hpp"

namespace MNN {

#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

#define CL_KERNEL_WAVE_SIZE_QCOM 0xAA02

enum GpuType { MALI = 0, ADRENO = 1, RADEON = 2, INTEL = 3, OTHER = 4 };
enum GpuLevel { UNDEFINED = 0, TOP = 1, MEDIUM = 2, LOW = 3 };
enum MaliAr { MIDGARD = 0, BIFROST = 1, VALHALL = 2 };
enum SvmType { FINE_BUFFER = 0, COARSE_BUFFER = 1, SVM_NONE = 2 };

struct RuntimeInitInfo {
    int platformSize;
    int platformId;
    int deviceId;
    void* contextPtr;
};

struct KernelPool {
    uint64_t maxWorkGroupSize;
    std::queue<std::shared_ptr<cl::Kernel>> recycle;
};

struct TuneInfo {
    std::string programName;
    std::string md5;
    std::vector<uint32_t> globalSize;
    std::vector<uint32_t> localSize;
    uint32_t timeCost;
};

class KernelWrap {
public:
    KernelWrap(std::shared_ptr<cl::Kernel> k, KernelPool* recycle) : mKernel(k), mRecycle(recycle) {
        // Do nothing
    }
    ~KernelWrap() {
        if (nullptr != mRecycle) {
            mRecycle->recycle.push(mKernel);
        }
    }
    cl::Kernel& get() { return *mKernel; }
    KernelPool* mRecycle;

private:
    std::shared_ptr<cl::Kernel> mKernel;
};

// A gemm tuning candidate whose program has already been compiled on the foreground thread
// into the shared mBuildProgramMap. The background worker turns each immutable cl::Program
// into its own cl::Kernel (clCreateKernel is thread-safe on a built program) and only measures.
struct GemmTuneCandidate {
    cl::Program program;
    std::string kernelName;
    std::vector<uint32_t> params;
};

class OpenCLRuntime {
public:
    OpenCLRuntime(int platformSize, int platformId, int deviceId, void* contextPtr, const RuntimeHint& hint);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime&) = delete;
    OpenCLRuntime& operator=(const OpenCLRuntime&) = delete;

    bool isSupportedFP16() const;
    bool isClCreateImageAvailable() const;
    bool isDeviceSupportedLowPower() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;
    bool isSupportedIntelSubgroup() const;
    ::cl::Context& context();
    ::cl::CommandQueue& commandQueue();
    ::cl::CommandQueue& recordableQueue();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t MaxThreadsPerDevice() const;
    uint32_t MaxWorkGroupSize() const;
    uint32_t maxFreq() const;
    uint64_t getMaxWorkGroupSize(std::shared_ptr<KernelWrap> kernel);
    uint64_t GetKernelWaveSize(std::shared_ptr<KernelWrap> kernel);
    std::vector<uint32_t> getMaxWorkItemSizes();
    uint64_t getMaxLocalMem() const;
    uint32_t getUseRecordableQueueSize() { return mUseRecordableQueueSize; }
    bool isSupportRecordQueue() { return mSupportRecordQueue; }
    GpuType getGpuType() { return mGpuType; }
    MaliAr getMaliAr() { return mMaliAr; }
    float getCLVersion() { return mCLVersion; }
    bool isSupportAHD() { return mIsSupportAHD; }
#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities getSvmCapabilities() { return mSvmCapabilities; }
#endif
    GpuLevel getGpuLevel() { return mGpuLevel; }
    std::string getDeviceName() { return mDeviceName; }
    void pushEvent(std::pair<std::string, cl::Event> data) { return mEvents.push_back(data); }
    unsigned int getEventTime(cl::Event& event);
    void printEventTime();
    void clearEvent() {
        mKernelTime = 0;
        mEvents.clear();
    }
    uint64_t maxAllocSize() const;
    void setCommandQueueProfileEnable();
    void setCommandQueueProfileDisable();

    unsigned int mQueueCount = 0;
    unsigned int getQueueNum();

    unsigned int mKernelTime = 0;

    std::map<std::vector<uint32_t>, std::vector<uint32_t>>& tunedGemmParamsMap();

    // Background gemm tuning: the foreground compiles each candidate program once into the
    // shared mBuildProgramMap (so inference and the disk cache reuse it), then hands the
    // precompiled cl::Programs to the worker. The worker only measures them on its own
    // profiling queue and merges the winner into mTunedGemmParams under mGemmParamsMutex.
    void setPrebuildResizeActive(bool active);
    void setPrebuildTuneActive(bool active);
    // Marks the shape as queued (under the mutex) so the foreground compiles candidates only
    // once. Returns false when the shape is already tuned or already queued.
    bool reserveGemmTuneSlot(const std::vector<uint32_t>& gemmSize, int precision);
    void submitGemmTuneJob(const std::vector<uint32_t>& gemmSize, int precision,
                           std::vector<GemmTuneCandidate> candidates);
    std::mutex& gemmParamsMutex() { return mGemmParamsMutex; }

    // A single worker serializes clBuildProgram. Speculative tasks enter at the back; an exact
    // foreground miss moves to the front and waits only for that key.
    void submitPrebuild(const std::string& programName, const std::set<std::string>& buildOptions, int precision,
                        const Tensor* input = nullptr, const Tensor* output = nullptr, bool highPriority = false,
                        bool exactFirst = false);

    // Compile (or fetch from the shared cache) the program for a build-option set and return it.
    cl::Program buildTuneProgram(const std::string& programName, const std::set<std::string>& buildOptions,
                                 int precisionLevel);

    std::map<std::pair<std::string, std::vector<uint32_t>>, TuneInfo>& tunedLwsMap();

    std::map<std::string, std::vector<TuneInfo>>& getTuneLwsMap();

    std::shared_ptr<KernelWrap> buildKernel(const std::string& programName, const std::string& kernelName,
                                            const std::set<std::string>& buildOptions, int precisionLevel,
                                            const Tensor* input = nullptr, const Tensor* output = nullptr);
    std::shared_ptr<KernelWrap> buildKernelWithCache(const std::string& programName, const std::string& kernelName,
                                                     const std::set<std::string>& buildOptions, int precisionLevel,
                                                     const Tensor* input = nullptr, const Tensor* output = nullptr,
                                                     bool useCache = true);
    std::shared_ptr<KernelWrap> buildKernelFromSource(const std::string&, const std::string& kernelName,
                                                      const std::set<std::string>& buildOptions, int precisionLevel);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const { return mIsCreateError; }

    float flops() const { return mFlops; }

    double getCostTime(const cl::Event* event);
    double getQueuedTime(const cl::Event* event);
    double getSubmitTime(const cl::Event* event);

    std::pair<const void*, size_t> makeCache(void* tuneInfo);
    bool setCache(std::pair<const void*, size_t> cache);

private:
    bool loadProgram(const std::string& programName, cl::Program* program);
    bool buildProgram(const std::string& buildOptionsStr, cl::Program* program);
    std::string makeBuildOptionsStr(const std::set<std::string>& buildOptions, int precisionLevel, const Tensor* input,
                                    const Tensor* output);
    bool getOrBuildProgram(const std::string& programName, const std::string& buildOptionsStr, cl::Program* outProgram);
    bool getDeviceSupportsExtension(const cl::Device& device, const char* extensionName);
    void gemmTuneWorker();
    void prebuildWorker();

private:
    std::vector<size_t> mMaxImageSize;
    std::vector<uint32_t> mMaxWorkIterms;
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueueTuning;
    struct ProgramWithKernel {
        cl::Program program;
        std::map<std::string, KernelPool> kernels;
        std::shared_ptr<char> Buffer;
        int BufferSize = 0;
    };
    cl::CommandQueue* mCurrentCommandQueue;
    std::mutex mProgramBuildMutex;
    std::mutex mBuildProgramMutex;
    std::map<std::tuple<std::string, std::string>, ProgramWithKernel> mBuildProgramMap;
    std::shared_ptr<::cl::CommandQueue> mRecordableQueuePtr;
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq;
    uint64_t mMaxMemAllocSize;
    uint64_t mMaxLocalMemSize;
    uint32_t mMaxThreadsPerDevice;
    uint32_t mMaxWorkGroupSize;
    uint32_t mUseRecordableQueueSize = 0;
    bool mSupportRecordQueue = false;
    bool mIsSupportedFP16 = false;
    bool mIsDeviceSupportedLowPower = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    bool mSupportedIntelSubgroup = false;
    bool mIsSupportAHD = false;
    GpuType mGpuType;
    MaliAr mMaliAr;
    GpuLevel mGpuLevel = UNDEFINED;
    float mCLVersion = 1.0f;
    std::vector<std::pair<std::string, cl::Event>> mEvents;

#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities mSvmCapabilities;
#endif
    std::string mDeviceName;
    std::string mDeviceInfo;
    std::string mDriverInfo;
    bool isSetWorkGroupAttribute = false;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};

    double mStartNanos;
    double mStopNanos;

    std::map<std::vector<uint32_t>, std::vector<uint32_t>> mTunedGemmParams;
    std::map<std::pair<std::string, std::vector<uint32_t>>, TuneInfo> mTunedLws;
    std::map<std::string, std::vector<TuneInfo>> mTuneLws;
    std::vector<uint8_t> mBuffer;
    RuntimeInitInfo mInitInfo;

    struct GemmTuneJob {
        std::vector<uint32_t> gemmSize;
        int precision;
        std::vector<GemmTuneCandidate> candidates;
    };
    std::mutex mGemmParamsMutex; // protects mTunedGemmParams and the tune job queue
    std::condition_variable mGemmTuneCV;
    std::vector<GemmTuneJob> mGemmTuneJobs;
    std::set<std::vector<uint32_t>> mGemmTuneQueued;
    std::thread mGemmTuneThread;
    bool mGemmTuneStop = false;

    // Speculative tasks enter at the back. If resize requests a missing key, it is moved
    // to the front and the caller waits only for that key; unrelated builds continue later.
    using PrebuildKey = std::tuple<std::string, std::string>;
    struct PrebuildTask {
        std::string programName;
        std::string buildOptionsStr;
        bool highPriority = false;
        bool exactFirst = false;
        bool required = false;
    };
    struct PrebuildKeyHash {
        size_t operator()(const PrebuildKey& k) const noexcept {
            return std::hash<std::string>{}(std::get<0>(k)) ^ (std::hash<std::string>{}(std::get<1>(k)) << 1);
        }
    };
    std::mutex mPrebuildMutex;
    std::condition_variable mPrebuildCV;
    std::deque<PrebuildTask> mPrebuildQueue;
    std::unordered_set<PrebuildKey, PrebuildKeyHash> mPrebuildSeen;
    std::unordered_set<PrebuildKey, PrebuildKeyHash> mPrebuildCompleted;
    std::unordered_set<PrebuildKey, PrebuildKeyHash> mPrebuildFailed;
    PrebuildKey mPrebuildInFlightKey;
    bool mPrebuildInFlight = false;
    int mPrebuildResizeDepth = 0;
    int mPrebuildTuneDepth = 0;
    std::thread mPrebuildThread;
    bool mPrebuildStop = false;
};

} // namespace MNN
#endif /* OpenCLRuntime_hpp */