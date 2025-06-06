//
//  OpenCLRuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp


#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <queue>
#include <string>
#include <vector>

#include <string>
#include <vector>
#include "core/Macro.h"
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
enum SvmType { FINE_BUFFER = 0, COARSE_BUFFER = 1, SVM_NONE = 2};

struct RuntimeInitInfo {
    int platformSize;
    int platformId;
    int deviceId;
    void *contextPtr;
};

struct KernelPool {
    uint64_t maxWorkGroupSize;
    std::queue<std::shared_ptr<cl::Kernel>> recycle;
};
class KernelWrap {
public:
    KernelWrap(std::shared_ptr<cl::Kernel> k, KernelPool* recycle) : mKernel(k), mRecycle(recycle) {
        // Do nothing
    }
    ~ KernelWrap() {
        if (nullptr != mRecycle) {
            mRecycle->recycle.push(mKernel);
        }
    }
    cl::Kernel& get() {
        return *mKernel;
    }
    KernelPool* mRecycle;
private:
    std::shared_ptr<cl::Kernel> mKernel;
};
class OpenCLRuntime {
public:
    OpenCLRuntime(int platformSize, int platformId, int deviceId, void *contextPtr, const RuntimeHint& hint);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    bool isDeviceSupportedLowPower() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;
    bool isSupportedIntelSubgroup() const;
    ::cl::Context &context();
    ::cl::CommandQueue &commandQueue();
    ::cl::CommandQueue &recordableQueue();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t MaxThreadsPerDevice() const;
    uint32_t MaxWorkGroupSize() const;
    uint32_t maxFreq() const;
    uint64_t getMaxWorkGroupSize(std::shared_ptr<KernelWrap> kernel);
    uint64_t GetKernelWaveSize(std::shared_ptr<KernelWrap> kernel);
    std::vector<uint32_t> getMaxWorkItemSizes();
    uint64_t getMaxLocalMem() const;
    uint32_t getUseRecordableQueueSize(){
        return mUseRecordableQueueSize;
    }
    bool isSupportRecordQueue(){
        return mSupportRecordQueue;
    }
    GpuType getGpuType() {
        return mGpuType;
    }
    MaliAr getMaliAr() {
        return mMaliAr;
    }
    float getCLVersion() {
        return mCLVersion;
    }
    bool isSupportAHD(){
        return mIsSupportAHD;
    }
#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities getSvmCapabilities() {
        return mSvmCapabilities;
    }
#endif
    GpuLevel getGpuLevel() {
        return mGpuLevel;
    }
    std::string getDeviceName() {
        return mDeviceName;
    }
    void pushEvent(std::pair<std::string, cl::Event> data) {
        return mEvents.push_back(data);
    }
    unsigned int getEventTime(cl::Event& event);
    void printEventTime();
    void clearEvent(){
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

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>, uint32_t>>& tunedLwsMap();
    
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>>>>& getTuneLwsMap();
    
    std::shared_ptr<KernelWrap> buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions, int precisionLevel, const Tensor *input = nullptr, const Tensor *output = nullptr);
    std::shared_ptr<KernelWrap> buildKernelWithCache(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions, int precisionLevel, const Tensor *input = nullptr, const Tensor *output = nullptr, bool useCache = true);
    std::shared_ptr<KernelWrap> buildKernelFromSource(const std::string&, const std::string &kernelName,
                                       const std::set<std::string> &buildOptions, int precisionLevel);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const {
        return mIsCreateError;
    }

    float flops() const {
        return mFlops;
    }
    
    bool canShareRuntime(int platformSize, int platformId, int deviceId, void *contextPtr){
        return (platformSize == mInitInfo.platformSize) && (platformId == mInitInfo.platformId) && (deviceId == mInitInfo.deviceId) && (contextPtr == mInitInfo.contextPtr);
    }

    double getCostTime(const cl::Event *event);
    double getQueuedTime(const cl::Event *event);
    double getSubmitTime(const cl::Event *event);

    std::pair<const void*, size_t> makeCache(void* tuneInfo);
    bool setCache(std::pair<const void*, size_t> cache);
private:
    bool loadProgram(const std::string &programName, cl::Program *program);
    bool buildProgram(const std::string &buildOptionsStr, cl::Program *program);
    bool getDeviceSupportsExtension(const cl::Device &device, const char *extensionName);

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
    bool isSetWorkGroupAttribute = false;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};
    
    double mStartNanos;
    double mStopNanos;

    std::map<std::vector<uint32_t>, std::vector<uint32_t>> mTunedGemmParams;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>,  uint32_t>> mTunedLws;
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>,  uint32_t>>>> mTuneLws;
    std::vector<uint8_t> mBuffer;
    RuntimeInitInfo mInitInfo;
};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */
