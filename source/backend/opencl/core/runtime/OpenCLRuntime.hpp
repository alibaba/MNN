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
#include <string>
#include <vector>

#include <string>
#include <vector>
#include "core/Macro.h"
#include "Type_generated.h"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include "MNN/MNNForwardType.h"

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
enum MaliAr { MIDGARD = 0, BIFROST = 1, VALHALL = 2 };
enum GpuMemObject { AUTO = 0, BUFFER = 1, IMAGE = 2};
enum CLTuneLevel { None = 0, Heavy = 1, Wide = 2, Normal = 3, Fast = 4};
enum SvmType { FINE_BUFFER = 0, COARSE_BUFFER = 1, SVM_NONE = 2};

class OpenCLRuntime {
public:
    OpenCLRuntime(const BackendConfig::PrecisionMode precision, const int cl_mode, int platformSize, int platformId, int deviceId);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    bool isWeightCpuTransHalf() const;
    bool isDeviceSupportedFP16() const;
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
    uint64_t getMaxWorkGroupSize(const ::cl::Kernel &kernel);
    uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
    std::vector<uint32_t> getMaxWorkItemSizes();
    uint64_t getMaxLocalMem() const;
    std::vector<cl_recording_qcom> *getRecordings(){
        return &mRecordings;
    }
    uint32_t getUseRecordableQueueSize(){
        return mUseRecordableQueueSize;
    }
    bool isUseRecordQueue(){
        return mUseRecordQueue;
    }
    bool isDevideOpRecord(){
        return mDevideOpRecord;
    }
    void setDevideOpRecord(){
        mDevideOpRecord = true;
    }
    void setRecordNum(int num){
        mRecordNums = num;
    }
    uint32_t getRecordNum(){
        return mRecordNums;
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
#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities getSvmCapabilities() {
        return mSvmCapabilities;
    }
#endif
    GpuMemObject getGpuMemType() {
        return mMemType;
    }
    CLTuneLevel getCLTuneLevel() {
        return mTuneLevel;
    }
    std::string getDeviceName() {
        return mDeviceName;
    }
    void pushEvent(std::pair<std::string, cl::Event> data) {
        return mEvents.push_back(data);
    }
    void printEventTime();
    void clearEvent(){
        mKernelTime = 0;
        mEvents.clear();
    }
    uint64_t maxAllocSize() const;
    void setCommandQueueProfileEnable();
    void setCommandQueueProfileDisable();
    void clearRecord();
    void enqeueRecord();
    void endRecord();
    void releaseRecord();

    unsigned int mQueueCount = 0;
    unsigned int getQueueNum();
    
    unsigned int mKernelTime = 0;

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>, uint32_t>>& tunedLwsMap();
    
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>>>>& getTuneLwsMap();
    
    ::cl::Kernel buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions);
    ::cl::Kernel buildKernelFromSource(const std::string&, const std::string &kernelName,
                                       const std::set<std::string> &buildOptions);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const {
        return mIsCreateError;
    }

    float flops() const {
        return mFlops;
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
    void setGpuMode(const int cl_mode_num);

private:
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
    std::map<std::tuple<std::string, std::string>, ::cl::Program> mBuildProgramMap;
    std::shared_ptr<::cl::CommandQueue> mRecordableQueuePtr;
    std::vector<cl_recording_qcom> mRecordings;
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq;
    uint32_t mMaxMemAllocSize;
    uint64_t mMaxLocalMemSize;
    uint32_t mMaxThreadsPerDevice;
    uint32_t mMaxWorkGroupSize;
    uint32_t mUseRecordableQueueSize;
    uint32_t mRecordNums = 0;
    bool mUseRecordQueue = false;
    bool mDevideOpRecord = true;
    bool mIsSupportedFP16     = false;
    bool mIsDeviceSupportedFP16 = false;
    bool mIsDeviceSupportedLowPower = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    bool mSupportedIntelSubgroup = false;
    GpuType mGpuType;
    MaliAr mMaliAr;
    float mCLVersion = 1.0f;
    std::vector<std::pair<std::string, cl::Event>> mEvents;

#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities mSvmCapabilities;
#endif
    GpuMemObject mMemType = AUTO;
    CLTuneLevel mTuneLevel = Wide;
    std::string mDeviceName;
    bool isSetWorkGroupAttribute = false;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};
    
    double mStartNanos;
    double mStopNanos;

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>,  uint32_t>> mTunedLws;
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>,  uint32_t>>>> mTuneLws;
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */
