//
//  OpenCLRuntime.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/Macro.h"
#include "OpenCLTuneInfo.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "CLCache_generated.h"
using namespace CLCache;
namespace MNN {

extern const std::map<std::string, std::vector<unsigned char>> OpenCLProgramMap;
extern std::mutex gCLMutex;

bool OpenCLRuntime::getDeviceSupportsExtension(const cl::Device &device, const char *extensionName) {
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto pos               = extensions.find(extensionName);
    return (pos != std::string::npos);
}

OpenCLRuntime::OpenCLRuntime(const BackendConfig::PrecisionMode precision, const int cl_mode, int platformSize, int platformId, int deviceId) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start OpenCLRuntime !\n");
#endif
    mDefaultBuildParams = " -cl-mad-enable";
    std::vector<cl::Platform> platforms;
    cl_int res = cl::Platform::get(&platforms, platformSize);
    MNN_CHECK_CL_SUCCESS(res, "getPlatform");
    if(platforms.size() > 0 && res == CL_SUCCESS) {
        if(platformId >= platforms.size() || platformId < 0) {
            platformId = 0;
        }
        cl::Platform::setDefault(platforms[platformId]);
        std::vector<cl::Device> gpuDevices;

        res = platforms[platformId].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
        if(1 <= gpuDevices.size() && res == CL_SUCCESS) {
            if(deviceId >= gpuDevices.size() || deviceId < 0) {
                deviceId = 0;
            }
            mFirstGPUDevicePtr = std::make_shared<cl::Device>(gpuDevices[deviceId]);
            if(mFirstGPUDevicePtr == nullptr) {
                mIsCreateError = true;
                return;
            }
            const std::string deviceName    = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NAME>();
            mDeviceName = deviceName;
            const std::string deviceVersion = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VERSION>();
            std::map<std::string, MNN::MaliAr> maliArMap {
                {"Mali-T860", MIDGARD},
                {"Mali-T880", MIDGARD},
                {"Mali-G31", BIFROST},
                {"Mali-G51", BIFROST},
                {"Mali-G52", BIFROST},
                {"Mali-G71", BIFROST},
                {"Mali-G72", BIFROST},
                {"Mali-G76", BIFROST},
                {"Mali-G57", VALHALL},
                {"Mali-G68", VALHALL},
                {"Mali-G77", VALHALL},
                {"Mali-G78", VALHALL},
                {"Mali-G310", VALHALL},
                {"Mali-G510", VALHALL},
                {"Mali-G610", VALHALL},
                {"Mali-G615", VALHALL},
                {"Mali-G710", VALHALL},
                {"Mali-G715", VALHALL},
            };
        
            const std::string deviceVendor  = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VENDOR>();
            cl_command_queue_properties properties = 0;

        #ifdef ENABLE_OPENCL_TIME_PROFILER
            properties |= CL_QUEUE_PROFILING_ENABLE;
        #endif
            cl_int res;
            // if device is QUALCOMM's and version is 2.0 , set spacial optimized param

            sscanf(deviceVersion.c_str(), "%*s%f%*s", &mCLVersion);
            
        #ifdef MNN_OPENCL_SVM_ENABLE
            if(mCLVersion > 1.99f && (false == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isSvmError())) {
                res = mFirstGPUDevicePtr->getInfo(CL_DEVICE_SVM_CAPABILITIES, &mSvmCapabilities);

                #ifdef LOG_VERBOSE
                if (res != CL_SUCCESS || mSvmCapabilities == 0) {
                    MNN_PRINT("SVM capalibilties: NONE\n");
                } else {
                    if (mSvmCapabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
                        MNN_PRINT("SVM capalibilties: SVM_FINE_GRAIN_BUFFER\n");
                        if (mSvmCapabilities & CL_DEVICE_SVM_ATOMICS) {
                            MNN_PRINT("SVM capalibilties: SVM_ATOMICS\n");
                        }
                    } else if (mSvmCapabilities & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
                        MNN_PRINT("SVM capalibilties: SVM_COARSE_GRAIN_BUFFER\n");
                    }
                }
                #endif
            }
        #endif
            
            if (deviceName == "QUALCOMM Adreno(TM)") {
                mGpuType = ADRENO;
                
                // if device is QUALCOMM's and version is 2.0 , set spacial optimized param
                //if Adreno version is less than Adreno512, donot set WorkGroupAttribute option
                std::string adrenoVersion = deviceVersion.substr(deviceVersion.size()-3);
                //printf("Adreno Version:%s\n", adrenoVersion.c_str());
                if(mCLVersion > 1.99f && adrenoVersion >= "512") {
                    isSetWorkGroupAttribute = true;
                }
            } else if (deviceName.find("Mali") != std::string::npos) {
                mGpuType = MALI;
                if(maliArMap.find(deviceName) != maliArMap.end()){
                    mMaliAr = maliArMap[deviceName];
                }else{
                    mMaliAr = VALHALL;
                }
            } else if (deviceVendor.find("Advanced Micro Devices") != std::string::npos) {
                // Radeon series GPU is main product of Advanced Micro Devices (AMD)
                mGpuType = RADEON;
                isSetWorkGroupAttribute = true;
            } 
            else if (deviceVendor.find("Intel") != std::string::npos) {
                mGpuType = INTEL;
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
                const std::string extensions = mFirstGPUDevicePtr->getInfo<CL_DEVICE_EXTENSIONS>();
                if (extensions.find("cl_intel_subgroups") != std::string::npos) {
                    mSupportedIntelSubgroup = true;
                    uint32_t execution_units_count = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                    uint32_t num_threads_per_eu = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NUM_THREADS_PER_EU_INTEL>();
                    uint32_t maxThreadsPerExecutionUnit = num_threads_per_eu > 0 ? num_threads_per_eu : 7;
                    mMaxThreadsPerDevice =  maxThreadsPerExecutionUnit * execution_units_count;
                    mMaxWorkGroupSize = mFirstGPUDevicePtr->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                }
#endif 
            }
            else {
                mGpuType = OTHER;
            }
            const std::string extensions = platforms[0].getInfo<CL_PLATFORM_EXTENSIONS>();
            bool isPriorityHint = (extensions.find("cl_khr_priority_hints") != std::string::npos);

            if(mGpuType == ADRENO && !isPriorityHint){
                std::vector<cl_context_properties> context_properties;
                context_properties.reserve(5);
                context_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
                context_properties.push_back(CL_PERF_HINT_HIGH_QCOM);
                context_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
                context_properties.push_back(CL_PRIORITY_HINT_LOW_QCOM);
                context_properties.push_back(0);
                mContext = std::shared_ptr<cl::Context>(new cl::Context(std::vector<cl::Device>({*mFirstGPUDevicePtr}), context_properties.data(), nullptr, nullptr, &res));
                mIsDeviceSupportedLowPower = true;
            }else{
                mContext = std::shared_ptr<cl::Context>(new cl::Context(std::vector<cl::Device>({*mFirstGPUDevicePtr}), nullptr, nullptr, nullptr, &res));
            }

            MNN_CHECK_CL_SUCCESS(res, "context");
            if (res != CL_SUCCESS) {
                mIsCreateError = true;
                return;
            }
            
            mIsDeviceSupportedLowPower = (mIsDeviceSupportedLowPower || isPriorityHint);
            
            #ifdef MNN_USE_LIB_WRAPPER
            if(isPriorityHint)
            {
                if(true == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isPropError())
                {
                    mIsCreateError = true;
                    return;
                }

                cl_queue_properties prop[] = {CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR,
#ifdef ENABLE_OPENCL_TIME_PROFILER
                    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
#endif
                    0};
                mCommandQueuePtr.reset(new cl::CommandQueue(clCreateCommandQueueWithProperties((*mContext).get(), (*mFirstGPUDevicePtr).get(), prop, &res)));
            }
            else
            #endif
            {
                mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &res);
            }
            MNN_CHECK_CL_SUCCESS(res, "commandQueue");
            if (res != CL_SUCCESS) {
                mIsCreateError = true;
                return;
            }
            
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &mGPUGlobalMemeryCacheSize);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &mGPUComputeUnits);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &mMaxFreq);
            cl_device_fp_config fpConfig;
            auto success = mFirstGPUDevicePtr->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fpConfig);
            mIsDeviceSupportedFP16     = CL_SUCCESS == success && fpConfig > 0;
            
            //set gpu mode, tuning level and memory object
            setGpuMode(cl_mode);
            
            if(mMemType == AUTO) {
                if(mGpuType == MALI || mGpuType == INTEL) {
                    mMemType = BUFFER;
                } else {
                    mMemType = IMAGE;
                }
            }

            auto permitFloat16 = false;
            if (precision == BackendConfig::Precision_Low || (mMemType == BUFFER && precision == BackendConfig::Precision_Normal)) {//buffer mode not support Normal Precision yet
                permitFloat16 = true;
            }
            mIsSupportedFP16 = mIsDeviceSupportedFP16 && permitFloat16;

            if(getDeviceSupportsExtension(*(mFirstGPUDevicePtr.get()), "cl_arm_integer_dot_product_int8")){
                mSupportDotInt8 = true;
            }
            if(getDeviceSupportsExtension(*(mFirstGPUDevicePtr.get()), "cl_arm_integer_dot_product_accumulate_int8")){
                mSupportDotAccInt8 = true;
            }
            
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
            {
                if((false == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isQcomError()) && getDeviceSupportsExtension(*(mFirstGPUDevicePtr.get()), "cl_qcom_recordable_queues")){
                    uint32_t MaxRecordableQueueSize = mFirstGPUDevicePtr->getInfo<CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE>();
                    cl_int err;
                    if(MaxRecordableQueueSize > 0 && IMAGE == mMemType){
                        // TODO: Use setSessionHint to set the number of mUseRecordableQueueSize
                        mUseRecordableQueueSize = 10;
                        mUseRecordableQueueSize = MaxRecordableQueueSize < mUseRecordableQueueSize ? MaxRecordableQueueSize : mUseRecordableQueueSize;
                        mUseRecordQueue = true;
                        mRecordableQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, CL_QUEUE_RECORDABLE_QCOM, &err);
                        if(err != CL_SUCCESS){
                            mIsCreateError = true;
                            return;
                        }
                    }
                }
            }
#endif
            
        }else{
            mIsCreateError = true;
            MNN_ASSERT(1 <= gpuDevices.size());
        }
    }else{
        mIsCreateError = true;
        MNN_ASSERT(platforms.size() > 0);
    }
}

void OpenCLRuntime::setGpuMode(const int cl_mode_num) {
    int totalSet = 0;
    bool isSet = (cl_mode_num & MNN_GPU_MEMORY_BUFFER);
    if(isSet) {
        mMemType = BUFFER;
        totalSet++;
    }
    isSet = (cl_mode_num & MNN_GPU_MEMORY_IMAGE);
    if(isSet) {
        mMemType = IMAGE;
        totalSet++;
    }
    if(totalSet > 1) {
        MNN_PRINT("set both BUFFER and IMAGE mode is not permitted, please check cl_mode:%x！\n", cl_mode_num);
    }
    
    totalSet = 0;
    isSet = (cl_mode_num & MNN_GPU_TUNING_NONE);
    if(isSet) {
        mTuneLevel = None;
        totalSet++;
    }
    
    isSet = (cl_mode_num & MNN_GPU_TUNING_FAST);
    if(isSet) {
        mTuneLevel = Fast;
        totalSet++;
    }
    
    isSet = (cl_mode_num & MNN_GPU_TUNING_NORMAL);
    if(isSet) {
        mTuneLevel = Normal;
        totalSet++;
    }
    
    isSet = (cl_mode_num & MNN_GPU_TUNING_HEAVY);
    if(isSet) {
        mTuneLevel = Heavy;
        totalSet++;
    }
    
    isSet = (cl_mode_num & MNN_GPU_TUNING_WIDE);
    if(isSet) {
        mTuneLevel = Wide;
        totalSet++;
    }

    if(totalSet != 1) {
        MNN_PRINT("set multi tuning mode is not permitted, please check cl_mode:%x！\n", cl_mode_num);
    }
    
    totalSet = 0;
    isSet = (cl_mode_num & MNN_GPU_RECORD_OP);
    if(isSet) {
        mDevideOpRecord = true;
        totalSet++;
    }
    
    isSet = (cl_mode_num & MNN_GPU_RECORD_BATCH);
    if(isSet) {
        mDevideOpRecord = false;
        totalSet++;
    }
    
    if(totalSet > 1) {
        MNN_PRINT("set multi record kernel mode is not permitted, please check cl_mode:%x！\n", cl_mode_num);
    }
}

void OpenCLRuntime::setCommandQueueProfileEnable() {
    mCommandQueuePtr->finish();
    mCommandQueuePtr.reset();
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

    cl_int res;
    mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &res);
    MNN_CHECK_CL_SUCCESS(res, "commandQueue");
}

void OpenCLRuntime::setCommandQueueProfileDisable() {
    mCommandQueuePtr->finish();
    mCommandQueuePtr.reset();
    cl_command_queue_properties properties = 0;

    cl_int res;
    mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &res);
    MNN_CHECK_CL_SUCCESS(res, "commandQueue");
}

unsigned int OpenCLRuntime::getQueueNum() {
    mQueueCount++;
    return mQueueCount;
}

std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>, uint32_t>>& OpenCLRuntime::tunedLwsMap() {
    return mTunedLws;
}

std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>>>>& OpenCLRuntime::getTuneLwsMap() {
    return mTuneLws;
}

OpenCLRuntime::~OpenCLRuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~OpenCLRuntime !\n");
#endif
    clearEvent();
    releaseRecord();
    mBuildProgramMap.clear();
    mRecordings.clear();
    mCommandQueuePtr.reset();
    mRecordableQueuePtr.reset();
    mContext.reset();
    mFirstGPUDevicePtr.reset();
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~OpenCLRuntime !\n");
#endif
}

std::vector<size_t> OpenCLRuntime::getMaxImage2DSize() {
    size_t max_height, max_width;
    cl_int res = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
    MNN_CHECK_CL_SUCCESS(res, "image2Dsize");
    res = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
    MNN_CHECK_CL_SUCCESS(res, "image2Dsize");
    return {max_height, max_width};
}

bool OpenCLRuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}
bool OpenCLRuntime::isWeightCpuTransHalf() const {
#ifdef USE_HALF_WEIGHT_MEMORY
    return mIsSupportedFP16;
#else
    return false;//most of time
#endif
}

bool OpenCLRuntime::isDeviceSupportedFP16() const {
    return mIsDeviceSupportedFP16;
}

bool OpenCLRuntime::isDeviceSupportedLowPower() const {
    return mIsDeviceSupportedLowPower;
}

bool OpenCLRuntime::isSupportedDotInt8() const {
    return mSupportDotInt8;
}

bool OpenCLRuntime::isSupportedDotAccInt8() const {
    return mSupportDotAccInt8;
}

bool OpenCLRuntime::isSupportedIntelSubgroup() const {
    return mSupportedIntelSubgroup;
 }
cl::Context &OpenCLRuntime::context() {
    return *mContext;
}

cl::CommandQueue &OpenCLRuntime::commandQueue() {
    return *mCommandQueuePtr;
}

cl::CommandQueue &OpenCLRuntime::recordableQueue(){
    return *mRecordableQueuePtr;
}

uint64_t OpenCLRuntime::deviceGlobalMemeryCacheSize() const {
    return mGPUGlobalMemeryCacheSize;
}

uint32_t OpenCLRuntime::deviceComputeUnits() const {
    return mGPUComputeUnits;
}

uint32_t OpenCLRuntime::MaxThreadsPerDevice() const {
    return mMaxThreadsPerDevice;
}
uint32_t OpenCLRuntime::MaxWorkGroupSize() const {
    return mMaxWorkGroupSize;
}

uint32_t OpenCLRuntime::maxFreq() const {
    return mMaxFreq;
}

uint64_t OpenCLRuntime::maxAllocSize() const {
    return mMaxMemAllocSize;
}

bool OpenCLRuntime::loadProgram(const std::string &programName, cl::Program *program) {
    std::lock_guard<std::mutex> lck(gCLMutex);
    auto it_source = OpenCLProgramMap.find(programName);
    if (it_source != OpenCLProgramMap.end()) {
        cl::Program::Sources sources;
        std::string source(it_source->second.begin(), it_source->second.end());
        sources.push_back(source);
        *program = cl::Program(context(), sources);
        return true;
    } else {
        MNN_PRINT("Can't find kernel source !\n");
        return false;
    }
}

bool OpenCLRuntime::buildProgram(const std::string &buildOptionsStr, cl::Program *program) {
    AUTOTIME;
    cl_int ret = program->build({*mFirstGPUDevicePtr}, buildOptionsStr.c_str());
    if (ret != CL_SUCCESS) {
        if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*mFirstGPUDevicePtr) == CL_BUILD_ERROR) {
            std::string buildLog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*mFirstGPUDevicePtr);
            MNN_PRINT("Program build log: %s \n", buildLog.c_str());
        }
        MNN_PRINT("Build program failed, err:%d ! \n", ret);
        return false;
    }
    return true;
}

cl::Kernel OpenCLRuntime::buildKernel(const std::string &programName, const std::string &kernelName,
                                      const std::set<std::string> &buildOptions) {
    std::string buildOptionsStr;
    if (mIsSupportedFP16) {
        buildOptionsStr = "-DFLOAT=half -DFLOAT2=half2 -DFLOAT3=half3 -DFLOAT4=half4 -DFLOAT8=half8 -DFLOAT16=half16 -DRI_F=read_imageh -DWI_F=write_imageh -DCONVERT_FLOAT4=convert_half4 -DMNN_SUPPORT_FP16";
    } else {
        buildOptionsStr = "-DFLOAT=float  -DFLOAT2=float2 -DFLOAT3=float3 -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4";
    }
    
    if(isSetWorkGroupAttribute) {
        buildOptionsStr += " -DSET_ATTRIBUTE=true";
    } else {
        buildOptionsStr += " -DSET_ATTRIBUTE=false";
    }
    for (auto &option : buildOptions) {
        buildOptionsStr += " " + option;
    }
    buildOptionsStr += mDefaultBuildParams;
    auto key = std::make_tuple(programName, buildOptionsStr);

    auto buildProgramInter = mBuildProgramMap.find(key);
    cl::Program program;
    if (buildProgramInter != mBuildProgramMap.end()) {
        program = buildProgramInter->second;
    } else {
        this->loadProgram(programName, &program);
        auto status = this->buildProgram(buildOptionsStr, &program);
        if (!status) {
            FUNC_PRINT_ALL(programName.c_str(), s);
        }
        mBuildProgramMap.emplace(key, program);
    }

    cl_int res;
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str(), &res);
    MNN_CHECK_CL_SUCCESS(res, "getKernel");
    return kernel;
}

cl::Kernel OpenCLRuntime::buildKernelFromSource(const std::string& source, const std::string &kernelName,
                                                const std::set<std::string> &buildOptions) {
    std::string buildOptionsStr;
    if (mIsSupportedFP16) {
        buildOptionsStr = "-DFLOAT=half -DFLOAT4=half4 -DFLOAT8=half8 -DFLOAT16=half16 -DRI_F=read_imageh -DWI_F=write_imageh -DCONVERT_FLOAT4=convert_half4 -DMNN_SUPPORT_FP16";
    } else {
        buildOptionsStr = "-DFLOAT=float -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4";
    }
    
    if(isSetWorkGroupAttribute) {
        buildOptionsStr += " -DSET_ATTRIBUTE=true";
    } else {
        buildOptionsStr += " -DSET_ATTRIBUTE=false";
    }
    for (auto &option : buildOptions) {
        buildOptionsStr += " " + option;
    }
    buildOptionsStr += mDefaultBuildParams;
    
    cl::Program::Sources sources;
    sources.push_back(source);
    cl::Program program = cl::Program(context(), sources);
    auto status = this->buildProgram(buildOptionsStr, &program);
    if (!status) {
        FUNC_PRINT_ALL(kernelName.c_str(), s);
    }
    // mBuildProgramMap.emplace(key, program);

    cl_int res;
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str(), &res);
    MNN_CHECK_CL_SUCCESS(res, "getKernel");
    return kernel;
}


uint64_t OpenCLRuntime::getMaxWorkGroupSize(const cl::Kernel &kernel) {
    uint64_t maxWorkGroupSize = 0;
    kernel.getWorkGroupInfo(*mFirstGPUDevicePtr, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize);
    return maxWorkGroupSize;
}

uint64_t OpenCLRuntime::GetKernelWaveSize(const cl::Kernel &kernel) {
    uint64_t kernelWaveSize = 0;
    kernel.getWorkGroupInfo(*mFirstGPUDevicePtr, CL_KERNEL_WAVE_SIZE_QCOM, &kernelWaveSize);
    return kernelWaveSize;
}

std::vector<uint32_t> OpenCLRuntime::getMaxWorkItemSizes() {
    int dims = 3;
    cl_int res = mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &dims);
    MNN_CHECK_CL_SUCCESS(res, "DeviceGetInfo");

    if(dims < 3) {
        std::vector<uint32_t> workItem(3, 8);
        return workItem;
    }
    
    cl::vector<cl::size_type> _workItems(dims, 1);
    res = mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &_workItems);
    MNN_CHECK_CL_SUCCESS(res, "DeviceGetInfo");
    
    std::vector<uint32_t> workItems(dims, 1);
    for (int i = 0; i < dims; ++i) {
        workItems[i] = _workItems[i];
    }
    return workItems;
}

double OpenCLRuntime::getCostTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res, "clEvent");
    mStartNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    mStopNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
    mKernelTime += (unsigned int)((mStopNanos - mStartNanos) / 1000.0);
    return (mStopNanos - mStartNanos) / 1000.0;
}

double OpenCLRuntime::getQueuedTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res, "clEvent");
    return (event->getProfilingInfo<CL_PROFILING_COMMAND_START>() - event->getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / 1000.0;
}

double OpenCLRuntime::getSubmitTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res, "clEvent");
    return (event->getProfilingInfo<CL_PROFILING_COMMAND_START>() - event->getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) / 1000.0;
}


std::pair<const void*, size_t> OpenCLRuntime::makeCache(void* tuneInfo) {
    auto tune = reinterpret_cast<MNN::OpenCL::TuneInfo*>(tuneInfo);
    std::unique_ptr<CacheT> cache(new CacheT);
    for (auto& p : tune->mInfos) {
        cache->tuned.emplace_back(std::move(p));
    }
    tune->mInfos.clear();
    // Get All program's binary
    for (auto& iter : mBuildProgramMap) {
        std::unique_ptr<ShaderT> pro(new ShaderT);
        auto program = iter.second;
        auto devicesNumber = program.getInfo<CL_PROGRAM_NUM_DEVICES>();
        auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
        auto binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
        if (binSizes.empty() || devices.empty()) {
            MNN_ERROR("Can't load binary, binarySize:%lu, deviceSize:%lu\n", binSizes.size(), devices.size());
            continue;
        }
        // Only use first one
        pro->program = std::get<0>(iter.first);
        pro->buildInfo = std::get<1>(iter.first);
        
        //MNN_PRINT("%s - %s - %s\n", pro->program.c_str(), pro->kernel.c_str(), pro->buildInfo.c_str());
        
        pro->buffer.resize(binSizes[0]);
        auto proRaw = program.get();
        auto c = pro->buffer.data();
        clGetProgramInfo(proRaw, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &c, nullptr);
        cache->programs.emplace_back(std::move(pro));
    }
    // Get All Autotuning cache
    for (auto& iter : mTunedLws) {
        std::unique_ptr<AutotuningT> tuning(new AutotuningT);
        tuning->gloablSize = iter.first.second;
        tuning->localSize = iter.second.first;
        tuning->timeCost = iter.second.second;
        tuning->key = iter.first.first;
        cache->tunings.emplace_back(std::move(tuning));
    }

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

bool OpenCLRuntime::setCache(std::pair<const void*, size_t> cache) {
    if (nullptr == cache.first) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return true;
    }

    mCacheOutsideSize = cache.second;
    mCacheOutside = cache.first;
    auto cacheBuffer = GetCache(cache.first);
    
    if(nullptr == cacheBuffer->programs() && nullptr == cacheBuffer->tunings()) {
        return false;
    }
    
    // Load Program
    if (nullptr != cacheBuffer->programs()) {
        auto programs = cacheBuffer->programs();
        for (int i=0; i<programs->size(); ++i) {
            auto shaderInfo = programs->GetAs<Shader>(i);
            if (nullptr == shaderInfo->program()|| nullptr == shaderInfo->buildInfo() || nullptr == shaderInfo->buffer()) {
                MNN_ERROR("Invalid Cache\n");
                return false;
            }
            auto program = shaderInfo->program()->str();
            // Builder Info
            std::string buildinfo = shaderInfo->buildInfo()->str();
            
            auto buffer = shaderInfo->buffer()->data();
            size_t bufferSize = shaderInfo->buffer()->size();
            auto deviceId = mFirstGPUDevicePtr->get();
            auto programRaw = clCreateProgramWithBinary(context().get(), 1, &deviceId, &bufferSize, (const unsigned char**)(&buffer), nullptr, nullptr);
            if (!programRaw) {
                MNN_ERROR("Can't load %s - %s load program\n", program.c_str(), buildinfo.c_str());
                return false;
            }
            auto pro = cl::Program(programRaw);
            auto res = buildProgram(buildinfo, &pro);
            if (!res) {
                MNN_ERROR("Can't build %s - %s load program\n", program.c_str(), buildinfo.c_str());
                return false;
            }
            mBuildProgramMap.insert(std::make_pair(std::make_tuple(program, buildinfo), pro));
        }
    }

    // Load Auto Tuning Info
    if (nullptr != cacheBuffer->tunings()) {
        auto tuningInfo = cacheBuffer->tunings();
        for (int i=0; i<tuningInfo->size(); ++i) {
            auto tun = tuningInfo->GetAs<Autotuning>(i);
            if (nullptr == tun->gloablSize() || nullptr == tun->localSize() || nullptr == tun->key()) {
                MNN_ERROR("Error tunning info\n");
                return false;
            }
            std::vector<uint32_t> glo(tun->gloablSize()->size());
            for (int v=0; v<glo.size(); ++v) {
                glo[v] = tun->gloablSize()->data()[v];
            }
            std::vector<uint32_t> loc(tun->localSize()->size());
            for (int v=0; v<loc.size(); ++v) {
                loc[v] = tun->localSize()->data()[v];
            }
            uint32_t cost = tun->timeCost();
            mTunedLws.insert(std::make_pair(std::make_pair(tun->key()->str(), glo), std::make_pair(loc, cost)));
            mTuneLws[tun->key()->str()].push_back(std::make_pair(glo, std::make_pair(loc, cost)));
        }
    }
    return true;
}

void OpenCLRuntime::clearRecord(){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue && mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = mCommandQueuePtr->EnqueueRecordingQCOM(mRecordings[i], 0, nullptr, 0, nullptr,
                  0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            MNN_CHECK_CL_SUCCESS(res, "EnqueueRecordingQCOM");
        }
        mCommandQueuePtr->finish();
        mRecordings.clear();
    }
#endif
}

void OpenCLRuntime::enqeueRecord(){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue && !mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = mCommandQueuePtr->EnqueueRecordingQCOM(mRecordings[i], 0, nullptr, 0, nullptr,
                  0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            MNN_CHECK_CL_SUCCESS(res, "EnqueueRecordingQCOM");
        }
        mCommandQueuePtr->finish();
    }
#endif
}

void OpenCLRuntime::endRecord(){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue  && !mDevideOpRecord){
        if(!mRecordings.empty()){
            cl_int res = clEndRecordingQCOM(mRecordings.back());
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
        }
    }
#endif
}

void OpenCLRuntime::releaseRecord(){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue  && !mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = clReleaseRecordingQCOM(mRecordings[i]);
            MNN_CHECK_CL_SUCCESS(res, "clReleaseRecordingQCOM");
        }
        mRecordings.clear();
    }
#endif
}

void OpenCLRuntime::printEventTime(){
#ifdef ENABLE_OPENCL_TIME_PROFILER
    if(mEvents.empty()){
        return;
    }
    for(int i = 0; i < mEvents.size(); ++i){
        auto event = &mEvents[i].second;
        cl_int res = event->wait();
        MNN_CHECK_CL_SUCCESS(res, "clEvent");
        auto StartNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto StopNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto kernel_time = (unsigned int)((StopNanos - StartNanos) / 1000.0);
        mKernelTime += kernel_time;
        MNN_PRINT("kernel time = %d    us %s\n", kernel_time, mEvents[i].first.c_str());
    }
    mEvents.clear();
    MNN_PRINT("total kernel time = %d  us\n", mKernelTime);
#endif
}
} // namespace MNN
