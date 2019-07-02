//
//  OpenCLRuntime.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/runtime/OpenCLRuntime.hpp"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Macro.h"
//#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"
namespace MNN {

extern const std::map<std::string, std::vector<unsigned char>> OpenCLProgramMap;

bool OpenCLRuntime::getDeviceSupportsExtension(const cl::Device &device, const char *extensionName) {
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto pos               = extensions.find(extensionName);
    return (pos != std::string::npos);
}

GpuType OpenCLRuntime::getGpuType() {
    return mGpuType;
}

bool OpenCLRuntime::isCreateError() const {
    return mIsCreateError;
}

OpenCLRuntime::OpenCLRuntime(bool permitFloat16) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start OpenCLRuntime !\n");
#endif
    mDefaultBuildParams = " -cl-mad-enable";
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() > 0){
        cl::Platform::setDefault(platforms[0]);
        std::vector<cl::Device> gpuDevices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);
        
        if(1 <= gpuDevices.size()){
            mFirstGPUDevicePtr              = std::make_shared<cl::Device>(gpuDevices[0]);
            const std::string deviceName    = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NAME>();
            const std::string deviceVersion = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VERSION>();

            cl_command_queue_properties properties = 0;

        #ifdef ENABLE_OPENCL_TURNING_PROFILER
            properties |= CL_QUEUE_PROFILING_ENABLE;
        #endif
            cl_int err;
            // if device is QUALCOMM's and version is 2.0 , set spacial optimized param
            if (deviceName == "QUALCOMM Adreno(TM)" && deviceVersion.substr(0, deviceVersion.find('2')) == "OpenCL ") {
                mGpuType = ADRENO;
            } else if (deviceName.find("Mali") != std::string::npos) {
                mGpuType = MALI;
            } else {
                mGpuType = OTHER;
            }
            mContext = std::shared_ptr<cl::Context>(new cl::Context({*mFirstGPUDevicePtr}, nullptr, nullptr, nullptr, &err));
            MNN_CHECK_CL_SUCCESS(err);

            mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &err);
            MNN_CHECK_CL_SUCCESS(err);

            mFirstGPUDevicePtr->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &mGPUGlobalMemeryCacheSize);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &mGPUComputeUnits);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &mMaxFreq);
            cl_device_fp_config fpConfig;
            auto success = mFirstGPUDevicePtr->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fpConfig);
            mIsSupportedFP16     = CL_SUCCESS == success && fpConfig > 0;
            mIsSupportedFP16     = mIsSupportedFP16 && permitFloat16;
        }else{
            mIsCreateError = true;
            MNN_ASSERT(1 <= gpuDevices.size());
        }
    }else{
        mIsCreateError = true;
        MNN_ASSERT(platforms.size() > 0);
    }
}

OpenCLRuntime::~OpenCLRuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~OpenCLRuntime !\n");
#endif
    mBuildProgramMap.clear();
    mCommandQueuePtr.reset();
    mContext.reset();
    mFirstGPUDevicePtr.reset();
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~OpenCLRuntime !\n");
#endif
}

std::vector<size_t> OpenCLRuntime::getMaxImage2DSize() {
    size_t max_height, max_width;
    cl_int err = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
    MNN_CHECK_CL_SUCCESS(err);
    err = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
    MNN_CHECK_CL_SUCCESS(err);
    return {max_height, max_width};
}

bool OpenCLRuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}

cl::Context &OpenCLRuntime::context() {
    return *mContext;
}

cl::CommandQueue &OpenCLRuntime::commandQueue() {
    return *mCommandQueuePtr;
}

uint64_t OpenCLRuntime::deviceGlobalMemeryCacheSize() const {
    return mGPUGlobalMemeryCacheSize;
}

uint32_t OpenCLRuntime::deviceComputeUnits() const {
    return mGPUComputeUnits;
}

uint32_t OpenCLRuntime::maxFreq() const {
    return mMaxFreq;
}

uint64_t OpenCLRuntime::maxAllocSize() const {
    return mMaxMemAllocSize;
}

bool OpenCLRuntime::loadProgram(const std::string &programName, cl::Program *program) {
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
        MNN_PRINT("Build program failed ! \n");
        return false;
    }
    return true;
}

cl::Kernel OpenCLRuntime::buildKernel(const std::string &programName, const std::string &kernelName,
                                      const std::set<std::string> &buildOptions) {
    std::string buildOptionsStr;
    if (mIsSupportedFP16) {
        buildOptionsStr = "-DFLOAT=half -DFLOAT4=half4 -DRI_F=read_imageh -DWI_F=write_imageh";
    } else {
        buildOptionsStr = "-DFLOAT=float -DFLOAT4=float4 -DRI_F=read_imagef -DWI_F=write_imagef";
    }
    for (auto &option : buildOptions) {
        buildOptionsStr += " " + option;
    }
    buildOptionsStr += mDefaultBuildParams;
    std::string buildProgramKey = programName + buildOptionsStr;

    auto buildProgramInter = mBuildProgramMap.find(buildProgramKey);
    cl::Program program;
    if (buildProgramInter != mBuildProgramMap.end()) {
        program = buildProgramInter->second;
    } else {
        this->loadProgram(programName, &program);
        auto status = this->buildProgram(buildOptionsStr, &program);
        if (!status) {
            FUNC_PRINT_ALL(programName.c_str(), s);
        }
        mBuildProgramMap.emplace(buildProgramKey, program);
    }

    cl_int err;
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str(), &err);
    MNN_CHECK_CL_SUCCESS(err);
    return kernel;
}

uint64_t OpenCLRuntime::getMaxWorkGroupSize(const cl::Kernel &kernel) {
    uint64_t maxWorkGroupSize = 0;
    MNN_ASSERT(0 == kernel.getWorkGroupInfo(*mFirstGPUDevicePtr, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    return maxWorkGroupSize;
}

} // namespace MNN
