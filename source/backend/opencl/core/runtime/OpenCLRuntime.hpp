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

#include <sstream>
#include <string>
#include <vector>
#include "Macro.h"
#include "Type_generated.h"
#include "core/runtime/OpenCLWrapper.hpp"

namespace MNN {

enum GpuType { MALI = 0, ADRENO = 1, OTHER = 2 };

class OpenCLRuntime {
public:
    OpenCLRuntime(bool permitFloat16);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    ::cl::Context &context();
    ::cl::CommandQueue &commandQueue();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t maxFreq() const;
    uint64_t getMaxWorkGroupSize(const ::cl::Kernel &kernel);
    GpuType getGpuType();
    uint64_t maxAllocSize() const;

    ::cl::Kernel buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const;

private:
    bool loadProgram(const std::string &programName, cl::Program *program);
    bool buildProgram(const std::string &buildOptionsStr, cl::Program *program);
    bool getDeviceSupportsExtension(const cl::Device &device, const char *extensionName);

private:
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
    std::map<std::string, ::cl::Program> mBuildProgramMap;
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq;
    uint32_t mMaxMemAllocSize;
    bool mIsSupportedFP16     = false;
    GpuType mGpuType;
    std::string mDefaultBuildParams;
    bool mIsCreateError{false}; 
};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */
