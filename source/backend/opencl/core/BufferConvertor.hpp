//
//  BufferConvertor.hpp
//  MNN
//
//  Created by MNN on 2020/09/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef BufferConvertor_hpp
#define BufferConvertor_hpp

#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

bool converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(const Tensor *input, Tensor *output, const std::string Name,
                                                    OpenCLRuntime *runtime, bool needInpTrans = false, bool needWait = false, bool svmFlag = false);

bool convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(const Tensor *input, Tensor *output, const std::string Name,
                                                    OpenCLRuntime *runtime, bool needOutTrans = false, bool needWait = false, bool svmFlag = false);

enum TransType {InpTrans = 0, OutTrans = 1, NoTrans = 2};

bool convertNC4HW4BufferToNC4HW4Buffer(const Tensor *input, Tensor *output,
                                       OpenCLRuntime *runtime, TransType formatTrans = NoTrans, bool needWait = false, bool svmFlag = false, bool srcswap = false, bool dstswap = false);

#ifdef MNN_SUPPORT_INTEL_SUBGROUP
bool convertNC4HW4BufferBetweenNC16HW16Buffer(const Tensor *input, Tensor *output, const std::string Name,
                                             OpenCLRuntime *runtime, TransType formatTrans = NoTrans, bool needWait = false,
                                             bool svmFlag = false, bool srcswap = false, bool dstswap = false);
#endif
                                       
class BufferConvertor {
public:
    explicit BufferConvertor(OpenCLRuntime *opencl_runtime) : mOpenCLRuntime(opencl_runtime) {
    }
    bool convertToNC4HW4Buffer(const Tensor *input, const OpenCLBufferFormat type, Tensor *output,
                               bool needTrans, bool needWait = false, bool lowMemory = false, int quantBit = 0);

private:
    OpenCLRuntime *mOpenCLRuntime;
    std::shared_ptr<KernelWrap> mBufferToImageKernel;
    std::string mBufferToImageKernelName;
};

} // namespace OpenCL
} // namespace MNN
#endif  /* BufferConvertor_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
