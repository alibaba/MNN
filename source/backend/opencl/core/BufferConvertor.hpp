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

bool convertNCHWBufferToNC4HW4Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                     OpenCLRuntime *runtime, bool needInpTrans = false, bool needWait = false, bool svmFlag = false);

bool convertNHWCBufferToNC4HW4Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                     OpenCLRuntime *runtime, bool needInpTrans = false, bool needWait = false, bool svmFlag = false);

enum TransType {InpTrans = 0, OutTrans = 1, NoTrans = 2};
bool convertNC4HW4BufferToNC4HW4Buffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                       OpenCLRuntime *runtime, TransType formatTrans = NoTrans, bool needWait = false, bool svmFlag = false, bool srcswap = false, bool dstswap = false);

bool convertNC4HW4BufferToNCHWBuffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                     OpenCLRuntime *runtime, bool needOutTrans = false, bool needWait = false, bool svmFlag = false);

bool convertNC4HW4BufferToNHWCBuffer(const Tensor *input, Tensor *output, cl::Kernel &convertBufferKernel,
                                     OpenCLRuntime *runtime, bool needOutTrans = false, bool needWait = false, bool svmFlag = false);

class BufferConvertor {
public:
    explicit BufferConvertor(OpenCLRuntime *opencl_runtime) : mOpenCLRuntime(opencl_runtime) {
    }
    bool convertToNC4HW4Buffer(const Tensor *input, const OpenCLBufferFormat type, Tensor *output,
                               bool needTrans, bool needWait = false);

private:
    OpenCLRuntime *mOpenCLRuntime;
    cl::Kernel mImageToBufferKernel;
    std::string mImageToBufferKernelName;
    cl::Kernel mBufferToImageKernel;
    std::string mBufferToImageKernelName;
};

} // namespace OpenCL
} // namespace MNN
#endif  /* BufferConvertor_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
