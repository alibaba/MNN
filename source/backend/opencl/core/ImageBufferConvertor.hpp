//
//  ImageBufferConvertor.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageBufferConvertor_hpp
#define ImageBufferConvertor_hpp

#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {
/**
 * @brief convert nchw buffer to image.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertNCHWBufferToImage(const Tensor *input, Tensor *output, 
                              OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);
/**
 * @brief convert nhwc buffer to image.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertNHWCBufferToImage(const Tensor *input, Tensor *output, 
                              OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);
/**
 * @brief convert image to nchw buffer.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertImageToNCHWBuffer(const Tensor *input, Tensor *output, 
                              OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);
/**
 * @brief convert nc/4hwc%4 buffer to image.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertNC4HW4BufferToImage(const Tensor *input, Tensor *output, 
                                OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);

/**
 * @brief convert image to nc/4hwc%4 buffer.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertImageToNC4HW4Buffer(const Tensor *input, Tensor *output, 
                                OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);
/**
 * @brief convert image to nhwc buffer.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertImageToNHWCBuffer(const Tensor *input, Tensor *output, 
                              OpenCLRuntime *runtime, bool needWait = false, bool svmFlag = false);

class ImageBufferConvertor {
public:
    explicit ImageBufferConvertor(OpenCLRuntime *opencl_runtime) : mOpenCLRuntime(opencl_runtime) {
    }
    bool convertImageToBuffer(const Tensor *input, const OpenCLBufferFormat type, Tensor *output,
                              bool needWait = false, bool svmFlag = false);
    bool convertBufferToImage(const Tensor *input, const OpenCLBufferFormat type, Tensor *output,
                              bool needWait = false, const std::string &buildOption = "");

private:
    OpenCLRuntime *mOpenCLRuntime;
    std::shared_ptr<KernelWrap> mImageToBufferKernel;
    std::string mImageToBufferKernelName;
    std::shared_ptr<KernelWrap> mBufferToImageKernel;
    std::string mBufferToImageKernelName;
};

} // namespace OpenCL
} // namespace MNN
#endif  /* ImageBufferConvertor_hpp */
