//
//  CommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonExecution_hpp
#define CommonExecution_hpp
#include "core/Execution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
namespace MNN {
namespace OpenCL {

// Check kernel after buildKernel in constructor; set mValid=false on failure
#define OPENCL_CHECK_KERNEL_CTOR(kernel)  \
    if (kernel == nullptr) {              \
        mValid = false;                   \
        return;                           \
    }

// Check kernel after buildKernel in onResize/onEncode; return NOT_SUPPORT on failure
#define OPENCL_CHECK_KERNEL(kernel)       \
    if (kernel == nullptr) {              \
        return NOT_SUPPORT;               \
    }

// Wrap 'new Execution' in creator: validate then return (or nullptr on failure)
inline Execution* checkExeValid(Execution* exe) {
    if (exe != nullptr && !exe->valid()) {
        delete exe;
        return nullptr;
    }
    return exe;
}
#define OPENCL_CREATOR_CHECK(p) return checkExeValid(p)

// Check onAcquireBuffer in constructor; set mValid=false on failure
#define OPENCL_CHECK_ALLOC_CTOR(expr)     \
    if (!(expr)) {                        \
        mValid = false;                   \
        return;                           \
    }

// Check onAcquireBuffer in onResize/onEncode; return OUT_OF_MEMORY on failure
#define OPENCL_CHECK_ALLOC(expr)          \
    if (!(expr)) {                        \
        return OUT_OF_MEMORY;             \
    }

// Check pointer (e.g. ConvolutionCommon::load) in constructor; set mValid=false on failure
#define OPENCL_CHECK_PTR_CTOR(ptr)        \
    if (ptr == nullptr) {                 \
        mValid = false;                   \
        return;                           \
    }

struct Unit {
    std::shared_ptr<KernelWrap> kernel;
    cl::NDRange globalWorkSize;
    cl::NDRange localWorkSize;
};

class CommonExecution : public Execution {
public:
    CommonExecution(Backend *backend, const MNN::Op *Op);
    virtual ~CommonExecution(){
        if(mRecording != NULL){
#ifdef MNN_USE_LIB_WRAPPER
            clReleaseRecordingQCOM(mRecording);
#endif
        }
    }
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    std::vector<Unit> mUnits;
    const MNN::Op *mOp;
    OpType mOpType;
    cl_recording_qcom mRecording{NULL};
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
};
} // namespace OpenCL
} // namespace MNN
#endif /* CommonExecution_hpp */
