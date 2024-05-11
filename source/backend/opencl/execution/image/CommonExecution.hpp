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
