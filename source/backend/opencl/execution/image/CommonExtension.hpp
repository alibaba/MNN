//
//  CommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonExtension_hpp
#define CommonExtension_hpp
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
namespace MNN {
namespace OpenCL {

class CommonExtension {
public:
    CommonExtension() = default;
    virtual ~CommonExtension(){
        if(mRecording != NULL){
#ifdef MNN_USE_LIB_WRAPPER
            clReleaseRecordingQCOM(mRecording);
#endif
        }
    }
    cl_recording_qcom mRecording{NULL};
};
} // namespace OpenCL
} // namespace MNN
#endif /* CommonExtension_hpp */
