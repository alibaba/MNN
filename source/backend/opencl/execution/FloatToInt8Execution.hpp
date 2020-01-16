//
//  FloatToInt8Execution.hpp
//  MNN
//
//  Created by MNN on 2019/6/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FloatToInt8Execution_hpp
#define FloatToInt8Execution_hpp

#include "CommonExecution.hpp"
#include <MNN_generated.h>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class FloatToInt8Execution : public Execution {
public:
    FloatToInt8Execution(Backend *backend, const MNN::Op *param);
    virtual ~FloatToInt8Execution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScales;
    std::shared_ptr<cl::Buffer> mScaleBuffer;
    OpenCLBackend *mOpenCLBackend;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
};
}
} // namespace MNN

#endif /* FloatToInt8Execution_hpp */
