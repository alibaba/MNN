//
//  ReductionExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ReductionExecution_hpp
#define ReductionExecution_hpp

#include "core/Execution.hpp"
#include <MNN_generated.h>
#include <vector>
#include <string.h>
#include <unordered_set>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/execution/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ReductionExecution : public CommonExecution {
public:
    ReductionExecution(const MNN::Op* op, Backend* backend);
    virtual ~ReductionExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode prepareReduction2Dkernel(const std::vector<int> &inputArray, const Tensor *input, const Tensor *output);
    ErrorCode prepareReduction1Dkernel(const std::vector<int> &inputArray, const Tensor *input, const Tensor *output);
    ErrorCode generateReductionGWSLWS(const std::vector<uint32_t> &paramArray);
private:
    cl::Kernel mReduct2DKernel;
    cl::Kernel mReduct1DKernel;
    OpenCLBackend *mOpenCLBackend;
    MNN::DataType mdataType;
    std::string mReductType;
    std::unordered_set<int> mAxis;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1};
};

} // namespace OpenCL
} // namespace MNN
#endif /* ReductionExecution_hpp */
