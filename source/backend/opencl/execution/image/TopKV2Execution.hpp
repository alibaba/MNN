//
//  TopKV2Execution.hpp
//  MNN
//
//  OpenCL image-path implementation of TopKV2.
//

#ifndef TopKV2Execution_hpp
#define TopKV2Execution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class TopKV2Execution : public CommonExecution {
public:
    TopKV2Execution(const MNN::Op* op, Backend* backend, int k);
    virtual ~TopKV2Execution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    OpenCLBackend* mOpenCLBackend = nullptr;
    bool mLargest = true;
    int mNumRows = 0;
    int mK = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* TopKV2Execution_hpp */
