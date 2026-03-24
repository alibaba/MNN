//
//  TopKV2BufExecution.hpp
//  MNN
//
//  OpenCL buffer-path implementation of TopKV2.
//

#ifndef TopKV2BufExecution_hpp
#define TopKV2BufExecution_hpp

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class TopKV2BufExecution : public CommonExecution {
public:
    TopKV2BufExecution(const MNN::Op *op, Backend *backend, int k);
    virtual ~TopKV2BufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    OpenCLBackend *mOpenCLBackend = nullptr;
    uint32_t mMaxWorkGroupSize = 0;
    std::vector<uint32_t> mGlobalWorkSize = {1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize  = {1, 1, 1};
    bool mLargest = true;
    int mNumRows = 0;
    int mK = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* TopKV2BufExecution_hpp */
