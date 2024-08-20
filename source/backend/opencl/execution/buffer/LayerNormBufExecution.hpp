//
//  LayerNormBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef LayerNormBufExecution_hpp
#define LayerNormBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LayerNormBufExecution : public CommonExecution {
public:
    LayerNormBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~LayerNormBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    int getLocalSize(int size, int maxGroupSize);
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    OpenCLBackend *mOpenCLBackend;
    int axis_size = 0;
    int group_ = 1;
    float epsilon_ = 0.001;
    bool RMSNorm = false;

    std::shared_ptr<cl::Buffer> mGammaBuffer;
    std::shared_ptr<cl::Buffer> mBetaBuffer;
    std::shared_ptr<Tensor> mInputPlain, mOutputPlain;
    bool has_gamma_beta_ = false;
    uint32_t mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LayerNormBufExecution_hpp */


#endif /* MNN_OPENCL_BUFFER_CLOSED */
