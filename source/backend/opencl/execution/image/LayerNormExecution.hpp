//
//  LayerNormExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LayerNormExecution_hpp
#define LayerNormExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class LayerNormExecution : public CommonExecution {
public:
    LayerNormExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~LayerNormExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    int getLocalSize(int size, int maxGroupSize);
    OpenCLBackend *mOpenCLBackend;
    int axis_size = 0;
    int group_ = 1;
    bool RMSNorm = false;
    float epsilon_ = 0.001;

    std::shared_ptr<cl::Buffer> mGammaBuffer;
    std::shared_ptr<cl::Buffer> mBetaBuffer;
    bool has_gamma_beta_ = false;
    uint32_t mMaxWorkGroupSize;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LayerNormExecution_hpp */
