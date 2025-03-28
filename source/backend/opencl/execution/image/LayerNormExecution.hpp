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
struct LayernormResource {
    std::shared_ptr<cl::Buffer> mGammaBuffer;
    std::shared_ptr<cl::Buffer> mBetaBuffer;
    bool has_gamma_beta_ = false;
    uint32_t mMaxWorkGroupSize;
    int axis_size ;
    int group_ ;
    float epsilon_;
    bool RMSNorm;
};
class LayerNormExecution : public CommonExecution {
public:
    LayerNormExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    LayerNormExecution(std::shared_ptr<LayernormResource> resource, const Op* op, Backend* backend);
    virtual ~LayerNormExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    int getLocalSize(int size, int maxGroupSize);
    OpenCLBackend *mOpenCLBackend;
    std::shared_ptr<LayernormResource> mResource;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LayerNormExecution_hpp */
