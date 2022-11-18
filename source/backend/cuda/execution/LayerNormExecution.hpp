//
//  LayerNormExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LayerNormExecution_hpp
#define LayerNormExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class LayerNormExecution : public Execution {
public:
    LayerNormExecution(const LayerNorm* layer_norm_param, Backend *backend);
    virtual ~LayerNormExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDARuntime *mRuntime;
    void *mDeviceGamma = nullptr;
    void *mDeviceBeta = nullptr;

    std::vector<int> mAxises;
    int mInside = 1;
    int mOutside = 1;

    float mEps = 0.001;
    int mGroup = 1;

    std::unique_ptr<Tensor> mGammaTensor;
    std::unique_ptr<Tensor> mBetaTensor;

    std::shared_ptr<Tensor> LayerNormTensor;
    std::shared_ptr<Tensor> biasTensor;

};

} // namespace CUDA
} // namespace MNN
#endif /* LayerNormExecution_hpp */
