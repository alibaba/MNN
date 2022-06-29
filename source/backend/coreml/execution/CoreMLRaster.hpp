//
//  CoreMLRaster.hpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLRASTER_HPP
#define MNN_COREMLRASTER_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLRaster : public CoreMLCommonExecution {
public:
    CoreMLRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLRaster() = default;
private:
    bool rasterOptimization(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    bool buildPermute(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
    bool buildReshape(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
    bool buildPad(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
    bool buildCrop(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
    bool buildSlice(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
    bool buildDepthToSpace(CoreML__Specification__NeuralNetworkLayer* layer, const Tensor* input, const Tensor* output);
};
} // namespace MNN

#endif // MNN_COREMLRASTER_HPP
