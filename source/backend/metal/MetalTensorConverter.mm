//
//  MetalTensorConverter.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalTensorConverter.hpp"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalTensorConverter::MetalTensorConverter(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalTensorConverter::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    backend->onCopyBuffer(inputs[0], outputs[0]);
    return NO_ERROR;
}

class MetalTensorConverterCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalTensorConverter(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalTensorConverterCreator, OpType_ConvertTensor);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
