//
//  MetalTensorConverter.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalTensorConverter.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalTensorConverter::MetalTensorConverter(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalTensorConverter::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        backend->onCopyBuffer(inputs[0], outputs[0]);

        auto context = (__bridge MNNMetalContext *)backend->context();
        if(context.isCommitEachShader) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);
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
