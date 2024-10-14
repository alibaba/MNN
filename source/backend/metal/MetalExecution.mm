//
//  MetalExecution.mm
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MetalExecution.hpp"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
MetalExecution::MetalExecution(Backend *backend) : Execution(backend) {
    // Do nothing
}

ErrorCode MetalExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    auto func = [=](){
        auto encoder           = backend->encoder_for_net();
        this->onEncode(inputs, outputs, encoder);
        if(backend->isCmdBufferCommit()) {
            backend->flushEncoder();
            backend->commit_net();
        }
    };
    func();

    return NO_ERROR;
}


};
#endif
