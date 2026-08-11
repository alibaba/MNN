//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include "core/MusaBackend.hpp"
#include "core/ConvolutionCommon.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace MUSA {

class ConvExecution : public Execution {
public:
    ConvExecution(const MNN::Op* op, Backend* backend);
    virtual ~ConvExecution() = default;
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    
private:
    std::shared_ptr<ConvolutionCommon::Resource> mResource;
    ConvolutionCommon::Im2ColParameters mIm2ColParams;
    int mThreadNumber{1};
    bool mIsDepthWise{false};
    bool mIsConv1x1{false};
};

} // namespace MUSA
} // namespace MNN

#endif /* ConvExecution_hpp */
