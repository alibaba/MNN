//
//  DeconvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvExecution_hpp
#define DeconvExecution_hpp

#include "ConvExecution.hpp"
namespace MNN {
namespace OpenCL {

class DeconvExecution : public ConvCommonExecution, public CommonExecution {
public:
    DeconvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    DeconvExecution(std::shared_ptr<ConvResource> resource, const Op* op, Backend* backend);
    virtual ~DeconvExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    std::vector<int> mPaddings{0, 0};
};

} // namespace OpenCL
} // namespace MNN
#endif /* DeconvExecution_hpp */
