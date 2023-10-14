//
//  CPUProposal.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUProposal_hpp
#define CPUProposal_hpp

#include <functional>
#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "core/BufferAllocator.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPUProposal : public Execution {
public:
    CPUProposal(Backend *backend, const Proposal *proposal);
    virtual ~CPUProposal() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Proposal *mProposal;
    AutoStorage<float> mAnchors;
    MemChunk mScoreBuffer;
};

} // namespace MNN

#endif /* CPUProposal_hpp */
