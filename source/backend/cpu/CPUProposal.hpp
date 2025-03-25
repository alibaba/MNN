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

    struct ProposalCache {
        int32_t featStride;
        int32_t preNmsTopN;
        int32_t minSize;
        int32_t afterNmsTopN;
        float nmsThreshold;
    };
private:
    ProposalCache mCache;
    AutoStorage<float> mAnchors;
    MemChunk mScoreBuffer;
};

} // namespace MNN

#endif /* CPUProposal_hpp */
