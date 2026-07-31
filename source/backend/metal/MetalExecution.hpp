//
//  MetalExecution.hpp
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalExecution_hpp
#define MetalExecution_hpp

#include "core/Execution.hpp"
#import "MetalDefine.h"
#import "MetalReplay.hpp"
#include <string>
#if MNN_METAL_ENABLED
namespace MNN {

class MetalExecution : public Execution {
public:
    MetalExecution(Backend *backend);
    virtual ~MetalExecution() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) = 0;

protected:
    // Encode-replay (see MetalReplay.hpp). Return false for ops whose encode
    // depends on per-token CPU state that buffer-binding replay can't capture
    // (e.g. attention's kv-length grids / param contents).
    virtual bool canRecordEncode() const {
        return true;
    }
    // Called before each replayed encode; ops with dynamic state patch their
    // recorded events here. Returning false falls back to normal onEncode.
    virtual bool onReplayUpdate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        return true;
    }
    std::vector<MetalReplayEvent> mReplayEvents;
    uint64_t mReplayKey = 0;
    int mReplayStable = 0;
    int mReplayState = 0; // 0 observing, 1 replaying, -1 banned
    // Consecutive replay failures; a recording that never survives more than a
    // couple of replays (e.g. an onReplayUpdate hook that always bails) is
    // banned after a few attempts so it stops paying the re-record cost.
    int mReplayFailCount = 0;
};
} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif
