//
//  CPUIndexedLinear.hpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUIndexedLinear_hpp
#define CPUIndexedLinear_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "core/Execution.hpp"
#include <memory>
#include <vector>

namespace MNN {
class CPUIndexedLinear : public Execution {
public:
    CPUIndexedLinear(Backend* backend, const Op* op);
    ~CPUIndexedLinear() override = default;
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* backend, const Op* op, Execution** dst) override;

private:
    struct Resource;
    struct Tile {
        int expert;
        int firstRoute;
        int rows;
    };
    CPUIndexedLinear(Backend* backend, std::shared_ptr<Resource> resource, int inputMode, bool fuseSwiGLU);
    std::shared_ptr<Resource> mResource;
    int mInputMode = 0;
    bool mFuseSwiGLU = false;
    int mTopK = 0;
    int mTokens = 0;
    int mThreads = 1;
    int mChannelTile = 0;
    int mInputTileBytes = 0;
    int mScratchBytes = 0;
    int mOutputTileBytes = 0;
    int mQuantInfoBytes = 0;
    Tensor mPackedInput;
    Tensor mScratch;
    Tensor mPackedOutput;
    Tensor mQuantInfo;
    Tensor mAccumulator;
    std::vector<int> mOffsets;
    std::vector<int> mRouteOrder;
    std::vector<Tile> mTiles;
};
} // namespace MNN
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif
