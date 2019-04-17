//
//  MetalSpaceToBatchND.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalSpaceToBatchND_hpp
#define MetalSpaceToBatchND_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalSpaceToBatchND : public Execution {
public:
    MetalSpaceToBatchND(Backend *backend, int blockHeight, int blockWidth, int paddingTop, int paddingLeft);
    virtual ~MetalSpaceToBatchND() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mBlockHeight;
    int mBlockWidth;
    int mPaddingTop;
    int mPaddingLeft;
    id<MTLBuffer> mConst;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalSpaceToBatchND_hpp */
