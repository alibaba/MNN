//
//  MetalTile.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalTile_hpp
#define MetalTile_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalTile : public Execution {
public:
    MetalTile(Backend *backend);
    virtual ~MetalTile() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalTile_hpp */
