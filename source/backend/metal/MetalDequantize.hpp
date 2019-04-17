//
//  MetalDequantize.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalDequantize_hpp
#define MetalDequantize_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalDequantize : public Execution {
public:
    MetalDequantize(Backend *backend, const Dequantize *dq);
    virtual ~MetalDequantize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode onTFLite(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onMinCombined(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onMinFirst(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onScaled(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    ModeFormat mModeFormat;
    DataType mType;
    QuantizeMode mMode;
};
} // namespace MNN

#endif /* MNN_METAL_ENABLED */
#endif /* MetalDequantize_hpp */
