//
//  MetalConvolutionActivation.metal
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

typedef enum : int {
    None  = 0,
    ReLU  = 1,
    ReLU6 = 2,
} conv_activation_type;

inline ftype4 activate(ftype4 value, conv_activation_type type) {
    switch (type) {
        case ReLU:
            return max(value, 0);
        case ReLU6:
            return clamp(value, 0, 6);
        default: // None
            return value;
    }
}
