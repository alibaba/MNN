//
//  IndexedLinearCommon.hpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef IndexedLinearCommon_hpp
#define IndexedLinearCommon_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "core/ConvolutionCommon.hpp"
#include <vector>

namespace MNN {

// Decode the existing weight format without introducing child executions.
class MNN_PUBLIC IndexedLinearCommon {
public:
    static bool dimensions(const Op* op, int& experts, int& inputChannels, int& outputChannels);
    // dimensions returns the physical projection width (2D for fused SwiGLU, whose output is D).
    // Fused banks are loaded as interleaved gate/up rows, including bias and quantization metadata.
    // Quantized weights stay packed (W4: high nibble first, signed value = nibble - 8).
    // alpha uses the existing signed-weight convention: asymmetric [offset, scale].
    // The returned quan pointer belongs to op; consumers must copy any metadata they retain.
    static std::shared_ptr<ConvolutionCommon::Int8Common> loadLinear(const Op* op, int expert, Backend* backend,
                                                                     bool forceFloat = false);
    static bool loadBias(const Op* op, int expert, std::vector<float>& bias);
};

} // namespace MNN
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif
