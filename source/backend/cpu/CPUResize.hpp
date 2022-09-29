//
//  CPUResize.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUResize_hpp
#define CPUResize_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class CPUResizeCommon : public Execution {
public:
    CPUResizeCommon(Backend *backend) : Execution(backend) {
        // Do nothing
    }
    virtual ~CPUResizeCommon()                                                                             = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)  = 0;

    void CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float wOffset, float hOffset);
    void CPUResizeBilinearC4(halide_buffer_t &input, halide_buffer_t &output, const int *widthPosition,
                             const float *widthFactor, const int *heightPosition, const float *heightFactor,
                             float *lineBuffer, int threadNumber);
    void CPUResizeNearestneighborC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float wOffset = 0.f, float hOffset = 0.f);
    void CPUResizeNearestneighborRoundC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float wOffset = 0.f, float hOffset = 0.f);

    void CPUResizeNearestneighbor3DC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float dScale,
                                      float wOffset = 0.f, float hOffset = 0.f, float dOffset = 0.f);
    void CPUResizeNearestneighbor3DRoundC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale, float dScale,
                                           float wOffset = 0.f, float hOffset = 0.f, float dOffset = 0.f);
};

} // namespace MNN

#endif /* CPUResize_hpp */
