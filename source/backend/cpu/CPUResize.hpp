//
//  CPUResize.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUResize_hpp
#define CPUResize_hpp

#include "AutoStorage.h"
#include "Execution.hpp"

namespace MNN {

class CPUResizeCommon : public Execution {
public:
    CPUResizeCommon(Backend *backend) : Execution(backend) {
    }
    virtual ~CPUResizeCommon()                                                                             = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)  = 0;

    void CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output);
    void CPUResizeBilinearC4(halide_buffer_t &input, halide_buffer_t &output, const int *widthPosition,
                             const float *widthFactor, const int *heightPosition, const float *heightFactor,
                             float *lineBuffer, int threadNumber);
    void CPUReiseNearstneighborC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale);
};

class CPUResize : public CPUResizeCommon {
public:
    CPUResize(Backend *backend, float xScale, float yScale);
    virtual ~CPUResize();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mWidthPosition;
    Tensor mWidthFactor;
    Tensor mHeightPosition;
    Tensor mHeightFactor;
    Tensor mLineBuffer;
    float mXScale;
    float mYScale;
};

} // namespace MNN

#endif /* CPUResize_hpp */
