//
//  CPUResize.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUResize_hpp
#define CPUResize_hpp

#include "Execution.hpp"
#include "AutoStorage.h"

namespace MNN {

void CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output);
void CPUResizeBilinearC4(halide_buffer_t &input, halide_buffer_t &output,
                         const int* widthPosition, const float* widthFactor,
                         const int* heightPosition, const float* heightFactor,
                         float* lineBuffer, int threadNumber);
void CPUReiseNearstneighborC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale);

class CPUResize : public Execution {
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
