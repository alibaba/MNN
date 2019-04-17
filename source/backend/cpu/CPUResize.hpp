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

namespace MNN {

void CPUResizeCubicC4(halide_buffer_t &input, halide_buffer_t &output);
void CPUResizeBilinearC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale);
void CPUReiseNearstneighborC4(halide_buffer_t &input, halide_buffer_t &output, float wScale, float hScale);

class CPUResize : public Execution {
public:
    CPUResize(Backend *backend, float xScale, float yScale);
    virtual ~CPUResize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    float mXScale;
    float mYScale;
};

} // namespace MNN

#endif /* CPUResize_hpp */
