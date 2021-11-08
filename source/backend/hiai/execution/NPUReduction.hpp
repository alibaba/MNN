//
//  NPUReduction.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUReduction_HPP
#define NPUDEMO_NPUReduction_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUReduction : public NPUCommonExecution {
public:
    NPUReduction(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUReduction() = default;

private:
    vector<int64_t> convertAxis(vector<int64_t> origAxis, Tensor * input);
    ge::op::Const mConstAxis;
 
    int axisMap[8][4] = {{0},{1},                   //mNCHW1d,mNHWC1d
                         {0, 1},{2, 1},             //mNCHW2d,mNHWC2d
                         {0, 1, 2},{2, 3, 1},       //mNCHW3d,mNHWC3d
                         {0, 1, 2, 3},{0, 2, 3, 1}, //mNCHW4d,mNHWC4d
                         };
};
} // namespace MNN

#endif // NPUDEMO_NPUReduction_HPP
