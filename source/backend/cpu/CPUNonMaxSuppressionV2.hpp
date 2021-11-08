//
//  CPUNonMaxSuppressionV2.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUNonMaxSuppressionV2_hpp
#define CPUNonMaxSuppressionV2_hpp

#include "core/Execution.hpp"

namespace MNN {

/**
 * @brief apply non_max_suppression, output the selected boxes index
 * @param decodedBoxes : Tensor, shape is [num_boxes, 4], where 4 represent [ymin, xmin, ymax, xmax]
 * @param scores : float*, length is [num_boxes]
 * @param maxDetections : int output maxDetections boxes
 * @param iouThreshold : float
 * @param scoreThreshold : float
 * @param selected : std::vector<int32_t>*
 */
void NonMaxSuppressionSingleClasssImpl(const Tensor* decodedBoxes, const float* scores, int maxDetections, float iouThreshold, float scoreThreshold, std::vector<int>* selected);


class CPUNonMaxSuppressionV2 : public Execution {
public:
    CPUNonMaxSuppressionV2(Backend *backend, const Op *op);
    virtual ~CPUNonMaxSuppressionV2() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUNonMaxSuppressionV2_hpp */
