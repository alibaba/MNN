//
//  InitNet.hpp
//  MNN
//
//  Created by MNN on 2018/09/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/Schedule.hpp"

namespace MNN {
MNN_PUBLIC bool needComputeOp(const Op* op);
MNN_PUBLIC bool initConstTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net, Backend* defaultBackend, bool netHold, ErrorCode& code);
// init Tensors by net
bool initTensors(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net);
// init Pipeline Infos by oplist and tensors
void initPipelineInfosFromOps(std::vector<Schedule::PipelineInfo>& infos, std::vector<const Op*>& ops, const std::vector<std::shared_ptr<Tensor>>& allTensors);
// set input and output for allTensors by ops info
void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors, const std::vector<const Op*>& ops, bool isStatic = false);
// init Pipeline Infos by net and tensors, set input and output info
void initPipelineInfosFromNet(std::vector<Schedule::PipelineInfo>& infos, const Net* net, std::vector<std::shared_ptr<Tensor>>& allTensors);
} // namespace MNN
