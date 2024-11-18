//
//  CPUUnique.cpp
//  MNN
//
//  Created by MNN on 2019/6/11.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include <unordered_map>
#include "backend/cpu/CPUUnique.hpp"
namespace MNN {

ErrorCode CPUUnique::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto outputPtr  = output->host<int32_t>();
    int outputSize  = 0;
    auto eleSize = input->elementSize();
    if (outputs.size() <= 2) {
        std::unordered_map<int, int> idx_map;
        for (int i = 0; i < eleSize; ++i) {
            auto value = input->host<int32_t>()[i];
            if (idx_map.find(value) == idx_map.end()) {
                outputPtr[outputSize] = value;
                idx_map[value] = outputSize++;
            }
        }
        outputSize  = 0;
        if (outputs.size() > 1) {
            auto outIdx = outputs[1]->host<int>();
            for (int i = 0; i < eleSize; ++i) {
                auto value = input->host<int32_t>()[i];
                if (idx_map.find(value) == idx_map.end()) {
                    outIdx[outputSize] = idx_map[value];
                    outputSize++;
                }
            }
        }
    } else {
        MNN_ASSERT(4 == outputs.size());
        auto outIdx = outputs[1]->host<int>();
        auto reverseIdx = outputs[2]->host<int>();
        auto count = outputs[3]->host<int>();
        ::memset(count, 0, outputs[3]->usize());
        std::unordered_map<int, int> idx_map;
        for (int i = 0; i < eleSize; ++i) {
            auto value = input->host<int32_t>()[i];
            auto iter = idx_map.find(value);
            int pos;
            if (iter == idx_map.end()) {
                outputPtr[outputSize] = value;
                outIdx[outputSize] = i;
                pos = outputSize;
                idx_map[value] = outputSize++;
            } else {
                pos = iter->second;
            }
            reverseIdx[i] = pos;
        }
    }
    return NO_ERROR;
}
class CPUUniqueCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUUnique(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUUniqueCreator, OpType_Unique);

}; // namespace MNN
