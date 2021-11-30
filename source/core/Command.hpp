//
//  Command.hpp
//  MNN
//
//  Created by MNN on b'2020/07/28'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Command_hpp
#define Command_hpp
#include <MNN/Tensor.hpp>
#include "AutoStorage.h"
#include <string>
#include <memory>
namespace MNN {
struct Op;
class Execution;
class OperatorInfo;
struct Command : public RefCount {
    const Op* op;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    std::shared_ptr<BufferStorage> buffer;
    std::shared_ptr<Execution> execution;
    std::shared_ptr<Execution> executionOrigin;
    std::shared_ptr<OperatorInfo> info;
};
struct CommandBuffer {
    std::vector<SharedPtr<Command>> command;
    std::vector<std::shared_ptr<Tensor>> extras;
};
}; // namespace MNN
#endif
