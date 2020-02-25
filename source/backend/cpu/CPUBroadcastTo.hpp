//
//  CPUBroadcastTo.hpp
//  MNN
//
//  Created by MNN on 2019/12/2.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBroadcastTo_hpp
#define CPUBroadcastTo_hpp


#include "core/Execution.hpp"

namespace MNN {

class CPUBroadcastTo : public Execution{
public:
    CPUBroadcastTo(Backend *b):Execution(b){
    }
    virtual ~CPUBroadcastTo() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUBroadcastTo_hpp */
