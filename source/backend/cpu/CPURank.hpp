//
//  CPURank.hpp
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURank_hpp
#define CPURank_hpp

#include "Execution.hpp"

namespace MNN {
class CPURank : public Execution {
public:
    CPURank(Backend *backend);
    virtual ~CPURank() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPURank_hpp */
