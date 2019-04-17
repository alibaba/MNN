//
//  CPUTile.hpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTile_hpp
#define CPUTile_hpp

#include "Execution.hpp"

namespace MNN {
class CPUTile : public Execution {
public:
    CPUTile(Backend *b, const MNN::Op *op);
    virtual ~CPUTile() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
} // namespace MNN

#endif /* CPUTile_hpp */
