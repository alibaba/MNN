//
//  CPUScatterNd.hpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUScatterNd_hpp
#define CPUScatterNd_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUScatterNd : public Execution {
public:
    CPUScatterNd(Backend *bn):Execution(bn){
    }
    virtual ~CPUScatterNd() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUScatterNd_hpp */
