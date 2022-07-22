//
//  CPUSvd.hpp
//  MNN
//
//  Created by MNN on 2022/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSvd_hpp
#define CPUSvd_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUSvd : public Execution {
public:
    CPUSvd(Backend *backend) : Execution(backend) {
        // Do nothing
    }
    virtual ~CPUSvd() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int mRow, mCol;
    std::shared_ptr<Tensor> mAt;
};

} // namespace MNN

#endif /* CPUSvd.hpp */
