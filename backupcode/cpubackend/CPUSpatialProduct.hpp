//
//  CPUSpatialProduct.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSpatialProduct_hpp
#define CPUSpatialProduct_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUSpatialProduct : public Execution {
public:
    CPUSpatialProduct(Backend *b);
    virtual ~CPUSpatialProduct() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUSpatialProduct_hpp */
