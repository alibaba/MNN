//
//  ConvertExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvertExecution_hpp
#define ConvertExecution_hpp

#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class ConvertExecution : public Execution {
public:
    ConvertExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvertExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    size_t mWidth  = 0;
    size_t mHeight = 0;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvertExecution_hpp */
