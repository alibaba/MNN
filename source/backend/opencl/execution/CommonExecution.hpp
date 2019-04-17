//
//  CommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CommonExecution_hpp
#define CommonExecution_hpp
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"
namespace MNN {
namespace OpenCL {

class CommonExecution : public Execution {
public:
    CommonExecution(Backend *backend);
    virtual ~CommonExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    struct Unit {
        cl::Kernel kernel;
        cl::NDRange globalWorkSize;
        cl::NDRange localWorkSize;
    };
    std::vector<Unit> mUnits;
};
} // namespace OpenCL
} // namespace MNN
#endif /* CommonExecution_hpp */
