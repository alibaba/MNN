//
//  LayerNormBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef LayerNormBufExecution_hpp
#define LayerNormBufExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class LayerNormBufExecution : public Execution {
public:
    LayerNormBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~LayerNormBufExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    int getLocalSize(int size, int maxGroupSize);
    cl::Kernel mKernel;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    OpenCLBackend *mOpenCLBackend;
    int axis_size = 0;
    int group_ = 1;
    float epsilon_ = 0.001;

    std::shared_ptr<cl::Buffer> mGammaBuffer;
    std::shared_ptr<cl::Buffer> mBetaBuffer;
    bool has_gamma_beta_ = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* LayerNormBufExecution_hpp */


#endif /* MNN_OPENCL_BUFFER_CLOSED */
