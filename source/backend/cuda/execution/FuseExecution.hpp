//
//  FuseExecution.hpp
//  MNN
//
//  Created by MNN on 2023/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_CODEGEN_CUDA

#ifndef FuseExecution_hpp
#define FuseExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "backend/cuda/core/compiler/CUDACompiler.hpp"

namespace MNN {
namespace CUDA {

class FuseExecution : public Execution {
public:
    FuseExecution(const Op* op, Backend *backend);
    virtual ~FuseExecution() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildFuseKernel(const Op* op);

private:
    CUfunction mKernel;
    std::string mSource;
    const char* mName;
    bool mVectorize;
    int batch, area = 1, channel, channel_pack;
    std::pair<void*, size_t> mDivAreaStorage;
    std::pair<void*, size_t> mDivChannelStorage;
    bool ignoreRaster = false;
};

}; // namespace CUDA
}; // namespace MNN

#endif
#endif