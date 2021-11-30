#ifndef BatchMatMulExecution_hpp
#define BatchMatMulExecution_hpp
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
namespace MNN {
namespace CUDA {
class BatchMatMulExecution : public Execution {
public:
    BatchMatMulExecution(bool transposeA, bool transposeB, Backend *backend);
    virtual ~BatchMatMulExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempOutput;
    bool mTransposeA;
    bool mTransposeB;
};
} // namespace CUDA
} // namespace MNN

#endif
