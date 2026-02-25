#ifndef _MUSA_EMBEDDING_EXECUTION_HPP_
#define _MUSA_EMBEDDING_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace CUDA {

class EmbeddingExecution : public Execution {
public:
    EmbeddingExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~EmbeddingExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    
    int mNumIndices;
    int mEmbeddingDim;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace CUDA
} // namespace MNN

#endif
