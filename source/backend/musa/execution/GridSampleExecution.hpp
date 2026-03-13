#ifndef _MUSA_GRIDSAMPLE_EXECUTION_HPP_
#define _MUSA_GRIDSAMPLE_EXECUTION_HPP_

#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

class GridSampleExecution : public Execution {
public:
    GridSampleExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend);
    virtual ~GridSampleExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MusaBackend* mBackend;
    const MNN::GridSample* mOp;
    
    int mBatch;
    int mChannels;
    int mInHeight;
    int mInWidth;
    int mOutHeight;
    int mOutWidth;
    bool mAlignCorners;
    
    dim3 mDim3Grid;
    dim3 mDim3Block;
};

} // namespace MUSA
} // namespace MNN

#endif
