#ifndef HexagonPooling_hpp
#define HexagonPooling_hpp

#include "MNN_generated.h"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonPooling : public HexagonExecution {
public:
    HexagonPooling(Backend* backend, const Pool* parameter);
    virtual ~HexagonPooling() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonPooling* create(Backend* backend, const Op* op);

private:
    const Pool* mParameter = nullptr;

    int mKernelX = 0;
    int mKernelY = 0;
    int mStrideX = 0;
    int mStrideY = 0;
    int mPadX = 0;
    int mPadY = 0;

    int mPadType = 0;
    int mCountType = 0;
    int mPoolType = 0;

    int mBatch = 0;
    int mInputHeight = 0;
    int mInputWidth = 0;
    int mOutputHeight = 0;
    int mOutputWidth = 0;

    int mChannelBlock = 0;
    int mPack = 4;
    int mBytes = 2;
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif
