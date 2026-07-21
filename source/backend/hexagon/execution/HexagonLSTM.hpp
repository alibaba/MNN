#ifndef HexagonLSTM_hpp
#define HexagonLSTM_hpp

#include "HexagonExecution.hpp"
#include <memory>

namespace MNN {

class HexagonLSTM : public HexagonExecution {
public:
    explicit HexagonLSTM(Backend* backend, int hiddenSize);
    virtual ~HexagonLSTM();

    static HexagonLSTM* create(Backend* backend, const Op* op,
                               const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
    void releasePackedWeights();

    int mHiddenSize = 0;
    std::shared_ptr<Tensor> mScratch;
    std::shared_ptr<Tensor> mPackedW;
    std::shared_ptr<Tensor> mPackedR;
    int mPackedInputSize = 0;
    int mPackedHiddenSize = 0;
    int mPackedGateSize = 0;
    int mPackedDirection = 0;
};

} // namespace MNN

#endif
