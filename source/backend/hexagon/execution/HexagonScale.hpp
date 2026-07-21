#ifndef HexagonScale_hpp
#define HexagonScale_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonScale : public HexagonExecution {
public:
    HexagonScale(const Op* op, Backend* bn);
    virtual ~HexagonScale();

    static HexagonScale* create(Backend* backend, const Op* op);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonScale(Backend* bn, const std::shared_ptr<Tensor>& scaleBias, bool hasBias);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<Tensor> mScaleBias;
    bool mHasBias = false;
};

} // namespace MNN

#endif // HexagonScale_hpp
