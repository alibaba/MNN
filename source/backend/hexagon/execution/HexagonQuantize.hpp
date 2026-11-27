#ifndef HexagonQuantize_hpp
#define HexagonQuantize_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonQuantize : public HexagonExecution {
public:
    static HexagonQuantize* create(Backend* backend, const Op* op,
                                    const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs);
    static bool validTensorContract(const Tensor* input, const Tensor* output,
                                    bool quantize);

    bool onClone(Backend* backend, const Op* op, Execution** dst) override;

private:
    HexagonQuantize(Backend* backend, bool quantize, float scale,
                    int32_t zero);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    bool mQuantize = false;
    float mScale = 0.0f;
    int32_t mZero = 0;
};

} // namespace MNN

#endif
