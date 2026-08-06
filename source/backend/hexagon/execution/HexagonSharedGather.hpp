#ifndef HexagonSharedGather_hpp
#define HexagonSharedGather_hpp

#include "HexagonConvolution.hpp"

namespace MNN {

class HexagonSharedGather : public HexagonExecution {
public:
    HexagonSharedGather(Backend* backend, std::shared_ptr<HexagonConvolution::Resource> res);
    virtual ~HexagonSharedGather() = default;

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

private:
    std::shared_ptr<HexagonConvolution::Resource> mResource;
};

} // namespace MNN

#endif
