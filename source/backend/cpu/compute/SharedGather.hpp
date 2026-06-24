#ifndef SharedGather_hpp
#define SharedGather_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "core/Execution.hpp"

namespace MNN {
class SharedGather : public Execution {
public:
    SharedGather(Backend* backend, std::shared_ptr<CPUConvolution::ResourceInt8> res);
    virtual ~SharedGather();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
    MemChunk mCacheBuffer;
};
} // namespace MNN

#endif
