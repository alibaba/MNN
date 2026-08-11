#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef SharedGatherBufExecution_hpp
#define SharedGatherBufExecution_hpp

#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class SharedGatherBufExecution : public CommonExecution {
public:
    SharedGatherBufExecution(std::shared_ptr<ConvBufResource> resource, const Op* op, Backend* backend);
    virtual ~SharedGatherBufExecution() = default;

    static bool validResource(const std::shared_ptr<ConvBufResource>& resource);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    OpenCLBackend* mOpenCLBackend = nullptr;
    std::shared_ptr<ConvBufResource> mResource;
    std::vector<uint32_t> mGWS{0, 0};
    std::vector<uint32_t> mLWS{0, 0};
};

} // namespace OpenCL
} // namespace MNN

#endif /* SharedGatherBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
