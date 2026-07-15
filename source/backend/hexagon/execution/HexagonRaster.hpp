#ifndef HexagonRaster_hpp
#define HexagonRaster_hpp

#include <cstdint>
#include <vector>
#include <memory>
#include <map>

#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonRaster : public HexagonExecution {
public:
    struct RasterRegion {
        int32_t srcIndex;
        int32_t srcOffset;
        int32_t dstOffset;
        int32_t size[3];
        int32_t srcStride[3];
        int32_t dstStride[3];
    } __attribute__((packed));

    virtual ~HexagonRaster();

    static HexagonRaster* create(Backend* backend, const Op* op);

private:
    explicit HexagonRaster(Backend* backend);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
    void releaseDynamicTemps();

    int mRegionCount = 0;
    int mBytes = 2;

    // NC4HW4 support
    OpCommonUtils::TensorConvertParameter mSingleConvert;
    bool mNeedZero = false;
    std::map<Tensor*, std::shared_ptr<Tensor>> mTempInput;
    std::vector<std::pair<Tensor*, Tensor::InsideDescribe::Region*>> mTempInputCopy;
    std::shared_ptr<Tensor> mTempOutput;
    std::vector<std::shared_ptr<Tensor::InsideDescribe::Region>> mCacheRegions;

};

} // namespace MNN

#endif
