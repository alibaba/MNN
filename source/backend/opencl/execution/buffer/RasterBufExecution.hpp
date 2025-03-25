//
//  RasterBufExecution.hpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef RaterBufExecution_hpp
#define RaterBufExecution_hpp
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class RasterBufExecution : public CommonExecution {
public:
    class CanCombineInfo {
    public:
        CanCombineInfo(const Tensor::InsideDescribe::Region region, const int src_offset, const int dst_offset, const int canCombineNum) : mRegion(region), mSrc_offset(src_offset), mDst_offset(dst_offset), mCanCombineNum(canCombineNum){
            // Do nothing
        }
        int mSrc_offset;
        int mDst_offset;
        int mCanCombineNum;
        Tensor::InsideDescribe::Region mRegion;
    };
    RasterBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~RasterBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void CanCombine(const std::vector<Tensor *> &outputs);
    std::map<Tensor*, cl::Buffer *> mTempInput;
    cl::Buffer *mTempOutput;
    OpenCLBackend *mOpenCLBackend;
    bool mNeedZero = false;
    bool mFast = false;
    std::vector<CanCombineInfo> mCombineInfo;
};

} // namespace OpenCL
} // namespace MNN

#endif
#endif /* MNN_OPENCL_BUFFER_CLOSED */
