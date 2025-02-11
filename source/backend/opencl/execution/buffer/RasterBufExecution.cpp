//
//  RasterBufExecution.cpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/RasterBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

RasterBufExecution::RasterBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = (OpenCLBackend *)backend;
    //nothing to do
}

ErrorCode RasterBufExecution::onEncode(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RasterBufExecution onResize !\n");
#endif
    mTempInput.clear();
    mTempOutput = nullptr;
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    if (!____inputs.empty()) {
        OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    }
    auto des = TensorUtils::getDescribe(output);
    auto outputDes = TensorUtils::getDescribe(output);
    auto regionNum = des->regions.size();
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    int kernel_idx = 0;
    auto outputShape = tensorShapeFormat(output);
    mFast = false;
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output, 4, true)) {
                mFast = false;
                break;
            }
        }
    }
    mNeedZero = !TensorUtils::regionIsFull(output);
    mNeedZero = mNeedZero || ((outputShape[3] % 4) != 0 && MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat && !mFast);
    if(mFast == false){
        CanCombine(outputs);
        regionNum = mCombineInfo.size();
    }
    mUnits.resize(regionNum);
    if(mNeedZero)
    {
        mUnits.resize(regionNum + 1);
        int region[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//nchw
        if(MNN_DATA_FORMAT_NC4HW4 == outputDes->dimensionFormat){
            region[1] = ROUND_UP(outputShape[3], 4);
        }
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster_buf", "buffer_set_zero", {}, output, output);
        unit.localWorkSize  = {8, 8};
        unit.globalWorkSize = {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
                                   (uint32_t)UP_DIV((region[0] * region[1]), 8)*8};
    
        int global_dim0 = region[2] * region[3];
        int global_dim1 = region[0] * region[1];
    
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, global_dim0);
        ret |= unit.kernel->get().setArg(idx++, global_dim1);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        mOpenCLBackend->recordKernel2d(unit.kernel, {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
            (uint32_t)UP_DIV((region[0] * region[1]), 8)*8},  {8, 8});
    }
    if(mFast)
    {
        // nc4hw4 buffer raster
        for (auto& slice : des->regions)
        {
            auto origin = slice.origin;
            auto inputShape = tensorShapeFormat(origin);
            Tensor::InsideDescribe::Region C4Region;
            OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4, true);
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster_buf", "raster_nc4hw4_buffer", {}, origin, output);

            const std::vector<uint32_t> gws =  {(uint32_t)C4Region.size[2],
                                                    (uint32_t)C4Region.size[1],
                                                    (uint32_t)C4Region.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
            
            auto outputShape    = tensorShapeFormat(output);
            auto sliceShape    = tensorShapeFormat(slice.origin);

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, gws[0]);
            ret |= unit.kernel->get().setArg(idx++, gws[1]);
            ret |= unit.kernel->get().setArg(idx++, gws[2]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(slice.origin));
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.offset);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[1]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[2]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[3]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.offset);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[1]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            std::string name = "raster_nc4hw4_buffer";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                                   ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                                   ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
            mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
        }
        return NO_ERROR;
    }
    
    for(auto& info : mCombineInfo){
        auto slice = info.mRegion;
        int nums = info.mCanCombineNum;
        int src_offset = info.mSrc_offset;
        int dst_offset = info.mDst_offset;
        std::set<std::string> buildOptions;
        auto origin = slice.origin;
        auto inputShape = tensorShapeFormat(origin);
        buildOptions.emplace("-DINPUT_FORMAT=" + std::to_string(TensorUtils::getDescribe(origin)->dimensionFormat));
        buildOptions.emplace("-DOUTPUT_FORMAT=" + std::to_string(outputDes->dimensionFormat));
        
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster_buf", "raster_direct_buffer", buildOptions, origin, output);
        const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2] * nums,
            (uint32_t)slice.size[1],
            (uint32_t)slice.size[0]};
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, gws[2]);
        ret |= unit.kernel->get().setArg(idx++, slice.size[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(origin));
        ret |= unit.kernel->get().setArg(idx++, slice.src.offset);
        ret |= unit.kernel->get().setArg(idx++, src_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[2]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[1]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[3]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[0]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
        ret |= unit.kernel->get().setArg(idx++, dst_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[2]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[1]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[0]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "raster_buffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        unit.globalWorkSize = {gws[0], gws[1], gws[2]};
        mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end RasterBufExecution onResize !\n");
#endif
    return NO_ERROR;
}

class RasterBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~RasterBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new RasterBufExecution(inputs, op, backend);
    }
};

void RasterBufExecution::CanCombine(const std::vector<Tensor *> &outputs){
    auto des = TensorUtils::getDescribe(outputs[0]);
    auto regions = des->regions;
    Tensor* origin;
    int size0, size1, size2, src_offset, dst_offset, last_src_offset, last_dst_offset, src_sride0, src_sride1, src_sride2, dst_sride0, dst_sride1, dst_sride2;
    int canCombineNum = 0;
    for(auto& slice : des->regions){
        bool res = true;
        if(canCombineNum == 0){
            origin = slice.origin;
            size0 = slice.size[0];
            size1 = slice.size[1];
            size2 = slice.size[2];
            src_sride0 = slice.src.stride[0];
            src_sride1 = slice.src.stride[1];
            src_sride2 = slice.src.stride[2];
            dst_sride0 = slice.dst.stride[0];
            dst_sride1 = slice.dst.stride[1];
            dst_sride2 = slice.dst.stride[2];
            canCombineNum++;
            // push back
            mCombineInfo.push_back(CanCombineInfo(slice, 0, 0, 1));
        } else if(canCombineNum == 1){
            res &= slice.origin == origin;
            res &= slice.size[0] == size0;
            res &= slice.size[1] == size1;
            res &= slice.size[2] == size2;
            res &= slice.src.stride[0] == src_sride0;
            res &= slice.src.stride[1] == src_sride1;
            res &= slice.src.stride[2] == src_sride2;
            res &= slice.dst.stride[0] == dst_sride0;
            res &= slice.dst.stride[1] == dst_sride1;
            res &= slice.dst.stride[2] == dst_sride2;
            if(res){
                src_offset = slice.src.offset - last_src_offset;
                dst_offset = slice.dst.offset - last_dst_offset;
                canCombineNum++;
                // change canCombineNum
                mCombineInfo.back().mSrc_offset = src_offset;
                mCombineInfo.back().mDst_offset = dst_offset;
                mCombineInfo.back().mCanCombineNum = canCombineNum;
            } else{
                origin = slice.origin;
                size0 = slice.size[0];
                size1 = slice.size[1];
                size2 = slice.size[2];
                src_sride0 = slice.src.stride[0];
                src_sride1 = slice.src.stride[1];
                src_sride2 = slice.src.stride[2];
                dst_sride0 = slice.dst.stride[0];
                dst_sride1 = slice.dst.stride[1];
                dst_sride2 = slice.dst.stride[2];
                // recover
                canCombineNum = 1;
                // push back
                mCombineInfo.push_back(CanCombineInfo(slice, 0, 0, 1));
            }
        } else{
            res &= slice.origin == origin;
            res &= slice.size[0] == size0;
            res &= slice.size[1] == size1;
            res &= slice.size[2] == size2;
            res &= slice.src.stride[0] == src_sride0;
            res &= slice.src.stride[1] == src_sride1;
            res &= slice.src.stride[2] == src_sride2;
            res &= slice.dst.stride[0] == dst_sride0;
            res &= slice.dst.stride[1] == dst_sride1;
            res &= slice.dst.stride[2] == dst_sride2;
            res &= slice.src.offset - last_src_offset == src_offset;
            res &= slice.dst.offset - last_dst_offset == dst_offset;
            if(res){
                canCombineNum++;
                // change canCombineNum
                mCombineInfo.back().mSrc_offset = src_offset;
                mCombineInfo.back().mDst_offset = dst_offset;
                mCombineInfo.back().mCanCombineNum = canCombineNum;
            } else{
                origin = slice.origin;
                size0 = slice.size[0];
                size1 = slice.size[1];
                size2 = slice.size[2];
                src_sride0 = slice.src.stride[0];
                src_sride1 = slice.src.stride[1];
                src_sride2 = slice.src.stride[2];
                dst_sride0 = slice.dst.stride[0];
                dst_sride1 = slice.dst.stride[1];
                dst_sride2 = slice.dst.stride[2];
                // recover
                canCombineNum = 1;
                // push back
                mCombineInfo.push_back(CanCombineInfo(slice, 0, 0, 1));
                
            }
        }
        last_src_offset = slice.src.offset;
        last_dst_offset = slice.dst.offset;
    }
}

REGISTER_OPENCL_OP_CREATOR(RasterBufCreator, OpType_Raster, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
