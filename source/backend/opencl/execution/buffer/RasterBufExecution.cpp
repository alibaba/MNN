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
    mNeedZero = !TensorUtils::regionIsFull(output);
    auto regionNum = des->regions.size();
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
   
    bool cancombine = CanCombine(outputs);
    if(cancombine){
        regionNum = 1;
    }
    int kernel_idx = 0;
    mUnits.resize(regionNum);
    auto outputShape = tensorShapeFormat(output);
    if(mNeedZero || (outputShape[3] % 4) != 0)
    {
        mUnits.resize(regionNum + 1);
        int region[] = {outputShape[0], ROUND_UP(outputShape[3], 4), outputShape[1], outputShape[2]};//nhwc
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {}, output, output);
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
    if(cancombine){
        auto regions = des->regions;
        auto slice = regions[0];
        int nums = regions.size();
        int src_offset = regions[1].src.offset - slice.src.offset;
        int dst_offset = regions[1].dst.offset - slice.dst.offset;
        std::set<std::string> buildOptions;
        auto origin = slice.origin;
        auto inputShape = tensorShapeFormat(origin);
        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC)
        {
            buildOptions.emplace(" -DINPUT_DATA_FORMAT_NHWC");
        }
        if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
        {
            buildOptions.emplace(" -DOUTPUT_DATA_FORMAT_NHWC");
        }
        
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster_buf", "raster_direct_buffer", buildOptions, output, output);
        
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
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
        ret |= unit.kernel->get().setArg(idx++, dst_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[2]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[1]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "raster_buffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        unit.globalWorkSize = {gws[0], gws[1], gws[2]};
        mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
    }else{
        for(auto& slice : des->regions){
            std::set<std::string> buildOptions;
            auto origin = slice.origin;
            auto inputShape = tensorShapeFormat(origin);
            int src_offset = 0;
            int dst_offset = 0;
            if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC)
            {
                buildOptions.emplace(" -DINPUT_DATA_FORMAT_NHWC");
            }
            if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
            {
                buildOptions.emplace(" -DOUTPUT_DATA_FORMAT_NHWC");
            }
            
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster_buf", "raster_direct_buffer", buildOptions, output, output);
            
            const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2],
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
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
            ret |= unit.kernel->get().setArg(idx++, dst_offset);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[1]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
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

bool RasterBufExecution::CanCombine(const std::vector<Tensor *> &outputs){
    auto des = TensorUtils::getDescribe(outputs[0]);
    auto regions = des->regions;
    if(regions.size() < 2)
        return false;
    auto origin = regions[0].origin;
    const int size0 = regions[0].size[0];
    const int size1 = regions[0].size[1];
    const int size2 = regions[0].size[2];
    const int src_offset = regions[1].src.offset - regions[0].src.offset;
    const int dst_offset = regions[1].dst.offset - regions[0].dst.offset;
    const int src_sride0 = regions[0].src.stride[0];
    const int src_sride1 = regions[0].src.stride[1];
    const int src_sride2 = regions[0].src.stride[2];
    const int dst_sride0 = regions[0].dst.stride[0];
    const int dst_sride1 = regions[0].dst.stride[1];
    const int dst_sride2 = regions[0].dst.stride[2];
    bool res = true;
    for(int i = 1; i < regions.size(); ++i){
        res &= regions[i].origin == origin;
        res &= regions[i].size[0] == size0;
        res &= regions[i].size[1] == size1;
        res &= regions[i].size[2] == size2;
        res &= regions[i].src.stride[0] == src_sride0;
        res &= regions[i].src.stride[1] == src_sride1;
        res &= regions[i].src.stride[2] == src_sride2;
        res &= regions[i].dst.stride[0] == dst_sride0;
        res &= regions[i].dst.stride[1] == dst_sride1;
        res &= regions[i].dst.stride[2] == dst_sride2;
        res &= (regions[i].src.offset - regions[i - 1].src.offset) == src_offset;
        res &= (regions[i].dst.offset - regions[i - 1].dst.offset) == dst_offset;
        if(res == false){
            return res;
        }
    }
    return res;
}

REGISTER_OPENCL_OP_CREATOR(RasterBufCreator, OpType_Raster, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
