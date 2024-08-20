//
//  RasterExecution.cpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/RasterExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {


RasterExecution::RasterExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = (OpenCLBackend *)backend;
    //nothing to do
}

ErrorCode RasterExecution::onEncode(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RasterExecution onResize !\n");
#endif
    mTempInput.clear();
    mTempOutput = nullptr;
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);

    auto des = TensorUtils::getDescribe(output);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(output);
    auto regionNum = des->regions.size();
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    
    
    bool cancombine = CanCombine(outputs);
    // Alloc Temp buffer
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    if(output->getType().code == halide_type_float && runtime->isSupportedFP16()) {
        mTempOutput         = bufferPool->alloc(output->usize()/2);
    }else{
        mTempOutput         = bufferPool->alloc(output->usize());
    }
    bufferPool->recycle(mTempOutput);
    
    auto originNum = mTempInput.size();
    if(cancombine){
        regionNum = 1;
    }
    mUnits.resize(regionNum + 1);
    
    int kernel_idx = 0;
    if(mNeedZero)
    {
        mUnits.resize(regionNum + 2);
        auto outputShape    = tensorShapeFormat(output);
        int region[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//nhwc
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {}, output, output);

        std::vector<uint32_t> gws = {(uint32_t)(region[2] * region[3]),
                                     (uint32_t)(region[0] * region[1])};
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_set_zero";
        std::vector<uint32_t> lws = localWS2DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
        
        mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    }
    
    // buffer raster
    if(cancombine){
        std::set<std::string> buildOptions;
        auto regions = des->regions;
        auto slice = regions[0];
        auto origin = slice.origin;
        auto inputShape = tensorShapeFormat(origin);
        int nums = regions.size();
        int src_offset = regions[1].src.offset - slice.src.offset;
        int dst_offset = regions[1].dst.offset - slice.dst.offset;
        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            buildOptions.emplace(" -DINPUT_DATA_FORMAT_NHWC");
        }
        
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "raster_buffer_direct", buildOptions, output, output);
        
        unit.globalWorkSize = {(uint32_t)slice.size[2] * nums,
            (uint32_t)slice.size[1],
            (uint32_t)slice.size[0]};
        
        const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2] * nums,
            (uint32_t)slice.size[1],
            (uint32_t)slice.size[0]};
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, gws[2]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(origin));
        ret |= unit.kernel->get().setArg(idx++, slice.src.offset);
        ret |= unit.kernel->get().setArg(idx++, src_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[2]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[1]);
        ret |= unit.kernel->get().setArg(idx++, inputShape[3]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
        ret |= unit.kernel->get().setArg(idx++, dst_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, slice.size[2]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "rasterBuffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
            ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
        mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
    }else{
        for (auto& slice : des->regions)
        {
            std::set<std::string> buildOptions;
            auto origin = slice.origin;
            auto inputShape = tensorShapeFormat(origin);
            int src_offset = 0;
            int dst_offset = 0;
            if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
                buildOptions.emplace(" -DINPUT_DATA_FORMAT_NHWC");
            }
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "raster_buffer_direct", buildOptions, output, output);
            
            unit.globalWorkSize = {(uint32_t)slice.size[2],
                (uint32_t)slice.size[1],
                (uint32_t)slice.size[0]};
            
            const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2],
                (uint32_t)slice.size[1],
                (uint32_t)slice.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
            
            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, gws[0]);
            ret |= unit.kernel->get().setArg(idx++, gws[1]);
            ret |= unit.kernel->get().setArg(idx++, gws[2]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(origin));
            ret |= unit.kernel->get().setArg(idx++, slice.src.offset);
            ret |= unit.kernel->get().setArg(idx++, src_offset);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, inputShape[2]);
            ret |= unit.kernel->get().setArg(idx++, inputShape[1]);
            ret |= unit.kernel->get().setArg(idx++, inputShape[3]);
            ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
            ret |= unit.kernel->get().setArg(idx++, dst_offset);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, slice.size[2]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            
            std::string name = "rasterBuffer";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
            mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
        }
    }
    
    //buffer to image
    {
        auto outputShape    = tensorShapeFormat(output);
        int wh[]     = {outputShape[2], outputShape[1]};
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        Unit &unit          = mUnits[kernel_idx++];
        if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", {}, output, output);
        }
        else //nchw buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nchw_buffer_to_image", {}, output, output);
        }
        
        std::vector<uint32_t> gws = {(uint32_t)(region[3] * region[1]),
                                     (uint32_t)(region[2] * region[0])};
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        ret |= unit.kernel->get().setArg(idx++, wh[1]);
        ret |= unit.kernel->get().setArg(idx++, wh[0]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_to_image";
        std::vector<uint32_t> lws = localWS2DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
        mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end RasterExecution onResize !\n");
#endif
    return NO_ERROR;
}

bool RasterExecution::CanCombine(const std::vector<Tensor *> &outputs){
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

using RasterCreator = TypedCreator<RasterExecution>;
REGISTER_OPENCL_OP_CREATOR(RasterCreator, OpType_Raster, IMAGE);
} // namespace OpenCL
} // namespace MNN
