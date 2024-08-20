//
//  LoopBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/LoopBufExecution.hpp"

namespace MNN {
namespace OpenCL {
        
static void _TileOrPackTensor(Tensor *input, Tensor *output, std::shared_ptr<KernelWrap>& kernelW, cl::NDRange &globalWorkSize,
                              cl::NDRange &localWorkSize, const int Width, const int Height, const int Channel,
                              const int Batch, OpenCLBackend *bn, const std::string& KernelName, std::set<std::string> buildOptions,
                              const int WidthPad, const int HeightPad, const int ChannelPad, OpenCLRuntime* runtime) {
    bool fastTileTranspose = false;
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC || TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
        buildOptions.emplace("-DMNN_NHWC");
    } else {
        if (KernelName == "tile_buf" && buildOptions.find("-DTRANSPOSE") != buildOptions.end() && (buildOptions.find("-DDIMENSION_3") != buildOptions.end() || buildOptions.find("-DDIMENSION_4") != buildOptions.end())) {
            fastTileTranspose = true;
        }
    }
    
    std::string runKernelName = KernelName;
    unsigned int tileW = 32;
    unsigned int tileC = 32;
    unsigned int tileH = 32;

    unsigned int localW = 8;
    unsigned int localC = 8;
    unsigned int localH = 8;
    if(fastTileTranspose) {
        // local memory limit
        uint32_t local_mem_size = 4;
        if(runtime->isSupportedFP16()) {
            local_mem_size = 2;
        }

        if(buildOptions.find("-DDIMENSION_4") != buildOptions.end()) {
            local_mem_size *= (64 * 64 * 4);
            if(local_mem_size <= runtime->getMaxLocalMem()) {
                if((WidthPad & 63) == 0) {
                    tileW = 64;
                }
                if((HeightPad & 63) == 0) {
                    tileH = 64;
                }
            }

            runKernelName = "tile_trans_4d_buf";
            // match with tileW tileH tileW/localW tileH/localH
            buildOptions.emplace("-DWGSW=" + std::to_string(tileW));
            buildOptions.emplace("-DWGSH=" + std::to_string(tileH));
            buildOptions.emplace("-DTSW=" + std::to_string(tileW/localW));
            buildOptions.emplace("-DTSH=" + std::to_string(tileH/localH));
        } else {
            local_mem_size *= (64 * 64);
            if(local_mem_size <= runtime->getMaxLocalMem()) {
                if((ChannelPad & 63) == 0) {
                    tileC = 64;
                }
                if((HeightPad & 63) == 0) {
                    tileH = 64;
                }
            }
            runKernelName = "tile_trans_3d_buf";
            // match with tileW tileH tileW/localW tileH/localH
            buildOptions.emplace("-DWGSC=" + std::to_string(tileC));
            buildOptions.emplace("-DWGSH=" + std::to_string(tileH));
            buildOptions.emplace("-DTSC=" + std::to_string(tileC/localC));
            buildOptions.emplace("-DTSH=" + std::to_string(tileH/localH));
        }

    }
    if(input->getType().code == halide_type_int){
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", runKernelName, buildOptions, input, input);
    }else if (output->getType().code == halide_type_int){
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", runKernelName, buildOptions, output, output);
    }else {
        kernelW = bn->getOpenCLRuntime()->buildKernel("loop_buf", runKernelName, buildOptions, input, output);
    }
    auto kernel = kernelW->get();
    
    uint32_t mMaxWorkGroupSize  = static_cast<uint32_t>(bn->getOpenCLRuntime()->getMaxWorkGroupSize(kernelW));
    
    if(fastTileTranspose) {
        int w_per_thread = tileW / localW;
        int h_per_thread = tileH / localH;
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)WidthPad/w_per_thread, (uint32_t)HeightPad/h_per_thread, (uint32_t)(UP_DIV(ChannelPad, 4)*Batch)};
        std::vector<uint32_t> mLocalWorkSize = {localW, localH, 1};

        if(buildOptions.find("-DDIMENSION_3") != buildOptions.end()) {
            int c_per_thread = tileC / localC;
            int h_per_thread = tileH / localH;
            mGlobalWorkSize = {(uint32_t)ChannelPad/c_per_thread, (uint32_t)HeightPad/h_per_thread, (uint32_t)Batch};
            mLocalWorkSize = {localC, localH, 1};
        }

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(index++, openCLBuffer(input));
        ret |= kernel.setArg(index++, openCLBuffer(output));
        ret |= kernel.setArg(index++, WidthPad);
        ret |= kernel.setArg(index++, HeightPad);
        ret |= kernel.setArg(index++, ChannelPad);
        ret |= kernel.setArg(index++, Batch);
        ret |= kernel.setArg(index++, Width);
        ret |= kernel.setArg(index++, Height);
        ret |= kernel.setArg(index++, Channel);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBuf _TileOrPackTensor tile_transpose_fast_buf");
                
        globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        bn->recordKernel3d(kernelW, mGlobalWorkSize, mLocalWorkSize);
    } else {
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)WidthPad, (uint32_t)HeightPad, (uint32_t)(UP_DIV(ChannelPad, 4)*Batch)};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel.setArg(index++, mGlobalWorkSize[0]);
        ret |= kernel.setArg(index++, mGlobalWorkSize[1]);
        ret |= kernel.setArg(index++, mGlobalWorkSize[2]);
        ret |= kernel.setArg(index++, openCLBuffer(input));
        ret |= kernel.setArg(index++, openCLBuffer(output));
        ret |= kernel.setArg(index++, WidthPad);
        ret |= kernel.setArg(index++, HeightPad);
        ret |= kernel.setArg(index++, ChannelPad);
        ret |= kernel.setArg(index++, Batch);
        ret |= kernel.setArg(index++, Width);
        ret |= kernel.setArg(index++, Height);
        ret |= kernel.setArg(index++, Channel);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBuf _TileOrPackTensor");
        
        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, bn->getOpenCLRuntime(), KernelName, kernelW).first;
        
        globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        bn->recordKernel3d(kernelW, mGlobalWorkSize, mLocalWorkSize);
    }
}

static void _setTensorStack(std::vector<Tensor *> &result, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs, const LoopParam *loop) {
    if (loop->inputIndexes() != nullptr) {
        for (int i = 0; i < loop->inputIndexes()->size(); ++i) {
            result[loop->inputIndexes()->data()[i]] = inputs[i];
        }
    }
    for (int i = 0; i < loop->outputIndexes()->size(); ++i) {
        result[loop->outputIndexes()->data()[i]] = outputs[i];
    }
}


LoopGatherBufExecution::LoopGatherBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
: CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
    auto cmd = loop->commands()->GetAs<RegionCommand>(0);
}
ErrorCode LoopGatherBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    mUnits.clear();
    mOffsetTensors.clear();
    int x = cmd->size()->data()[0];
    int y = cmd->size()->data()[1];
    int z = cmd->size()->data()[2];
    int n = mLoop->loopNumber();
    int inputSize = mTensors[cmd->indexes()->data()[1]]->elementSize();
    
    auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
    for (int i = 0; i < 3; ++i) {
        mStride_src[i] = srcStride[i];
        mStride_dst[i] = dstStride[i];
    }
    
    mStride_src[3] = cmd->view()->GetAs<View>(1)->offset();
    mStride_dst[3] = cmd->view()->GetAs<View>(0)->offset();
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    
    // gather
    {
        auto input = mTensors[cmd->indexes()->data()[1]];
        auto output = mTensors[cmd->indexes()->data()[0]];
        std::vector<int> inputShape = tensorShapeFormat(input);
        std::vector<int> outputShape = tensorShapeFormat(output);
        int inputShapeVec[4] = {inputShape[2], inputShape[1], inputShape[3], inputShape[0]};
        int outputShapeVec[4] = {outputShape[2], outputShape[1], outputShape[3], outputShape[0]};
        int offset_index = 0;
        
        Unit unit;
        std::set<std::string> buildOptions;
        if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
            buildOptions.emplace("-DGATHER_OUTPUT_NHWC");
        }
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC){
            buildOptions.emplace("-DGATHER_INPUT_NHWC");
        }

        if (mIter[0] >= 0) {
            buildOptions.emplace("-DOFFSET_DST");
        }
        if (mIter[1] >= 0) {
            buildOptions.emplace("-DOFFSET_SRC");
        }
        
        unit.kernel = runTime->buildKernel("gather_buf", "batch_gather_buf", buildOptions, input, output);
        uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
        std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(x * y), (uint32_t)(z), (uint32_t)(n)};
        
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(index++, openCLBuffer(input));
        for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
            if (mIter[i] >= 0) {
                auto tensor = mTensors[cmd->iterIndexes()->data()[i]];
                std::vector<int> shape = tensorShapeFormat(tensor);
                int offsetShapeVec[4] = {shape[2], shape[1], shape[3], shape[0]};// WHCN
                ret |= unit.kernel->get().setArg(index++, openCLBuffer(tensor));
                ret |= unit.kernel->get().setArg(index++, sizeof(offsetShapeVec), offsetShapeVec);
            }
        }
        ret |= unit.kernel->get().setArg(index++, x);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_src), mStride_src);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStride_dst), mStride_dst);
        ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
        ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
        ret |= unit.kernel->get().setArg(index++, sizeof(outputShapeVec), outputShapeVec);
        ret |= unit.kernel->get().setArg(index++, sizeof(inputShapeVec), inputShapeVec);
        ret |= unit.kernel->get().setArg(index++, inputSize);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopGatherBufExecution");
        
        std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, "batch_gather_buf", unit.kernel).first;
        
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        mUnits.emplace_back(unit);
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }
    return NO_ERROR;
}


LoopBatchMatMulBufExecution::LoopBatchMatMulBufExecution(const LoopParam *loop, const MNN::Op *op, Backend *bn)
: CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
}
    
static std::tuple<int, int, int> getTileDimensionSize(std::tuple<int, int, int> shape, std::tuple<int, int, int> tile, MNN_DATA_FORMAT format, int dimension, bool transpose, int index) {
    if(index > 2 || index < 0) {
        MNN_ERROR("Error getTileDimensionSize index, only support 1 for input_1, 2 for input_2, 0 for output!\n");
        return shape;
    }
    // tile: {e, l, h}
    int tile_e = std::get<0>(tile);
    int tile_l = std::get<1>(tile);
    int tile_h = std::get<2>(tile);
    // shape: {w, h, c}
    int pad_w =  std::get<0>(shape);
    int pad_h =  std::get<1>(shape);
    int pad_c =  std::get<2>(shape);
                  
    // output
    if(index == 0) {
        if (format == MNN::MNN_DATA_FORMAT_NHWC) {
            if(dimension == 3) {
                // [N, H, W] -> (n, e, h)
                pad_h = ROUND_UP(pad_h, tile_e);
                pad_w = ROUND_UP(pad_w, tile_h);
            } else {
                // [N*H, W, C] -> [n, e, h]
                pad_w = ROUND_UP(pad_w, tile_e);
                pad_c = ROUND_UP(pad_c, tile_h);
            }
        } else {
            if(dimension == 3) {
                // [N, C, H] -> (n, e, h)
                pad_c = ROUND_UP(pad_c, tile_e);
                pad_h = ROUND_UP(pad_h, tile_h);
            } else {
                // [N*C, H, W] -> [n, e, h]
                pad_h = ROUND_UP(pad_h, tile_e);
                pad_w = ROUND_UP(pad_w, tile_h);
            }
        }
        return std::make_tuple(pad_w, pad_h, pad_c);
    }

    if (format == MNN::MNN_DATA_FORMAT_NHWC) {
        if(dimension == 3) {
            if(transpose) {
                if(index == 1) {
                    // [N, H, W] -> (n, l, e)
                    pad_h = ROUND_UP(pad_h, tile_l);
                    pad_w = ROUND_UP(pad_w, tile_e);
                } else {
                    // [N, H, W] -> (n, h, l)
                    pad_h = ROUND_UP(pad_h, tile_h);
                    pad_w = ROUND_UP(pad_w, tile_l);
                }
            } else {
                if(index == 1) {
                    // [N, H, W] -> (n, e, l)
                    pad_h = ROUND_UP(pad_h, tile_e);
                    pad_w = ROUND_UP(pad_w, tile_l);
                } else {
                    // [N, H, W] -> (n, l, h)
                    pad_h = ROUND_UP(pad_h, tile_l);
                    pad_w = ROUND_UP(pad_w, tile_h);
                }
            }
        } else {
            if(transpose) {
                if(index == 1) {
                    // [N*H, W, C] -> (n, l, e)
                    pad_w = ROUND_UP(pad_w, tile_l);
                    pad_c = ROUND_UP(pad_c, tile_e);
                } else {
                    // [N*H, W, C] -> (n, h, l)
                    pad_w = ROUND_UP(pad_w, tile_h);
                    pad_c = ROUND_UP(pad_c, tile_l);
                }
            } else {
                if(index == 1) {
                    // [N*H, W, C] -> [n, e, l]
                    pad_w = ROUND_UP(pad_w, tile_e);
                    pad_c = ROUND_UP(pad_c, tile_l);
                } else {
                    // [N*H, W, C] -> [n, l, h]
                    pad_w = ROUND_UP(pad_w, tile_l);
                    pad_c = ROUND_UP(pad_c, tile_h);
                }
            }
        }
    } else {
        if(dimension == 3) {
            if(transpose) {
                if(index == 1) {
                    // [N, C, H] -> (n, l, e)
                    pad_c = ROUND_UP(pad_c, tile_l);
                    pad_h = ROUND_UP(pad_h, tile_e);
                } else {
                    // [N, C, H] -> (n, h, l)
                    pad_c = ROUND_UP(pad_c, tile_h);
                    pad_h = ROUND_UP(pad_h, tile_l);
                }
            } else {
                if(index == 1) {
                    // [N, C, H] -> (n, e, l)
                    pad_c = ROUND_UP(pad_c, tile_e);
                    pad_h = ROUND_UP(pad_h, tile_l);
                } else {
                    // [N, C, H] -> (n, l, h)
                    pad_c = ROUND_UP(pad_c, tile_l);
                    pad_h = ROUND_UP(pad_h, tile_h);
                }
            }
        } else {
            if(transpose) {
                if(index == 1) {
                    // [N*C, H, W] -> (n, l, e)
                    pad_h = ROUND_UP(pad_h, tile_l);
                    pad_w = ROUND_UP(pad_w, tile_e);
                } else {
                    // [N*C, H, W] -> (n, h, l)
                    pad_h = ROUND_UP(pad_h, tile_h);
                    pad_w = ROUND_UP(pad_w, tile_l);
                }
            } else {
                if(index == 1) {
                    // [N*C, H, W] -> [n, e, l]
                    pad_h = ROUND_UP(pad_h, tile_e);
                    pad_w = ROUND_UP(pad_w, tile_l);
                } else {
                    // [N*C, H, W] -> [n, l, h]
                    pad_h = ROUND_UP(pad_h, tile_l);
                    pad_w = ROUND_UP(pad_w, tile_h);
                }
            }
        }
    }
    return std::make_tuple(pad_w, pad_h, pad_c);
}
    
ErrorCode LoopBatchMatMulBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd     = mLoop->commands()->GetAs<RegionCommand>(0);
    mHasBias = cmd->indexes()->size() > 3;
    mTransposeA = cmd->op()->main_as_MatMul()->transposeA();
    mTransposeB = cmd->op()->main_as_MatMul()->transposeB();
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    
    mOffset[0] = cmd->view()->GetAs<View>(0)->offset();
    mOffset[1] = cmd->view()->GetAs<View>(1)->offset();
    mOffset[2] = cmd->view()->GetAs<View>(2)->offset();
    mUnits.clear();
    mOffsetTensors.clear();
    mTmpTensors.resize(3);
    if (mHasBias) {
        mTmpTensors.resize(4);
        mOffset[3] = cmd->view()->GetAs<View>(3)->offset();
    }
    
    ::memcpy(mStep, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
    ::memcpy(mIter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
    int e = cmd->size()->data()[0];
    int l = cmd->size()->data()[1];
    int h = cmd->size()->data()[2];
    int n = mLoop->loopNumber();
    
    int tileM = 32;
    int tileN = 32;
    int tileK = 4;
    bool isTotalLarge = (e * 1.0 / 512 * l / 512 * h / 512 > 0.5);
    bool isDimLarge = (e > 256 && l > 256 && h > 256);
    int max_eh = std::max(e, h);
    int min_eh = std::min(e, h);
    isDimLarge = isDimLarge || (l >= 512 && (max_eh > 1024 || min_eh > 32));

    mBatchGemmOpt = isTotalLarge && isDimLarge;
    for(int i = 0; i < cmd->iterIndexes()->size(); ++i){
        if (mIter[i] >= 0) {
            mBatchGemmOpt = false;
            break;
        }
    }
    
    if(mHasBias) {
        mBatchGemmOpt = false;
    }
   
    bool needRearrangeA = false;
    if(mBatchGemmOpt && !mTransposeA) {
        // rearrange to [n, l, e]
        needRearrangeA = true;
    }
    bool needRearrangeB = false;
    if(mBatchGemmOpt && mTransposeB) {
        // rearrange to [n, l, h]
        needRearrangeB = true;
    }
   
    // tile input
    for (int i = 1; i < cmd->indexes()->size(); ++i) {
       auto input = mTensors[cmd->indexes()->data()[i]];
       std::vector<int> Shape = tensorShapeFormat(input);
       const int Channel = Shape.at(3);
       const int Width = Shape.at(2);
       const int Height = Shape.at(1);
       const int Batch = Shape.at(0);
       bool needTranspose = false;
       if(i == 1) {
           needTranspose = needRearrangeA;
       } else if(i == 2) {
           needTranspose = needRearrangeB;
       }

       Unit unit;
       std::set<std::string> buildOptions = mBuildOptions;
       if(needTranspose) {
           buildOptions.emplace("-DTRANSPOSE");
       }
       if(input->buffer().dimensions == 3) {
           buildOptions.emplace("-DDIMENSION_3");
       }
       if(input->buffer().dimensions == 4) {
           buildOptions.emplace("-DDIMENSION_4");
       }
       
       int WidthPad = Width;
       int HeightPad = Height;
       int ChannelPad = Channel;
        
        if(mBatchGemmOpt) {
            auto shape = getTileDimensionSize(std::make_tuple(Width, Height, Channel), std::make_tuple(tileM, tileK, tileN), TensorUtils::getDescribe(input)->dimensionFormat, input->buffer().dimensions, needTranspose, i);
            WidthPad   = std::get<0>(shape);
            HeightPad  = std::get<1>(shape);
            ChannelPad = std::get<2>(shape);
        }
        
        mTmpTensors[i] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, ChannelPad, HeightPad, WidthPad}, Tensor::CAFFE));
        // MNN_PRINT("input%d, %d %d %d %d\n", i, Batch, ChannelPad, HeightPad, WidthPad);

        mOpenCLBackend->onAcquireBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);
       _TileOrPackTensor(input, mTmpTensors[i].get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "tile_buf", buildOptions, WidthPad, HeightPad, ChannelPad, runTime);
       mUnits.emplace_back(unit);
    }

    for(int i = 0; i < cmd->iterIndexes()->size(); ++i){
       if (mIter[i] >= 0) {
           auto input = mTensors[cmd->iterIndexes()->data()[i]];
           std::vector<int> Shape = tensorShapeFormat(input);
           const int Channel = Shape.at(3);
           const int Width = Shape.at(2);
           const int Height = Shape.at(1);
           const int Batch = Shape.at(0);
           mOffsetTensors.emplace_back(std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{Batch, Channel, Height, Width}, Tensor::CAFFE)));
           mOpenCLBackend->onAcquireBuffer(mOffsetTensors.back().get(), Backend::DYNAMIC);
           // MNN_PRINT("input%d offset, %d %d %d %d\n", i, Batch, Channel, Height, Width);

           Unit unit;
           _TileOrPackTensor(input, mOffsetTensors.back().get(), unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "tile_buf", mBuildOptions, Width, Height, Channel, runTime);
           mUnits.emplace_back(unit);
       }
    }

    mBatch = n;
    mM = e;
    mN = h;
    mK = l;
    if(mBatchGemmOpt) {
        // matmul
        int e_pack = ROUND_UP(e, tileM);
        int l_pack = ROUND_UP(l, tileK);
        int h_pack = ROUND_UP(h, tileN);
        mTmpTensors[0] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{n * e_pack * h_pack}, Tensor::CAFFE));
        mOpenCLBackend->onAcquireBuffer(mTmpTensors[0].get(), Backend::DYNAMIC);

        
        std::set<std::string> buildOptions;
        
        uint32_t layout = 0;
        auto param = getGemmParams({(uint32_t)e_pack, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)n, (uint32_t)0}, {openCLBuffer(mTmpTensors[1].get()), openCLBuffer(mTmpTensors[2].get()), openCLBuffer(mTmpTensors[0].get())}, mOpenCLBackend->getOpenCLRuntime());

        int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
        buildOptions.emplace("-DKWG=" + std::to_string(KWG));
        buildOptions.emplace("-DKWI=" + std::to_string(KWI));
        buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
        buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
        buildOptions.emplace("-DMWG=" + std::to_string(MWG));
        buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
        buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
        buildOptions.emplace("-DNWG=" + std::to_string(NWG));
        buildOptions.emplace("-DSA=" + std::to_string(SA));
        buildOptions.emplace("-DSB=" + std::to_string(SB));
        buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
        buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
        buildOptions.emplace("-DVWM=" + std::to_string(VWM));
        buildOptions.emplace("-DVWN=" + std::to_string(VWN));
        if(layout >= 4) {
            buildOptions.emplace("-DOUTPUTMN");
        }
        
        tileM = MWG;
        tileN = NWG;
        int localM = MDIMC;
        int localN = NDIMC;
        
        if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
            buildOptions.emplace("-DUSE_CL_MAD=1");
            buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
        }
        
        Unit unit;
        unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions);
        
        int out_per_thread_m = tileM / localM;
        int out_per_thread_n = tileN / localN;
        
        std::vector<uint32_t>  globalWorkSize = {static_cast<uint32_t>(e_pack/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(n)};
        std::vector<uint32_t>  localWorkSize = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};
        
        float alpha = 1.0;
        float beta = 0.0f;
        int batch_offset_a = e_pack * l_pack;
        int batch_offset_b = h_pack * l_pack;
        int batch_offset_c = e_pack * h_pack;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(e_pack));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(h_pack));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(l_pack));
        ret |= unit.kernel->get().setArg(idx++, alpha);
        ret |= unit.kernel->get().setArg(idx++, beta);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTmpTensors[1].get()));
        ret |= unit.kernel->get().setArg(idx++, batch_offset_a);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTmpTensors[2].get()));
        ret |= unit.kernel->get().setArg(idx++, batch_offset_b);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mTmpTensors[0].get()));
        ret |= unit.kernel->get().setArg(idx++, batch_offset_c);
        MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBuf GemmTile Kernel");

        unit.globalWorkSize = {globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]};
        unit.localWorkSize  = {localWorkSize[0], localWorkSize[1], localWorkSize[2]};
        mUnits.emplace_back(unit);
        mOpenCLBackend->recordKernel3d(unit.kernel, globalWorkSize, localWorkSize);
        
    } else {
       // matmul
       mTmpTensors[0] = std::make_shared<Tensor>(Tensor::createDevice<float>(std::vector<int>{1, n, e, h}, Tensor::CAFFE));
       mOpenCLBackend->onAcquireBuffer(mTmpTensors[0].get(), Backend::DYNAMIC);
       int offset_index = 0;

       // MNN_PRINT("batchgemm:%d, %d %d %d, transAB %d %d, bias:%d, inputsize:%d\n", n, e, h, l, mTransposeA, mTransposeB, mHasBias, cmd->indexes()->size());
       Unit unit;
       std::string KernelName = "batch_matmul";
       std::set<std::string> buildOptions = mBuildOptions;
       if (mHasBias) {
           buildOptions.emplace("-DBIAS");
       }
       if (mTransposeA) {
           buildOptions.emplace("-DTRANSPOSE_A");
       }
       if (mTransposeB) {
           buildOptions.emplace("-DTRANSPOSE_B");
       }
       buildOptions.emplace("-DH_LEAVES=" + std::to_string(h % 4));
       unit.kernel = runTime->buildKernel("loop", KernelName, buildOptions, mTensors[cmd->indexes()->data()[1]], mTensors[cmd->indexes()->data()[0]]);
       uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));
       std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(UP_DIV(h, 4)), (uint32_t)(UP_DIV(e, 4)),(uint32_t)(n)};

       uint32_t index = 0;
       cl_int ret = CL_SUCCESS;
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
       ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[0].get()));
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[1].get()));
       ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[2].get()));
       if (mHasBias) {
           ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTmpTensors[3].get()));
       }
       for (int i = 0; i < cmd->iterIndexes()->size(); ++i) {
           if (mIter[i] >= 0) {
               ret |= unit.kernel->get().setArg(index++, openCLBuffer(mOffsetTensors[offset_index++].get()));
           } else {
               ret |= unit.kernel->get().setArg(index++, openCLBuffer(mTensors[cmd->indexes()->data()[1]]));
           }
       }
       ret |= unit.kernel->get().setArg(index++, e);
       ret |= unit.kernel->get().setArg(index++, l);
       ret |= unit.kernel->get().setArg(index++, h);
       ret |= unit.kernel->get().setArg(index++, sizeof(mOffset), mOffset);
       ret |= unit.kernel->get().setArg(index++, sizeof(mIter), mIter);
       ret |= unit.kernel->get().setArg(index++, sizeof(mStep), mStep);
       MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBatchMatMulBufExecution");

       std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

       unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
       unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
       mUnits.emplace_back(unit);
       mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    }

    //pack output
    {
       auto output = mTensors[cmd->indexes()->data()[0]];
       std::vector<int> Shape = tensorShapeFormat(output);
       const int Channel = Shape.at(3);
       const int Width = Shape.at(2);
       const int Height = Shape.at(1);
       const int Batch = Shape.at(0);
       // MNN_PRINT("output, %d %d %d %d\n", Batch, Channel, Height, Width);

       Unit unit;
       std::set<std::string> buildOptions = mBuildOptions;
       if(mBatchGemmOpt) {
           buildOptions.emplace("-DTRANSPOSE");
           if (mHasBias) {
               buildOptions.emplace("-DBIAS");
           }
           if(output->buffer().dimensions == 3) {
               buildOptions.emplace("-DDIMENSION_3");
           }
           if(output->buffer().dimensions == 4) {
               buildOptions.emplace("-DDIMENSION_4");
           }
       }
        
        int WidthPad = Width;
        int HeightPad = Height;
        int ChannelPad = Channel;
        if(mBatchGemmOpt) {
            auto shape = getTileDimensionSize(std::make_tuple(Width, Height, Channel), std::make_tuple(tileM, tileK, tileN), TensorUtils::getDescribe(output)->dimensionFormat, output->buffer().dimensions, false, 0);
            WidthPad   = std::get<0>(shape);
            HeightPad  = std::get<1>(shape);
            ChannelPad = std::get<2>(shape);
        }
       _TileOrPackTensor(mTmpTensors[0].get(), output, unit.kernel, unit.globalWorkSize, unit.localWorkSize, Width, Height, Channel, Batch, mOpenCLBackend, "pack_buf", buildOptions, WidthPad, HeightPad, ChannelPad, runTime);
       mUnits.emplace_back(unit);
    }

    for (int i = 0; i < cmd->indexes()->size(); ++i) {
         mOpenCLBackend->onReleaseBuffer(mTmpTensors[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < mOffsetTensors.size(); ++i) {
         mOpenCLBackend->onReleaseBuffer(mOffsetTensors[i].get(), Backend::DYNAMIC);
    }

    return NO_ERROR;
}

ErrorCode LoopBatchMatMulBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#else
    if(openCLBackend->isUseRecordQueue()){
        openCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    for (auto &unit : mUnits) {
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize,
                                                    nullptr,
                                                    &event);
        std::string name = "While-gemm";

        if(mBatchGemmOpt) {
            if(idx == 2) {
                name += "-batchgemm";
            } else if(idx == 0) {
                name += "-rearrangeA";
            } else if(idx == 1) {
                name += "-rearrangeB";
            } else {
                name += "-rearrangeC";
            }
        } else {
            if(idx == mUnits.size()-2) {
                name += "-batchgemm";
            } else if(idx == 0) {
                name += "-rearrangeA";
            } else if(idx == 1) {
                name += "-rearrangeB";
            } else {
                name += "-rearrangeC";
            }
        }
        std::string b = std::to_string(mBatch);
        std::string m = std::to_string(mM);
        std::string n = std::to_string(mN);
        std::string k = std::to_string(mK);
        std::string total = std::to_string(1.0 / 1000000 * mBatch * mM * mN * mK);
        name += "-b" + b + "m" + m + "n" + n + "k" + k + "-total:" + total + "*10^6";
        runtime->pushEvent({name.c_str(), event});
        idx++;
    #else
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize);
    #endif
        MNN_CHECK_CL_SUCCESS(res, "While-gemm execute");
    }
    return NO_ERROR;
}
LoopBinaryBufExecution::LoopBinaryBufExecution(const LoopParam *loop, const std::string &compute, const MNN::Op *op, Backend *bn)
    : CommonExecution(bn, op) {
    mLoop = loop;
    mTensors.resize(mLoop->tensorNumber());
    mBuildOptions.emplace("-DLOOP_BINARY_OPERATOR=" + compute);
}

ErrorCode LoopBinaryBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto cmd                      = mLoop->commands()->GetAs<RegionCommand>(0);
    OpenCLBackend *mOpenCLBackend = (OpenCLBackend *)backend();
    auto runTime                  = mOpenCLBackend->getOpenCLRuntime();
    _setTensorStack(mTensors, inputs, outputs, mLoop);
    mUnits.clear();
    
    Unit unit;
    auto input0 = mTensors[cmd->indexes()->data()[1]];
    std::vector<int> input0C4Shape = tensorShapeFormat(input0);
    int input0C4Size[4] = {input0C4Shape.at(0), input0C4Shape.at(3),input0C4Shape.at(1),input0C4Shape.at(2)};
         
    auto input1 = mTensors[cmd->indexes()->data()[2]];
    std::vector<int> input1C4Shape = tensorShapeFormat(input1);
    int input1C4Size[4] = {input1C4Shape.at(0), input1C4Shape.at(3),input1C4Shape.at(1),input1C4Shape.at(2)};
         
    auto output = mTensors[cmd->indexes()->data()[0]];
    std::vector<int> outputC4Shape = tensorShapeFormat(output);
    
    int input0Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int input1Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int outputShape[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    int offset0 = output->dimensions() - input0->dimensions();
    int offset1 = output->dimensions() - input1->dimensions();
    for (int i = 0; i < input0->dimensions(); ++i) {
        input0Shape[i + offset0] = input0->length(i);
    }
    for (int i = 0; i < input1->dimensions(); ++i) {
        input1Shape[i + offset1] = input1->length(i);
    }
    for(int i =0;i<output->dimensions();++i){
        outputShape[i] = output->length(i);
    }
    if (TensorUtils::getDescribe(input0)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = input0Shape[0];
        int iH = input0Shape[1];
        int iW = input0Shape[2];
        int iC = input0Shape[3];
            
        if(input0->dimensions() > 4)
        {
            for(int i = 4; i < input0->dimensions(); i++)
            {
                iC *= input0Shape[i];
            }
        }
        input0Shape[0] = iN;
        input0Shape[1] = iC;
        input0Shape[2] = iH;
        input0Shape[3] = iW;
        input0Shape[4] = 1;
    }
    if (TensorUtils::getDescribe(input1)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = input1Shape[0];
        int iH = input1Shape[1];
        int iW = input1Shape[2];
        int iC = input1Shape[3];
            
        if(input1->dimensions() > 4)
        {
            for(int i = 4; i < input1->dimensions(); i++)
            {
                iC *= input1Shape[i];
            }
        }
        input1Shape[0] = iN;
        input1Shape[1] = iC;
        input1Shape[2] = iH;
        input1Shape[3] = iW;
        input1Shape[4] = 1;
    }
    if (TensorUtils::getDescribe(output)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        int iN = outputShape[0];
        int iH = outputShape[1];
        int iW = outputShape[2];
        int iC = outputShape[3];
            
        if(input1->dimensions() > 4)
        {
            for(int i = 4; i < output->dimensions(); i++)
            {
                iC *= outputShape[i];
            }
        }
        outputShape[0] = iN;
        outputShape[1] = iC;
        outputShape[2] = iH;
        outputShape[3] = iW;
        outputShape[4] = 1;
    }
    auto BuildOptions = mBuildOptions;
    for(int i = 0; i < 4; ++i){
        if(input1C4Shape[i] != outputC4Shape[i]){
            BuildOptions.emplace("-DBROADCAST_INPUT1");
            break;
        }
    }
   
    const int Channel = outputC4Shape.at(3);
    const int Width = outputC4Shape.at(2);
    const int Height = outputC4Shape.at(1);
    const int Batch = outputC4Shape.at(0);
    const int ChannelBlock = UP_DIV(Channel, 4);
    std::string KernelName = "broadcast_binary_buf";
    if(input0Shape[1] == input1Shape[1] && input0C4Size[1] == input1C4Size[1]){
        KernelName = "broadcast_binary_channel_equall_buf";
    } else if((input0->dimensions() == 1 && input0Shape[1] == 1) || (input1->dimensions() == 1 && input1Shape[1] == 1)){
        KernelName = "broadcast_binary_dimmision1_channel1_buf";
    }
    unit.kernel = runTime->buildKernel("loop_buf", KernelName, BuildOptions, input0, output);
    uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runTime->getMaxWorkGroupSize(unit.kernel));

    std::vector<uint32_t> mGlobalWorkSize = {(uint32_t)(Width), (uint32_t)(Height), (uint32_t)(Batch * ChannelBlock)};

    uint32_t index = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(index++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input0));
    ret |= unit.kernel->get().setArg(index++, openCLBuffer(input1));
    ret |= unit.kernel->get().setArg(index++, sizeof(input0Shape), input0Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(input0C4Size), input0C4Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(input1Shape), input1Shape);
    ret |= unit.kernel->get().setArg(index++, sizeof(input1C4Size), input1C4Size);
    ret |= unit.kernel->get().setArg(index++, sizeof(outputShape), outputShape);
    ret |= unit.kernel->get().setArg(index++, Width);
    ret |= unit.kernel->get().setArg(index++, Height);
    ret |= unit.kernel->get().setArg(index++, Channel);
    ret |= unit.kernel->get().setArg(index++, ChannelBlock);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LoopBinaryBufExecution");

    std::vector<uint32_t> mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runTime, KernelName, unit.kernel).first;

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    mUnits.emplace_back(unit);
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    return NO_ERROR;
}

class LoopBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
         for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
         }
         for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
         }
        auto loop = op->main_as_LoopParam();
        if (nullptr == loop || loop->commands() == nullptr) {
            return nullptr;
        }
        if (nullptr != loop->initCommand()) {
            return nullptr;
        }
        // Make Tensor Stack
        if (1 == loop->commands()->size()) {
            auto cmd   = loop->commands()->GetAs<RegionCommand>(0);
            auto subop = cmd->op();
            if (OpType_UnaryOp == subop->type() && nullptr == subop->main() && cmd->fuse() < 0) {
                return new LoopGatherBufExecution(loop, op, backend);
            }
            if (OpType_MatMul == subop->type() && loop->parallel()) {
                return new LoopBatchMatMulBufExecution(loop, op, backend);
            }
            if (OpType_BinaryOp == subop->type() && loop->parallel()) {
                switch (subop->main_as_BinaryOp()->opType()) {
                    case BinaryOpOperation_MUL:
                        return new LoopBinaryBufExecution(loop, "in0*in1", op, backend);
                    case BinaryOpOperation_ADD:
                        return new LoopBinaryBufExecution(loop, "in0+in1", op, backend);
                    case BinaryOpOperation_SUB:
                        return new LoopBinaryBufExecution(loop, "in0-in1", op, backend);
                    case BinaryOpOperation_REALDIV:
                        return new LoopBinaryBufExecution(loop, "sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001))", op, backend);
                    case BinaryOpOperation_MINIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in1:in0", op, backend);
                    case BinaryOpOperation_MAXIMUM:
                        return new LoopBinaryBufExecution(loop, "in0>in1?in0:in1", op, backend);
                    case BinaryOpOperation_GREATER:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isgreater(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isless(in0,in1))", op, backend);
                    case BinaryOpOperation_LESS_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-islessequal(in0,in1))", op, backend);
                    case BinaryOpOperation_GREATER_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isgreaterequal(in0,in1))", op, backend);
                    case BinaryOpOperation_EQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isequal(in0,in1))", op, backend);
                    case BinaryOpOperation_FLOORDIV:
                        return new LoopBinaryBufExecution(loop, "floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))", op, backend);
                    case BinaryOpOperation_FLOORMOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    case BinaryOpOperation_POW:
                        return new LoopBinaryBufExecution(loop, "pow(in0,in1)", op, backend);
                    case BinaryOpOperation_SquaredDifference:
                        return new LoopBinaryBufExecution(loop, "(in0-in1)*(in0-in1)", op, backend);
                    case BinaryOpOperation_ATAN2:
                        return new LoopBinaryBufExecution(loop, "(in1==(float4)0?(sign(in0)*(float4)(PI/2)):(atan(in0/in1)+(in1>(float4)0?(float4)0:sign(in0)*(float4)PI)))", op, backend);
                    case BinaryOpOperation_NOTEQUAL:
                        return new LoopBinaryBufExecution(loop, "convert_float4(-isnotequal(in0,in1))", op, backend);
                    case BinaryOpOperation_MOD:
                        return new LoopBinaryBufExecution(loop, "in0-floor(sign(in1)*in0/(fabs(in1)>(float4)((float)0.0000001)?fabs(in1):(float4)((float)0.0000001)))*in1", op, backend);
                    default:
                        break;
                }
                return nullptr;
            }
        }
        return nullptr;
    }
};

REGISTER_OPENCL_OP_CREATOR(LoopBufCreator, OpType_While, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
