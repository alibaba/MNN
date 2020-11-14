//
//  CPUConvolution3D.cpp
//  MNN
//
//  Created by MNN on 2019/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstdint>
#include "backend/cpu/CPUConvolution3D.hpp"
#include "backend/cpu/compute/ConvolutionWinograd.hpp"
#include "backend/cpu/compute/ConvolutionWinograd3D.hpp"
#include "backend/cpu/compute/Convolution1x1Strassen.hpp"
#include "backend/cpu/compute/ConvolutionTiledExecutor.hpp"
#include "backend/cpu/compute/Convolution3D3x3.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/ConvolutionFloatFactory.h"

#define MIN_CON_PLANESIZE 256

namespace MNN {
    // outsideNumber = N*C, planeNumber = H*W
    // when C4 == true, NC4DHW4 --> DNC4HW4
    // when C4 == false, NCDHW --> DNCHW, used by kernel transform.
    void CPUConvolution3D::convertToDepthMajor(float* dst, const float* src, uint32_t planeNumber,
                                                   uint32_t depth, uint32_t outsideNumber) {
        if (depth == 1 && planeNumber == 1) {
            memcpy(dst, src, outsideNumber * sizeof(float));
            return;
        }
        for (uint32_t d = 0; d < depth; ++d) {
            auto dstData = dst + d * outsideNumber * planeNumber;
            auto srcData = src + d * planeNumber;
            for (uint32_t o = 0; o < outsideNumber; ++o) {
                memcpy(dstData + o * planeNumber, srcData + o * depth * planeNumber, planeNumber * sizeof(float));
            }
        }
    }
    // outsideNumber = N*C, planeNumber = H*W
    void CPUConvolution3D::convertDNC4HW4toNC4DHW4(float* dst, const float* src, uint32_t planeNumber,
                                                   uint32_t depth, uint32_t outsideNumber, bool add) {
        const int threadNumber = ((CPUBackend*)backend())->threadNumber();
        for (uint32_t o = 0; o < outsideNumber; ++o) {
            auto dstData = dst + o * depth * planeNumber;
            auto srcData = src + o * planeNumber;
            for (uint32_t d = 0; d < depth; ++d) {
                auto _dstData = dstData + d * planeNumber;
                auto _srcData = srcData + d * outsideNumber * planeNumber;
                if (add) {
                    if (planeNumber >= MIN_CON_PLANESIZE * threadNumber) {
                        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                            const int step = UP_DIV(planeNumber / 4, threadNumber);
                            auto __dstData = _dstData + tId * step * 4;
                            auto __srcData = _srcData + tId * step * 4;
                            MNNMatrixAdd(__dstData, __dstData, __srcData, ALIMIN(planeNumber / 4 - tId * step, step), 0, 0, 0, 1);
                        }
                        MNN_CONCURRENCY_END()
                    } else {
                        MNNMatrixAdd(_dstData, _dstData, _srcData, planeNumber / 4, 0, 0, 0, 1);
                    }
                } else {
                    memcpy(_dstData, _srcData, planeNumber * sizeof(float));
                }
            }
        }
    }

    static Convolution2DCommon* createConvolution2DCommon(flatbuffers::FlatBufferBuilder& fbb, int kernelY, int kernelX,
                                                          PadMode padMode, int padY, int padX, int inputChannel, int outputChannel) {
        auto builder = Convolution2DCommonBuilder(fbb);
        builder.add_kernelX(kernelX);
        builder.add_kernelY(kernelY);
        builder.add_inputCount(inputChannel);
        builder.add_outputCount(outputChannel);
        builder.add_padX(padX);
        builder.add_padY(padY);
        builder.add_padMode(padMode);
        auto offset = builder.Finish();
        return reinterpret_cast<Convolution2DCommon*>(fbb.GetCurrentBufferPointer() + fbb.GetSize() - offset.o);
    }

    CPUConvolution3D::CPUConvolution3D(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                       const MNN::Op *op, Backend *b) : MNN::Execution(b) {
        auto convOp = op->main_as_Convolution3D();
        mCommon = convOp->common();
        mPadMode = mCommon->padMode();
        for (int32_t kernel: *(mCommon->kernels())) {
            mKernels.push_back(kernel);
        }
        for (int32_t stride: *(mCommon->strides())) {
            MNN_ASSERT(stride == 1);
            mStrides.push_back(stride);
        }
        if (mPadMode != PadMode_SAME) {
            for (int32_t pad: *(mCommon->pads())) {
                mPads.push_back(pad);
            }
        }
        for (int32_t dilate: *(mCommon->dilates())) {
            MNN_ASSERT(dilate == 1);
            mDilates.push_back(dilate);
        }
        mInputCount = mCommon->inputCount();
        mOutputCount = mCommon->outputCount();
        mPostFunction = getPostFunction(mCommon);

        int kernelDepth = mKernels[0];
        mWeights.reset(Tensor::createDevice<float>({kernelDepth, (int)convOp->weight()->size() / kernelDepth}));
        mBias.reset(Tensor::createDevice<float>({ALIGN_UP4(mOutputCount)}));
        bool valid = b->onAcquireBuffer(mWeights.get(), Backend::STATIC);
        valid = valid && b->onAcquireBuffer(mBias.get(), Backend::STATIC);
        if (!valid) {
            return;
        }
        convertToDepthMajor(mWeights->host<float>(), convOp->weight()->data(), mKernels[1] * mKernels[2], kernelDepth, mInputCount * mOutputCount);
        memset(mBias->host<float>(), 0, mBias->size());
        memcpy(mBias->host<float>(), convOp->bias()->data(), convOp->bias()->size() * sizeof(float));
    }

    CPUConvolution3D::~CPUConvolution3D() {
        backend()->onReleaseBuffer(mWeights.get(), Backend::STATIC);
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }

    ErrorCode CPUConvolution3D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        auto input = inputs[0];
        auto output = outputs[0];

        if (mPadMode == PadMode_SAME) {
            mPads.clear();
            for (int i = 0; i < 3; ++i) {
                int inputNeeded = (output->length(i + 2) - 1) * mStrides[i] + (mKernels[i] - 1) * mDilates[i] + 1;
                mPads.push_back((inputNeeded - input->length(i + 2)) / 2);
            }
        }

        const int batch = input->length(0), inputChannel = input->length(1), outputChannel = output->length(1);
        const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
        const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
        const int depthPad = mPads[0], kernelDepth = mKernels[0], kernelHeight = mKernels[1], kernelWidth = mKernels[2];
        auto cpuBackend = (CPUBackend*)backend();

        mBreakDown = true;
        mSubInputTensors.clear();
        mSubExecution.clear();

        do {
            bool useWinograd = ConvolutionWinograd3D::canUseWinograd(mCommon) || cpuBackend->memoryMode() != BackendConfig::Memory_Low;
            if (!useWinograd) {
                break;
            }
            auto unit = ConvolutionWinograd3D::bestWinogradUnit(mCommon, input, output, cpuBackend->threadNumber());
            if (unit > 4) {
                mSubExecution.emplace_back(
                            new ConvolutionWinograd3D(mCommon, input, output, cpuBackend, mWeights->host<float>(),
                                                      mWeights->elementSize(), mBias->host<float>(), outputChannel, unit));
            } else if (unit > 1 && kernelHeight == 3 && kernelWidth == 3) {
                mSubExecution.emplace_back(new Convolution3D3x3(mCommon, cpuBackend, mWeights->host<float>(), mWeights->elementSize(),
                                                                mBias->host<float>(), outputChannel));
            } else {
                break;
            }
            mSubExecution[0]->onResize(inputs, outputs);
            mBreakDown = false;
            return NO_ERROR;
        } while(0);

        mCrossDepth = (kernelDepth != 1 || kernelHeight != 1 || depthPad != 0 || mPads[1] != 0);

        if (!mCrossDepth) {
            mSubInputTensors.emplace_back(Tensor::create<float>({batch, inputChannel, inputDepth * inputHeight, inputWidth},
                                                                (void*)(input->host<float>()), Tensor::CAFFE_C4));
            mSubOutputTensor.reset(Tensor::create<float>({batch, outputChannel, outputDepth * outputHeight, outputWidth},
                                                               (void*)(output->host<float>()), Tensor::CAFFE_C4));
        } else {
            mInputStorage.reset(Tensor::createDevice<float>({inputDepth + 2 * depthPad, batch, ALIGN_UP4(inputChannel), inputHeight, inputWidth}));
            mSubOutputTensor.reset(Tensor::createDevice<float>({outputDepth * batch, outputChannel, outputHeight, outputWidth}, Tensor::CAFFE_C4));
            bool valid = true;
            valid = valid && backend()->onAcquireBuffer(mInputStorage.get(), Backend::DYNAMIC);
            valid = valid && backend()->onAcquireBuffer(mSubOutputTensor.get(), Backend::DYNAMIC);
            if (!valid) {
                return OUT_OF_MEMORY;
            }
            const float* data = mInputStorage->host<float>();
            for (int d = 0; d < kernelDepth; ++d) {
                mSubInputTensors.emplace_back(Tensor::create<float>({outputDepth * batch, inputChannel, inputHeight, inputWidth}, (void*)data, Tensor::CAFFE_C4));
                data += mInputStorage->stride(0);
            }
        }

        {
            std::shared_ptr<Tensor> zerosLikeBias(Tensor::createDevice<float>({mOutputCount}));
            bool valid = backend()->onAcquireBuffer(zerosLikeBias.get(), Backend::DYNAMIC);
            if (!valid) {
                return OUT_OF_MEMORY;
            }
            memset(zerosLikeBias->host<float>(), 0, mOutputCount * sizeof(float));
            for (int d = 0; d < kernelDepth; ++d) {
                flatbuffers::FlatBufferBuilder fbb;
                auto common = createConvolution2DCommon(fbb, kernelHeight, kernelWidth, mPadMode, mPads[1], mPads[2], inputChannel, outputChannel);
                auto originWeightSize = mWeights->stride(0), biasSize = mOutputCount;
                auto originWeight = mWeights->host<float>() + d * originWeightSize, bias = zerosLikeBias->host<float>();

                Execution* subExec = nullptr;
                if (common->kernelX() == 1 && common->kernelY() == 1) {
                    subExec = new Convolution1x1Strassen(common, backend(), originWeight, originWeightSize, bias, biasSize);
                } else {
                    subExec = new ConvolutionTiledExecutor(common, backend(), originWeight, originWeightSize, bias, biasSize);
                }
                mSubExecution.emplace_back(subExec);
                mSubExecution[d]->onResize({mSubInputTensors[d].get()}, {mSubOutputTensor.get()});
            }
            backend()->onReleaseBuffer(zerosLikeBias.get(), Backend::DYNAMIC);
        }

        if (mCrossDepth) {
            backend()->onReleaseBuffer(mInputStorage.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mSubOutputTensor.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

    ErrorCode CPUConvolution3D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        if (!mBreakDown) {
            auto code = mSubExecution[0]->onExecute(inputs, outputs);
            return code;
        }

        auto input = inputs[0];
        auto output = outputs[0];
        const int batch = input->length(0), inputChannel = input->length(1), outputChannel = output->length(1);
        const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
        const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
        const int depthPad = mPads[0], kernelDepth = mKernels[0];

        if (mCrossDepth) {
            float* data = mInputStorage->host<float>();
            const int stride = mInputStorage->stride(0);
            memset(data, 0, depthPad * stride * sizeof(float));
            data += depthPad * stride;
            convertToDepthMajor(data, input->host<float>(), 4 * inputHeight * inputWidth, inputDepth, batch * UP_DIV(inputChannel, 4));
            data += inputDepth * stride;
            memset(data, 0, depthPad * stride * sizeof(float));
        }

        for (unsigned int d = 0; d < kernelDepth; ++d) {
            mSubExecution[d]->onExecute({mSubInputTensors[d].get()}, {mSubOutputTensor.get()});
            if (mCrossDepth) {
                convertDNC4HW4toNC4DHW4(output->host<float>(), mSubOutputTensor->host<float>(),
                                        4 * outputHeight * outputWidth, outputDepth, batch * UP_DIV(outputChannel, 4), d != 0);
            }
        }

        for (int b = 0; b < batch; ++b) {
            mPostFunction(output->host<float>() + b * output->stride(0), mBias->host<float>(),
                          outputDepth * outputHeight * outputWidth, UP_DIV(outputChannel, 4));
        }

        return NO_ERROR;
    }

    CPUConvolution3D::POSTFUNCTION CPUConvolution3D::getPostFunction(const Convolution3DCommon* common) {
        if (common->relu()) {
            return MNNAddBiasRelu;
        }
        if (common->relu6()) {
            return MNNAddBiasRelu6;
        }
        return MNNAddBias;
    }

    class Convolution3DCreator : public CPUBackend::Creator {
    public:
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                    const MNN::Op *op, Backend *backend) const override {
            return new CPUConvolution3D(inputs, outputs, op, backend);
        }
    };

    REGISTER_CPU_OP_CREATOR(Convolution3DCreator, OpType_Convolution3D);
} // namespace MNN
