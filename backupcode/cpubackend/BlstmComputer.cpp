//
//  BLSTM.cpp
//  MNN
//
//  Created by MNN on 2020/04/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "MNN/ErrorCode.hpp"
#include "MNN_generated.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/BlstmComputer.hpp"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Matrix.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

using std::shared_ptr;
using std::vector;

namespace MNN {

void BlstmComputer::trimTensor(Tensor *src_tensor, Tensor *tgt_tensor) {
  MNN_ASSERT(src_tensor->shape() == tgt_tensor->shape());
  auto src_format = TensorUtils::getDescribe(src_tensor)->dimensionFormat;
  if (src_format == MNN_DATA_FORMAT_NCHW ||
      src_format == MNN_DATA_FORMAT_NHWC) {
    memcpy(tgt_tensor->host<float>(), src_tensor->host<float>(),
           size_t(src_tensor->size()));
  } else if (src_format == MNN_DATA_FORMAT_NC4HW4) {
    mBackend->onCopyBuffer(src_tensor, tgt_tensor);
  } else {
    MNN_ERROR("src_tensor format not supported\n");
  }
}

BlstmComputer::~BlstmComputer() {
  for (int i = 0; i < mWeights.size(); i++) {
    backend()->onReleaseBuffer(mWeights[i].get(), Backend::DYNAMIC);
  }
  for (int i = 0; i < mHiddenStates.size(); i++) {
    backend()->onReleaseBuffer(mHiddenStates[i].get(), Backend::DYNAMIC);
  }
  for (int i = 0; i < mCellStates.size(); i++) {
    backend()->onReleaseBuffer(mCellStates[i].get(), Backend::DYNAMIC);
  }
  for (int i = 0; i < mGateInputs.size(); i++) {
    backend()->onReleaseBuffer(mGateInputs[i].get(), Backend::DYNAMIC);
  }
  for (int i = 0; i < mGateOutputs.size(); i++) {
    backend()->onReleaseBuffer(mGateOutputs[i].get(), Backend::DYNAMIC);
  }
  if (mInput) {
    backend()->onReleaseBuffer(mInput.get(), Backend::DYNAMIC);
  }
  if (mOutput) {
    backend()->onReleaseBuffer(mOutput.get(), Backend::DYNAMIC);
  }
}

float BlstmComputer::sigmoid(float x) { return 1. / (1. + expf(-x)); }

BlstmComputer::BlstmComputer(int inDim, int stateSize, bool bidirectional,
                             CPUBackend *backend)
    : mInDim(inDim), mStateSize(stateSize), mBidirectional(bidirectional),
      mBackend(backend) {}

ErrorCode
BlstmComputer::importWeights(const vector<shared_ptr<Tensor>> &weightsVec) {
  if (mBidirectional) {
    MNN_ASSERT(weightsVec.size() == 24)
  } else {
    MNN_ASSERT(weightsVec.size() == 12)
  }
  mWeights.clear();
  // initialize mWeights
  for (int b = 0; b < (mBidirectional ? 2 : 1); b++) {
    // b = 0 -> forward, b = 1 -> backward
    // Wi, Wn, Wf, Wo
    for (int i = 0; i < 4; i++) {
      mWeights.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{mInDim, mStateSize}, Tensor::CAFFE)));
    }
    // Ui, Un, Uf, Uo
    for (int i = 0; i < 4; i++) {
      mWeights.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{mStateSize, mStateSize}, Tensor::CAFFE)));
    }
    // Bi, Bn, Bf, Bo
    for (int i = 0; i < 4; i++) {
      mWeights.push_back(shared_ptr<Tensor>(
          Tensor::createDevice<float>(vector<int>{mStateSize}, Tensor::CAFFE)));
    }
  }
  // alloc space for mWeights
  for (int i = 0; i < mWeights.size(); i++)
    backend()->onAcquireBuffer(mWeights[i].get(), Backend::DYNAMIC);

  // copy weight data
  for (int b = 0; b < (mBidirectional ? 2 : 1); b++) {
    // b = 0 -> forward, b = 1 -> backward
    for (int i = 0 + b * 12; i < 4 + b * 12; i++) {
      MNN_ASSERT(weightsVec[i]->dimensions() == 2);
      MNN_ASSERT(weightsVec[i]->buffer().dim[0].extent == mInDim);
      MNN_ASSERT(weightsVec[i]->buffer().dim[1].extent == mStateSize);
      trimTensor(weightsVec[i].get(), mWeights[i].get());
    }
    for (int i = 4 + b * 12; i < 8 + b * 12; i++) {
      // Ui, Un, Uf, Uo
      MNN_ASSERT(weightsVec[i]->dimensions() == 2);
      MNN_ASSERT(weightsVec[i]->buffer().dim[0].extent == mStateSize);
      MNN_ASSERT(weightsVec[i]->buffer().dim[1].extent == mStateSize);
      trimTensor(weightsVec[i].get(), mWeights[i].get());
    }
    for (int i = 8 + b * 12; i < 12 + b * 12; i++) {
      // Bi, Bn, Bf, Bo
      MNN_ASSERT(weightsVec[i]->dimensions() == 1);
      MNN_ASSERT(weightsVec[i]->buffer().dim[0].extent == mStateSize);
      trimTensor(weightsVec[i].get(), mWeights[i].get());
    }
  }
  return NO_ERROR;
}

ErrorCode BlstmComputer::onResize(int timeSteps, int batchSize) {
  if (batchSize != mBatchSize) {
    // Reinitialize mHiddenStates & mCellStates
    for (int i = 0; i < mHiddenStates.size(); i++) {
      backend()->onReleaseBuffer(mHiddenStates[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < mCellStates.size(); i++) {
      backend()->onReleaseBuffer(mCellStates[i].get(), Backend::DYNAMIC);
    }
    mHiddenStates.clear();
    mCellStates.clear();
    for (int i = 0; i < (mBidirectional ? 2 : 1); i++) {
      mHiddenStates.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{batchSize, mStateSize}, Tensor::CAFFE)));
      backend()->onAcquireBuffer(mHiddenStates[i].get(), Backend::DYNAMIC);
      mCellStates.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{batchSize, mStateSize}, Tensor::CAFFE)));
      backend()->onAcquireBuffer(mCellStates[i].get(), Backend::DYNAMIC);
    }
  }

  if (batchSize != mBatchSize || timeSteps != mTimeSteps) {
    // Reinitialize mInput, mGateInputs, mGateOutputs, mOutput
    backend()->onReleaseBuffer(mInput.get(), Backend::DYNAMIC);
    mInput.reset(Tensor::createDevice<float>(
        vector<int>{batchSize, timeSteps, mInDim}, Tensor::CAFFE));
    backend()->onAcquireBuffer(mInput.get(), Backend::DYNAMIC);

    for (int i = 0; i < mGateInputs.size(); i++) {
      backend()->onReleaseBuffer(mGateInputs[i].get(), Backend::DYNAMIC);
    }
    for (int i = 0; i < mGateOutputs.size(); i++) {
      backend()->onReleaseBuffer(mGateOutputs[i].get(), Backend::DYNAMIC);
    }
    mGateInputs.clear();
    mGateOutputs.clear();
    for (int i = 0; i < (mBidirectional ? 8 : 4); i++) {
      mGateInputs.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{batchSize * timeSteps, mStateSize}, Tensor::CAFFE)));
      backend()->onAcquireBuffer(mGateInputs[i].get(), Backend::DYNAMIC);
      mGateOutputs.push_back(shared_ptr<Tensor>(Tensor::createDevice<float>(
          vector<int>{batchSize, mStateSize}, Tensor::CAFFE)));
      backend()->onAcquireBuffer(mGateOutputs[i].get(), Backend::DYNAMIC);
    }

    backend()->onReleaseBuffer(mOutput.get(), Backend::DYNAMIC);
    mOutput.reset(Tensor::createDevice<float>(
        vector<int>{batchSize * timeSteps,
                    mBidirectional ? 2 * mStateSize : mStateSize},
        Tensor::CAFFE));
    backend()->onAcquireBuffer(mOutput.get(), Backend::DYNAMIC);
  }
  mBatchSize = batchSize;
  mTimeSteps = timeSteps;
  return NO_ERROR;
}

ErrorCode BlstmComputer::onExecute(Tensor *input,
                                   const vector<int> &batchLengths,
                                   const vector<shared_ptr<Tensor>> &initH,
                                   const vector<shared_ptr<Tensor>> &initC) {

  MNN_ASSERT(input->buffer().dimensions == 3);
  MNN_ASSERT(input->length(0) == mBatchSize);
  MNN_ASSERT(input->length(1) == mTimeSteps);
  MNN_ASSERT(input->length(2) == mInDim);

  vector<int> lengths = batchLengths;
  if (lengths.size() == 0) {
    // no batchLengths provided
    for (int i = 0; i < mBatchSize; i++) {
      lengths.push_back(mTimeSteps);
    }
  }
  MNN_ASSERT(mBatchSize == lengths.size());

  if (!initH.empty()) {
    MNN_ASSERT(initH.size() == (mBidirectional ? 2 : 1));
    for (int i = 0; i < initH.size(); i++) {
      MNN_ASSERT(initH[i]->length(0) == mBatchSize);
      MNN_ASSERT(initH[i]->length(1) == mStateSize);
    }
  }
  for (int i = 0; i < (mBidirectional ? 2 : 1); i++) {
    // initialize mHiddenStates
    if (initH.empty()) {
      memset(mHiddenStates[i]->host<float>(), 0, mHiddenStates[i]->size());
    } else {
      trimTensor(initH[i].get(), mHiddenStates[i].get());
    }
  }
  if (!initC.empty()) {
    MNN_ASSERT(initC.size() == (mBidirectional ? 2 : 1));
    for (int i = 0; i < initH.size(); i++) {
      MNN_ASSERT(initC[i]->length(0) == mBatchSize);
      MNN_ASSERT(initC[i]->length(1) == mStateSize);
    }
  }
  for (int i = 0; i < (mBidirectional ? 2 : 1); i++) {
    // initialize mCellStates
    if (initC.empty()) {
      memset(mCellStates[i]->host<float>(), 0, mCellStates[i]->size());
    } else {
      trimTensor(initC[i].get(), mCellStates[i].get());
    }
  }

  // copy input to mInput
  trimTensor(input, mInput.get());

  // reshape mInput from (B, T, F) to (B * F, C)
  auto reshaped_input = shared_ptr<Tensor>(Tensor::create(
      vector<int>{mBatchSize * mTimeSteps, mInDim}, halide_type_of<float>(),
      mInput->host<float>(), Tensor::CAFFE));

  // pre-calculate all input related matrix across all timesteps and store
  // results in mGateInputs
  MNN_CONCURRENCY_BEGIN(i, (mBidirectional ? 8 : 4)) {
    int weightIndex = i < 4 ? i : i + 8;
    Math::Matrix::multi(mGateInputs[i].get(), reshaped_input.get(),
                        mWeights[weightIndex].get());
  }
  MNN_CONCURRENCY_END();

  for (int t = 0; t < mTimeSteps; t++) {
    // compute 4(8) gates' output, and store results in mGateOutputs
    MNN_CONCURRENCY_BEGIN(i, (mBidirectional ? 8 : 4)) {
      int weightIndex = i < 4 ? i + 4 : i + 12;
      int biasIndex = i < 4 ? i + 8 : i + 16;
      int tIndex = i < 4 ? t : mTimeSteps - 1 - t; // real timeStep index
      int stateIndex = i < 4 ? 0 : 1;
      // Ht * U
      Math::Matrix::multi(mGateOutputs[i].get(),
                          mHiddenStates[stateIndex].get(),
                          mWeights[weightIndex].get());
      // + bias
      Math::Matrix::add(mGateOutputs[i].get(), mGateOutputs[i].get(),
                        mWeights[biasIndex].get());
      // + Xt * W, obtain from mGateInputs
      for (int b = 0; b < mBatchSize; b++) {
        auto aRowSrc = mGateInputs[i]->host<float>() +
                       (b * mTimeSteps + tIndex) * mStateSize;
        auto bRowSrc = mGateOutputs[i]->host<float>() + b * mStateSize;
        int w = 0;
#ifdef MNN_USE_NEON
        for (; w <= mStateSize - 16; w += 16) {
          float32x4_t a0 = vld1q_f32(aRowSrc + w);
          float32x4_t a1 = vld1q_f32(aRowSrc + w + 4);
          float32x4_t a2 = vld1q_f32(aRowSrc + w + 8);
          float32x4_t a3 = vld1q_f32(aRowSrc + w + 12);
          float32x4_t b0 = vld1q_f32(bRowSrc + w);
          float32x4_t b1 = vld1q_f32(bRowSrc + w + 4);
          float32x4_t b2 = vld1q_f32(bRowSrc + w + 8);
          float32x4_t b3 = vld1q_f32(bRowSrc + w + 12);
          float32x4_t sum0 = vaddq_f32(a0, b0);
          float32x4_t sum1 = vaddq_f32(a1, b1);
          float32x4_t sum2 = vaddq_f32(a2, b2);
          float32x4_t sum3 = vaddq_f32(a3, b3);
          vst1q_f32(bRowSrc + w, sum0);
          vst1q_f32(bRowSrc + w + 4, sum1);
          vst1q_f32(bRowSrc + w + 8, sum2);
          vst1q_f32(bRowSrc + w + 12, sum3);
        }
        for (; w <= mStateSize - 4; w += 4) {
          float32x4_t aa = vld1q_f32(aRowSrc + w);
          float32x4_t bb = vld1q_f32(bRowSrc + w);
          float32x4_t sum = vaddq_f32(aa, bb);
          vst1q_f32(bRowSrc + w, sum);
        }
#endif
        for (; w < mStateSize; ++w) {
          bRowSrc[w] = aRowSrc[w] + bRowSrc[w];
        }
      }
      // activation
      auto src = mGateOutputs[i]->host<float>();
      for (int j = 0; j < mBatchSize * mStateSize; j++) {
        if (i == 1 || i == 5) {
          src[j] = tanhf(src[j]);
        } else {
          src[j] = sigmoid(src[j]);
        }
      }
    }
    MNN_CONCURRENCY_END();

    MNN_CONCURRENCY_BEGIN(i, (mBidirectional ? 2 : 1)) {
      // compute Ct = Nt * It + Ct-1 * Ft
      int gateBase = i * 4;
      // note this is a inplace dot product. Values in next gate will be
      // changed. We just temporally store result in the next gate.
      Math::Matrix::dot(mGateOutputs[gateBase + 1].get(),
                        mGateOutputs[gateBase + 1].get(),
                        mGateOutputs[gateBase].get());
      // also a inplace dot product, same as above
      Math::Matrix::dot(mGateOutputs[gateBase + 2].get(),
                        mGateOutputs[gateBase + 2].get(), mCellStates[i].get());
      Math::Matrix::add(mCellStates[i].get(), mGateOutputs[gateBase + 1].get(),
                        mGateOutputs[gateBase + 2].get());

      // Ht = tanh(Ct) * Ot
      auto hSrc = mHiddenStates[i]->host<float>();
      memcpy(hSrc, mCellStates[i]->host<float>(),
             mStateSize * mBatchSize * sizeof(float));
      for (int j = 0; j < mBatchSize * mStateSize; j++) {
        hSrc[j] = tanhf(hSrc[j]);
      }
      Math::Matrix::dot(mHiddenStates[i].get(), mHiddenStates[i].get(),
                        mGateOutputs[gateBase + 3].get());

      // store hidden states into mOutput
      int tIndex = (i == 0 ? t : mTimeSteps - 1 - t);
      int outDim = mBidirectional ? 2 * mStateSize : mStateSize;
      for (int b = 0; b < mBatchSize; b++) {
        auto hSrc = mHiddenStates[i]->host<float>() + b * mStateSize;
        auto out = mOutput->host<float>() + (b * mTimeSteps + tIndex) * outDim +
                   i * mStateSize;
        if (tIndex >= lengths[b]) {
          // padding, need to reset hidden/cell state and make output zero
          if (!initH.empty()) {
            memcpy(hSrc, initH[i]->host<float>() + b * initH[i]->stride(0),
                   mStateSize * sizeof(float));
          } else {
            memset(hSrc, 0, mStateSize * sizeof(float));
          }
          if (!initC.empty()) {
            memcpy(mCellStates[i]->host<float>() + b * mStateSize,
                   initC[i]->host<float>() + b * initC[i]->stride(0),
                   mStateSize * sizeof(float));
          } else {
            memset(mCellStates[i]->host<float>() + b * mStateSize, 0,
                   mStateSize * sizeof(float));
          }
          // set output to zero
          memset(out, 0, mStateSize * sizeof(float));
        } else {
          // copy hidden state to output
          memcpy(out, hSrc, mStateSize * sizeof(float));
        }
      }
    }
    MNN_CONCURRENCY_END();
  }
  return NO_ERROR;
}

CPUBackend *BlstmComputer::backend() { return mBackend; }

shared_ptr<Tensor> BlstmComputer::output() { return mOutput; }

} // namespace MNN
