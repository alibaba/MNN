//
//  BlstmComputer.hpp
//  MNN
//
//  Created by MNN on 2020/04/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BLSTMCOMPUTER_hpp
#define BLSTMCOMPUTER_hpp

#include <memory>
#include <vector>

#include "MNN/ErrorCode.hpp"
#include "MNN_generated.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

using std::shared_ptr;
using std::vector;

namespace MNN {

class BlstmComputer {
  /**
      Blstm:
      Xt = input at timestep t
      Ct-1 = cell state of last time step
      O =  sigmoid activation
      x = matrix product
      * =  matrix dot product
      Input gate:   It = Og(Xt x Wi + Ht-1 x Ui + Bi)
      Next gate:    Nt = tanh(Xt x Wn + Ht-1 x Un + Bn)
      Forget gate:  Ft = Og(Xt x Wf + Ht-1 x Uf + Bf)
      Output gate:  Ot = Og(Xt x Wo + Ht-1 x Uo + Bo)
      Cell state:   Ct = Nt * It + Ct-1 * Ft
      Hidden state: Ht = tanh(Ct) * Ot
      output : Ht

      Suppose input is a (Batch, Timestep, Feature) tensor
      General usage:
          (1). Construct a BlstmComputer* blstm = new BlstmComputer();
          (2). Call blstm.importWeights() to import weight into this blstm.
          (3). Upon every execution, first blstm.onResize(), then
     blstm.onExecute() This is a single layer blstm. If you want to construct a
     multi-layer blstm, you can just construct multiple blstm instances with
     proper args and connect them together.
  */

public:
  /**
   * @brief construct the BlstmComputer instance.
   * @param inDim input dimension, correspond to 'Feature' in input(Batch,
   * Timestep, Feature)
   * @param stateSize hidden state & cell state size.
   * @param bidirectional if this is a bidirectional or unidirectional lstm
   * @param backend backend
   */
  BlstmComputer(int inDim, int stateSize, bool bidirectional,
                MNN::CPUBackend *backend);
  virtual ~BlstmComputer();
  /**
   * @brief sigmoid activation function
   */
  static float sigmoid(float x);

  /**
   * @brief trim tensor into correct storage order. For NCHW and NHWC, data will
   * be directly copied, interal storage order will not be changed. For NC4HW4,
   * onCopyBuffer() will be used, interal storage order will be changed.
   */
  void trimTensor(Tensor *src_tensor, Tensor *tgt_tensor);

  /**
   * @brief allocate space for all the weights and bias. And import data from
   weightsVec.
   * @param weightsVec
      WeightsVec must has the same order as mWeights. This method will copy each
   tensor in WeightsVec to corresponding mWeight. for weightsVec[0-3, 12-15],
   shape = (mInDim, mStateSize) for weightsVec[4-7, 16-19], shape = (mStateSize,
   mStateSize) for weightsVec[8-11, 20-23], shape = (mStateSize) For
   bidirectional blstm, WeightsVec's size must equals to 24. For unidirectional
   lstm, WeightsVec's size must equals to 12. By default, tensor in weightsVec
   should be a NCHW or NHWC format tensor. If a NC4HW4 is passed, this method
   will transform it into NCHW format tensor, and the internel storage order
   might be changed. Thus, user should handle the data storage correctly.
   */
  ErrorCode importWeights(const vector<shared_ptr<Tensor>> &weightsVec);
  /**
   * @brief Need to be called before every onExecute(). This method will resize
   * the internal tensors' memory which are used by calculation process.
   * @param timeSteps, input's timeSteps
   * @param batchSize, input's batchSize
   */
  ErrorCode onResize(int timeSteps, int batchSize);
  /**
   * @param input input tensor, shape = (B, T, F). Should be a NCHW or
   * NHWCtensor. If a NC4HW4 is passed, this method will transform it into NCHW
   * format, thus the internal storage order will be changed. User
   * should handle the data storage order correctly.
   * @param batchLengths length for each data slot in this batch. If current
   * timestep > length, this data slot's output will be set to 0.
   * @param initH initial HiddenState of this blstm. Each element of initH
   * should be a (Batch, mStateSize) tensor. If bidirectional, initH.size() must
   * = 2. If unidirectional, initH.size() must = 1. If not provide
   * initH, it will be initialized to all 0.
   * @param initC initial CellState of this blstm. Each element of initC should
   * be a (Batch, mStateSize) tensor. If bidirectional, initC.size() must = 2.
   * If unidirectional, initC.size() must = 1. If not provide
   * initC, it will be initialized to all 0.
   */
  ErrorCode onExecute(Tensor *input, const vector<int> &batchLengths = {},
                      const vector<shared_ptr<Tensor>> &initH = {},
                      const vector<shared_ptr<Tensor>> &initC = {});

  /**
   * @brief get the output tensor of this blstm.
   */
  shared_ptr<Tensor> output();
  /**
   * @brief get backend instance stored in this blstm instance.
   */
  CPUBackend *backend();

private:
  int mInDim;          //  dimension for input' Feature
  int mStateSize;      // dimension for hidden state and cell_state.
  bool mBidirectional; // uni or bidirectional of this blstm
  int mBatchSize = 0;
  int mTimeSteps = 0;
  shared_ptr<Tensor> mInput;  // (B, T, F) tensor
  shared_ptr<Tensor> mOutput; // (B, T, F) tensor
  vector<shared_ptr<Tensor>> mGateInputs;
  vector<shared_ptr<Tensor>> mGateOutputs;
  // mHiddenStates[0] is hidden state forward. mHiddenStates[1] = hidden state
  // backward if bidirectional
  vector<shared_ptr<Tensor>> mHiddenStates;
  // mCellStates[0] is cell state forward. mCellStates[1] = cell state backward
  // if bidirectional
  vector<shared_ptr<Tensor>> mCellStates;
  /*
  mWeights[0] : Wi forward, shape = (mInDim, mStateSize)
  mWeights[1] : Wn forward, shape = (mInDim, mStateSize)
  mWeights[2] : Wf forward, shape = (mInDim, mStateSize)
  mWeights[3] : Wo forward, shape = (mInDim, mStateSize)
  mWeights[4] : Ui forward, shape = (mStateSize, mStateSize)
  mWeights[5] : Un forward, shape = (mStateSize, mStateSize)
  mWeights[6] : Uf forward, shape = (mStateSize, mStateSize)
  mWeights[7] : Uo forward, shape = (mStateSize, mStateSize)
  mWeights[8] : Bi forward, shape = (mStateSize)
  mWeights[9] : Bn forward, shape = (mStateSize)
  mWeights[10] : Bf forward, shape = (mStateSize)
  mWeights[11] : Bo forward, shape = (mStateSize)
  mWeights[12] : Wi backward if bidirectional, shape = (mInDim, mStateSize)
  mWeights[13] : Wn backward if bidirectional, shape = (mInDim, mStateSize)
  mWeights[14] : Wf backward if bidirectional, shape = (mInDim, mStateSize)
  mWeights[15] : Wo backward if bidirectional, shape = (mInDim, mStateSize)
  mWeights[16] : Ui backward if bidirectional, shape = (mStateSize, mStateSize)
  mWeights[17] : Un backward if bidirectional, shape = (mStateSize, mStateSize)
  mWeights[18] : Uf backward if bidirectional, shape = (mStateSize, mStateSize)
  mWeights[19] : Uo backward if bidirectional, shape = (mStateSize, mStateSize)
  mWeights[20] : Bi backward if bidirectional, shape = (mStateSize)
  mWeights[21] : Bn backward if bidirectional, shape = (mStateSize)
  mWeights[22] : Bf backward if bidirectional, shape = (mStateSize)
  mWeights[23] : Bo backward if bidirectional, shape = (mStateSize)
  */
  vector<shared_ptr<Tensor>> mWeights;
  MNN::CPUBackend *mBackend;

  /*
      To make it more clear for users about how to wrap weights and input of
    this blstm , we provide a simple example below.

      Suppose we have a blstm and input, with
      Batch = 2, Timestep = 2, F(inDim) = 3, stateSize = 2, bidirectional =
    true.

      Say if you want a input like this:
                           | - F - |
      timestep1, batch1    1,  2,  3
      timestep2, batch1    4,  5,  6
      timestep1, batch2    7,  8,  9
      timestep2, batch2    10, 11, 12

      If you pass a NCHW or NHWC tensor as input, the internal storage order
    should be: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12

      Also, if you want Wi to be like this:
      | - stateSize - |
    --|    1,    2    |
    F |    3,    4    |
    --|    5,    6    |

      If you pass a NCHW or NHWC tensor as Wi, the internal storage order should
    be: 1, 2, 3, 4, 5, 6

      Then input matrix can then multiply with Wi.

      So the general principle of warpping input, weight, initH/initC is:
          1. if you use NCHW/NHWC as source, make sure interal data is stored -1
    dim, then -2 dim ....
          2. if you use NC4HW4 as source, make sure after using onCopyBuffer(),
    the resulting tensor is stored -1 dim, then -2 dim .... interally.
  */
};

} // namespace MNN

#endif /* BLSTMCOMPUTER_hpp */
