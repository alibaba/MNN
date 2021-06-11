//
//  DepthToSpaceTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
using namespace MNN::Express;
class DepthToSpaceTest : public MNNTestCase {
public:
    virtual ~DepthToSpaceTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 8, 2, 3}, NCHW);
        input->setName("input");
        // set input data
        const float input_data[] = {
             0.,  1.,  2.,  3.,  4.,  5.,
             9., 10., 11., 12., 13., 14.,
            18., 19., 20., 21., 22., 23.,
            27., 28., 29., 30., 31., 32.,
            36., 37., 38., 39., 40., 41.,
            45., 46., 47., 48., 49., 50.,
            54., 55., 56., 57., 58., 59.,
            63., 64., 65., 66., 67., 68.
        };
        auto inputPtr = input->writeMap<float>();
        memcpy(inputPtr, input_data, 48 * sizeof(float));
        input->unMap();
        std::unique_ptr<MNN::OpT> depthToSpaceOp(new MNN::OpT);
        depthToSpaceOp->type = MNN::OpType_DepthToSpace;
        auto depthtospaceParam = new MNN::DepthSpaceParamT;
        depthtospaceParam->blockSize = 2;
        depthToSpaceOp->main.type = MNN::OpParameter_DepthSpaceParam;
        depthToSpaceOp->main.value = depthtospaceParam;
        depthtospaceParam->mode = MNN::DepthToSpaceMode_DCR;
        auto outputDCR = Variable::create(Expr::create(depthToSpaceOp.get(), {input}));
        const std::vector<float> expectedOutputDCR = {
            0., 18.,  1., 19.,  2., 20.,
            36., 54., 37., 55., 38., 56.,
            3., 21.,  4., 22.,  5., 23.,
            39., 57., 40., 58., 41., 59.,
            9., 27., 10., 28., 11., 29.,
            45., 63., 46., 64., 47., 65.,
            12., 30., 13., 31., 14., 32.,
            48., 66., 49., 67., 50., 68.
        };
        depthtospaceParam->mode = MNN::DepthToSpaceMode_CRD;
        auto outputCRD  = Variable::create(Expr::create(depthToSpaceOp.get(), {input}));
        const std::vector<float> expectedOutputCRD = {
            0., 9., 1., 10., 2., 11.,
            18., 27., 19., 28., 20., 29.,
            3., 12., 4., 13., 5., 14.,
            21., 30., 22., 31., 23., 32.,
            36., 45., 37., 46., 38., 47.,
            54., 63., 55., 64., 56., 65.,
            39., 48., 40., 49., 41., 50.,
            57., 66., 58., 67., 59., 68.
        };
        auto check = [](VARP output, const std::vector<float>& expectedOutput) {
            const std::vector<int> expectedDim      = {1, 2, 4, 6};
            auto gotOutput                          = output->readMap<float>();
            auto gotDim                             = output->getInfo()->dim;
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 48, 0)) {
                MNN_ERROR("DepthToSpaceTest test failed!\n");
                return false;
            }
            if (!checkVector<int>(gotDim.data(), expectedDim.data(), 4, 0)) {
                MNN_ERROR("DepthToSpaceTest test failed!\n");
                return false;
            }
            return true;
        };
        if (!check(outputDCR, expectedOutputDCR)) {
            MNN_ERROR("DCR mode failed!\n");
            return false;
        }
        if (!check(outputCRD, expectedOutputCRD)) {
            MNN_ERROR("CRD mode failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(DepthToSpaceTest, "op/depthtospace");
