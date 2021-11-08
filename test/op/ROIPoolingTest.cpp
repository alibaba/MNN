//
//  ROIPoolingTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/27.
//  Copyright ? 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
// #include <MNN/expr/Optimizer.hpp>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _ROIPooling(VARP input, VARP roi, int pooledHeight, int pooledWidth, float spatialScale) {
    std::unique_ptr<RoiPoolingT> roiPooling(new RoiPoolingT);
    roiPooling->pooledHeight = pooledHeight;
    roiPooling->pooledWidth  = pooledWidth;
    roiPooling->spatialScale = spatialScale;

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_ROIPooling;
    op->main.type  = OpParameter_RoiPooling;
    op->main.value = roiPooling.release();

    return (Variable::create(Expr::create(op.get(), {input, roi})));
}

class ROIPoolingTest : public MNNTestCase {
public:
    virtual ~ROIPoolingTest() = default;
    virtual bool run(int precision) override { return testOnBackend(MNN_FORWARD_CPU, "CPU", "ROIPooling"); }

protected:
    static bool testOnBackend(MNNForwardType type, const std::string& deviceName, const std::string& testOpName) {
        const int n = 1, c = 1, h = 16, w = 16;
        const int pooledHeight = 7, pooledWidth = 7;
        const float spatialScale = 1.f / 16;

        const std::vector<float> inputData = {// h = 0
                                              -0.4504, -1.6300, -1.1528, 2.0047, -0.7722, 1.2869, 0.9607, -0.1543,
                                              -1.7898, -0.4389, 1.0762, -0.5312, -0.3816, 0.5593, 2.1539, -0.8473,
                                              // h = 1
                                              1.2878, -0.3931, -0.5860, -2.2671, 0.1664, -0.1624, 0.7083, -0.9036,
                                              -1.8571, -0.9804, 0.4889, -0.7063, -0.3265, -0.3187, -0.4380, 0.6685,
                                              // h = 2
                                              -1.0542, 0.2055, 0.9351, -0.2695, 1.0169, 0.9110, -0.3597, 0.9373,
                                              -0.6850, 0.4412, -0.7418, 0.2520, -0.6617, -1.2510, -2.0578, 1.5503,
                                              // h = 3
                                              -0.0070, -0.6194, 1.1525, -0.1175, -0.5980, 0.6628, -1.5139, 0.5271,
                                              -1.7624, -0.8540, 2.1995, 0.0201, 0.1946, 0.9929, 0.3413, -1.4626,
                                              // h = 4
                                              2.4488, 0.1626, 0.3751, 0.7000, -0.1860, -1.0407, -1.0444, 0.0756,
                                              -1.4499, 0.2524, 0.3682, 1.2193, -1.3560, 2.3694, 0.5913, -1.1003,
                                              // h = 5
                                              -0.7432, -2.1867, -0.9452, -1.4011, 0.2582, 0.4201, 0.1170, 3.1787,
                                              -0.4540, -1.9947, -1.9697, 1.9943, 1.2668, 0.4033, -0.1934, 1.4952,
                                              // h = 6
                                              -1.1622, -0.3598, 0.1791, -0.5496, 0.2055, -0.9481, -0.6539, -1.3166,
                                              -0.2553, 1.1040, -1.1132, 0.6486, 1.3773, 0.4321, -0.6301, -0.0220,
                                              // h = 7
                                              0.7045, -1.3188, 0.9659, 0.3345, 0.1435, 1.4948, -1.3958, 0.8596, -0.2846,
                                              -1.6227, 3.0450, 0.6862, -1.2075, 0.6156, -0.2682, -0.4627,
                                              // h = 8
                                              0.4168, -0.9499, 0.2084, 2.2114, -1.1819, -0.8128, -1.0761, -0.0629,
                                              1.4855, -0.0506, 0.7821, -2.1390, -0.0286, 0.2027, 0.7717, -1.3940,
                                              // h = 9
                                              0.2336, -0.2081, 0.4518, 0.5892, 1.6259, 1.4382, 1.3699, -0.3971, -1.0778,
                                              0.3523, 1.3481, 0.0274, 0.8596, -1.3746, -1.5045, -0.0377,
                                              // h = 10
                                              0.6351, -0.8386, -0.7822, -0.2523, -0.3953, 0.0625, -0.9319, -0.4681,
                                              -1.0337, -0.4972, -2.3686, -0.0097, -0.4136, 1.6763, 0.2910, -1.6629,
                                              // h = 11
                                              -1.4581, 0.6477, -0.9243, -0.7744, -1.4067, -0.4087, -0.3171, 1.6140,
                                              -0.1184, -1.4282, -0.1889, -1.5489, 0.9621, 0.0987, 0.0585, 0.5535,
                                              // h = 12
                                              0.1638, 1.4905, -0.7721, -0.6452, 1.3665, -2.0732, -0.0865, 1.2407,
                                              -1.0586, 0.5204, 1.2189, -0.5717, -0.3990, 0.7323, -0.5211, 0.4576,
                                              // h = 13
                                              -0.6822, -0.0130, 0.6325, 1.7409, -0.4098, -0.1671, 1.3580, -1.3922,
                                              -1.1549, -0.5770, 0.0470, 1.8368, 0.4054, -1.2064, 1.1032, -0.4081,
                                              // h = 14
                                              -1.6945, -0.3223, -0.5065, -0.4902, 0.3292, 0.7854, -0.7723, -0.4000,
                                              0.8361, -2.2352, 0.8832, -0.6669, 0.8367, 0.2200, 0.6050, -0.8180,
                                              // h = 15
                                              1.2200, 1.3076, -0.8782, 1.5257, -0.7750, 0.0775, -1.5619, 0.6683,
                                              -0.3300, 1.3241, -0.0514, 0.3862, 1.1214, 0.0751, 0.0594, -0.4008};
        const std::vector<float> roiData   = {0, 5 / spatialScale, 10 / spatialScale, 10 / spatialScale,
                                            15 / spatialScale};
        // the output data calculated by torchvision.ops.roi_pool function using same input data
        const std::vector<float> outputData = {//
                                               0.0625, 0.0625, -0.4681, -0.4681, -0.4972, -0.4972, -2.3686,
                                               //
                                               0.0625, 0.0625, 1.6140, 1.6140, -0.1184, -0.1889, -0.1889,
                                               //
                                               -0.4087, -0.0865, 1.6140, 1.6140, 0.5204, 1.2189, 1.2189,
                                               //
                                               -0.1671, 1.3580, 1.3580, 1.2407, 0.5204, 1.2189, 1.2189,
                                               //
                                               0.7854, 1.3580, 1.3580, 0.8361, 0.8361, 0.8832, 0.8832,
                                               //
                                               0.7854, 0.7854, 0.6683, 0.8361, 1.3241, 1.3241, 0.8832,
                                               //
                                               0.0775, 0.0775, 0.6683, 0.6683, 1.3241, 1.3241, -0.0514};

        auto input = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
        auto roi   = _Input({1, 1, 1, 5}, NCHW, halide_type_of<float>());
        auto output =
            _ROIPooling(_Convert(input, NC4HW4), _Convert(roi, NC4HW4), pooledHeight, pooledWidth, spatialScale);
        output = _Convert(output, NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        ::memcpy(roi->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(), 0.001)) {
            MNN_ERROR("%s(%s) test failed!\n", testOpName.c_str(), deviceName.c_str());
            return false;
        }

        return true;
    }
};

MNNTestSuiteRegister(ROIPoolingTest, "op/ROIPooling");
