//
//  ROIAlignTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/11/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _ROIAlign(VARP intput, VARP rois, int pooledWidth, int pooledHeight, int samplingRatio, float spatialScale,
                      bool aligned, PoolType poolType) {
    std::unique_ptr<RoiParametersT> roiAlign(new RoiParametersT);
    roiAlign->pooledWidth   = pooledWidth;
    roiAlign->pooledHeight  = pooledHeight;
    roiAlign->samplingRatio = samplingRatio;
    roiAlign->spatialScale  = spatialScale;
    roiAlign->aligned       = aligned;
    roiAlign->poolType      = poolType;

    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_ROIAlign;
    op->main.type  = OpParameter_RoiParameters;
    op->main.value = roiAlign.release();

    return (Variable::create(Expr::create(op.get(), {intput, rois})));
}

class ROIAlignTest : public MNNTestCase {
public:
    virtual ~ROIAlignTest() = default;
    virtual bool run(int precision) override { return testOnBackend(MNN_FORWARD_CPU, "CPU", "ROIAlign"); }

protected:
    static bool testOnBackend(MNNForwardType type, const std::string& deviceName, const std::string& testOpName) {
        // case1
        {
            const int n = 3, c = 5, h = 4, w = 4;
            const int pooledHeight = 3, pooledWidth = 3;
            const int samplingRatio  = 2;
            const float spatialScale = 1.f / 16;
            const bool aligned       = true;
            const PoolType poolType  = PoolType_AVEPOOL;

            const std::vector<float> inputData = {// [0, 0, :, :]
                                                  -0.2280, 1.0844, -0.5641, 1.0726, -1.3492, 1.7201, 0.9563, -0.0467,
                                                  0.6715, 1.9948, -0.8688, 0.7443, 1.2673, -0.8053, -0.5623, -1.0905,
                                                  // [0, 1, :, :]
                                                  1.3876, 0.0851, -0.2216, 0.9205, -0.2518, 0.3693, -0.8745, -0.8473,
                                                  -1.2684, -0.5659, 1.7629, -1.1017, -0.6808, -0.3398, 0.3905, -1.4468,
                                                  // [0, 2, :, :]
                                                  0.3544, -1.3111, -0.4820, -0.7897, -0.6373, 0.2016, 0.6290, -0.4031,
                                                  -1.1968, 1.2157, 0.7829, 0.0650, 1.2640, -0.1800, 0.5405, 0.5089,
                                                  // [0, 3, :, :]
                                                  -1.0524, -0.3499, -1.0782, 2.3311, 0.3991, -1.0441, 1.4817, -0.1174,
                                                  -1.2182, 1.6637, 0.0859, 0.2630, 0.1632, -0.7833, 1.9403, 2.7318,
                                                  // [0, 4, :, :]
                                                  1.9020, -0.6179, -1.9794, -0.5269, 1.7679, -1.0336, 0.6070, 0.9047,
                                                  0.5333, 0.0791, -1.3838, -2.0784, 0.9168, -0.2555, 1.6408, 0.7012,
                                                  // [1, 0, :, :]
                                                  -1.3412, 0.1940, 1.1810, -0.1982, -0.6593, 1.3539, -1.0911, 0.7103,
                                                  0.0096, 0.4430, -0.0726, 0.3056, -0.5561, -0.0727, 1.3280, -0.1100,
                                                  // [1, 1, :, :]
                                                  2.1148, -1.5199, -1.7310, -1.2742, -0.5361, -0.1515, 0.3356, -0.4218,
                                                  -0.0041, -0.1434, 1.1420, 0.5182, -0.0388, 1.4477, -0.0358, -1.3596,
                                                  // [1, 2, :, :]
                                                  -0.5037, -1.2696, 0.1072, 0.4771, 1.0347, -0.8084, 0.0574, -0.2152,
                                                  -0.4784, -0.1899, -0.4062, -1.2771, -0.1168, 0.9634, 0.8064, 1.3298,
                                                  // [1, 3, :, :]
                                                  0.9451, 0.5655, 1.3067, -2.3284, -1.0857, 0.0211, 0.2268, -0.4762,
                                                  0.4864, -0.0473, 0.1158, 0.1223, -0.2063, -0.6610, -1.3957, 0.5784,
                                                  // [1, 4, :, :]
                                                  -0.6520, 1.7823, 1.6848, -1.3009, 0.8962, 0.7669, -0.1907, 0.0051,
                                                  -0.2227, 0.0253, -0.3897, -1.2509, 1.0388, 1.2419, 1.3504, 0.9589,
                                                  // [2, 0, :, :]
                                                  -0.6821, -1.3088, 0.6674, 0.3037, -0.6739, 0.7570, 1.2381, 0.9869,
                                                  0.6168, 0.3319, 0.0818, -1.4019, 0.4963, -0.6783, -0.3395, 0.1034,
                                                  // [2, 1, :, :]
                                                  -1.6103, 0.1561, -1.8178, -0.6959, 0.2309, -2.1099, 0.2700, 1.1527,
                                                  -0.1562, -0.8010, -1.7923, 1.9186, 0.0420, 1.0442, 0.4630, -1.7146,
                                                  // [2, 2, :, :]
                                                  2.0714, 0.6615, 0.4553, -0.2865, -0.5504, 1.7192, 1.1452, -0.1363,
                                                  0.8048, 0.9660, 1.3715, -1.0151, -1.2480, -0.3135, -0.3928, -0.2055,
                                                  // [2, 3, :, :]
                                                  0.9049, -2.9842, 0.1725, 0.6841, -0.7629, 0.0941, 0.1685, 0.2651,
                                                  0.2957, 0.7492, 1.0405, -1.3762, -0.2437, 1.3722, 0.0890, 0.2810,
                                                  // [2, 4, :, :]
                                                  0.4548, -0.4797, 0.3163, -0.3530, 1.0514, -0.9240, 1.1464, 0.1866,
                                                  -0.1598, 0.1525, 0.9954, 2.0155, -0.0096, -0.3440, 1.0122, 0.5473};
            const std::vector<float> roiData   = {
                //
                0, 0 / spatialScale, 1 / spatialScale, 2 / spatialScale, 3 / spatialScale,
                //
                2, 0.5f / spatialScale, 1 / spatialScale, 1.5f / spatialScale, 2 / spatialScale};
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                -1.1623, 0.2259, 1.4623, -0.3388, 0.7593, 1.5552, 0.7708, 1.1495, 1.1371,
                // [0, 1, :, :]
                0.0214, 0.1717, 0.1407, -0.7601, -0.4292, -0.0079, -1.1705, -0.8493, -0.1845,
                // [0, 2, :, :]
                -0.4720, -0.2613, 0.0319, -0.9171, -0.1042, 0.7082, -0.7867, 0.0982, 0.9430,
                // [0, 3, :, :]
                0.1572, -0.3856, -0.5978, -0.4096, -0.0499, 0.3888, -0.9880, 0.1340, 1.1124,
                // [0, 4, :, :]
                1.7902, 0.4130, -0.7743, 1.1506, 0.3367, -0.4624, 0.5972, 0.3103, -0.1272,
                // [1, 0, :, :]
                -0.5525, -0.3041, -0.0558, -0.4082, 0.0164, 0.4409, -0.1005, 0.1858, 0.4721,
                // [1, 1, :, :]
                -0.5448, -0.8687, -1.1926, -0.2118, -0.9114, -1.6111, -0.1940, -0.7859, -1.3777,
                // [1, 2, :, :]
                0.4974, 0.8451, 1.1928, -0.0466, 0.6295, 1.3057, 0.1625, 0.6847, 1.2070,
                // [1, 3, :, :]
                -0.3278, -0.5695, -0.8112, -0.5422, -0.3281, -0.1139, -0.2896, -0.0488, 0.1921,
                // [1, 4, :, :]
                0.5811, 0.0383, -0.5045, 0.6700, 0.0577, -0.5545, 0.4455, 0.0412, -0.3630};

            auto input  = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
            auto rois   = _Input({2, 5}, NCHW, halide_type_of<float>());
            auto output = _ROIAlign(_Convert(input, NC4HW4), rois, pooledWidth, pooledHeight, samplingRatio,
                                    spatialScale, aligned, poolType);
            output      = _Convert(output, NCHW);
            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(rois->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
            if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                                   0.001)) {
                MNN_ERROR("%s(%s) test case1 failed!\n", testOpName.c_str(), deviceName.c_str());
                return false;
            }
        }

        // case2
        {
            const int n = 2, c = 3, h = 5, w = 5;
            const int pooledHeight = 3, pooledWidth = 3;
            const int samplingRatio  = -1;
            const float spatialScale = 1.f / 8;
            const bool aligned       = false;
            const PoolType poolType  = PoolType_MAXPOOL;

            const std::vector<float> inputData = {
                // [0, 0, :, :]
                -1.1024, -0.0894, -1.0598, 0.6663, 0.2478, 0.4226, -0.7333, 0.3157, -0.5616, 0.6324, -1.0922, 1.3422,
                -0.8091, 1.0461, -1.0291, -1.2514, -1.2338, -1.1748, 0.2466, -0.5955, -0.4100, 0.6049, -0.1460, 0.7214,
                -0.8371,
                // [0, 1, :, :]
                1.3101, -1.0658, -0.2126, 0.0553, 1.2623, 1.0555, 0.1225, 0.5201, 1.6511, -0.2565, 0.1568, -0.1875,
                -0.2221, 0.9786, 1.4340, 0.2992, 0.6380, -0.5892, 0.5646, 0.7691, -0.7870, 0.0738, 0.3016, -1.4887,
                0.3936,
                // [0, 2, :, :]
                -1.9837, -0.5665, -1.0169, 0.4224, -0.1374, 1.0267, 1.0494, 0.2903, 0.4389, -0.1971, 0.3844, -0.5516,
                0.7392, 1.8795, 0.7050, 1.3164, -0.7506, -0.1962, 0.5041, -0.9073, -0.4222, -0.2292, -0.4276, 0.3254,
                -0.7846,
                // [1, 0, :, :]
                -1.1626, 0.7566, 1.5080, -0.3796, -0.1129, -0.0329, 1.2424, 1.6193, 0.0834, 1.4959, 0.3852, 1.3772,
                -1.2080, -2.0377, 0.6375, -1.8957, -1.1267, -0.1389, -1.2036, 0.4766, -2.4038, 0.9020, 0.7487, -1.1495,
                0.7606,
                // [1, 1, :, :]
                0.5755, 1.5765, 0.4307, 0.3414, -1.8102, 0.8271, -1.3906, 0.9912, -0.8921, -0.3487, -0.8493, 0.0274,
                -0.4842, -0.7655, -0.6322, -1.0093, -0.0490, -0.7346, 0.0005, 0.7290, -1.8860, 0.4296, -0.5291, -1.6982,
                0.6107,
                // [1, 2, :, :]
                1.1784, 1.6321, -0.0142, -1.0323, -0.7908, 0.9712, 1.1432, 1.2515, 0.7435, -0.3449, -0.9092, -0.4958,
                1.1608, 1.5287, 0.8189, -0.6357, 0.0325, 0.2310, 0.3238, 2.3113, 0.0518, -0.3411, -1.5270, 1.8309,
                0.3836};
            const std::vector<float> roiData = {
                //
                0, 0.f / spatialScale, 1.5f / spatialScale, 4.f / spatialScale, 3.5f / spatialScale,
                //
                1, 0.5f / spatialScale, 2.f / spatialScale, 4.f / spatialScale, 4.f / spatialScale};
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                0.9963, -0.0823, 0.7782, 0.0542, -0.4458, 0.6464, -0.9273, -0.5603, 0.3257,
                // [0, 1, :, :]
                0.1591, 0.2980, 1.1317, 0.2271, -0.0132, 0.9916, 0.5440, -0.1125, 0.5451,
                // [0, 3, :, :]
                0.2327, 0.9894, 1.6394, 0.3499, 0.5783, 1.1918, 0.4632, 0.0016, 0.4743,
                // [1, 0, :, :]
                0.3514, -0.7935, -0.0997, -0.7563, -0.1801, -0.0135, 0.3109, 0.4434, 0.1312,
                // [1, 1, :, :]
                -0.1865, -0.5365, -0.2752, -0.2491, -0.3364, 0.5165, -0.0553, -0.5614, 0.1304,
                // [1, 2, :, :]
                0.1193, 1.0005, 1.2612, 0.1069, 0.2813, 1.7316, -0.2084, 0.2883, 1.2907};

            auto input  = _Input({n, c, h, w}, NCHW, halide_type_of<float>());
            auto rois   = _Input({2, 5}, NCHW, halide_type_of<float>());
            auto output = _ROIAlign(_Convert(input, NC4HW4), rois, pooledWidth, pooledHeight, samplingRatio,
                                    spatialScale, aligned, poolType);
            output      = _Convert(output, NCHW);
            ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
            ::memcpy(rois->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
            if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                                   0.001)) {
                MNN_ERROR("%s(%s) test case2 failed!\n", testOpName.c_str(), deviceName.c_str());
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(ROIAlignTest, "op/ROIAlign");