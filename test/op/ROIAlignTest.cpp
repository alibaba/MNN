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


class ROIAlignTest : public MNNTestCase {
public:
    virtual ~ROIAlignTest() = default;
    virtual bool run(int precision) override { return testOnBackend(MNN_FORWARD_CPU, "CPU", "ROIAlign", precision); }

protected:
    static bool test(MNNForwardType type, const std::string& deviceName, const std::vector<float>& inputData,
                     const std::vector<float>& roiData, const std::vector<float>& outputData,
                     int numRoi, int batch, int channel, int height, int width, int outputHeight, int outputWidth,
                     PoolType poolType, float samplingRatio, float spatialScale, bool aligned, int precision) {
        auto input  = _Input({batch, channel, height, width}, NCHW, halide_type_of<float>());
        auto rois   = _Input({numRoi, 5}, NCHW, halide_type_of<float>());
        auto output = _ROIAlign(_Convert(input, NC4HW4), rois, outputHeight, outputWidth, spatialScale, samplingRatio, aligned, (PoolingMode)poolType);
        output      = _Convert(output, NCHW);
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        ::memcpy(rois->writeMap<float>(), roiData.data(), roiData.size() * sizeof(float));
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 200;
        if (!checkVectorByRelativeError<float>(output->readMap<float>(), outputData.data(), outputData.size(),
                                               0.001 * errorScale)) {
            MNN_ERROR("ROIAlign(%s) on %s test failed!, errorScale:%f\n", EnumNamePoolType(poolType), deviceName.c_str(), errorScale);
            return false;
        }
        return true;
    }
    static bool testOnBackend(MNNForwardType type, const std::string& deviceName, const std::string& testOpName, int precision) {
        const int n = 2, c = 3, h = 4, w = 4, roi = 2;
        const int pooledHeight = 3, pooledWidth = 3;
        const int samplingRatio  = 2;
        const float spatialScale = 1.f / 16;
        const std::vector<float> inputData = {
            // [0, 0, :, :]
            0.021263, -2.472217, -2.701763, -4.781910, -0.080594, 3.634799, -4.453195, 0.487596,
            -7.652394, -6.803094, -9.063873, 9.414629, -9.922793, -6.428401, 2.257335, -8.372608,
            // [0, 1, :, :]
            7.637930, 4.392403, 9.327800, 0.152711, -3.991926, 0.990011, 8.616374, 0.415229,
            -4.655859, 7.547976, -2.561625, -9.972333, -5.046299, -3.635330, 7.175550, -0.829937,
            // [0, 2, :, :]
            -1.108254, -3.277955, 7.613563, 8.900536, 9.837807, -2.465175, 9.322948, 5.837591,
            3.513783, -5.102211, -5.670855, -6.679043, 8.455132, -4.118467, -0.938115, -0.120843,
            // [1, 0, :, :]
            5.563432, 6.884699, -7.218546, -1.461913, 6.857098, 6.360666, -7.951725, -6.872333,
            -3.916026, -8.492819, -1.506740, -7.847646, 1.364352, -5.068861, 1.928661, -7.649487,
            // [1, 1, :, :]
            9.517677, 8.651224, -2.164061, -5.156428, -4.992036, -0.332129, -9.200144, 2.794102,
            -1.833942, -2.451869, 6.187299, 4.180709, 9.086677, -2.961275, 7.950855, 5.399343,
            // [1, 2, :, :]
            -2.851507, 2.433309, -4.228601, 7.487998, -7.751454, -5.751313, -6.339334, -1.939480,
            4.904659, 0.538149, -0.246474, -9.989080, -1.491966, -8.728925, -5.834935, 8.647879
        };
        const std::vector<float> roiData   = {
            0, 32.692989, 42.911362, 60.044643, 56.948814,
            1, 25.419289, 36.165802, 61.743431, 49.595341
        };
        // case1 avepool align=false
        {
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                -1.499167, -4.919542, -5.680096, -1.231677, -7.073600, -8.372608, -1.231677, -7.073600, -8.372608,
                // [0, 1, :, :]
                3.103553, -1.246561, -2.213851, 4.547950, 0.148356, -0.829937, 4.547950, 0.148356, -0.829937,
                // [0, 2, :, :]
                -1.476973, -1.179685, -1.113580, -0.669866, -0.220716, -0.120843, -0.669866, -0.220716, -0.120843,
                // [1, 0, :, :]
                -1.419553, -5.630120, -7.763026, -0.359157, -5.266063, -7.696973, 0.403175, -5.004337, -7.649487,
                // [1, 1, :, :]
                5.698225, 5.319514, 4.701102, 6.187745, 5.775888, 5.107313, 6.539666, 6.103982, 5.399343,
                // [1, 2, :, :]
                -2.673161, -2.196904, -2.030554, -4.041203, 1.785344, 4.181765, -5.024705, 4.648232, 8.647879,
            };
            bool pass = test(type, deviceName, inputData, roiData, outputData,
                 roi, n, c, h, w, pooledHeight, pooledWidth, PoolType_AVEPOOL, samplingRatio, spatialScale, false, precision);
            if (!pass) {
                return false;
            }
        }
        // case2 avepool align=true
        {
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                -5.577190, -1.795600, 2.798040, -2.816240, -1.873230, -1.660720, -0.055281, -1.950850, -6.119470,
                // [0, 1, :, :]
                1.191170, -2.393580, -6.308030, 2.987830, 0.384787, -3.619180, 4.784500, 3.163160, -0.930332,
                // [0, 2, :, :]
                -4.231310, -4.280490, -4.491040, -3.035570, -2.683920, -2.619710, -1.839820, -1.087340, -0.748386,
                // [1, 0, :, :]
                -4.740460, -3.403250, -7.166200, -4.612350, -2.438020, -7.089820, -3.652890, -1.679590, -6.939920,
                // [1, 1, :, :]
                0.979112, 4.515910, 4.106030, 1.682760, 6.033700, 4.619630, 1.837270, 6.492980, 4.976480,
                // [1, 2, :, :]
                -0.446526, -2.719200, -8.317140, -1.187400, -2.457090, -6.072510, -3.299430, -2.503520, -1.565150
            };
            bool pass = test(type, deviceName, inputData, roiData, outputData,
                 roi, n, c, h, w, pooledHeight, pooledWidth, PoolType_AVEPOOL, samplingRatio, spatialScale, true, precision);
            if (!pass) {
                return false;
            }
        }
        // case3 maxpool align=false
        {
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                -0.119745, -3.269570, -4.197830, 0.282628, -5.774590, -8.372610, 0.282628, -5.774590, -8.372610,
                // [0, 1, :, :]
                5.033390, 0.494710, -1.451980, 5.688380, 1.126650, -0.829937, 5.688380, 1.126650, -0.829937,
                // [0, 2, :, :]
                -0.933916, -0.567062, -0.567062, -0.553440, -0.120843, -0.120843, -0.553440, -0.120843, -0.120843,
                // [1, 0, :, :]
                -1.002520, -4.026910, -7.746510, -0.026014, -3.458700, -7.680460, 0.431881, -3.192260, -7.649490,
                // [1, 1, :, :]
                6.730320, 5.865850, 4.802650, 7.289790, 6.356580, 5.208870, 7.552130, 6.586700, 5.399340,
                // [1, 2, :, :]
                -2.273750, -0.705452, -0.477473, -2.989470, 4.804520, 5.734850, -3.571700, 7.388210, 8.647880,
            };
            bool pass = test(type, deviceName, inputData, roiData, outputData,
                 roi, n, c, h, w, pooledHeight, pooledWidth, PoolType_MAXPOOL, samplingRatio, spatialScale, false, precision);
            if (!pass) {
                return false;
            }
        }
        // case4 maxpool align=true
        {
            const std::vector<float> outputData = {
                // [0, 0, :, :]
                -4.583230, -0.201513, 4.877600, -1.366240, -1.491830, -0.324216, 1.850760, -0.513223, -4.483520,
                // [0, 1, :, :]
                1.884550, -0.609283, -4.968110, 4.064350, 2.193860, -2.264070, 6.732580, 4.997010, 0.439960,
                // [0, 2, :, :]
                -3.799060, -3.842080, -3.999150, -2.447130, -2.247990, -2.129340, -1.095200, -0.575362, -0.211416,
                // [1, 0, :, :]
                -3.759420, -1.910840, -6.541990, -3.050260, -0.895684, -6.306590, -2.090200, 0.034099, -6.062210,
                // [1, 1, :, :]
                2.938530, 5.673670, 4.470930, 3.463310, 6.553910, 4.932330, 3.738130, 7.042040, 5.305090,
                // [1, 2, :, :]
                0.134304, -0.750994, -7.091430, -0.587566, -1.106860, -4.561140, -2.648540, -1.930330, -0.115905,
            };
            bool pass = test(type, deviceName, inputData, roiData, outputData,
                 roi, n, c, h, w, pooledHeight, pooledWidth, PoolType_MAXPOOL, samplingRatio, spatialScale, true, precision);
            if (!pass) {
                return false;
            }
        }

        return true;
    }
};
MNNTestSuiteRegister(ROIAlignTest, "op/ROIAlign");
