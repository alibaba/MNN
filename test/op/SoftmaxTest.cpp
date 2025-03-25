//
//  SoftmaxTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

// axis=0
std::vector<int> expectedOrder0 = {24, 0, 25, 1, 2, 26, 3, 27, 4, 28, 29, 5, 6, 30,31, 7, 8,
                                   32, 33, 9, 34, 10, 35, 11, 12, 36, 13, 37, 14, 38, 39, 15,
                                   40, 16, 41, 17, 18, 42, 19, 43, 20, 44, 21, 45, 46, 22, 23, 47};
std::vector<float> expectedOutput0 = {0.8476,0.5572,0.0111,0.0577,0.0677,0.7076,0.1672,0.9817,0.0977,0.9950,0.9799,
                                      0.6407,0.0136,0.4876,0.3803,0.9829,0.9887,0.8233,0.0055,0.3753,0.0351,0.3318,0.9816,
                                      0.1788,0.1524,0.4428,0.9889,0.9423,0.9323,0.2924,0.8328,0.0183,0.9023,0.0050,0.0201,
                                      0.3593,0.9864,0.5124,0.6197,0.0171,0.0113,0.1767,0.9945,0.6247,0.9649,0.6682,0.0184,0.8212};

// axis=1
std::vector<int> expectedOrder1    = {12, 0, 1, 13, 2, 14, 3, 15, 4, 16, 17, 5, 6, 18, 7, 19, 20, 8, 21,
                                      9, 22, 10, 23, 11, 24, 36, 25, 37, 38, 26, 39, 27, 40,
                                      28, 41, 29, 30, 42, 31, 43, 44, 32, 33, 45, 46, 34, 35, 47};
std::vector<float> expectedOutput1 = {0.9821,0.0270,0.2704,0.0171,0.0254,0.6831,0.4209,0.2000,0.7778,0.9001,0.7266,
                                      0.6005,0.0179,0.9730,0.7296,0.9829,0.9746,0.3169,0.5791,0.8000,0.2222,0.0999,
                                      0.2734,0.3995,0.1200,0.0205,0.9532,0.9424,0.9693,0.8059,0.0197,0.0028,
                                      0.5407,0.0221,0.7436,0.1551,0.8800,0.9795,0.0468,0.0576,0.0307,0.1941,
                                      0.9803,0.9972,0.4593,0.9779,0.2564,0.8449};

// axis=2
std::vector<int> expectedOrder2 = {8, 4, 0, 1, 5, 9, 2, 6, 10, 3, 11, 7, 20, 12, 16, 17, 21, 13, 18,
                                   14, 22, 23, 15, 19, 24, 32, 28, 33, 25, 29, 34, 30, 26, 35, 31, 27,
                                   40, 44, 36, 41, 45, 37, 46, 38, 42, 39, 47, 43};
std::vector<float> expectedOutput2 = {0.8900,0.0196,0.0073,0.0079,0.0624,0.0967,0.0131,0.9669,0.0476,0.8837,0.9796,
                                      0.0252,0.0067,0.8317,0.0483,0.1046,0.9877,0.0528,0.0445,0.8915,0.0056,0.1155,0.9072,
                                      0.0039,0.1097,0.2595,0.8838,0.8002,0.5890,0.6661,0.0890,0.1120,0.3013,0.0743,0.0273,
                                      0.0878,0.7454,0.7818,0.0097,0.0012,0.0173,0.0101,0.9882,0.9870,0.2373,0.2080,0.0021,0.0118};

// axis=3
std::vector<int> expectedOrder3 = {3, 2, 1, 0, 6, 4, 5, 7, 11, 8, 10, 9, 12, 14, 15, 13, 18, 17,
                                   16, 19, 20, 23, 21, 22, 24, 25, 27, 26, 31, 30, 29, 28, 35, 33,
                                   34, 32, 39, 38, 36, 37, 40, 41, 42, 43, 46, 47, 44, 45};
std::vector<float> expectedOutput3 = {0.7560,0.2089,0.0226,0.0125,0.0199,0.3879,0.0154,0.5768,0.0032,0.7505,
                                      0.2431,0.0032,0.0017,0.9046,0.0073,0.0864,0.2334,0.0550,0.0065,0.7051,0.0052,0.4685,0.5144,
                                      0.0119,0.0537,0.0656,0.8001,0.0807,0.5256,0.3069,0.1469,0.0206,0.7381,0.0940,0.1236,
                                      0.0443,0.1104,0.8772,0.0110,0.0014,0.0011,0.0050,0.4956,0.4982,0.1235,0.8205,0.0084,0.0476};

int* orders[] = {expectedOrder0.data(), expectedOrder1.data(), expectedOrder2.data(), expectedOrder3.data()};
float* outputs[] = {expectedOutput0.data(), expectedOutput1.data(), expectedOutput2.data(), expectedOutput3.data()};

static bool checkProbAndOrder(float* gotOutput, const float* expectedOutput, const int* expectedOrder, int size,
                              std::vector<int> shape = {}, int axis = -1) {
    float expectedSum = 0, gotSum = 0;
    std::vector<int> gotOrder(size, 0);
    
    int outside = 1, inside = 1;
    for (int i = 0; i < axis; ++i) {
        outside *= shape[i];
    }
    for (int i = axis + 1; i < shape.size(); ++i) {
        inside *= shape[i];
    }
    
    float errorCase = 0;
    for (int z = 0; z < outside; ++z) {
        for (int x = 0; x < inside; ++x) {
            std::vector<int> orderY(shape[axis], 0);
            float expectedSumY = 0;
            float gotSumY      = 0;
            
            int xz             = x + z * inside * shape[axis];
            for (int y = 0; y < shape[axis]; ++y) {
                int idx = xz + y * inside;
                orderY[y] = idx;
                expectedSumY += expectedOutput[idx];
                gotSumY      += gotOutput[idx];
            }
            sort(orderY.begin(), orderY.end(), [&](const int &a, const int &b) {
                return gotOutput[a] < gotOutput[b];
            });
            float rateY        = 0;
            for (int y = 0; y < shape[axis]; ++y) {
                if (expectedOrder[(x + z *inside) * shape[axis] + y] == orderY[y]) {
                    rateY += 1;
                }
            }
            rateY /= shape[axis];
            float pointRate = gotSumY / expectedSumY;
            if (rateY < 0.5 || pointRate < 0.5 || pointRate > 2.0) {
                errorCase += 1;
            }
        }
    }
    if (errorCase / size > 0.03) {
        MNN_PRINT("softmaxInt8 test on axis = %d, ErrorRate = %f, failed\n", axis, errorCase/size);
        return false;
    }

    return true;
}

static std::vector<float> naiveSoftmax(const float* input, const int outside, const int axis, const int inside) {
    std::vector<float> output(outside * axis * inside, 0);
    for(int y = 0; y < outside; y++) {
        for(int x = 0; x < inside; x++) {
            const float* src = input + y * axis * inside + x;
            float* dst = (float *)(output.data()) + y * axis * inside + x;
            float maxValue = (float)src[0];
            for (int z=1; z<axis; ++z) {
                maxValue = maxValue > src[z * inside] ? maxValue : src[z * inside];
            }
            float sumValue = 0.0;
            for (int z=0; z<axis; ++z) {
                sumValue = sumValue + exp((float)src[z * inside] - maxValue);
            }
            sumValue = 1.0 / sumValue;
            for (int z=0; z<axis; ++z) {
                dst[z*inside] = (exp((float)src[z * inside] - maxValue) * sumValue);
            }
        }
    }
    return output;
}

class SoftmaxTest : public MNNTestCase {
public:
    virtual ~SoftmaxTest() = default;
    virtual bool run(int precision) {
        // testcase 0
        {
            auto input = _Input({1, 4}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 4 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0320586, 0.0871443, 0.236883, 0.643914};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.001)) {
                MNN_ERROR("SoftmaxTest0 test failed!\n");
                return false;
            }
        }
        // testcase 1
        {
            auto input = _Input({2, 4}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0320586, 0.0871443, 0.236883,  0.643914,
                                                       0.643914,  0.236883,  0.0871443, 0.0320586};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 8, 0.001)) {
                MNN_ERROR("SoftmaxTest1 test failed!\n");
                return false;
            }
        }
        // testcase 2
        {
            auto input = _Input({2, 5}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 10 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.0116558, 0.0316853, 0.0861187, 0.234124,  0.636416,
                                                       0.636416,  0.234124,  0.0861187, 0.0316853, 0.0116558};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 10, 0.001)) {
                MNN_ERROR("SoftmaxTest2 test failed!\n");
                return false;
            }
        }
        // testcase 3
        {
            auto input = _Input({2, 2}, NCHW);
            input->setName("input_tensor");
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 4 * sizeof(float));
            input->unMap();
            auto output                             = _Softmax(input);
            const std::vector<float> expectedOutput = {0.7310586, 0.26894143, 0.26894143, 0.7310586};
            auto gotOutput                          = output->readMap<float>();
            if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.001)) {
                MNN_ERROR("SoftmaxTest test failed!\n");
                return false;
            }
        }
        // testcase
        if(1)
        {
            const std::vector<int> axis_vec    = {1, 4, 32, 128, 256, 576, 1024};
            const std::vector<int> outside_vec = {1, 32, 1024, 65536};
            const std::vector<int> inside_vec = {1, 4, 7};
            const int limitSize = 3;

            for(int k = 0; k < outside_vec.size() && k < limitSize; k++) {
                for(int j = 0; j < axis_vec.size() && j < limitSize; j++) {
                    for(int i = 0; i < inside_vec.size(); i++) {
                        int outside = outside_vec[k];
                        int axis = axis_vec[j];
                        int inside = inside_vec[i];
                        auto input = _Input({outside, axis, inside}, NCHW);
                        // set input data
                        auto total = outside * axis * inside;
                        std::vector<float> inpudata(total);
                        for(int i = 0; i < total; i++) {
                            inpudata[i] = 1.0 * i / total;
                        }
                        auto inputPtr          = input->writeMap<float>();
                        memcpy(inputPtr, inpudata.data(), total * sizeof(float));
                        input->unMap();
                        auto output                             = _Softmax(input, 1);

                        std::vector<float> expectedOutput = naiveSoftmax((float *)inpudata.data(), outside, axis, inside);
                        auto gotOutput                          = output->readMap<float>();
                        int count = 0;
                        for (int i = 0; i < expectedOutput.size(); ++i) {
                            if(gotOutput[i] - expectedOutput[i] > 0.01 || gotOutput[i] - expectedOutput[i] < -0.01) {
                                count++;
                            }
                        }
                        if (!checkVector<float>(gotOutput, expectedOutput.data(), total, 0.01)) {
                            MNN_ERROR("SoftmaxTest test failed! error count:%d\n", count);
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};

class SoftmaxInt8Test: public MNNTestCase {
public:
    virtual ~SoftmaxInt8Test() = default;
    virtual bool run(int precision) {
        // testcase 1
        {
            std::vector<int> dimensions = {2, 2, 3, 4};
            auto input = _Input(dimensions, NCHW);
            input->setName("input_tensor");
            // set input data
            float inputData[] = {7.2129,5.9265,3.7045,3.1111,4.5548,7.5229,4.2968,7.9198,4.2842,9.7357,8.6082,
                                  4.2730,3.2067,9.5121,4.6973,7.1634,8.2003,6.7548,4.6160,9.3058,3.0313,7.5376,7.6309,3.8655,
                                  5.4967,5.6967,8.1985,5.9047,7.1774,6.6393,5.9027,3.9387,6.5073,4.4462,4.7199,3.6948,7.4889,
                                  9.5616,5.1855,3.1104,3.7267,5.2157,9.8103,9.8155,6.3442,8.2376,3.6553,5.3901};

            const float quantScales[] = {0.102, 0.00784};
            const float zeroPoints[]  = {1., 2.};
            input->writeScaleMap(quantScales[0], zeroPoints[0]);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inputData, 48 * sizeof(float));
            input->unMap();
            VARP output;
            for (int axis = 0; axis < dimensions.size(); ++axis) {
                output = _Softmax(input, axis);
                output->writeScaleMap(quantScales[1], zeroPoints[1]);
                auto gotOutput                          = output->readMap<float>();

                
                bool result = checkProbAndOrder((float*)gotOutput, outputs[axis], orders[axis], 48, dimensions, axis);
                if (!result) {
                    MNN_PRINT("when axis = %d, SoftmaxInt8 case1 failed!\n", axis);
                    return false;
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxTest, "op/softmax");
MNNTestSuiteRegister(SoftmaxInt8Test, "op/softmaxInt8");
