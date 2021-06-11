//
//  StridedSliceTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/1/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class StridedSliceTest : public MNNTestCase {
public:
    virtual ~StridedSliceTest() = default;
    virtual bool run(int precision) {
        auto input  = _Input({1, 3, 2, 3}, NCHW);
        auto begin  = _Input({3}, NCHW);
        auto end    = _Input({3}, NCHW);
        auto strided = _Input({3}, NCHW);
        const float input_data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
        memcpy(input->writeMap<float>(), input_data, 18 * sizeof(float));
        const int begin_data[] = {0, 0, 0};
        memcpy(begin->writeMap<int>(), begin_data, 3 * sizeof(int));
        const int end_data[] = {1, 2, 2};
        memcpy(end->writeMap<int>(), end_data, 3 * sizeof(int));
        const int stride_data[] = {1, 1, 1};
        memcpy(strided->writeMap<int>(), stride_data, 3 * sizeof(int));
        // 1. all mask = 0
        auto output_1 = _StridedSlice(input, begin, end, strided, 0, 0, 0, 0, 0);
        const std::vector<int> expectedShape_1 = {1, 2, 2, 3};
        const std::vector<float> expectedOutput_1 = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
        if (!checkVector<int>(output_1->getInfo()->dim.data(), expectedShape_1.data(), expectedShape_1.size(), 0) ||
            !checkVector<float>(output_1->readMap<float>(), expectedOutput_1.data(), expectedOutput_1.size(), 0.01)) {
            MNN_ERROR("stridedslice (all mask=0) test failed!\n");
            return false;
        }
        // 2. ellipsisMask = 2
        auto output_2 = _StridedSlice(input, begin, end, strided, 0, 0, 2, 0, 0);
        const std::vector<int> expectedShape_2 = {1, 3, 2, 2};
        const std::vector<float> expectedOutput_2 = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
        if (!checkVector<int>(output_2->getInfo()->dim.data(), expectedShape_2.data(), expectedShape_2.size(), 0) ||
            !checkVector<float>(output_2->readMap<float>(), expectedOutput_2.data(), expectedOutput_2.size(), 0.01)) {
            MNN_ERROR("stridedslice (ellipsisMask=2) test failed!\n");
            return false;
        }
        // 3. newAxisMask = 2
        auto output_3 = _StridedSlice(input, begin, end, strided, 0, 0, 0, 2, 0);
        const std::vector<int> expectedShape_3 = {1, 1, 2, 2, 3};
        const std::vector<float> expectedOutput_3 = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
        if (!checkVector<int>(output_3->getInfo()->dim.data(), expectedShape_3.data(), expectedShape_3.size(), 0) ||
            !checkVector<float>(output_3->readMap<float>(), expectedOutput_3.data(), expectedOutput_3.size(), 0.01)) {
            MNN_ERROR("stridedslice (newAxisMask=2) test failed!\n");
            return false;
        }
        // 4. shrinkAxisMask = 2
        auto output_4 = _StridedSlice(input, begin, end, strided, 0, 0, 0, 0, 2);
        const std::vector<int> expectedShape_4 = {1, 2, 3};
        const std::vector<float> expectedOutput_4 = {1, 1, 1, 2, 2, 2};
        if (!checkVector<int>(output_4->getInfo()->dim.data(), expectedShape_4.data(), expectedShape_4.size(), 0) ||
            !checkVector<float>(output_4->readMap<float>(), expectedOutput_4.data(), expectedOutput_4.size(), 0.01)) {
            MNN_ERROR("stridedslice (shrinkAxisMask=2) test failed!\n");
            return false;
        }
        // 5. ellipsisMask = 2, shrinkAxisMask = 4(0b0100)
        auto output_5 = _StridedSlice(input, begin, end, strided, 0, 0, 2, 0, 4);
        const std::vector<int> expectedShape_5 = {1, 3, 2};
        const std::vector<float> expectedOutput_5 = {1, 2, 3, 4, 5, 6};
        if (!checkVector<int>(output_5->getInfo()->dim.data(), expectedShape_5.data(), expectedShape_5.size(), 0) ||
            !checkVector<float>(output_5->readMap<float>(), expectedOutput_5.data(), expectedOutput_5.size(), 0.01)) {
            MNN_ERROR("stridedslice (ellipsisMask=2, shrinkAxisMask=4) test failed!\n");
            return false;
        }
        // 6. beginMask = 9, endMask = 15
        const int begin_data6[] = {0, 1, 1, 0};
        memcpy(begin->writeMap<int>(), begin_data6, 4 * sizeof(int));
        const int end_data6[] = {0, 0, 0, 0};
        memcpy(end->writeMap<int>(), end_data6, 4 * sizeof(int));
        const int stride_data6[] = {1, 1, 1, 1};
        memcpy(strided->writeMap<int>(), stride_data6, 4 * sizeof(int));
        auto output_6 = _StridedSlice(input, begin, end, strided, 9, 15, 0, 0, 0);
        const std::vector<int> expectedShape_6 = {1, 2, 1, 3};
        const std::vector<float> expectedOutput_6 = {4, 4, 4, 6, 6, 6};
        if (!checkVector<int>(output_6->getInfo()->dim.data(), expectedShape_6.data(), expectedShape_6.size(), 0) ||
            !checkVector<float>(output_6->readMap<float>(), expectedOutput_6.data(), expectedOutput_6.size(), 0.01)) {
            MNN_ERROR("stridedslice (beginMask=9, endMask=15) test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(StridedSliceTest, "op/stridedslice");
