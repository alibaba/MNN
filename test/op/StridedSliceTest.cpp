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
        auto begin  = _Input({4}, NCHW);
        auto end    = _Input({4}, NCHW);
        auto strided = _Input({4}, NCHW);
        const float input_data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
        memcpy(input->writeMap<float>(), input_data, 18 * sizeof(float));
        const int begin_data[] = {0, 0, 0, 0};
        memcpy(begin->writeMap<int>(), begin_data, 4 * sizeof(int));
        const int end_data[] = {1, 2, 2, 3};
        memcpy(end->writeMap<int>(), end_data, 4 * sizeof(int));
        const int stride_data[] = {1, 1, 1, 1};
        memcpy(strided->writeMap<int>(), stride_data, 4 * sizeof(int));
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
        const std::vector<int> expectedShape_2 = {1, 3, 2, 3};
        const std::vector<float> expectedOutput_2 = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
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
        // 5. ellipsisMask = 2, shrinkAxisMask = 8(0b01000)
        auto output_5 = _StridedSlice(input, begin, end, strided, 0, 0, 2, 0, 8);
        const std::vector<int> expectedShape_5 = {1, 3, 2};
        const std::vector<float> expectedOutput_5 = {1, 2, 3, 4, 5, 6};
        if (!checkVector<int>(output_5->getInfo()->dim.data(), expectedShape_5.data(), expectedShape_5.size(), 0) ||
            !checkVector<float>(output_5->readMap<float>(), expectedOutput_5.data(), expectedOutput_5.size(), 0.01)) {
            MNN_ERROR("stridedslice (ellipsisMask=2, shrinkAxisMask=8) test failed!\n");
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
        // 7. dim = 2, stride = -1
        const int begin_data7[] = {0, 0, 0, 0};
        memcpy(begin->writeMap<int>(), begin_data7, 4 * sizeof(int));
        const int end_data7[] = {1, 3, 2, 3};
        memcpy(end->writeMap<int>(), end_data7, 4 * sizeof(int));
        const int stride_data7[] = {1, 1, -1, 1};
        memcpy(strided->writeMap<int>(), stride_data7, 4 * sizeof(int));
        auto output_7 = _StridedSlice(input, begin, end, strided, 4, 4, 0, 0, 0);
        const std::vector<int> expectedShape_7 = {1, 3, 2, 3};
        const std::vector<float> expectedOutput_7 = {2, 2, 2, 1, 1, 1, 4, 4, 4, 3, 3, 3, 6, 6, 6, 5, 5, 5};
        if (!checkVector<int>(output_7->getInfo()->dim.data(), expectedShape_7.data(), expectedShape_7.size(), 0) ||
            !checkVector<float>(output_7->readMap<float>(), expectedOutput_7.data(), expectedOutput_7.size(), 0.01)) {
            MNN_ERROR("stridedslice dim=2, stride=-1 test failed!\n");
            return false;
        }
        // 8. dim = 3, stride = -1
        auto input8  = _Input({1, 2, 2, 4}, NCHW);
        const float input_data8[] = { 0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15 };
        memcpy(input8->writeMap<float>(), input_data8, 16 * sizeof(float));
        const int begin_data8[] = {0, 0, 0, 0};
        memcpy(begin->writeMap<int>(), begin_data8, 4 * sizeof(int));
        const int end_data8[] = {0, 0, 0, 0};
        memcpy(end->writeMap<int>(), end_data8, 4 * sizeof(int));
        const int stride_data8[] = {1, 1, 1, -1};
        memcpy(strided->writeMap<int>(), stride_data8, 4 * sizeof(int));
        auto output_8 = _StridedSlice(input8, begin, end, strided, 15, 15, 0, 0, 0);
        const std::vector<int> expectedShape_8 = {1,2,2,4};
        const std::vector<float> expectedOutput_8 = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
        auto info = output_8->getInfo();
        if (!checkVector<int>(output_8->getInfo()->dim.data(), expectedShape_8.data(), expectedShape_8.size(), 0) ||
            !checkVector<float>(output_8->readMap<float>(), expectedOutput_8.data(), expectedOutput_8.size(), 0.01)) {
            MNN_ERROR("stridedslice dim = 3, stride=-1 test failed!\n");
            return false;
        }
        // 9. dim = 0
        input = _Input({2, 1, 3, 3}, NCHW);
        begin  = _Input({1}, NCHW);
        end    = _Input({1}, NCHW);
        strided = _Input({1}, NCHW);
        const float input_data_[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
        memcpy(input->writeMap<float>(), input_data_, 18 * sizeof(float));
        const int begin_data10[] = {1};
        memcpy(begin->writeMap<int>(), begin_data10, 1 * sizeof(int));
        const int end_data10[] = {2};
        memcpy(end->writeMap<int>(), end_data10, 1 * sizeof(int));
        const int stride_data10[] = {1};
        memcpy(strided->writeMap<int>(), stride_data10, 1 * sizeof(int));
        auto output_10 = _StridedSlice(input, begin, end, strided, 0, 0, 0, 0, 1);
        const std::vector<int> expectedShape_10 = {1, 3, 3};
        const std::vector<float> expectedOutput_10 = {4, 4, 4, 5, 5, 5, 6, 6, 6};
        if (!checkVector<int>(output_10->getInfo()->dim.data(), expectedShape_10.data(), expectedShape_10.size(), 0) ||
            !checkVector<float>(output_10->readMap<float>(), expectedOutput_10.data(), expectedOutput_10.size(), 0.01)) {
            MNN_ERROR("stridedslice dim=0, stride=1 test failed!\n");
            return false;
        }
        // 11. dim = 0, zero shape
        input = _Input({0, 1}, NCHW);
        begin  = _Input({2}, NCHW);
        end    = _Input({2}, NCHW);
        strided = _Input({2}, NCHW);
        input->writeMap<int>();
        const int begin_data11[] = {0, 0};
        memcpy(begin->writeMap<int>(), begin_data11, 2 * sizeof(int));
        const int end_data11[] = {1, 1};
        memcpy(end->writeMap<int>(), end_data11, 2 * sizeof(int));
        const int stride_data11[] = {1, 1};
        memcpy(strided->writeMap<int>(), stride_data11, 2 * sizeof(int));
        auto output_11 = _StridedSlice(input, begin, end, strided, 0, 0, 0, 0, 3);
        const std::vector<int> expectedShape_11 = {};
        const std::vector<float> expectedOutput_11 = {0.0f};
        if ((!output_11->getInfo()->dim.empty()) ||
            !checkVector<float>(output_11->readMap<float>(), expectedOutput_11.data(), expectedOutput_11.size(), 0.01)) {
            MNN_ERROR("stridedslice dim=0, stride=1 test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(StridedSliceTest, "op/stridedslice");

class SplitC4Test : public MNNTestCase {
public:
    virtual ~SplitC4Test() = default;
    virtual bool run(int precision) {
        int N = 1; int C = 32; int W = 3; int H = 4;
        auto x = _Input({N, C, H, W}, NCHW, halide_type_of<int>());
        auto xPtr = x->writeMap<int>();
        for (int x=0; x<N; ++x) {
            for (int y=0; y<C; ++y) {
                for (int z=0; z<H; ++z) {
                    for (int w=0; w<W; ++w) {
                        auto pos = x * C * H * W + y * H * W + z * W + w;
                        xPtr[pos] = pos;
                    }
                }
            }
        }
        x = _Convert(x, NC4HW4);
        x.fix(VARP::CONSTANT);

        auto y = _Split(x, {2}, 1)[1];
        auto yInfo = y->getInfo();
        if (yInfo->dim[0] != N || yInfo->dim[1] != C/2 || yInfo->dim[2] != H || yInfo->dim[3] != W) {
            FUNC_PRINT(1);
            return false;
        }
        y = _Add(y, _Scalar<int>(0));
        y = _Convert(y, NCHW);
        {
            auto yPtr = y->readMap<int>();
            for (int x=0; x<N; ++x) {
                for (int y=0; y<C/2; ++y) {
                    for (int z=0; z<H; ++z) {
                        for (int w=0; w<W; ++w) {
                            auto pos = x * C/2 * H * W + y * H * W + z * W + w;
                            auto value = x * C * H * W + (y+C/2) * H * W + z * W + w;
                            if (yPtr[pos] != value) {
                                FUNC_PRINT(1);
                                return false;
                            }
                        }
                    }
                }
            }
        }
        if (1 == N) {
            auto y2 = _RasterRaw({x}, {C/2*H*W, 0, 0, 1, 0, 0, 0, 1, 1, 1, C/2*H*W}, {N, C/2, H, W}, halide_type_of<int>(), NC4HW4);
            y2 = _Add(y2, _Scalar<int>(0));
            y2 = _Convert(y2, NCHW);
            auto yPtr = y2->readMap<int>();
            for (int x=0; x<N; ++x) {
                for (int y=0; y<C/2; ++y) {
                    for (int z=0; z<H; ++z) {
                        for (int w=0; w<W; ++w) {
                            auto pos = x * C/2 * H * W + y * H * W + z * W + w;
                            auto value = x * C * H * W + (y+C/2) * H * W + z * W + w;
                            if (yPtr[pos] != value) {
                                FUNC_PRINT(1);
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(SplitC4Test, "op/splitc4");

class StrideSliceWriteTest: public MNNTestCase {
    virtual bool run(int precision) {
        {
            auto input   = _Input({2, 3, 2, 12}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({20}, NCHW);

            // initialize
            auto size = input->getInfo()->size;
            const float inputData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, size * sizeof(float));
            const int beginData[] = {0, 1, 0, 2};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {2, 3, 2, 12};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 2, 2};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 20 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {2, 3, 2, 12};
            const std::vector<float> expectedOutput = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                       3, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3,
                                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                       5, 5, 9, 5, 9, 5, 9, 5, 9, 5, 9, 5,
                                                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                       3, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3,
                                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                       5, 5, 9, 5, 9, 5, 9, 5, 9, 5, 9, 5,
                                                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite shape test0 error\n");
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test0 result error\n");
                return false;
            }
        }
        {
            auto input   = _Input({1, 3, 2, 12}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({12}, NCHW);

            // initialize
            auto size = input->getInfo()->size;
            const float inputData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, size * sizeof(float));
            const int beginData[] = {0, 1, 0, 2};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 3, 2, 12};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 2, 2};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 12 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 12};
            const std::vector<float> expectedOutput = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                       3, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3,
                                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                       5, 5, 9, 5, 9, 5, 9, 5, 9, 5, 9, 5,
                                                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite shape test1 error\n");
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test1 result error\n");
                return false;
            }
        }
        {
            int originSize = 300;
            int sliceSize = 250;
            INTS originShape = {3, originSize, originSize};
            INTS sliceShape = {3, sliceSize, sliceSize};
            auto input = _Input(sliceShape, NCHW, halide_type_of<float>());
            auto src = _Input(originShape, NCHW, halide_type_of<float>());
            // initialize 0
            memset(src->writeMap<float>(), 0, src->getInfo()->size * sizeof(float));
            // initialize 1
            for (int i = 0; i < input->getInfo()->size; ++i) {
                input->writeMap<float>()[i] = 1.f;
            }

            // slice write
            std::vector<int> begin = {0,0,0};
            std::vector<int> stride = {1,1,1};
            VARP startVar = _Const(begin.data(), {(int)begin.size()}, Dimensionformat::NCHW, halide_type_of<int>());
            VARP endVar = _Const(sliceShape.data(), {(int)sliceShape.size()}, Dimensionformat::NCHW, halide_type_of<int>());
            VARP strideVar = _Const(stride.data(), {(int)stride.size()}, Dimensionformat::NCHW, halide_type_of<int>());
            auto dst = _StridedSliceWrite(src, startVar, endVar, strideVar, input, 0, 0, 0, 0, 0);

            // validate
            if (_ReduceSum(input)->readMap<float>()[0] != _ReduceSum(dst)->readMap<float>()[0]) {
                MNN_PRINT("StrideSliceWrite test2 Error\n");
                return false;
            }
        }

        {
            auto input   = _Input({1, 3, 2, 3}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({3}, NCHW);

            // initialize
            const float inputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, 18 * sizeof(float));
            const int beginData[] = {0, 1, 0, 0};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 3, 2, 3};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 2, 2};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 3 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 3};
            const std::vector<float> expectedOutput = {1, 1, 1, 2, 2, 2, 9, 3, 9, 4, 4, 4, 9, 5, 9, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite test3 shape error\n");
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test3 result error\n");
                return false;
            }
        }

        {
            auto input   = _Input({1, 3, 2, 3}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({3}, NCHW);

            // initialize
            const float inputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, 18 * sizeof(float));
            const int beginData[] = {0, 1, 0, 0};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 3, 2, 3};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 2, 1};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 3 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 3};
            const std::vector<float> expectedOutput = {1, 1, 1, 2, 2, 2, 9, 9, 9, 4, 4, 4, 9, 9, 9, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite test4 shape error\n");
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test4 result error\n");
                return false;
            }
        }

        {
            auto input   = _Input({1, 3, 2, 3}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({3}, NCHW);

            // initialize
            const float inputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, 18 * sizeof(float));
            const int beginData[] = {0, 1, 0, 0};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 3, 2, 3};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 1, 1};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 3 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 3};
            const std::vector<float> expectedOutput = {1, 1, 1, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite test5 shape error\n");
                return false;
            }
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test5 result error\n");
                return false;
            }
        }
        {
            auto input   = _Input({1, 3, 2, 3}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({3}, NCHW);

            // initialize
            const float inputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, 18 * sizeof(float));
            const int beginData[] = {0, 1, 0, 0};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 2, 2, 3};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 1, 1};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 3 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 3};
            const std::vector<float> expectedOutput = {1, 1, 1, 2, 2, 2, 9, 9, 9, 9, 9, 9, 5, 5, 5, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite test6 shape error\n");
                return false;
            }
            auto ptr = output->readMap<float>();
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test6 result error\n");
                return false;
            }
        }

        {
            auto input   = _Input({1, 3, 2, 3}, NCHW);
            auto begin   = _Input({4}, NCHW);
            auto end     = _Input({4}, NCHW);
            auto strided = _Input({4}, NCHW);
            auto write   = _Input({3}, NCHW);

            // initialize
            const float inputData[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
            memcpy(input->writeMap<float>(), inputData, 18 * sizeof(float));
            const int beginData[] = {0, 0, 0, 0};
            memcpy(begin->writeMap<int>(), beginData, 4 * sizeof(int));
            const int endData[] = {1, 2, 2, 3};
            memcpy(end->writeMap<int>(), endData, 4 * sizeof(int));
            const int strideData[] = {1, 1, 1, 1};
            memcpy(strided->writeMap<int>(), strideData, 4 * sizeof(int));
            const float writeData[] = {9, 9, 9};
            memcpy(write->writeMap<float>(), writeData, 3 * sizeof(float));
            auto output= _StridedSliceWrite(input, begin, end, strided, write, 0, 0, 0, 0, 0);
            const std::vector<int> expectedShape = {1, 3, 2, 3};
            const std::vector<float> expectedOutput = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 6, 6, 6};

            // validate
            if (!checkVector<int>(output->getInfo()->dim.data(), expectedShape.data(), expectedShape.size(), 0)) {
                MNN_PRINT("StrideSliceWrite test7 shape error\n");
                return false;
            }
            auto ptr = output->readMap<float>();
            if (!checkVector<float>(output->readMap<float>(), expectedOutput.data(), expectedOutput.size(), 0.01)) {
                MNN_PRINT("StrideSliceWrite test7 result error\n");
                return false;
            }
        }

        {
            {
                // For strideslicewrite shape compute
                auto tinput  = _Input({1, 3, 2, 3}, NCHW);
                auto tbegin  = _Input({4}, NCHW);
                auto tend    = _Input({4}, NCHW);
                auto tstrided = _Input({4}, NCHW);
                auto twrite = _Input({3}, NCHW);
                auto toutput= _StridedSliceWrite(tinput, tbegin, tend, tstrided, twrite, 0, 0, 0, 0, 0);
                auto info = toutput->getInfo();
                if (nullptr == info) {
                    FUNC_PRINT(1);
                    return false;
                }
                if (info->dim != tinput->getInfo()->dim) {
                    MNN_PRINT("StrideSliceWrite test8 shape error\n");
                    return false;
                }
            }
        }

        return true;
    }
};
MNNTestSuiteRegister(StrideSliceWriteTest, "op/strideSliceWrite");
