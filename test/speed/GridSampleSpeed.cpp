//
//  CropAndResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <cfenv>
#include <cmath>
#include <random>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

#define BATCH 8
#define DEPTH 4
#define WIDTH 720
#define HEIGHT 720
#define TIME 10

class GridSampleSpeed : public MNNTestCase {
public:
    virtual ~GridSampleSpeed() = default;

    virtual bool run(int precision) {
        const int batch = BATCH;
        const int inHeight = HEIGHT;
        const int inWidth = WIDTH;
        const int outHeight = HEIGHT;
        const int outWidth = WIDTH;
        const int depth = DEPTH;
        auto input = _Input({batch, depth, inHeight, inWidth}, NCHW);
        auto grid = _Input({batch, outHeight, outWidth, 2}, NHWC);

        std::vector<InterpolationMethod> modes({BILINEAR});
        std::vector<GridSamplePaddingMode> paddingModes({GRID_SAMPLE_PADDING_ZEROS});
        std::vector<bool> alignCornersVec({false});

        std::vector<float> expectedOutput(batch * outHeight * outWidth * depth);
        for (auto mode : modes) {
            std::string modeStr = mode == BILINEAR ? "bilinear" : "nearest";
            for (auto paddingMode : paddingModes) {
                std::string paddingModeStr = paddingMode == GRID_SAMPLE_PADDING_ZEROS ?
                                             "zeros" : (paddingMode == GRID_SAMPLE_PADDING_BORDER ? "border"
                                                                                                  : "reflection");
                for (auto alignCorners : alignCornersVec) {
                    std::string alignCornersStr = alignCorners ? "true" : "false";

//                    grid->unMap();
//                    input->unMap();
//                    input = _Convert(input, NC4HW4);
                    auto output = _GridSample(input, grid, mode, paddingMode, alignCorners);
                    MNN_PRINT("Test GridSample for NCHW (%d, %d, %d, %d) x %d with setting %s %s %s \n",
                              BATCH, DEPTH, HEIGHT, WIDTH, TIME,
                              modeStr.c_str(), paddingModeStr.c_str(), alignCornersStr.c_str());
                    {
                        AUTOTIME;
                        for (int i = 0; i < TIME; ++i) {
                            auto inputPtr = input->writeMap<float>();
                            auto gridPtr = grid->writeMap<float>();

                            output->readMap<float>();
                        }
                    }
                }
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(GridSampleSpeed, "speed/GridSample");
