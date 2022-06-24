//
//  opencv_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "gtest/gtest.h"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();
    auto instance = testing::UnitTest::GetInstance();
    printf("\nTEST_NAME_OPENCV_UNIT: OpenCV单元测试\nTEST_CASE_AMOUNT_OPENCV_UNIT: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":%d}\n",
           instance->failed_test_count(), instance->successful_test_count(), instance->skipped_test_count());
    return res;
}
