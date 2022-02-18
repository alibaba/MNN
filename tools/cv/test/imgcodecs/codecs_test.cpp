//
//  codecs_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include "cv/imgcodecs.hpp"
#include "test_env.hpp"

#ifdef MNN_CODECS_TEST

static Env<uint8_t> testEnv(img_name, false);

// haveImageReader
TEST(haveImageReader, jpg) {
    auto cvHave = cv::haveImageReader(img_name);
    auto mnnHave = haveImageReader(img_name);
    EXPECT_EQ(cvHave, mnnHave);
}

TEST(haveImageReader, jpg_wrongname) {
    const char* name = "./imgs/cat__";
    auto cvHave = cv::haveImageReader(name);
    auto mnnHave = haveImageReader(name);
    EXPECT_EQ(cvHave, mnnHave);
}

TEST(haveImageReader, fake_jpg) {
    const char* name = "./imgs/fake.jpg";
    auto cvHave = cv::haveImageReader(name);
    auto mnnHave = haveImageReader(name);
    EXPECT_EQ(cvHave, mnnHave);
}

// haveImageWriter
TEST(haveImageWriter, jpg) {
    const char* name = "x.jpg";
    auto cvHave = cv::haveImageWriter(name);
    auto mnnHave = haveImageWriter(name);
    EXPECT_EQ(cvHave, mnnHave);
}

TEST(haveImageWriter, JPEG) {
    const char* name = "x.JPEG";
    auto cvHave = cv::haveImageWriter(name);
    auto mnnHave = haveImageWriter(name);
    EXPECT_EQ(cvHave, mnnHave);
}

TEST(haveImageWriter, txt) {
    const char* name = "x.txt";
    auto cvHave = cv::haveImageWriter(name);
    auto mnnHave = haveImageWriter(name);
    EXPECT_EQ(cvHave, mnnHave);
}

// imdecode
TEST(imdecode, IMREAD_COLOR) {
    FILE* pFile = fopen(img_name, "rb");
    fseek(pFile, 0, SEEK_END);
    long lSize = ftell(pFile);
    rewind(pFile);
    std::vector<unsigned char> data(lSize);
    fread(data.data(), sizeof(char), lSize, pFile);
    fclose(pFile);
    auto cvImg = cv::imdecode(data, cv::IMREAD_COLOR);
    auto mnnImg = imdecode(data, IMREAD_COLOR);
    EXPECT_TRUE(testEnv.equal(cvImg, mnnImg));
}

// imencode
TEST(imencode, jpg) {
    std::vector<uint8_t> cvBuf;
    cv::imencode(".jpg", testEnv.cvSrc, cvBuf);
    auto mnnRes = imencode(".jpg", testEnv.mnnSrc);
    // stbi_encode NOT fully equal to opencv, but is right
    EXPECT_EQ(memcmp(cvBuf.data(), mnnRes.second.data(), 16), 0);
}


// imread
TEST(imread, IMREAD_COLOR) {
    auto cvImg = cv::imread(img_name);
    auto mnnImg = imread(img_name);
    EXPECT_TRUE(testEnv.equal(cvImg, mnnImg));
}

TEST(imread, IMREAD_GRAYSCALE) {
    auto cvImg = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
    auto mnnImg = imread(img_name, IMREAD_GRAYSCALE);
    EXPECT_TRUE(testEnv.equal(cvImg, mnnImg));
}

// imwrite
TEST(imwrite, jpg) {
    bool cvRes = cv::imwrite("cv.jpg", testEnv.cvSrc);
    bool mnnRes = imwrite("mnn.jpg", testEnv.mnnSrc);
    EXPECT_EQ(cvRes, mnnRes);
}

#endif
