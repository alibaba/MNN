//
//  draw_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv/imgcodecs.hpp"
#include "test_env.hpp"

#ifdef MNN_DRAW_TEST

static Env<uint8_t> testEnv(img_name, false);

// arrowedLine
TEST(arrowedLine, basic) {
    cv::arrowedLine(testEnv.cvSrc, {10, 10}, {300, 200}, {0, 0, 255}, 1);
    arrowedLine(testEnv.mnnSrc, {10, 10}, {300, 200}, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

TEST(arrowedLine, thickness) {
    cv::arrowedLine(testEnv.cvSrc, {10, 10}, {30, 20}, {0, 0, 255}, 5);
    arrowedLine(testEnv.mnnSrc, {10, 10}, {30, 20}, {0, 0, 255}, 5);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

// circle
TEST(circle, basic) {
    cv::circle(testEnv.cvSrc, {50, 50}, 10, {0, 0, 255}, 1);
    circle(testEnv.mnnSrc, {50, 50}, 10, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

TEST(circle, thickness) {
    cv::circle(testEnv.cvSrc, {100, 100}, 10, {0, 0, 255}, 5);
    circle(testEnv.mnnSrc, {100, 100}, 10, {0, 0, 255}, 5);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

TEST(circle, fill) {
    cv::circle(testEnv.cvSrc, {150, 150}, 10, {0, 0, 255}, -1);
    circle(testEnv.mnnSrc, {150, 150}, 10, {0, 0, 255}, -1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

// line
TEST(line, basic) {
    // cv::line(testEnv.cvSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    // line(testEnv.mnnSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    cv::line(testEnv.cvSrc, {10, 10}, {50, 50}, {0, 0, 255}, 1);
    line(testEnv.mnnSrc, {10, 10}, {50, 50}, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

TEST(line, thickness) {
    cv::line(testEnv.cvSrc, {10, 10}, {20, 20}, {0, 0, 255}, 5);
    line(testEnv.mnnSrc, {10, 10}, {20, 20}, {0, 0, 255}, 5);
    // cv::imwrite("cv_line.jpg", testEnv.cvSrc);
    // imwrite("mnn_line.jpg", testEnv.mnnSrc);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

// rectangle
TEST(rectangle, basic) {
    cv::rectangle(testEnv.cvSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    rectangle(testEnv.mnnSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}
// drawContours
TEST(drawContours, basic) {
    cv::Mat gray, binary;
    cv::cvtColor(testEnv.cvSrc, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(testEnv.cvSrc, cv_contours, -1, {0, 0, 255}, -1);
    std::vector<std::vector<Point>> mnn_contours(cv_contours.size());
    for (int i = 0; i < cv_contours.size(); i++) {
        for (int j = 0; j < cv_contours[i].size(); j++) {
            Point p;
            p.set(cv_contours[i][j].x, cv_contours[i][j].y);
            mnn_contours[i].push_back(p);
        }
    }
    drawContours(testEnv.mnnSrc, mnn_contours, -1, {0, 0, 255}, -1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

TEST(fillPoly, basic) {
    cv::Mat gray, binary;
    cv::cvtColor(testEnv.cvSrc, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::fillPoly(testEnv.cvSrc, cv_contours, {0, 0, 255});
    std::vector<std::vector<Point>> mnn_contours(cv_contours.size());
    for (int i = 0; i < cv_contours.size(); i++) {
        for (int j = 0; j < cv_contours[i].size(); j++) {
            Point p;
            p.set(cv_contours[i][j].x, cv_contours[i][j].y);
            mnn_contours[i].push_back(p);
        }
    }
    fillPoly(testEnv.mnnSrc, mnn_contours, {0, 0, 255});
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}
#endif
