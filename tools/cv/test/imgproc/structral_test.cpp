//
//  structral_test.cpp
//  MNN
//
//  Created by MNN on 2021/12/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "test_env.hpp"

#define MNN_STRUCTRAL_TEST
#ifdef MNN_STRUCTRAL_TEST

static Env<uint8_t> testEnv(img_name, false);

static std::vector<uint8_t> img = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,1,1,0,0,
    0,0,1,0,0,1,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,1,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0
};
static void cmpContours(std::vector<std::vector<Point>> x, std::vector<std::vector<cv::Point>> y) {
    ASSERT_EQ(x.size(), y.size());
    for (int i = 0; i < x.size(); i++) {
        ASSERT_EQ(x[i].size(), y[i].size());
        for (int j = 0; j < x[i].size(); j++) {
            ASSERT_EQ(x[i][j].fX, (float)y[i][j].x);
            ASSERT_EQ(x[i][j].fY, (float)y[i][j].y);
        }
    }
}
// findContours
TEST(findContours, external_none) {
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC);
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    auto mnn_contours = findContours(x, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cv::findContours(mask, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cmpContours(mnn_contours, cv_contours);
}
TEST(findContours, external_simple) {
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC);
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    auto mnn_contours = findContours(x, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    cv::findContours(mask, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cmpContours(mnn_contours, cv_contours);
}
TEST(findContours, list_none) {
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC);
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    auto mnn_contours = findContours(x, RETR_LIST, CHAIN_APPROX_NONE);
    cv::findContours(mask, cv_contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cmpContours(mnn_contours, cv_contours);
}
TEST(findContours, list_simple) {
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC);
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    auto mnn_contours = findContours(x, RETR_LIST, CHAIN_APPROX_SIMPLE);
    cv::findContours(mask, cv_contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cmpContours(mnn_contours, cv_contours);
}

TEST(contourArea, basic) {
    std::vector<cv::Point> cv_contour;
    cv_contour.push_back(cv::Point2i(0, 0));
    cv_contour.push_back(cv::Point2i(10, 0));
    cv_contour.push_back(cv::Point2i(10, 10));
    cv_contour.push_back(cv::Point2i(5, 4));
    std::vector<Point> mnn_contour;
    mnn_contour.push_back({0, 0});
    mnn_contour.push_back({10, 0});
    mnn_contour.push_back({10, 10});
    mnn_contour.push_back({5, 4});
    double x = contourArea(mnn_contour);
    double y = cv::contourArea(cv_contour);
    ASSERT_EQ(x, y);
}

#define TEST_POINTS { {0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3} }
TEST(convexHull, indices) {
    std::vector<cv::Point> cv_contour = TEST_POINTS;
    std::vector<Point> mnn_contour = TEST_POINTS;
    auto x = convexHull(mnn_contour, false, false);
    std::vector<int> y;
    cv::convexHull(cv_contour, y, false, false);
    ASSERT_TRUE(x == y);
}
TEST(convexHull, pointers) {
    std::vector<cv::Point> cv_contour = TEST_POINTS;
    std::vector<Point> mnn_contour = TEST_POINTS;
    auto x = convexHull(mnn_contour, false, true);
    cv::Mat y = cv::Mat(1, 4, CV_32S);
    cv::convexHull(cv_contour, y, false, true);
    auto ptr = reinterpret_cast<int*>(y.data);
    std::vector<int> z(ptr , ptr + 8);
    ASSERT_TRUE(x == z);
}
TEST(minAreaRect, basic) {
    std::vector<cv::Point> cv_contour = TEST_POINTS;
    std::vector<Point> mnn_contour = TEST_POINTS;
    auto x = minAreaRect(mnn_contour);
    auto y = cv::minAreaRect(cv_contour);
    ASSERT_NEAR(x.center.x, y.center.x, 1e-4);
    ASSERT_NEAR(x.center.y, y.center.y, 1e-4);
    if ((x.size.width == y.size.width) && (x.size.height == y.size.height)) {
        ASSERT_NEAR(x.angle, y.angle, 1e-4);
    } else if ((x.size.width == y.size.height) && (x.size.height == y.size.width)) {
        ASSERT_NEAR(std::abs(std::abs(x.angle) + std::abs(y.angle)), 90.0, 1e-4);
    } else {
        ASSERT_TRUE(false);
    }
}
TEST(boundingRect, basic) {
    std::vector<cv::Point> cv_contour = TEST_POINTS;
    std::vector<Point> mnn_contour = TEST_POINTS;
    auto x = boundingRect(mnn_contour);
    auto y = cv::boundingRect(cv_contour);
    ASSERT_EQ(x.x, y.x);
    ASSERT_EQ(x.y, y.y);
    ASSERT_EQ(x.width, y.width);
    ASSERT_EQ(x.height, y.height);
}
TEST(connectedComponentsWithStats, basic) {
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC, halide_type_of<uint8_t>());
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    VARP mnn_label, mnn_statsv, mnn_centroids;
    int mnn_nlabels = connectedComponentsWithStats(x, mnn_label, mnn_statsv, mnn_centroids);
    cv::Mat cv_label, cv_statsv, cv_centroids;
    int cv_nlables = cv::connectedComponentsWithStats(mask, cv_label, cv_statsv, cv_centroids);
    ASSERT_EQ(mnn_nlabels, cv_nlables);
    ASSERT_TRUE(_equal<int>(cv_label, mnn_label));
    ASSERT_TRUE(_equal<int>(cv_statsv, mnn_statsv));
    ASSERT_TRUE((_equal<double, float>(cv_centroids, mnn_centroids)));
}
TEST(boxPoints, basic) {
    std::vector<cv::Point> cv_contour = TEST_POINTS;
    std::vector<Point> mnn_contour = TEST_POINTS;
    auto x = minAreaRect(mnn_contour);
    auto y = cv::minAreaRect(cv_contour);
    auto mnn_points = boxPoints(x);
    cv::Mat _cv_points;
    cv::boxPoints(y, _cv_points);
    auto ptr = reinterpret_cast<float*>(_cv_points.data);
    std::vector<Point> cv_points(4);
    for (int i = 0; i < 4; i++) {
        cv_points[i].fX = ptr[2 * i + 0];
        cv_points[i].fY = ptr[2 * i + 1];
    }
    auto comp = [](Point p1, Point p2) { return p1.fX < p2.fX; };
    std::sort(mnn_points.begin(), mnn_points.end(), comp);
    std::sort(cv_points.begin(), cv_points.end(), comp);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(cv_points[i].fX, mnn_points[i].fX, 1e-4);
        ASSERT_NEAR(cv_points[i].fY, mnn_points[i].fY, 1e-4);
    }
}
#endif
