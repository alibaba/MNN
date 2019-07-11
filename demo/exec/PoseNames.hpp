//
//  PoseNames.hpp
//  MNN
//
//  Created by MNN on 2019/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef POSENAMES_HPP
#define POSENAMES_HPP
#include <vector>
const std::vector<std::string> PoseNames{"nose",         "leftEye",       "rightEye",  "leftEar",    "rightEar",
                                         "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist",
                                         "rightWrist",   "leftHip",       "rightHip",  "leftKnee",   "rightKnee",
                                         "leftAnkle",    "rightAnkle"};

const std::vector<std::pair<std::string, std::string>> PoseChain{
    {"nose", "leftEye"},          {"leftEye", "leftEar"},        {"nose", "rightEye"},
    {"rightEye", "rightEar"},     {"nose", "leftShoulder"},      {"leftShoulder", "leftElbow"},
    {"leftElbow", "leftWrist"},   {"leftShoulder", "leftHip"},   {"leftHip", "leftKnee"},
    {"leftKnee", "leftAnkle"},    {"nose", "rightShoulder"},     {"rightShoulder", "rightElbow"},
    {"rightElbow", "rightWrist"}, {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
    {"rightKnee", "rightAnkle"}};

#endif // POSENAMES_HPP
