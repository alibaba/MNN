#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct FLAMEOuput {
    int frame_id;
    std::vector<float> expr;
    std::vector<float> jaw_pose;
public:
    bool IsEmpty() const
    {
        return expr.empty();
    }

    void Reset() {
        frame_id = -1;
        expr.clear();
        jaw_pose.clear();
    }
};

struct BodyParamsInput{
    int frame_id;
    std::vector<float> expression;
    std::vector<float> Rh;
    std::vector<float> Th;
    std::vector<float> body_pose;
    std::vector<float> jaw_pose;
    std::vector<float> leye_pose;
    std::vector<float> reye_pose;
    std::vector<float> left_hand_pose;
    std::vector<float> right_hand_pose;
    std::vector<float> eye_verts;
    std::vector<float> pose;
};


struct FullTypeOutput{
    int frame_id;
    std::vector<float> expr;
    std::vector<float> joints_transform;
    std::vector<float> local_joints_transform;
    std::vector<float> pose_z;
    std::vector<float> app_pose_z;
    std::vector<float> pose;
};

std::vector<BodyParamsInput> ReadFramesFromBinary(const std::string& binFilePath, std::string& errorMessage);
void WriteFramesToBinary(const std::string& binFilePath, const std::vector<BodyParamsInput>& frames);
std::vector<BodyParamsInput> ParseInputsFromJson(const std::string &json_path);

std::vector<float> interp_linear(const std::vector<float> &x, const std::vector<float> &y,
                                 const std::vector<float> &x_new);

std::vector<std::vector<float>> resample_bs_params(const std::vector<std::vector<float>> &bs_params, int L2);

std::vector<std::vector<float>> convert_to_2d(const std::vector<float> &flat_vector, int N, int M);

std::vector<float> resampleAudioData(const std::vector<float> &input, unsigned int sourceSampleRate,
                                     unsigned int targetSampleRate);
float calculateMean(const std::vector<float> &audio);
float calculateVariance(const std::vector<float> &audio, float mean);
std::vector<float> normalizeAudio(std::vector<float> &audio);


