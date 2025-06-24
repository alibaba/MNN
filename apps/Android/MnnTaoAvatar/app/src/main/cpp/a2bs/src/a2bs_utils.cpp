#include "a2bs/a2bs_utils.hpp"
#include <iostream>
#include <sstream>
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio/miniaudio.h>


#include <fstream>

std::vector<BodyParamsInput> ReadFramesFromBinary(const std::string& binFilePath, std::string& errorMessage) {
    std::vector<BodyParamsInput> frames;
    std::ifstream inFile(binFilePath, std::ios::binary);
    if (!inFile.is_open()) {
        errorMessage = "Could not open file for reading: " + binFilePath;
        return frames;
    }

    // 1. Read the number of frames
    size_t frameCount = 0;
    inFile.read(reinterpret_cast<char*>(&frameCount), sizeof(frameCount));
    frames.resize(frameCount);

    // 2. For each frame, read frame_id and vectors
    for (size_t i = 0; i < frameCount; ++i)
    {
        // Read frame_id
        inFile.read(reinterpret_cast<char*>(&frames[i].frame_id), sizeof(frames[i].frame_id));

        // Helper lambda to read a vector of floats
        auto readFloatVector = [&](std::vector<float>& vec) {
            size_t vecSize = 0;
            inFile.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));
            vec.resize(vecSize);
            if (vecSize > 0) {
                inFile.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(float));
            }
        };

        // Read each vector
        readFloatVector(frames[i].expression);
        readFloatVector(frames[i].Rh);
        readFloatVector(frames[i].Th);
        readFloatVector(frames[i].body_pose);
        readFloatVector(frames[i].jaw_pose);
        readFloatVector(frames[i].leye_pose);
        readFloatVector(frames[i].reye_pose);
        readFloatVector(frames[i].left_hand_pose);
        readFloatVector(frames[i].right_hand_pose);
        readFloatVector(frames[i].pose);
    }

    inFile.close();
    return frames;
}

void WriteFramesToBinary(const std::string& binFilePath, const std::vector<BodyParamsInput>& frames) {
    std::ofstream outFile(binFilePath, std::ios::binary);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + binFilePath);
    }

    // 1. Write the number of frames
    size_t frameCount = frames.size();
    outFile.write(reinterpret_cast<const char*>(&frameCount), sizeof(frameCount));

    // 2. For each frame, write frame_id and each vector
    for (const auto& frame : frames)
    {
        // Write frame_id
        outFile.write(reinterpret_cast<const char*>(&frame.frame_id), sizeof(frame.frame_id));

        // Helper lambda to write a vector of floats
        auto writeFloatVector = [&](const std::vector<float>& vec) {
            size_t vecSize = vec.size();
            outFile.write(reinterpret_cast<const char*>(&vecSize), sizeof(vecSize));
            if (!vec.empty()) {
                outFile.write(reinterpret_cast<const char*>(vec.data()), vecSize * sizeof(float));
            }
        };

        // Write each vector
        writeFloatVector(frame.expression);
        writeFloatVector(frame.Rh);
        writeFloatVector(frame.Th);
        writeFloatVector(frame.body_pose);
        writeFloatVector(frame.jaw_pose);
        writeFloatVector(frame.leye_pose);
        writeFloatVector(frame.reye_pose);
        writeFloatVector(frame.left_hand_pose);
        writeFloatVector(frame.right_hand_pose);
        writeFloatVector(frame.pose);
    }
    outFile.close();
}


// 辅助函数：将多维向量flatten为一维向量
template <typename T>
std::vector<T> flattenVector(const std::vector<std::vector<T>> &vec2D)
{
    std::vector<T> flattened;
    for (const auto &subVec : vec2D)
    {
        flattened.insert(flattened.end(), subVec.begin(), subVec.end());
    }
    return flattened;
}

// 函数将JSON数组转换为FrameData类型的vector
std::vector<BodyParamsInput> ParseInputsFromJson(const std::string &jsonFilePath)
{
    std::vector<BodyParamsInput> frames;
    std::ifstream file(jsonFilePath);

    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << jsonFilePath << std::endl;
        return frames;
    }

    json jsonData;
    try
    {
        file >> jsonData; // 尝试读取并解析文件中的JSON数据
    }
    catch (json::parse_error &e)
    {
        std::cerr << "解析错误: " << e.what() << std::endl;
        return frames;
    }

    for (const auto &item : jsonData)
    {
        BodyParamsInput frame;
        frame.frame_id = item["frame_id"].get<int>();
        frame.expression = flattenVector(item["expression"].get<std::vector<std::vector<float>>>());
        frame.Rh = flattenVector(item["Rh"].get<std::vector<std::vector<float>>>());
        frame.Th = flattenVector(item["Th"].get<std::vector<std::vector<float>>>());
        frame.body_pose = flattenVector(item["body_pose"].get<std::vector<std::vector<float>>>());
        frame.jaw_pose = flattenVector(item["jaw_pose"].get<std::vector<std::vector<float>>>());
        frame.leye_pose = flattenVector(item["leye_pose"].get<std::vector<std::vector<float>>>());
        frame.reye_pose = flattenVector(item["reye_pose"].get<std::vector<std::vector<float>>>());
        frame.left_hand_pose = flattenVector(item["left_hand_pose"].get<std::vector<std::vector<float>>>());
        frame.right_hand_pose = flattenVector(item["right_hand_pose"].get<std::vector<std::vector<float>>>());
        frame.pose = flattenVector(item["pose"].get<std::vector<std::vector<float>>>());
        frames.push_back(frame);
    }

    return frames;
}


// 使用 std::vector 实现的线性插值函数
std::vector<float> interp_linear(const std::vector<float> &x, const std::vector<float> &y,
                                 const std::vector<float> &x_new)
{
    std::vector<float> y_new(x_new.size());
    for (size_t i = 0; i < x_new.size(); ++i)
    {
        // 查找x_new[i]在哪两个x值之间
        size_t j = 0;
        while (j < x.size() - 1 && x[j + 1] < x_new[i])
        {
            j++;
        }

        // 计算线性插值
        float t = (x_new[i] - x[j]) / (x[j + 1] - x[j]);
        y_new[i] = (1 - t) * y[j] + t * y[j + 1];
    }
    return y_new;
}

// 重新采样函数
std::vector<std::vector<float>> resample_bs_params(const std::vector<std::vector<float>> &bs_params, int L2)
{
    int L1 = bs_params.size();
    int N = bs_params[0].size();
    std::vector<std::vector<float>> new_bs_params(L2, std::vector<float>(N, 0.0));

    // 生成插值因子
    std::vector<float> factors(L2);
    for (int i = 0; i < L2; ++i)
    {
        factors[i] = static_cast<float>(i) * (L1 - 1) / (L2 - 1);
    }
    std::vector<float> L1_range(L1);
    for (int i = 0; i < L1_range.size(); i++)
    {
        L1_range[i] = i;
    }

    // 对每个维度执行线性插值
    for (int i = 0; i < N; ++i)
    {
        std::vector<float> y(L1);
        for (int j = 0; j < L1; ++j)
        {
            y[j] = bs_params[j][i];
        }
        std::vector<float> y_new = interp_linear(L1_range, y, factors);
        for (int j = 0; j < L2; ++j)
        {
            new_bs_params[j][i] = y_new[j];
        }
    }

    return new_bs_params;
}

std::vector<std::vector<float>> convert_to_2d(const std::vector<float> &flat_vector, int N, int M)
{
    std::vector<std::vector<float>> two_dim_vector(N, std::vector<float>(M));

    for (int n = 0; n < N; ++n)
    {
        for (int m = 0; m < M; ++m)
        {
            two_dim_vector[n][m] = flat_vector[n * M + m];
        }
    }

    return two_dim_vector;
}

std::vector<float> resampleAudioData(const std::vector<float> &input, unsigned int sourceSampleRate,
                                     unsigned int targetSampleRate)
{
    // 初始化miniaudio重采样器
    ma_resampler_config resamplerConfig =
        ma_resampler_config_init(ma_format_f32, 1, sourceSampleRate, targetSampleRate, ma_resample_algorithm_linear);
    ma_resampler resampler;

    if (ma_resampler_init(&resamplerConfig, nullptr, &resampler) != MA_SUCCESS)
    {
        // 初始化失败处理
        return {};
    }

    // 准备输入和输出缓冲区
    std::vector<float> output(input.size() * targetSampleRate / sourceSampleRate);
    ma_uint64 inputFrameCount = input.size();
    ma_uint64 outputFrameCount = output.size();

    // 执行重采样
    if (ma_resampler_process_pcm_frames(&resampler, input.data(), &inputFrameCount, output.data(), &outputFrameCount) !=
        MA_SUCCESS)
    {
        // 重采样失败处理
        ma_resampler_uninit(&resampler, nullptr);
        return {};
    }

    // 释放重采样器资源
    ma_resampler_uninit(&resampler, nullptr);

    // 返回重采样后的音频数据
    return output;
}

// 函数计算均值
float calculateMean(const std::vector<float> &audio)
{
    float sum = std::accumulate(audio.begin(), audio.end(), 0.0f);
    return sum / audio.size();
}

// 函数计算方差
float calculateVariance(const std::vector<float> &audio, float mean)
{
    float sumSquares = std::accumulate(audio.begin(), audio.end(), 0.0f,
                                       [mean](float acc, float val)
                                       { return acc + (val - mean) * (val - mean); });
    return sumSquares / audio.size();
}

// 标准化音频数据
std::vector<float> normalizeAudio(std::vector<float> &audio)
{
    float mean = calculateMean(audio);
    float variance = calculateVariance(audio, mean);
    float std_dev = std::sqrt(variance + 1e-7f);

    std::transform(audio.begin(), audio.end(), audio.begin(),
                   [mean, std_dev](float val)
                   { return (val - mean) / std_dev; });
    return audio; // 返回新的标准化音频数据
}
