#ifndef UTILS_HPP
#define UTILS_HPP
#include <vector>
#include <dlfcn.h>
#include <cassert>
#include "file_loader.hpp"
#include "miniaudio/miniaudio.h"
#include "common/Common.hpp"

struct AudioData {
    std::vector<ma_int16> samples;          // PCM样本数据
    std::vector<float> normalized_samples;  // 归一化到[-1, 1]的PCM样本数据
    ma_uint32 sampleRate;                   // 采样率
    ma_uint32 channels;                     // 声道数
};

class file_utils {
public:
    static std::vector<char> LoadFileToBuffer(const char *fileName);
};

#endif //UTILS_HPP
