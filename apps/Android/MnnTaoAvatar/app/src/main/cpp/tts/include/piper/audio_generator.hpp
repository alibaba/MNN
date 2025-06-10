#pragma once

/**
 * @file audio_generator.hpp
 * @author PixelAI Team
 * @date 2025-05-27
 * @version 1.0
 * @brief piper tts模型的生成器，传入音素，合成音频
 *
 */

#include <llm/llm.hpp>
#include <audio/audio.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>

using namespace MNN;
using namespace MNN::Transformer;
using namespace MNN::Express;
using namespace MNN::AUDIO;

class AudioGenerator
{
public:
    AudioGenerator();
    AudioGenerator(const std::string &model_path);
    std::vector<float> Process(const std::vector<int> &input, int input_length, const std::vector<float> &scales);

private:
    std::shared_ptr<Module> module_;
    std::shared_ptr<Executor> executor_;
    std::vector<std::string> input_names{"input", "input_lengths", "scales"};
    std::vector<std::string> output_names{"output"};
};
