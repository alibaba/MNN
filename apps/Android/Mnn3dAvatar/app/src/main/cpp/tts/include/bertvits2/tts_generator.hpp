/**
 * @file tts_generator.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 基于音素和文本特征输入，得到音频的生成网络
 *
 * MNN网络，实现音频的合成
 */
#ifndef _HEADER_MNN_TTS_SDK_TTS_GENERATOR_H_
#define _HEADER_MNN_TTS_SDK_TTS_GENERATOR_H_

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "utils.hpp"

using namespace MNN;
using namespace MNN::Express;

class TTSGenerator
{
public:
    TTSGenerator();
    TTSGenerator(const std::string &tts_generator_model_path, const std::string &mnn_mmap_dir);
    std::vector<int16_t> Process(const phone_data &g2p_data_, const std::vector<std::vector<float>> &cn_bert, const std::vector<std::vector<float>> &en_bert);

private:
    // 资源文件根目录
    std::string resource_root_;

    // NOTE 为了保证TTSGenerator类支持复制初始化, 这里不能用unique_ptr，只能用shared_ptr
    std::shared_ptr<Module> module_;
    std::shared_ptr<Executor> executor_;
    std::vector<std::string> input_names{"phone", "tone", "lang_id", "cn_bert", "en_bert"};
    std::vector<std::string> output_names{"audio"};
    std::shared_ptr<Executor::RuntimeManager> rtmgr_;
    int bert_feature_dim_ = 1024;
};
#endif
