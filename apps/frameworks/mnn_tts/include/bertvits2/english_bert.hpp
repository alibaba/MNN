#ifndef _HEADER_MNN_TTS_SDK_ENGLISH_BERT_H_
#define _HEADER_MNN_TTS_SDK_ENGLISH_BERT_H_
/**
 * @file english_bert.hpp
 * @author PixelAI Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 英文bert特征提取，目前没有使用
 *
 *
 */

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "utils.hpp"

namespace fs = std::filesystem;

using namespace MNN;
using namespace MNN::Express;

class EnglishBert
{
public:
    EnglishBert();
    EnglishBert(const std::string &local_resource_root);
    std::vector<std::vector<float>> Process(const std::string &text, const std::vector<int> &word2ph);

private:
    std::vector<int> ObtainBertTokens(const std::string &text);
    void ParseBertTokenJsonFile(const std::string &json_path);

private:
    std::string local_resource_root_;

    // tokenizer
    // BertTokenizer bert_tokenizer_;

    bert_token bert_token_;

    // MNN 网络相关变量
    int bert_feature_dim_ = 1024;
    std::shared_ptr<Module> module; // module
    std::vector<std::string> input_names{"input_ids", "token_type_ids", "attention_mask"};
    std::vector<std::string> output_names{"hidden_states"};
};
#endif // _HEADER_MNN_TTS_SDK_ENGLISH_BERT_H_
