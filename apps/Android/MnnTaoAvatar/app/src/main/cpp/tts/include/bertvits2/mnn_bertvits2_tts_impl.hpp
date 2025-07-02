/**
 * @file mnn_tts_sdk.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief MNN C++ 版本的TTS SDK入口类
 *
 * SDK 入口类，使用说明和实现细节见文档：https://aliyuque.antfin.com/tqwzle/sdqvv3/orgq09hlu2lrtltk
 */
#ifndef _HEADER_MNN_TTS_SDK_mnn_tts_SDK_H_
#define _HEADER_MNN_TTS_SDK_mnn_tts_SDK_H_

#include "mnn_tts_impl_base.hpp"

#include "chinese_bert.hpp"
#include "chinese_g2p.hpp"
#include "english_bert.hpp"
#include "english_g2p.hpp"
#include "text_preprocessor.hpp"
#include "tts_generator.hpp"
#include "utils.hpp"

using json = nlohmann::json;

typedef std::vector<int16_t> Audio;

class MNNBertVits2TTSImpl : public MNNTTSImplBase
{
public:
    MNNBertVits2TTSImpl(const std::string &local_resource_root, const std::string &tts_generator_model_path, const std::string &mnn_mmap_dir);

    // 对一个字符串，合成对应的音频
    std::tuple<int, Audio> Process(const std::string& text) override;

private:
    // 提取文本对应的音素和bert特征，拿到特征后调用generator来合成音频
    std::tuple<phone_data, std::vector<std::vector<float>>, std::vector<std::vector<float>>> ExtractPhoneTextFeatures(const std::vector<SentLangPair> &word_list_by_lang);

private:
    // 资源文件根目录
    std::string resource_root_;

    // 采样率
    int sample_rate_ = 44100;

    // 中间处理步骤对应的变量
    TextPreprocessor text_preprocessor_;
    ChineseG2P cn_g2p_;
    EnglishG2P en_g2p_;
    ChineseBert cn_bert_model_;
    EnglishBert en_bert_model_;
    TTSGenerator tts_generator_;
};
#endif // _HEADER_MNN_TTS_SDK_mnn_tts_SDK_H_
