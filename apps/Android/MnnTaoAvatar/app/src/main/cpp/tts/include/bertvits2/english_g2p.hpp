/**
 * @file english_g2p.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 英文 Grapheme-to-Phoneme
 *
 * 将英文文本转换为音素，对英文单词，按照词典进行发音，别的缩写等按照字母来发音
 */
#ifndef _HEADER_MNN_TTS_SDK_ENGLISH_G2P_H_
#define _HEADER_MNN_TTS_SDK_ENGLISH_G2P_H_

#include "utils.hpp"

class EnglishG2P
{
public:
    EnglishG2P();
    EnglishG2P(const std::string &local_resource_root);
    std::tuple<std::string, phone_data> Process(SentLangPair &sent_lang);

private:
    // 解析英文单词到音素的字典
    void ParseEnglishDict(const std::string &json_path);

    // 英文G2P
    g2p_data G2P(const std::string &input_text);

    // 音素到index，并差值
    phone_data CleanedTextToSequence(const g2p_data &g2p_data_, const std::string &language);

private:
    // 资源文件根目录
    std::string resource_root_;

    // 保存英文发音的字典
    english_dict eng_dict_;

    std::unordered_map<std::string, std::string> punctuation_mapping_;
    std::unordered_map<std::string, int> _symbol_to_id;
};
#endif
