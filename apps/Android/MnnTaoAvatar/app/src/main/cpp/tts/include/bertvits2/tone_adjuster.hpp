/**
 * @file tone_adjuster.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 拼音音调调节类
 *
 * 对预测的pinyin根据文本内容进行进一步调节，比如叠词，两个三声在一起的变音，"不","一"等字的变音
 */
#ifndef _HEADER_MNN_TTS_SDK_TONE_ADJUSTER_H_
#define _HEADER_MNN_TTS_SDK_TONE_ADJUSTER_H_

#include "utils.hpp"
#include "word_spliter.hpp"

using namespace std;

class ToneAdjuster
{
public:
    ToneAdjuster(const std::string &local_resource_root);

    // 解析默认的必须和必不调整音节的词语json文件，key是'must_neural_tone_words'和‘must_not_neural_tone_words’
    std::vector<std::string> Process(const string &word, const string &pos, std::vector<std::string> &finals);

private:
    // 解析默认的必须和必不调整音节的词语json文件，key是'must_neural_tone_words'和‘must_not_neural_tone_words’
    void ParseDefaultToneWordJson(const std::string &json_path);

    // 判断一个字符串是否以指定前缀开头
    bool StartsWith(const std::string &str, const std::string &prefix);

    // 判断某个汉字是否为中文的数字，包含大写和小写形式
    bool IsChineseDigital(const std::string &hanzi);

    // 判断某个词语中的对应索引的汉字是否为中文的数字，包含大写和小写形式
    bool IsChineseDigital(const std::string &word, int index);

    // 判断一个汉字是否在所给的模式中，如 HanziIn("我", {"我"， “你", "他"}) = true
    bool HanziIn(const std::string &hanzi, const std::vector<std::string> &pattern);

    // 判断一个词语是否在所给的模式中，如 PhraseIn("阿里巴巴", {"北京", “阿里巴巴", "他"}) = true
    bool PhraseIn(const std::string &hanzi, const std::vector<std::string> &pattern);

    // 判断一个词语的某一部分是否在所给的模式中，如 PhraseIn("阿里巴巴", 2, 2, {"北京", “巴巴", "他"}) = true
    bool PhrasePartIn(const std::string &word, int start_index, int size, const std::vector<std::string> &pattern);

    // 判断词性是否在所在的模式中，如 PosIn("n", {"a", "n", "v"}) = true
    // 注意词性可能是多个字母，因此采用char类型表示不合适，应该采用std::string
    bool PosIn(const std::string &pos, const std::vector<std::string> &pattern);

    // 寻找词语中的第一个字在原始句子中的位置索引
    int FindSubstring(const std::vector<std::string> &str_list, const std::vector<std::string> &sub_str_list);

    // 寻找词语中的第一个字在原始句子中的位置索引
    int FindSubstring(const std::string &str, const std::string &sub_str);

    // 将短句进行分词
    std::vector<std::string> SplitWord(const std::string &word);

    // 判断音调中是否都是3声
    bool AllToneThree(const std::vector<std::string> &finals);

    // 对轻声进行处理
    std::vector<std::string> NeuralSandhi(const string &word, const string &pos, std::vector<std::string> &finals);

    // 对 “不”进行处理，如看不懂
    std::vector<std::string> BuSandhi(const string &word, const string &pos, std::vector<std::string> &finals);

    // 对 “一“ 在词语中的音调进行调整
    std::vector<std::string> YiSandhi(const string &word, const string &pos, std::vector<std::string> &finals);

    // 对词语中的三声进行特殊处理
    std::vector<std::string> ThreeToneSandhi(const string &word, const string &pos, std::vector<std::string> &finals);

private:
    // 资源文件根目录
    std::string resource_root_;

    std::vector<std::string> not_neural_words_;
    std::vector<std::string> must_neural_words_;
    std::vector<std::string> yuqici_pattern_;
    std::vector<std::string> de_pattern_;
    std::vector<std::string> men_pattern_;
    std::vector<std::string> shang_pattern_;
    std::vector<std::string> lai_pattern_;
    std::vector<std::string> lai_aux_pattern_;
    std::vector<std::string> ge_aux_pattern_;
    std::vector<std::string> digital_pattern_;
    std::vector<std::string> punc_pattern_;

    WordSpliter &word_spliter_;
};
#endif // _HEADER_MNN_TTS_SDK_TONE_ADJUSTER_H_
