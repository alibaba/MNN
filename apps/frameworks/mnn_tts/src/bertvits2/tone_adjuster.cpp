#include "tone_adjuster.hpp"

ToneAdjuster::ToneAdjuster(const std::string &local_resource_root) : word_spliter_(WordSpliter::GetInstance(local_resource_root))
{
    auto t0 = clk::now();
    PLOG(INFO, "ToneAdjuster 开始初始化...");
    resource_root_ = local_resource_root;

    yuqici_pattern_ = SplitUtf8String("吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶");
    de_pattern_ = SplitUtf8String("的地得");
    men_pattern_ = SplitUtf8String("们子");
    shang_pattern_ = SplitUtf8String("上下里");
    lai_pattern_ = SplitUtf8String("来去");
    lai_aux_pattern_ = SplitUtf8String("上下进出回过起开");
    ge_aux_pattern_ = SplitUtf8String("几有两半多各整每做是");
    digital_pattern_ = SplitUtf8String("一二三四五六七八九壹贰叁肆伍陆柒捌玖");
    punc_pattern_ = SplitUtf8String("：，；。？！“”‘’':,;.?!");

    // spliter_ = WordSpliter::GetInstance(local_resource_root);

    auto json_path = resource_root_ + "common/text_processing_jsons/default_tone_words.json";
    ParseDefaultToneWordJson(json_path);

    auto t1 = clk::now();
    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);

    PLOG(INFO, " ToneAdjuster 初始化完成, timecost: " + std::to_string(d1.count()) + "ms");
}

void ToneAdjuster::ParseDefaultToneWordJson(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);

    must_neural_words_ = json_obj["must_neural_tone_words"].get<std::vector<std::string>>();
    not_neural_words_ = json_obj["must_not_neural_tone_words"].get<std::vector<std::string>>();

    PLOG(PDEBUG, "tone_adjuster 必须轻声的词语: " + ConcatStrList(must_neural_words_, "|"));
    PLOG(PDEBUG, "tone_adjuster 不能轻声的词语: " + ConcatStrList(not_neural_words_, "|"));
}

// 判断一个字符串是否以指定前缀开头
bool ToneAdjuster::StartsWith(const std::string &str, const std::string &prefix)
{
    // 如果字符串长度小于前缀长度，直接返回 false
    if (str.size() < prefix.size())
    {
        return false;
    }
    // 使用 std::string::compare 检查前缀
    return str.compare(0, prefix.size(), prefix) == 0;
}

// 判断某个汉字是否为中文的数字，包含大写和小写形式
bool ToneAdjuster::IsChineseDigital(const std::string &hanzi)
{
    return HanziIn(hanzi, digital_pattern_);
}

// 判断某个词语中的对应索引的汉字是否为中文的数字，包含大写和小写形式
bool ToneAdjuster::IsChineseDigital(const std::string &word, int index)
{
    return PhrasePartIn(word, index, 1, digital_pattern_);
}

// 判断一个汉字是否在所给的模式中，如 HanziIn("我", {"我"， “你", "他"}) = true
bool ToneAdjuster::HanziIn(const std::string &hanzi, const std::vector<std::string> &pattern)
{
    return std::find(pattern.begin(), pattern.end(), hanzi) != pattern.end();
}

// 判断一个词语是否在所给的模式中，如 PhraseIn("阿里巴巴", {"北京", “阿里巴巴", "他"}) = true
bool ToneAdjuster::PhraseIn(const std::string &hanzi, const std::vector<std::string> &pattern)
{
    return std::find(pattern.begin(), pattern.end(), hanzi) != pattern.end();
}

// 判断一个词语的某一部分是否在所给的模式中，如 PhrasePartIn("阿里巴巴", 2, 2, {"北京", “巴巴", "他"}) = true
bool ToneAdjuster::PhrasePartIn(const std::string &word, int start_index, int size, const std::vector<std::string> &pattern)
{
    // 将词语拆分为单个的汉字组成的vector
    auto hanzi_list = SplitUtf8String(word);

    // 为了支持类似python中的负数索引，例如word[-1]表示最后一个汉字
    if (start_index < 0)
    {
        start_index += hanzi_list.size();
    }

    std::string target_phrase = "";
    for (int i = 0; i < size; i++)
    {
        target_phrase += hanzi_list[start_index + i];
    }
    return std::find(pattern.begin(), pattern.end(), target_phrase) != pattern.end();
}

// 判断词性是否在所在的模式中，如 PosIn("n", {"a", "n", "v"}) = true
// 注意词性可能是多个字母，因此采用char类型表示不合适，应该采用std::string
bool ToneAdjuster::PosIn(const std::string &pos, const std::vector<std::string> &pattern)
{
    return std::find(pattern.begin(), pattern.end(), pos) != pattern.end();
}

// 寻找词语中的第一个字在原始句子中的位置索引
int ToneAdjuster::FindSubstring(const std::vector<std::string> &str_list, const std::vector<std::string> &sub_str_list)
{

    int pos = -1;
    for (int i = 0; i < str_list.size(); i++)
    {
        if (str_list[i] == sub_str_list[0])
        {
            pos = i;
            break;
        }
    }

    return static_cast<int>(pos); // 将 size_t 类型的 pos 转换为 int 类型并返回
}

int ToneAdjuster::FindSubstring(const std::string &str, const std::string &sub_str)
{
    auto str_list = SplitUtf8String(str);
    auto sub_str_list = SplitUtf8String(sub_str);
    return FindSubstring(str_list, sub_str_list);
}

std::vector<std::string> ToneAdjuster::SplitWord(const std::string &word)
{
    auto utf8_word = SplitUtf8String(word);
    auto word_list = word_spliter_.ProcessWoPos(word, true);

    // 输入的word里面没有汉字，则直接返回空字符串数组
    if (word_list.size() < 1)
    {
        return {};
    }

    std::sort(word_list.begin(), word_list.end(), [](const std::string &a, const std::string &b)
              { return a.size() < b.size(); });

    auto first_subword = word_list[0];
    auto first_subword_utf8 = SplitUtf8String(first_subword);
    auto first_begin_idx = FindSubstring(utf8_word, first_subword_utf8);
    std::vector<std::string> new_word_list;
    if (first_begin_idx == 0)
    {
        std::string sencond_subword = "";
        for (int i = first_subword_utf8.size(); i < utf8_word.size(); i++)
        {
            sencond_subword += utf8_word[i];
        }

        // 不添加空的字符串
        if (!first_subword.empty())
        {
            new_word_list.push_back(first_subword);
        }
        if (!sencond_subword.empty())
        {
            new_word_list.push_back(sencond_subword);
        }
    }
    else
    {
        std::string sencond_subword = "";
        for (int i = 0; i < utf8_word.size() - first_subword_utf8.size(); i++)
        {
            sencond_subword += utf8_word[i];
        }
        // 不添加空的字符串
        if (!sencond_subword.empty())
        {
            new_word_list.push_back(sencond_subword);
        }
        if (!first_subword.empty())
        {
            new_word_list.push_back(first_subword);
        }
    }

    return new_word_list;
}

// 判断音调中是否都是3声
bool ToneAdjuster::AllToneThree(const std::vector<std::string> &finals)
{
    bool is_all_tone_three = true;
    for (int i = 0; i < finals.size(); i++)
    {
        if (finals[i].back() != '3')
        {
            is_all_tone_three = false;
            break;
        }
    }
    return is_all_tone_three;
}

// 对轻声进行处理
std::vector<std::string> ToneAdjuster::NeuralSandhi(const string &raw_word, const string &pos, std::vector<std::string> &finals)
{
    auto word = SplitUtf8String(raw_word);
    // 输入文本中没有汉字，直接返回
    if (word.size() < 1)
    {
        PLOG(WARNING, "tone_adjuster NeuralSandhi 没有中文: " + raw_word);
        return finals;
    }

    // 处理叠词，例如奶奶, 试试, 旺旺，爸爸等，将第二个字转换为轻声
    for (size_t j = 0; j < word.size(); ++j)
    {
        if (j >= 1 && word[j] == word[j - 1] && (PosIn(pos, {"n", "a", "v"})) && PhraseIn(raw_word, not_neural_words_))
        {
            PLOG(TRACE, "tone_adjuster NeuralSandhi 叠词: " + raw_word);
            finals[j].back() = '5';
        }
    }

    int ge_idx = FindSubstring(raw_word, "个");

    // 对语气词进行处理
    if (word.size() >= 1 && PhrasePartIn(raw_word, -1, 1, yuqici_pattern_))
    {
        PLOG(TRACE, "tone_adjuster NeuralSandhi 语气词: " + raw_word);
        finals.back().back() = '5';
    }
    // 对"的地得"进行处理
    else if (word.size() >= 1 && PhrasePartIn(raw_word, -1, 1, de_pattern_))
    {
        //        PLOG(TRACE, "tone_adjuster NeuralSandhi 的地得: " + raw_word);
        // TODO 这里会有问题，例如 ‘机会难得’，
        //        finals.back().back() = '5';
    }
    else if (word.size() > 1 && PhrasePartIn(raw_word, -1, 1, men_pattern_) && PosIn(pos, {"r", "n"}) && PhraseIn(raw_word, not_neural_words_))
    {
        PLOG(TRACE, "tone_adjuster NeuralSandhi 子们: " + raw_word);
        finals.back().back() = '5';
    }
    else if (word.size() > 1 && PhrasePartIn(raw_word, -1, 1, shang_pattern_) && PosIn(pos, {"s", "l", "f"}))
    {
        PLOG(TRACE, "tone_adjuster NeuralSandhi 上下里: " + raw_word);
        finals.back().back() = '5';
    }
    else if (word.size() > 1 && PhrasePartIn(raw_word, -1, 1, lai_pattern_) and PhrasePartIn(raw_word, -2, 1, lai_aux_pattern_))
    {
        PLOG(TRACE, "tone_adjuster NeuralSandhi 来去: " + raw_word);
        finals.back().back() = '5';
    }
    else if ((ge_idx >= 1 && (IsChineseDigital(raw_word, ge_idx - 1) || PhrasePartIn(raw_word, ge_idx - 1, 1, ge_aux_pattern_))) || raw_word == "个")
    {
        PLOG(TRACE, "tone_adjuster NeuralSandhi 个: " + raw_word);
        finals[ge_idx].back() = '5';
    }
    else
    {
        if (PhraseIn(raw_word, must_neural_words_) || (word.size() > 2 && PhrasePartIn(raw_word, word.size() - 2, 2, must_neural_words_)))
        {
            PLOG(TRACE, "tone_adjuster NeuralSandhi 必须轻声的词: " + raw_word);
            finals.back().back() = '5';
        }
    }

    auto word_list = SplitWord(raw_word);
    if (word_list.size() < 1)
    {
        PLOG(WARNING, "tone_adjuster NeuralSandhi 没有汉字: " + raw_word);
        return finals;
    }

    auto first_subword_length = SplitUtf8String(word_list[0]).size();

    std::vector<std::vector<std::string>> finals_list;
    finals_list.push_back(std::vector<std::string>(finals.begin(), finals.begin() + first_subword_length));
    finals_list.push_back(std::vector<std::string>(finals.begin() + first_subword_length, finals.end()));

    for (size_t i = 0; i < word_list.size(); ++i)
    {
        auto cur_word_utf8 = SplitUtf8String(word_list[i]);
        if (PhraseIn(word_list[i], must_neural_words_) || (cur_word_utf8.size() >= 2 && PhrasePartIn(word_list[i], cur_word_utf8.size() - 2, 2, must_neural_words_)))
        {
            PLOG(TRACE, "tone_adjuster NeuralSandhi 细分后必须轻声的词: " + word_list[i]);
            finals_list[i].back().back() = '5';
        }
    }

    std::vector<std::string> merged_finals;
    for (const auto &part : finals_list)
    {
        merged_finals.insert(merged_finals.end(), part.begin(), part.end());
    }
    return merged_finals;
}

// 对 “不”进行处理，如看不懂
std::vector<std::string> ToneAdjuster::BuSandhi(const string &word, const string &pos, std::vector<std::string> &finals)
{
    auto utf8_word = SplitUtf8String(word);
    // 看不懂
    if (utf8_word.size() == 3 && utf8_word[1] == "不")
    {
        finals[1].back() = '5';
    }
    else
    {
        // "不" before tone4 should be bu2, e.g. 不怕
        for (int i = 0; i < utf8_word.size(); i++)
        {
            if (utf8_word[i] == "不" && i + 1 < utf8_word.size() && finals[i + 1].back() == '4')
            {
                finals[i].back() = '2';
            }
        }
    }

    return finals;
}

// 对 “一“ 在词语中的音调进行调整
std::vector<std::string> ToneAdjuster::YiSandhi(const string &word, const string &pos, std::vector<std::string> &finals)
{
    auto utf8_word = SplitUtf8String(word);
    // 包含一且全部都是数字，一照样发一声。如一百五十，五三一七，
    if (HanziIn("一", utf8_word))
    {
        bool has_no_digital = false;
        for (int i = 0; i < utf8_word.size(); i++)
        {
            if (!IsChineseDigital(utf8_word[i]))
            {
                has_no_digital = true;
                break;
            }
        }

        if (has_no_digital)
        {
            return finals;
        }
    }

    // "一" between reduplication words should be yi5, e.g. 看一看
    if (utf8_word.size() == 3 && utf8_word[1] == "一" && utf8_word[0] == utf8_word[2])
    {
        finals[1].back() = '5';
    }
    else if (StartsWith(word, "第一"))
    {
        finals[1].back() = '1';
    }
    else
    {
        for (int i = 0; i < utf8_word.size(); i++)
        {
            if (utf8_word[i] == "一" && i + 1 < utf8_word.size())
            {
                // "一" before tone4 should be yi2, e.g.一段
                if (finals[i + 1].back() == '4')
                {
                    finals[i].back() = '2';
                }
                // "一" before non - tone4 should be yi4, e.g.一天
                // "一" 后面如果是标点，还读一声
                else if (!HanziIn(utf8_word[i + 1], punc_pattern_))
                {
                    finals[i].back() = '4';
                }
            }
        }
    }
    return finals;
}

// 对词语中的三声进行特殊处理
std::vector<std::string> ToneAdjuster::ThreeToneSandhi(const string &word, const string &pos, std::vector<std::string> &finals)
{
    auto utf8_word = SplitUtf8String(word);
    // 如果两个字都是三声，第一个三声要变成二声，如你好
    if (utf8_word.size() == 2 && AllToneThree(finals))
    {
        finals[0].back() = '2';
    }
    else if (utf8_word.size() == 3)
    {
        auto word_list = SplitWord(word);
        if (AllToneThree(finals))
        {
            // disyllabic + monosyllabic, e.g. 蒙古/包
            if (word_list[0].size() == 2)
            {
                finals[0].back() = '2';
                finals[1].back() = '2';
            }
            // monosyllabic + disyllabic, e.g. 纸/老虎
            else if (word_list[0].size() == 1)
            {
                finals[1].back() = '2';
            }
        }
        else
        {
            std::vector<std::vector<std::string>> finals_list;
            int first_word_size = SplitUtf8String(word_list[0]).size();
            finals_list.push_back(std::vector<std::string>(finals.begin(), finals.begin() + first_word_size));
            finals_list.push_back(std::vector<std::string>(finals.begin() + first_word_size, finals.end()));
            if (finals_list.size() == 2)
            {

                std::vector<std::string> updated_finals;
                for (int i = 0; i < finals_list.size(); i++)
                {
                    auto sub = finals_list[i];
                    // e.g. 所有/人
                    if (sub.size() == 2 && AllToneThree(sub))
                    {
                        sub[0].back() = '2';
                    }
                    else if (i == 1 && !AllToneThree(sub) && finals_list[i][0].back() == '3' && finals_list[0][finals_list[0].size() - 1].back() == '3')
                    {
                        // TODO
                        //  finals_list[0] = finals_list[0][finals_list[0].size() - 1];
                        finals_list[0].back() = '2';
                    }
                    updated_finals.insert(updated_finals.end(), finals_list[i].begin(), finals_list[i].end());
                }
                finals = updated_finals;
            }
        }
    }
    // split idiom into two words who's length is 2
    else if (utf8_word.size() == 4)
    {
        std::vector<std::vector<std::string>> finals_list;
        finals_list.push_back(std::vector<std::string>(finals.begin(), finals.begin() + 2));
        finals_list.push_back(std::vector<std::string>(finals.begin() + 2, finals.end()));

        std::vector<std::string> updated_finals;
        for (int i = 0; i < finals_list.size(); i++)
        {
            if (AllToneThree(finals_list[i]))
            {
                finals_list[i][0].back() = '2';
            }
            updated_finals.insert(updated_finals.end(), finals_list[i].begin(), finals_list[i].end());
        }

        finals = updated_finals;
    }

    return finals;
}

std::vector<std::string> ToneAdjuster::Process(const string &word, const string &pos, std::vector<std::string> &finals)
{
    PLOG(PDEBUG, "tone_adjuster 输入韵母: " + ConcatStrList(finals, "|"));

    finals = BuSandhi(word, pos, finals);
    PLOG(PDEBUG, "tone_adjuster 处理'不'后: " + ConcatStrList(finals, "|"));

    finals = YiSandhi(word, pos, finals);
    PLOG(PDEBUG, "tone_adjuster 处理'一'后: " + ConcatStrList(finals, "|"));

    finals = NeuralSandhi(word, pos, finals);
    PLOG(PDEBUG, "tone_adjuster 处理轻声后: " + ConcatStrList(finals, "|"));

    finals = ThreeToneSandhi(word, pos, finals);
    PLOG(PDEBUG, "tone_adjuster 处理三声后: " + ConcatStrList(finals, "|"));
    return finals;
}
