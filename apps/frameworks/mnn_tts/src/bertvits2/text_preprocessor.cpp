#include "text_preprocessor.hpp"

TextNormalizer::TextNormalizer()
{
    // 句子分割符号，这里的符号是中英文句尾的表示符号
    SENTENCE_SPLITOR = std::regex(R"((：.。？！?!”’?))");
}

std::string TextNormalizer::RemoveSpecialChineseSymbols(const std::string &text)
{
    std::vector<std::string> specical_chinese_symbols = {"——",
                                                         " ",
                                                         "《",
                                                         "》",
                                                         "<"
                                                         ">",
                                                         "{"
                                                         "}",
                                                         "(",
                                                         ")",
                                                         "#",
                                                         "&",
                                                         "@",
                                                         "“",
                                                         "”",
                                                         "^",
                                                         "_",
                                                         "|",
                                                         "…",
                                                         "\\"};
    auto texts = SplitUtf8String(text);
    std::string processed_texts = "";
    for (auto t : texts)
    {
        if (!HanListContains(specical_chinese_symbols, t))
        {
            processed_texts += t;
        }
    }
    return processed_texts;
}

std::vector<std::string> TextNormalizer::SplitCnPart(const std::string &text)
{
    // 这里只处理纯中文文本
    auto processed_text = RemoveSpecialChineseSymbols(text);

    std::vector<std::string> sentences;
    std::string current_sentence;

    auto words = SplitUtf8String(processed_text);

    for (const auto &word : words)
    {
        current_sentence += word; // 将当前汉字追加到当前句子

        // 检查当前字符串是否为句子结束标点
        if (IsChineseEnding(word))
        {
            sentences.push_back(current_sentence); // 保存当前句子
            current_sentence.clear();              // 开始新的句子
        }
    }

    // 添加最后一个句子（如果有）
    if (!current_sentence.empty())
    {
        sentences.push_back(current_sentence);
    }

    return sentences;
}

std::vector<std::string> TextNormalizer::SplitEnPart(const std::string &text)
{
    std::vector<std::string> sentences;
    std::string current_sentence;
    for (const auto &word : text)
    {
        current_sentence += word; // 将当前单词追加到当前句子
        // 检查当前字符串是否为句子结束标点
        if (IsEnglishEnding(std::string({word})))
        {
            sentences.push_back(current_sentence); // 保存当前句子
            std::cout <<"==> cur sen:"<< current_sentence << std::endl;
            current_sentence.clear();              // 开始新的句子
        }
    }
    // 添加最后一个句子（如果有）
    if (!current_sentence.empty())
    {
        sentences.push_back(current_sentence);
    }
    return sentences;
}

// 对句子进行规范化，包括繁体转简体、全角转半角、日期、时间转换，电话号码转换等
std::string TextNormalizer::NormalizeSentence(const std::string &sentence)
{
    PLOG(TRACE, "textnorm input: " + sentence);
    auto sentence1 = TraditionalToSimplified(sentence);
    PLOG(TRACE, "textnorm 繁转简: " + sentence1);
    sentence1 = FullWidthToHalfWidth(sentence1);
    PLOG(TRACE, "textnorm 全角转半角: " + sentence1);
    sentence1 = ReplaceCompare(sentence1);
    PLOG(TRACE, "textnorm 去除冒号: " + sentence1);
    sentence1 = ReplaceTyperead(sentence1);
    sentence1 = ReplaceTyperead2(sentence1);
    PLOG(TRACE, "textnorm 去除冒号: " + sentence1);
    sentence1 = ReplaceSpecificCommaNumbers(sentence1);
    PLOG(TRACE, "textnorm 去除数字中逗号: " + sentence1);
    sentence1 = ReplaceCurrency(sentence1);
    PLOG(TRACE, "textnorm 货币转换: " + sentence1);
    sentence1 = RegexReplaceDate(sentence1);
    PLOG(TRACE, "textnorm 日期转换: " + sentence1);
    sentence1 = RegexReplaceDate2(sentence1);
    PLOG(TRACE, "textnorm 日期转换2: " + sentence1);
    sentence1 = RegexReplaceTimeRange(sentence1);
    PLOG(TRACE, "textnorm 时间范围: " + sentence1);
    sentence1 = RegexReplaceTime(sentence1);
    PLOG(TRACE, "textnorm 时间: " + sentence1);
    sentence1 = RegexReplaceTemperature(sentence1);
    PLOG(TRACE, "textnorm 温度: " + sentence1);
    sentence1 = ReplaceMeasure(sentence1);
    PLOG(TRACE, "textnorm 计量: " + sentence1);
    sentence1 = RegexReplaceFrac(sentence1);
    PLOG(TRACE, "textnorm 分数: " + sentence1);
    sentence1 = RegexReplacePercentage(sentence1);
    PLOG(TRACE, "textnorm 百分比: " + sentence1);
    sentence1 = RegexReplaceMobile(sentence1);
    PLOG(TRACE, "textnorm 手机号: " + sentence1);
    sentence1 = RegexReplacePhone(sentence1);
    PLOG(TRACE, "textnorm 电话号码: " + sentence1);
    sentence1 = RegexReplaceNationalUniformNumber(sentence1);
    PLOG(TRACE, "textnorm 国标码: " + sentence1);
    sentence1 = RegexReplaceRange(sentence1);
    PLOG(TRACE, "textnorm 范围: " + sentence1);
    sentence1 = RegexReplaceNegativeNum(sentence1);
    PLOG(TRACE, "textnorm 负数: " + sentence1);
    sentence1 = RegexReplaceDecimalNum(sentence1);
    PLOG(TRACE, "textnorm 小数: " + sentence1);
    sentence1 = RegexReplace2WithQuantifier(sentence1);
    PLOG(TRACE, "textnorm 2->两: " + sentence1);
    sentence1 = RegexReplacePositiveQuantifier(sentence1);
    PLOG(TRACE, "textnorm 单位: " + sentence1);
    sentence1 = RegexReplaceDefaultNum(sentence1);
    PLOG(TRACE, "textnorm 一转幺: " + sentence1);
    sentence1 = RegexReplaceNumber(sentence1);
    PLOG(TRACE, "textnorm 数字替换: " + sentence1);
    sentence1 = PostReplace(sentence1);
    PLOG(TRACE, "textnorm 特殊符号: " + sentence1);
    return sentence1;
}
std::vector<std::string> TextNormalizer::Process(const std::string &text)
{
    auto sentences = SplitCnPart(text);

    std::vector<std::string> results;
    for (auto &sentence : sentences)
    {
        results.push_back(NormalizeSentence(sentence));
    }
    return results;
}

TextPreprocessor::TextPreprocessor() { normalizer = TextNormalizer(); }

std::string TextPreprocessor::ReplaceSpecialText(std::string &text)
{
    text = StrReplaceAll(text, "嗯", "恩");
    text = StrReplaceAll(text, "呣", "母");
    text = StrReplaceAll(text, "咯", "落");
    text = StrReplaceAll(text, "嘞", "勒");
    text = StrReplaceAll(text, "B2B", "BtoB");
    text = StrReplaceAll(text, "B2C", "BtoC");
    text = StrReplaceAll(text, "C2C", "CtoC");
    text = ReReplace(text, R"((\d+)\s?ml)", "$1 毫升");
    text = ReReplace(text, R"((m|M)22)", "$1 two two");

    std::regex pattern(R"([Nn][oO]\.? ?1)");
    // Replace matches of the pattern in the text with "Number One"
    text = std::regex_replace(text, pattern, "Number One");

    // 替换1w为1万
    std::regex pattern1(R"((\d+)[wW])");
    // Use regex_replace to replace the pattern with the digit followed by '万'
    text = std::regex_replace(text, pattern1, "$1万");
    return text;
}

// 分离字符串，返回段落和语言的对
std::vector<std::pair<std::string, std::string>> TextPreprocessor::SplitByLang(const std::string &text,
                                                                               const std::string &digit_type)
{
    std::vector<std::pair<std::string, std::string>> segments;
    std::vector<std::string> types;

    // 确定每个字符的类型：中文、字母、其他
    auto characters = SplitUtf8String(text);
    for (int i = 0; i < characters.size(); ++i)
    {
        auto cur_char = characters[i];
        uint32_t unicode_char = Utf8ToUnicode(cur_char);

        if (IsChinese(unicode_char))
        {
            types.push_back("zh");
        }
        else if (IsAlphabet(unicode_char))
        {
            types.push_back("en");
        }
        else
        {
            if (!digit_type.empty() && cur_char >= "0" && cur_char <= "9")
            {
                types.push_back(digit_type);
            }
            else
            {
                types.push_back("other");
            }
        }
    }
    assert(types.size() == characters.size());

    std::string temp_seg;
    std::string temp_lang;
    bool flag = false;

    for (size_t i = 0; i < characters.size(); ++i)
    {
        // 找到段落的第一个字符
        if (!flag)
        {
            temp_seg += characters[i];
            temp_lang = types[i];
            flag = true;
        }
        else
        {
            if (temp_lang == "other")
            {
                // 文本开始不是语言
                temp_seg += characters[i];
                if (types[i] != temp_lang)
                {
                    temp_lang = types[i];
                }
            }
            else
            {
                if (types[i] == temp_lang || types[i] == "other")
                {
                    // 合并相同语言或其他
                    temp_seg += characters[i];
                }
                else
                {
                    // 语言变更
                    segments.push_back({temp_seg, temp_lang});
                    temp_seg = characters[i];
                    temp_lang = types[i]; // 新语言
                }
            }
        }
    }
    if (!temp_seg.empty())
    {
        segments.push_back({temp_seg, temp_lang});
    }
    return segments;
}

// 合并相邻的中文/英文segments
std::vector<std::pair<std::string, std::string>> TextPreprocessor::MergeAdjacent(
    const std::vector<std::pair<std::string, std::string>> &lang_splits)
{
    std::vector<std::pair<std::string, std::string>> segments;
    std::pair<std::string, std::string> currentSeg{"", ""};

    for (const auto &seg : lang_splits)
    {
        if (seg.second == "en" || seg.second == "other")
        {
            if (currentSeg.first.empty())
            {
                // 第一次遇到非中文
                segments.push_back(seg);
            }
            else
            {
                // 如果有累积的中文段落，先将它们添加到 segments
                segments.push_back(currentSeg);
                // 然后添加当前的非中文段落
                segments.push_back(seg);
                // 重置累积的中文段落
                currentSeg = {"", ""};
            }
        }
        else
        {
            // 中文
            if (currentSeg.first.empty())
            {
                // 第一次遇到中文
                currentSeg = seg;
            }
            else
            {
                // 合并累积的中文段落
                currentSeg.first += seg.first;
            }
        }
    }

    if (!currentSeg.first.empty())
    {
        // 如果最后有累积的中文段落，将它们添加到 segments
        segments.push_back(currentSeg);
    }

    return segments;
}

std::vector<std::vector<SentLangPair>> TextPreprocessor::Process(std::string &text, int split_max_len)
{
    // 删除空行
    std::string norm_text = RemoveEmptyLines(text);
    // 句尾增加句号
    norm_text = AddPeriodBeforeNewline(norm_text);

    // 替换一些需要特殊处理的字符，如B2C替换为BToC，这些规则可以根据实际使用情况进行添加
    norm_text = ReplaceSpecialText(text);

    norm_text = normalizer.NormalizeSentence(norm_text);
    // 根据语言将输入的文本拆分成中文部分、英文部分，数字部分当作中文
    auto lang_splits = SplitByLang(norm_text, "zh");

    auto segments = lang_splits;
    std::vector<std::vector<SentLangPair>> norm_list;
    std::vector<SentLangPair> cur_sent_ls;
    int cur_len = 0;
    for (const auto &seg : segments)
    {
        std::string lang = seg.second;
        std::string content = seg.first;
        std::vector<SentLangPair> tmp;
        if (lang == "en")
        {
            // 将英文部分拆分成独立的句子
            for (const auto &x : normalizer.SplitEnPart(content))
            {
                tmp.emplace_back(x, lang);
            }
        }
        else
        {
            // 将中文部分拆分成独立的句子，并进行规范化，如数字时间日期等的转换
            for (const auto &x : normalizer.Process(content))
            {
                tmp.emplace_back(x, lang);
            }
        }
        for (const auto &split_words : tmp)
        {
            cur_sent_ls.push_back(split_words);
            cur_len += split_words.sent.length();
            // if (cur_len > split_max_len && SENTENCE_SPLITOR.find(split_words.lang.back()) != std::string::npos)
            if (cur_len > split_max_len)
            {
                norm_list.push_back(cur_sent_ls);
                cur_sent_ls.clear();
                cur_len = 0;
            }
        }
    }
    if (!cur_sent_ls.empty())
    {
        norm_list.push_back(cur_sent_ls);
    }
    // auto res = FlattenVector<SentLangPair>(norm_list);
    return norm_list;
}
