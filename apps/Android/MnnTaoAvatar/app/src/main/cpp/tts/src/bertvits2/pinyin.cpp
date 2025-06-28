#include "pinyin.hpp"

const std::vector<std::string> _INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j",
                                            "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"};

std::map<std::string, std::string> PHONETIC_SYMBOL_DICT = {
    {"ā", "a1"},
    {"á", "a2"},
    {"ǎ", "a3"},
    {"à", "a4"},
    {"ē", "e1"},
    {"é", "e2"},
    {"ě", "e3"},
    {"è", "e4"},
    {"ō", "o1"},
    {"ó", "o2"},
    {"ǒ", "o3"},
    {"ò", "o4"},
    {"ī", "i1"},
    {"í", "i2"},
    {"ǐ", "i3"},
    {"ì", "i4"},
    {"ū", "u1"},
    {"ú", "u2"},
    {"ǔ", "u3"},
    {"ù", "u4"},
    {"ü", "v"},
    {"ǖ", "v1"},
    {"ǘ", "v2"},
    {"ǚ", "v3"},
    {"ǜ", "v4"},
    {"ń", "n2"},
    {"ň", "n3"},
    {"ǹ", "n4"},
    {"m̄", "m1"},
    {"ḿ", "m2"},
    {"m̀", "m4"},
    {"ê̄", "ê1"},
    {"ế", "ê2"},
    {"ê̌", "ê3"},
    {"ề", "ê4"},
};

// 将带有声调符号的字符替换为使用数字表示声调的字符
std::string replace_symbol_to_number(const std::string &pinyin)
{
    std::string value = pinyin;
    for (const auto &symbol_pair : PHONETIC_SYMBOL_DICT)
    {
        std::regex reg(symbol_pair.first);
        value = std::regex_replace(value, reg, symbol_pair.second);
    }
    return value;
}

std::string to_tone3(const std::string &pinyin)
{
    std::string tone2 = replace_symbol_to_number(pinyin);
    std::regex reg("([a-zêü]+)([1-4])([a-zêü]*)");
    return std::regex_replace(tone2, reg, "$1$3$2");
}

std::tuple<std::string, std::string> find_initial_final(const std::string &pinyin)
{
    std::string initial = "";
    std::string final = "";
    for (const auto &cur_initial : _INITIALS)
    {
        if (pinyin.compare(0, cur_initial.length(), cur_initial) == 0)
        {
            initial = cur_initial;
            break;
        }
    }

    if (!initial.empty())
    {
        final = pinyin.substr(initial.length());
    }
    else
    {
        final = pinyin;
    }

    return std::make_tuple(initial, final); // 如果没有找到匹配的声母，返回空字符串
}
Pinyin::Pinyin() {}
Pinyin::Pinyin(const std::string &local_resource_root)
{
    auto t0 = clk::now();
    PLOG(INFO, "Pinyin 开始初始化...");
    resource_root_ = local_resource_root;

    // 读取大json速度太慢，修改为读取bin
    // auto pinyin_json_path = resource_root_ + "common/text_processing_jsons/pinyin_dict.json";
    // auto phrase_json_path = resource_root_ + "common/text_processing_jsons/phrases_dict.json";
    // 解析单字和词组的拼音数据
    // ParsePinyinJsonFile(pinyin_json_path);
    // ParsePhraseJsonFile(phrase_json_path);

    auto pinyin_bin_path = resource_root_ + "common/text_processing_jsons/pinyin_dict.bin";
    auto phrase_bin_path = resource_root_ + "common/text_processing_jsons/phrases_dict.bin";

    // SavePinyinMapToBin(pinyin_bin_path, pinyin_map_);
    // SavePhraseMapToBin(phrase_bin_path, phrase_map_);
    LoadPinyinMapFromBin(pinyin_bin_path, pinyin_map_);
    LoadPhraseMapFromBin(phrase_bin_path, phrase_map_);

    // hotwords load不耗时，且会动态更新，不转换为bin
    auto hotwords_json_path = resource_root_ + "common/text_processing_jsons/hotwords_cn.json";
    ParseHotwordsCNFile(hotwords_json_path);

    // 这部分比较耗时，600ms+
    PrepareSegPrefixSet();

    auto t1 = clk::now();
    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);

    PLOG(INFO, "Pinyin 初始化成功, timecost: " + std::to_string(d1.count()) + "ms");
}

bool Pinyin::AddCustomPhrases(const std::map<std::string, std::vector<std::string>> &map)
{
    for (auto &m : map)
    {
        phrase_map_[m.first] = m.second;
    }
    PrepareSegPrefixSet();
    return true;
}

bool Pinyin::IsPhrase(const std::string &text) { return phrase_map_.find(text) != phrase_map_.end(); }

bool Pinyin::IsInPrefixSet(const std::string &text) { return prefix_set_.find(text) != prefix_set_.end(); }

phrase_type Pinyin::ObtainPinyinOfPhrase(const std::string &text)
{
    auto it = phrase_map_.find(text);
    return it->second;
}

bool Pinyin::IsSingleHan(std::string &text)
{
    auto parts = SplitUtf8String(text);
    if (parts.size() > 1)
    {
        return false;
    }

    auto unicode_idx = Utf8ToUnicode(parts[0]);
    return pinyin_map_.find(std::to_string(unicode_idx)) != pinyin_map_.end();
}

pinyin_type Pinyin::ObtainPinyinOfSingleHan(const std::string &text)
{
    auto parts = SplitUtf8String(text);
    auto unicode_idx = Utf8ToUnicode(parts[0]);
    auto it = pinyin_map_.find(std::to_string(unicode_idx));
    return it->second;
}

void Pinyin::PrepareSegPrefixSet()
{
    // 遍历phrase_map中的每个键
    for (const auto &pair : phrase_map_)
    {
        const std::string &word_raw = pair.first; // 获取键，即word

        // word_raw 中一个汉字的size为3，不方便处理，因此需要转换为单个字的数组
        auto words = SplitUtf8String(word_raw);

        // 遍历words的每个前缀并添加到prefix_set中
        for (size_t index = 0; index < words.size(); ++index)
        {
            std::string tmp_words = "";
            for (int i = 0; i < index + 1; i++)
            {
                tmp_words += words[i];
            }
            prefix_set_.insert(tmp_words);
        }
    }
}

std::tuple<std::vector<std::string>, std::vector<std::string>> Pinyin::Process(const std::string &text)
{
    // 获取所有拼音
    std::vector<pinyin_type> pinyin_list;
    std::vector<bool> is_pinyin_valid_list;

    auto small_parts = SentenceSplit(text);

    PLOG(PDEBUG, DELIMITER_LINE);
    PLOG(PDEBUG, "拼音输入文本: " + text);
    PLOG(PDEBUG, "拼音分词结果: " + ConcatStrList(small_parts, "|"));

    for (auto &small_part : small_parts)
    {
        PLOG(TRACE, "拼音开始处理|" + small_part + "|");
        bool is_pinyin_valid = false;
        if (IsPhrase(small_part))
        {
            PLOG(TRACE, "是固定词语:" + small_part);
            // 是否为固定词语
            is_pinyin_valid = true;
            auto cur_pinyin_list = ObtainPinyinOfPhrase(small_part);
            PLOG(TRACE, "拼音:" + ConcatStrList(cur_pinyin_list, "|"));
            for (auto &pinyin : cur_pinyin_list)
            {
                pinyin_list.push_back(pinyin);
                is_pinyin_valid_list.push_back(is_pinyin_valid);
            }
        }
        else if (IsSingleHan(small_part))
        {
            // 是否为单个汉字
            PLOG(TRACE, "是单个汉字:" + small_part);
            is_pinyin_valid = true;
            auto cur_pinyin_raw = ObtainPinyinOfSingleHan(small_part);
            PLOG(TRACE, "所有拼音:" + cur_pinyin_raw);
            // 多音字选择第一个发音
            auto cur_pinyin = SplitString(cur_pinyin_raw, ',')[0];
            PLOG(TRACE, "选用拼音:" + cur_pinyin);
            pinyin_list.push_back(cur_pinyin);
            is_pinyin_valid_list.push_back(is_pinyin_valid);
        }
        else
        {
            PLOG(TRACE, "非汉字:" + small_part);
            pinyin_list.push_back(small_part);
            is_pinyin_valid_list.push_back(is_pinyin_valid);
        }
    }

    // 调整输出音节形式
    PLOG(PDEBUG, "音节调整前拼音:" + ConcatStrList(pinyin_list, "|"));
    auto adjusted_pinyin_list = AdjustTones(pinyin_list, is_pinyin_valid_list);
    PLOG(PDEBUG, "音节调整后拼音:" + ConcatStrList(adjusted_pinyin_list, "|"));

    // 获取声母韵母
    std::vector<std::string> initial_list;
    std::vector<std::string> final_list;
    for (int i = 0; i < adjusted_pinyin_list.size(); i++)
    {
        auto pinyin = adjusted_pinyin_list[i];
        bool has_pinyin = is_pinyin_valid_list[i];
        if (has_pinyin)
        {
            auto initial_final = find_initial_final(pinyin);
            auto cur_initial = std::get<0>(initial_final);
            auto cur_final = std::get<1>(initial_final);
            initial_list.push_back(cur_initial);
            final_list.push_back(cur_final);
        }
        else
        {
            initial_list.push_back(pinyin);
            final_list.push_back(pinyin);
        }
    }
    PLOG(PDEBUG, "所有声母:" + ConcatStrList(initial_list, "|"));
    PLOG(PDEBUG, "所有韵母:" + ConcatStrList(final_list, "|"));
    PLOG(PDEBUG, DELIMITER_LINE);

    return std::make_tuple(initial_list, final_list);
}

void Pinyin::ParsePinyinJsonFile(const std::string &pinyin_json_path)
{
    auto json_obj = LoadJson(pinyin_json_path);

    // 将json对象转换为std::map
    pinyin_map_ = json_obj.get<pinyin_map>();
}

void Pinyin::ParsePhraseJsonFile(const std::string &phrase_json_path)
{
    auto json_obj = LoadJson(phrase_json_path);

    // 将json对象转换为std::map
    phrase_map_ = json_obj.get<phrase_map>();
}

void Pinyin::ParseHotwordsCNFile(const std::string &hotwords_cn_json_path)
{
    auto json_obj = LoadJson(hotwords_cn_json_path);

    std::vector<std::string> hot_word_list;
    // 由于hotwords_cn.json文件中还有别的类型的数据，这里只获取我们想要的pinyin字段
    for (auto &item : json_obj)
    {
        std::string cur_word = item["hotword"];
        hot_word_list.push_back(cur_word);
        std::vector<std::string> pinyin = item["pinyin"];
        phrase_map_[cur_word] = pinyin;
    }

    PLOG(PDEBUG, "自定义热词列表:" + ConcatStrList(hot_word_list, "|"));
}

void Pinyin::SavePinyinMapToBin(const std::string &filename, const pinyin_map &pinyin_map)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = pinyin_map.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : pinyin_map)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        size_t value_size = pair.second.size();
        ofs.write(reinterpret_cast<const char *>(&value_size), sizeof(value_size));
        ofs.write(pair.second.c_str(), value_size);
    }

    ofs.close();
}

void Pinyin::SavePhraseMapToBin(const std::string &filename, const phrase_map &phrase_map)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = phrase_map.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : phrase_map)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        const phrase_type &phrases = pair.second;
        size_t phrases_size = phrases.size();
        ofs.write(reinterpret_cast<const char *>(&phrases_size), sizeof(phrases_size));

        for (const auto &phrase : phrases)
        {
            size_t phrase_size = phrase.size();
            ofs.write(reinterpret_cast<const char *>(&phrase_size), sizeof(phrase_size));
            ofs.write(phrase.c_str(), phrase_size);
        }
    }

    ofs.close();
}

void Pinyin::SavePinyinToSymbolMapToBin(const std::string &filename, const pinyin_to_symbol_map &pinyin_to_symbol)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = pinyin_to_symbol.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : pinyin_to_symbol)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        size_t value_size = pair.second.size();
        ofs.write(reinterpret_cast<const char *>(&value_size), sizeof(value_size));
        ofs.write(pair.second.c_str(), value_size);
    }

    ofs.close();
}

void Pinyin::LoadPinyinMapFromBin(const std::string &filename, pinyin_map &pinyin_map)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Unable to open file for reading:" + filename);

    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::string key(key_size, '\0');
        ifs.read(&key[0], key_size);

        size_t value_size;
        ifs.read(reinterpret_cast<char *>(&value_size), sizeof(value_size));
        std::string value(value_size, '\0');
        ifs.read(&value[0], value_size);

        pinyin_map[key] = value;
    }

    ifs.close();
}

void Pinyin::LoadPhraseMapFromBin(const std::string &filename, phrase_map &phrase_map)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Unable to open file for reading");

    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::string key(key_size, '\0');
        ifs.read(&key[0], key_size);

        size_t phrases_size;
        ifs.read(reinterpret_cast<char *>(&phrases_size), sizeof(phrases_size));
        phrase_type phrases(phrases_size);

        for (size_t j = 0; j < phrases_size; ++j)
        {
            size_t phrase_size;
            ifs.read(reinterpret_cast<char *>(&phrase_size), sizeof(phrase_size));
            phrases[j].resize(phrase_size);
            ifs.read(&phrases[j][0], phrase_size);
        }

        phrase_map[key] = phrases;
    }

    ifs.close();
}

void Pinyin::LoadPinyinToSymbolMapFromBin(const std::string &filename, pinyin_to_symbol_map &pinyin_to_symbol)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Unable to open file for reading");

    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::string key(key_size, '\0');
        ifs.read(&key[0], key_size);

        size_t value_size;
        ifs.read(reinterpret_cast<char *>(&value_size), sizeof(value_size));
        std::string value(value_size, '\0');
        ifs.read(&value[0], value_size);

        pinyin_to_symbol[key] = value;
    }

    ifs.close();
}

// 增加音调
std::vector<std::string> Pinyin::AdjustTones(const std::vector<std::string> &pinyin_list,
                                             const std::vector<bool> &is_pinyin_valid_list)
{
    std::vector<std::string> adjusted_pinyin_list;
    for (int i = 0; i < pinyin_list.size(); i++)
    {
        auto pinyin = pinyin_list[i];
        auto is_pinyin_valid = is_pinyin_valid_list[i];

        if (is_pinyin_valid)
        {
            PLOG(TRACE, "转音调前:" + pinyin);
            auto adjusted_pinyin = to_tone3(pinyin);
            PLOG(TRACE, "转三声后:" + adjusted_pinyin);
            if (!ContainsNumber(adjusted_pinyin))
            {
                adjusted_pinyin += "5";
                PLOG(TRACE, "数字优化后:" + adjusted_pinyin);
            }
            adjusted_pinyin_list.push_back(adjusted_pinyin);
        }
        else
        {
            adjusted_pinyin_list.push_back(pinyin);
        }
    }
    return adjusted_pinyin_list;
}

std::vector<std::string> Pinyin::SentenceSplit(const std::string &text)
{
    std::vector<std::string> result;
    std::string matched;
    std::vector<std::string> remain = SplitUtf8String(text);
    while (remain.size() > 0)
    {
        matched.clear();
        bool finish_for_loop = true;
        for (size_t index = 0; index < remain.size(); ++index)
        {
            std::string word = "";
            int word_size = 0;
            for (int i = 0; i < index + 1; i++)
            {
                word += remain[i];
                word_size += 1;
            }
            if (prefix_set_.find(word) != prefix_set_.end())
            {
                // 当前词语在短语或者短语的前缀集合中
                matched = word;
            }
            else
            {
                if (!matched.empty() && IsPhrase(matched))
                {
                    // 前面的字符串是个词语
                    result.push_back(matched);
                    std::vector<std::string> words;
                    for (int i = index; i < remain.size(); i++)
                    {
                        words.push_back(remain[i]);
                    }
                    remain = words;
                    matched = "";
                    finish_for_loop = false;
                    break;
                }
                else
                {
                    // 前面为空或不是真正的词语
                    result.push_back(remain[0]);
                    std::vector<std::string> words;
                    for (int i = index + 2 - word_size; i < remain.size(); i++)
                    {
                        words.push_back(remain[i]);
                    }
                    remain = words;
                    finish_for_loop = false;
                    matched = "";
                    break;
                }
            }
        }

        // 整个文本就是一个词语，或者不包含任何词语
        if (finish_for_loop)
        {
            auto remain_str = ConcatStrList(remain);
            if (!IsPhrase(remain_str))
            {
                for (auto &x : remain)
                {
                    result.push_back(x);
                }
            }
            else
            {
                result.push_back(remain_str);
            }
            break;
        }
    }
    return result;
}
