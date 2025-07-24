#include "chinese_g2p.hpp"

extern std::unordered_map<std::string, std::string> rep_map = {
    {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"}, {"\n", "."}, {"·", ","}, {"、", ","}, {"...", "…"}, {"$", "."}, {"“", "'"}, {"”", "'"}, {"‘", "'"}, {"’", "'"}, {"（", "'"}, {"）", "'"}, {"(", "'"}, {")", "'"}, {"《", "'"}, {"》", "'"}, {"【", "'"}, {"】", "'"}, {"[", "'"}, {"]", "'"}, {"—", "-"}, {"～", "-"}, {"~", "-"}, {"「", "\""}, {"」", "\""}};

std::vector<std::string> punctuation = {"!", "?", ",", ".", "\"", "-", "…", "【", "】", "", ":", "「", "」", "'", "“"};

ChineseG2P::ChineseG2P(const std::string &local_resource_root) : pinyin_(local_resource_root), word_spliter_(WordSpliter::GetInstance(local_resource_root)), tone_adjuster_(local_resource_root)
{
    auto t0 = clk::now();
    PLOG(INFO, "ChineseG2P 开始初始化...");
    resource_root_ = local_resource_root;
    an2cn_ = An2Cn();

    //    pinyin_.AddCustomPhrases(custom_pharse);

    auto pinyin_to_symbol_bin_path = resource_root_ + "common/text_processing_jsons/pinyin_to_symbol_map.bin";
    LoadBinToMap(pinyin_to_symbol_bin_path, pinyin_to_symbol_map_);

    auto hotwords_bin_path = resource_root_ + "common/text_processing_jsons/hotwords_cn.bin";
    LoadHotwordsFromBin(hotwords_bin_path);

    // 创建映射
    int id = 0;
    for (auto &symbol : symbols)
    {
        _symbol_to_id[symbol] = id++;
    }

    auto t1 = clk::now();

    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
    PLOG(INFO, "ChineseG2P 初始化成功, timecost: " + std::to_string(d1.count()) + "ms");
}

void ChineseG2P::ParseHotwordsCNFile(const std::string &hotwords_cn_json_path)
{
    auto json_obj = LoadJson(hotwords_cn_json_path);

    std::vector<std::string> hot_word_list;
    // 由于hotwords_cn.json文件中还有别的类型的数据，这里只获取我们想要的pinyin字段
    for (auto &item : json_obj)
    {
        std::string cur_word = item["hotword"];
        hot_word_list.push_back(cur_word);
        std::vector<std::string> initials = SplitString(item["initials"], ',');
        std::vector<std::string> finals = SplitString(item["finals"], ',');
        custom_phone_[cur_word] = std::make_tuple(initials, finals);
    }
}

void ChineseG2P::SaveHotwordsToBin(const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Unable to open file for writing");
    }

    // 写入热词数量
    size_t size = custom_phone_.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // 写入每个热词及其拼音信息
    for (const auto &pair : custom_phone_)
    {
        const std::string &hotword = pair.first;
        const auto &phonemes = pair.second;
        const auto &initials = std::get<0>(phonemes);
        const auto &finals = std::get<1>(phonemes);

        // 写入热词长度和内容
        size_t hotword_size = hotword.size();
        ofs.write(reinterpret_cast<const char *>(&hotword_size), sizeof(hotword_size));
        ofs.write(hotword.c_str(), hotword_size);

        // 写入声母
        size_t initials_size = initials.size();
        ofs.write(reinterpret_cast<const char *>(&initials_size), sizeof(initials_size));
        for (const auto &initial : initials)
        {
            size_t initial_size = initial.size();
            ofs.write(reinterpret_cast<const char *>(&initial_size), sizeof(initial_size));
            ofs.write(initial.c_str(), initial_size);
        }

        // 写入韵母
        size_t finals_size = finals.size();
        ofs.write(reinterpret_cast<const char *>(&finals_size), sizeof(finals_size));
        for (const auto &final : finals)
        {
            size_t final_size = final.size();
            ofs.write(reinterpret_cast<const char *>(&final_size), sizeof(final_size));
            ofs.write(final.c_str(), final_size);
        }
    }

    ofs.close();
}

void ChineseG2P::LoadHotwordsFromBin(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Unable to open file for reading");
    }

    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size)); // 读取热词数量

    for (size_t i = 0; i < size; ++i)
    {
        // 读取热词
        size_t hotword_size;
        ifs.read(reinterpret_cast<char *>(&hotword_size), sizeof(hotword_size));
        std::string hotword(hotword_size, '\0');
        ifs.read(&hotword[0], hotword_size);

        // 读取声母
        size_t initials_size;
        ifs.read(reinterpret_cast<char *>(&initials_size), sizeof(initials_size));
        std::vector<std::string> initials(initials_size);
        for (size_t j = 0; j < initials_size; ++j)
        {
            size_t initial_size;
            ifs.read(reinterpret_cast<char *>(&initial_size), sizeof(initial_size));
            initials[j].resize(initial_size);
            ifs.read(&initials[j][0], initial_size);
        }

        // 读取韵母
        size_t finals_size;
        ifs.read(reinterpret_cast<char *>(&finals_size), sizeof(finals_size));
        std::vector<std::string> finals(finals_size);
        for (size_t j = 0; j < finals_size; ++j)
        {
            size_t final_size;
            ifs.read(reinterpret_cast<char *>(&final_size), sizeof(final_size));
            finals[j].resize(final_size);
            ifs.read(&finals[j][0], final_size);
        }

        custom_phone_[hotword] = std::make_tuple(initials, finals);
    }

    ifs.close();
}

void ChineseG2P::ParsePinyToSymbolJsonFile(const std::string &pinyin_json_path)
{
    auto json_obj = LoadJson(pinyin_json_path);

    // 将json对象转换为std::map
    pinyin_to_symbol_map_ = json_obj.get<pinyin_to_symbol_map>();
    SaveJsonToBin("pinyin_to_symbol_map.bin", pinyin_to_symbol_map_);
}

g2p_data ChineseG2P::ConvertInitialFinalToPhoneTone(
    const std::vector<std::string> &initials, const std::vector<std::string> &finals)
{
    std::vector<std::string> phones_list;
    std::vector<int> tones_list;
    std::vector<int> word2ph;

    for (size_t i = 0; i < initials.size(); ++i)
    {
        assert(finals.size() > i); // Ensure there are enough finals
        const std::string &c = initials[i];
        const std::string &v = finals[i];
        std::string raw_pinyin = c + v;

        // NOTE: post process for pypinyin outputs
        // we discriminate i, ii and iii
        if (c == v)
        {
            phones_list.push_back(std::string(1, c[0]));
            tones_list.push_back(0);
            word2ph.push_back(1);
        }
        else
        {
            std::string v_without_tone = v.substr(0, v.size() - 1);
            char tone = v.back();
            assert(std::string("12345").find(tone) != std::string::npos);
            std::string pinyin = c + v_without_tone;

            if (!c.empty())
            {
                // 多音节
                std::unordered_map<std::string, std::string> v_rep_map = {{"uei", "ui"}, {"iou", "iu"}, {"uen", "un"}};
                auto it = v_rep_map.find(v_without_tone);
                if (it != v_rep_map.end())
                {
                    pinyin = c + it->second;
                }
            }
            else
            {
                // 单音节
                std::unordered_map<std::string, std::string> pinyin_rep_map = {
                    {"ing", "ying"}, {"i", "yi"}, {"in", "yin"}, {"u", "wu"}};
                auto it = pinyin_rep_map.find(pinyin);
                if (it != pinyin_rep_map.end())
                {
                    pinyin = it->second;
                }
                else
                {
                    std::unordered_map<char, std::string> single_rep_map = {
                        {'v', "yu"}, {'e', "e"}, {'i', "y"}, {'u', "w"}};
                    auto it2 = single_rep_map.find(pinyin[0]);
                    if (it2 != single_rep_map.end())
                    {
                        pinyin = it2->second + pinyin.substr(1);
                    }
                }
            }

            std::vector<std::string> phone = SplitString(pinyin_to_symbol_map_[pinyin], ' ');
            word2ph.push_back(phone.size());

            for (const auto &ph : phone)
            {
                phones_list.push_back(ph);
            }
            tones_list.insert(tones_list.end(), phone.size(), tone - '0');
        }
    }
    return std::make_tuple(phones_list, tones_list, word2ph);
}

std::string ChineseG2P::ReplacePunctuations(const std::string &text)
{
    std::string modified_text = text;

    // 将英文的"..."替换为"…"
    StrReplaceAll(modified_text, "...", "…");

    //  // 替换先定义的字符映射
    for (const auto &pair : rep_map)
    {
        auto word_splits = SplitUtf8String(modified_text);
        if (HanListContains(word_splits, pair.first))
        {
            StrReplaceAll(modified_text, pair.first, pair.second);
        }
    }

    return modified_text;
}

std::string ChineseG2P::TextAn2cn(std::string &text)
{
    // 使用正则表达式查找匹配的数字
    std::regex pattern(R"(\d+(?:\.?\d+)?)");
    std::sregex_iterator iter(text.begin(), text.end(), pattern);
    std::sregex_iterator end;

    std::vector<std::string> numbers;

    // 将找到的数字存储到 vector 中
    while (iter != end)
    {
        numbers.push_back(iter->str());
        ++iter;
    }

    // 替换每个数字
    for (const std::string &number : numbers)
    {
        text.replace(text.find(number), number.length(), an2cn_.Process(number));
    }
    return text;
}

// Function to normalize text by converting numbers and replacing punctuation.
std::string ChineseG2P::TextNormalize(const std::string &text)
{
    std::string normalized_text = text;
    PLOG(PDEBUG, "g2p text_norm 输入: " + normalized_text);

    // 进行阿拉伯数字到汉字数字的转换
    normalized_text = TextAn2cn(normalized_text);
    PLOG(PDEBUG, "g2p text_norm an2cn后: " + normalized_text);

    // 替换中文标点符号为英文标点符号
    normalized_text = ReplacePunctuations(normalized_text);
    PLOG(PDEBUG, "g2p text_norm 替换标点后: " + normalized_text);
    return normalized_text;
}

g2p_data ChineseG2P::G2P(std::string &text)
{
    PLOG(PDEBUG, "g2p 输入句子: " + text);

    std::vector<std::string> phones;
    std::vector<int> tones;
    std::vector<int> word2ph;

    // text是从原始句子中切出来的中文部分，可能包含标点符号，如 '你好啊.你是,我的男朋友?'
    // sentences是根据标点符号切分的小的分句，标点符号跟随前面的文本，如 ['你好啊.', '你是,', '我的男朋友?']
    auto segments = SplitChineseSentenceToParts(text);
    PLOG(PDEBUG, "g2p 标点分句: " + ConcatStrList(segments, "|"));

    std::tie(phones, tones, word2ph) = G2PInternal(segments);

    // Prepend and append special tokens to the lists
    phones.insert(phones.begin(), "_");
    phones.push_back("_");
    tones.insert(tones.begin(), 0);
    tones.push_back(0);
    word2ph.insert(word2ph.begin(), 1);
    word2ph.push_back(1);

    return std::make_tuple(phones, tones, word2ph);
}

g2p_data ChineseG2P::G2PInternal(const std::vector<std::string> &segments)
{
    std::vector<std::string> all_initials;
    std::vector<std::string> all_finals;

    std::vector<std::string> phones;
    std::vector<int> tones;
    std::vector<int> word2ph;

    // 对每个标点符号分割的小分句进行分词，并返回对应的词性，如 ['你好啊'] -> [[(‘你好', 'l‘)， ('啊', 'zg')]]
    for (auto &seg : segments)
    {
        PLOG(TRACE, DELIMITER_LINE);
        PLOG(TRACE, "g2p processing seg " + seg);
        // 保存当前分句的声母和韵母
        std::vector<std::string> initials;
        std::vector<std::string> finals;

        // 返回值类型: vector<WordPosPair>
        auto seg_cut_list = word_spliter_.Process(seg);
        PLOG(TRACE, "g2p 分词结果: " + ConcatList(seg_cut_list, "|"));

        // 计算拼音
        for (auto &[word, pos] : seg_cut_list)
        {
            PLOG(TRACE, "g2p 开始处理词语: " + word + "/" + pos);
            // eng表示英文字母和数字
            if (pos == "eng")
            {
                PLOG(TRACE, "g2p 是英文和数字: " + word);
                continue;
            }
            // 如果词语在自定义词表里面，采用自定义的发音
            if (custom_phone_.find(word) != custom_phone_.end())
            {
                PLOG(TRACE, "g2p 是自定义词语: " + word);
                auto [sub_initials, sub_finals] = custom_phone_[word];
                initials.insert(initials.end(), sub_initials.begin(), sub_initials.end());
                finals.insert(finals.end(), sub_finals.begin(), sub_finals.end());
            }
            else
            {
                PLOG(TRACE, "g2p 提取拼音: " + word);
                auto [sub_initials, sub_finals] = pinyin_.Process(word);
                PLOG(TRACE, "g2p 原始声母: " + ConcatStrList(sub_initials, "|"));
                PLOG(TRACE, "g2p 原始韵母: " + ConcatStrList(sub_finals), "|");

                sub_finals = tone_adjuster_.Process(word, pos, sub_finals);
                PLOG(TRACE, "g2p 修改声调后韵母: " + ConcatStrList(sub_finals));

                initials.insert(initials.end(), sub_initials.begin(), sub_initials.end());
                finals.insert(finals.end(), sub_finals.begin(), sub_finals.end());
            }
        }

        PLOG(TRACE, "g2p 该seg全部声母: " + ConcatStrList(initials, "|"));
        PLOG(TRACE, "g2p 该seg全部韵母: " + ConcatStrList(finals, "|"));
        all_initials.insert(all_initials.end(), initials.begin(), initials.end());
        all_finals.insert(all_finals.end(), finals.begin(), finals.end());

        // 将拼音转换为音素和音调
        auto [cur_phones, cur_tones, cur_word2ph] = ConvertInitialFinalToPhoneTone(initials, finals);

        phones.insert(phones.end(), cur_phones.begin(), cur_phones.end());
        tones.insert(tones.end(), cur_tones.begin(), cur_tones.end());
        word2ph.insert(word2ph.end(), cur_word2ph.begin(), cur_word2ph.end());
        PLOG(TRACE, DELIMITER_LINE);
    }

    PLOG(PDEBUG, "g2p 整句内容: " + ConcatStrList(segments, "|"));
    PLOG(PDEBUG, "g2p 整句声母: " + ConcatStrList(all_initials, "|"));
    PLOG(PDEBUG, "g2p 整句韵母: " + ConcatStrList(all_finals, "|"));
    PLOG(PDEBUG, "g2p 整句phones: " + ConcatStrList(phones, "|"));
    PLOG(PDEBUG, "g2p 整句tones: " + ConcatIntList(tones, "|"));
    PLOG(PDEBUG, "g2p 整句word2ph: " + ConcatIntList(word2ph, "|"));

    return make_tuple(phones, tones, word2ph);
}

phone_data ChineseG2P::CleanedTextToSequence(const g2p_data &g2p_data_, const std::string &language)
{
    auto phones = std::get<0>(g2p_data_);
    auto tones = std::get<1>(g2p_data_);
    auto word2ph = std::get<2>(g2p_data_);

    std::vector<int> int_phones;
    for (auto &symbol : phones)
    {
        // 假设 symbol 在 _symbol_to_id 中一定能找到。
        int_phones.push_back(_symbol_to_id[symbol]);
    }

    int tone_start = language_tone_start_map[language];
    std::vector<int> adjusted_tones;
    for (int tone : tones)
    {
        adjusted_tones.push_back(tone + tone_start);
    }

    std::vector<int> lang_ids;
    for (int i = 0; i < int_phones.size(); i++)
    {
        lang_ids.push_back(language_id_map[language]);
    }

    // 增加空白
    int_phones = Intersperse<int>(int_phones, 0);
    adjusted_tones = Intersperse<int>(adjusted_tones, 0);
    lang_ids = Intersperse<int>(lang_ids, 0);

    // 下面的循环操作将每个元素值翻倍，然后将第一个元素增加1
    for (size_t i = 0; i < word2ph.size(); ++i)
    {
        word2ph[i] = word2ph[i] * 2;
    }
    word2ph[0] += 1;

    PLOG(PDEBUG, "g2p 调整后phones(size=" + std::to_string(int_phones.size()) + "):" + ConcatIntList(int_phones, "|"));
    PLOG(PDEBUG, "g2p 调整后tones(size=" + std::to_string(adjusted_tones.size()) + "):" + ConcatIntList(adjusted_tones, "|"));
    PLOG(PDEBUG, "g2p 调整后word2ph(size=" + std::to_string(word2ph.size()) + "):" + ConcatIntList(word2ph, "|"));
    PLOG(PDEBUG, "g2p lang_ids(size=" + std::to_string(lang_ids.size()) + "):" + ConcatIntList(lang_ids, "|"));

    return std::make_tuple(int_phones, adjusted_tones, lang_ids, word2ph);
}

std::tuple<std::string, phone_data> ChineseG2P::Process(SentLangPair &sent_lang)
{
    PLOG(PDEBUG, "g2p 输入文本: " + sent_lang.sent);
    auto norm_text = TextNormalize(sent_lang.sent);
    PLOG(PDEBUG, "g2p norm后文本: " + norm_text);

    auto g2p_res = G2P(norm_text);
    auto res = CleanedTextToSequence(g2p_res, sent_lang.lang);

    return std::make_tuple(norm_text, res);
}
