#include "word_spliter.hpp"
// 辅助函数：将多字节 UTF-8 字符串转换为字符向量
std::vector<std::string> split_utf8(const std::string &str)
{
    std::vector<std::string> result;
    size_t i = 0;
    while (i < str.size())
    {
        size_t length = 1;
        unsigned char c = str[i];
        if ((c & 0x80) == 0)
        { // 1-byte UTF-8
            length = 1;
        }
        else if ((c & 0xE0) == 0xC0)
        { // 2-byte UTF-8
            length = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        { // 3-byte UTF-8
            length = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        { // 4-byte UTF-8
            length = 4;
        }
        result.push_back(str.substr(i, length));
        i += length;
    }
    return result;
}

std::unique_ptr<WordSpliter> WordSpliter::instance = nullptr;
std::mutex WordSpliter::mtx;

WordSpliter &WordSpliter::GetInstance(const std::string &local_resource_root)
{
    static WordSpliter inst(local_resource_root);
    return inst;
}

WordSpliter::WordSpliter(const std::string &local_resource_root)
{
    auto t0 = clk::now();
    PLOG(INFO, "WordSpliter 开始初始化...");
    resource_root_ = local_resource_root;

    LoadAllFromBin();

    auto hotwords_json_path = resource_root_ + "common/text_processing_jsons/hotwords_cn.json";
    ParseHotwordsCNFile(hotwords_json_path);

    auto t1 = clk::now();
    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);

    PLOG(INFO, "WordSpliter 初始化完成, timecost: " + std::to_string(d1.count()) + "ms");
};

void WordSpliter::ParseWordFreq(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    word_freq_ = json_obj.get<word_freq_map>();
    total = 60101967;
}

void WordSpliter::ParseWordTag(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    word_tag_ = json_obj.get<word_tag_map>();
}

void WordSpliter::ParseCharState(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    char_state_ = json_obj.get<char_state_map>();
}

void WordSpliter::ParseProbEmit(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    prob_emit_ = json_obj.get<prob_emit_map>();
}

void WordSpliter::ParseProbStart(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    prob_start_ = json_obj.get<prob_start_map>();
}

void WordSpliter::ParseProbTrans(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);
    prob_trans_ = json_obj.get<prob_trans_map>();
}

void WordSpliter::ParseHotwordsCNFile(const std::string &hotwords_cn_json_path)
{
    auto json_obj = LoadJson(hotwords_cn_json_path);

    std::vector<std::string> hot_word_list;
    // 由于hotwords_cn.json文件中还有别的类型的数据，这里只获取我们想要的hotword, word_freq, word_type字段
    for (auto &item : json_obj)
    {
        std::string cur_word = item["hotword"];
        int word_freq = item["word_freq"];
        std::string word_type = item["word_type"];

        int actual_word_freq = SuggestFreq(cur_word);
        actual_word_freq = std::max(word_freq, actual_word_freq);

        word_freq_[cur_word] = actual_word_freq;
        total += actual_word_freq;

        auto segs = SplitUtf8String(cur_word);
        std::string forward_parts = "";
        for (int i = 0; i < segs.size(); i++)
        {
            forward_parts += segs[i];
            if (word_freq_.find(forward_parts) == word_freq_.end())
            {
                word_freq_[forward_parts] = 0;
            }
        }
    }
}

void WordSpliter::SaveWordFreqToBin(const std::string &filename, const word_freq_map &word_freq)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = word_freq.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : word_freq)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);
        ofs.write(reinterpret_cast<const char *>(&pair.second), sizeof(pair.second));
    }

    ofs.close();
}

void WordSpliter::SaveWordTagToBin(const std::string &filename, const word_tag_map &word_tag)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = word_tag.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : word_tag)
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

void WordSpliter::SaveCharStateToBin(const std::string &filename, const char_state_map &char_state)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = char_state.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : char_state)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        const auto &vector = pair.second;
        size_t vector_size = vector.size();
        ofs.write(reinterpret_cast<const char *>(&vector_size), sizeof(vector_size));

        for (const auto &item : vector)
        {
            size_t item_size = item.size();
            ofs.write(reinterpret_cast<const char *>(&item_size), sizeof(item_size));
            ofs.write(item.c_str(), item_size);
        }
    }

    ofs.close();
}

void WordSpliter::SaveProbEmitToBin(const std::string &filename, const prob_emit_map &prob_emit)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = prob_emit.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : prob_emit)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        const auto &inner_map = pair.second;
        size_t inner_size = inner_map.size();
        ofs.write(reinterpret_cast<const char *>(&inner_size), sizeof(inner_size));

        for (const auto &inner_pair : inner_map)
        {
            size_t inner_key_size = inner_pair.first.size();
            ofs.write(reinterpret_cast<const char *>(&inner_key_size), sizeof(inner_key_size));
            ofs.write(inner_pair.first.c_str(), inner_key_size);

            double value = inner_pair.second;
            ofs.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
    }

    ofs.close();
}

void WordSpliter::SaveProbStartToBin(const std::string &filename, const prob_start_map &prob_start)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = prob_start.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : prob_start)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        double value = pair.second;
        ofs.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    ofs.close();
}

void WordSpliter::SaveProbTransToBin(const std::string &filename, const prob_trans_map &prob_trans)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    size_t size = prob_trans.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : prob_trans)
    {
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        const auto &inner_map = pair.second;
        size_t inner_size = inner_map.size();
        ofs.write(reinterpret_cast<const char *>(&inner_size), sizeof(inner_size));

        for (const auto &inner_pair : inner_map)
        {
            size_t inner_key_size = inner_pair.first.size();
            ofs.write(reinterpret_cast<const char *>(&inner_key_size), sizeof(inner_key_size));
            ofs.write(inner_pair.first.c_str(), inner_key_size);

            double value = inner_pair.second;
            ofs.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
    }

    ofs.close();
}

void WordSpliter::LoadWordFreqFromBin(const std::string &filename, word_freq_map &word_freq)
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

        int value;
        ifs.read(reinterpret_cast<char *>(&value), sizeof(value));

        word_freq[key] = value;
    }

    ifs.close();
}

void WordSpliter::LoadWordTagFromBin(const std::string &filename, word_tag_map &word_tag)
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

        word_tag[key] = value;
    }

    ifs.close();
}

void WordSpliter::LoadCharStateFromBin(const std::string &filename, char_state_map &char_state)
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

        size_t vector_size;
        ifs.read(reinterpret_cast<char *>(&vector_size), sizeof(vector_size));
        std::vector<std::string> vector(vector_size);

        for (size_t j = 0; j < vector_size; ++j)
        {
            size_t item_size;
            ifs.read(reinterpret_cast<char *>(&item_size), sizeof(item_size));
            vector[j].resize(item_size);
            ifs.read(&vector[j][0], item_size);
        }

        char_state[key] = vector;
    }

    ifs.close();
}

void WordSpliter::LoadProbEmitFromBin(const std::string &filename, prob_emit_map &prob_emit)
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

        size_t inner_size;
        ifs.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));

        std::unordered_map<std::string, double> inner_map;
        for (size_t j = 0; j < inner_size; ++j)
        {
            size_t inner_key_size;
            ifs.read(reinterpret_cast<char *>(&inner_key_size), sizeof(inner_key_size));
            std::string inner_key(inner_key_size, '\0');
            ifs.read(&inner_key[0], inner_key_size);

            double value;
            ifs.read(reinterpret_cast<char *>(&value), sizeof(value));

            inner_map[inner_key] = value;
        }

        prob_emit[key] = inner_map;
    }

    ifs.close();
}

void WordSpliter::LoadProbStartFromBin(const std::string &filename, prob_start_map &prob_start)
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

        double value;
        ifs.read(reinterpret_cast<char *>(&value), sizeof(value));

        prob_start[key] = value;
    }

    ifs.close();
}

void WordSpliter::LoadProbTransFromBin(const std::string &filename, prob_trans_map &prob_trans)
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

        size_t inner_size;
        ifs.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));

        std::unordered_map<std::string, double> inner_map;
        for (size_t j = 0; j < inner_size; ++j)
        {
            size_t inner_key_size;
            ifs.read(reinterpret_cast<char *>(&inner_key_size), sizeof(inner_key_size));
            std::string inner_key(inner_key_size, '\0');
            ifs.read(&inner_key[0], inner_key_size);

            double value;
            ifs.read(reinterpret_cast<char *>(&value), sizeof(value));

            inner_map[inner_key] = value;
        }

        prob_trans[key] = inner_map;
    }

    ifs.close();
}

// 保存所有到二进制文件
void WordSpliter::SaveAllToBin()
{
    auto word_freq_bin_path = resource_root_ + "common/text_processing_jsons/word_freq.bin";
    auto word_tag_bin_path = resource_root_ + "common/text_processing_jsons/word_tag.bin";
    auto char_state_bin_path = resource_root_ + "common/text_processing_jsons/char_state.bin";
    auto prob_emit_bin_path = resource_root_ + "common/text_processing_jsons/prob_emit.bin";
    auto prob_start_bin_path = resource_root_ + "common/text_processing_jsons/prob_start.bin";
    auto prob_trans_bin_path = resource_root_ + "common/text_processing_jsons/prob_trans.bin";

    SaveWordFreqToBin(word_freq_bin_path, word_freq_);
    SaveWordTagToBin(word_tag_bin_path, word_tag_);
    SaveCharStateToBin(char_state_bin_path, char_state_);
    SaveProbEmitToBin(prob_emit_bin_path, prob_emit_);
    SaveProbStartToBin(prob_start_bin_path, prob_start_);
    SaveProbTransToBin(prob_trans_bin_path, prob_trans_);
}

// 从二进制文件加载所有数据
void WordSpliter::LoadAllFromBin()
{
    auto word_freq_bin_path = resource_root_ + "common/text_processing_jsons/word_freq.bin";
    auto word_tag_bin_path = resource_root_ + "common/text_processing_jsons/word_tag.bin";
    auto char_state_bin_path = resource_root_ + "common/text_processing_jsons/char_state.bin";
    auto prob_emit_bin_path = resource_root_ + "common/text_processing_jsons/prob_emit.bin";
    auto prob_start_bin_path = resource_root_ + "common/text_processing_jsons/prob_start.bin";
    auto prob_trans_bin_path = resource_root_ + "common/text_processing_jsons/prob_trans.bin";

    LoadWordFreqFromBin(word_freq_bin_path, word_freq_);
    LoadWordTagFromBin(word_tag_bin_path, word_tag_);
    LoadCharStateFromBin(char_state_bin_path, char_state_);
    LoadProbEmitFromBin(prob_emit_bin_path, prob_emit_);
    LoadProbStartFromBin(prob_start_bin_path, prob_start_);
    LoadProbTransFromBin(prob_trans_bin_path, prob_trans_);
}

int WordSpliter::SuggestFreq(const std::string &segment)
{
    int word_freq = 1.0;
    if (word_freq_.find(segment) != word_freq_.end())
    {
        word_freq = word_freq_[segment];
    }

    auto segment_utf8 = SplitUtf8String(segment);
    auto segs = CutDAGNoHMM(segment_utf8, segment);
    float ftotal = float(total);
    int freq = 1;
    for (const auto &seg : segs)
    {

        float freq_ratio = 1.0;
        if (word_freq_.find(seg) != word_freq_.end())
        {
            freq_ratio = word_freq_[seg];
        }
        freq *= freq_ratio / ftotal;

        freq = std::max(int(freq * total) + 1, word_freq);
    }
    return freq;
}

std::pair<double, std::vector<std::string>> WordSpliter::Viterbi(const std::vector<std::string> &obs)
{
    std::vector<std::unordered_map<std::string, double>> V(1);
    std::vector<std::unordered_map<std::string, std::string>> mem_path(1);

    std::vector<std::string> all_states;

    // 默认是包含256个状态的states
    for (const auto &kv : prob_trans_)
    {
        all_states.push_back(kv.first);
    }

    // 如果当前的汉字不再char_state_ dict的key中，则采用默认的256个状态的state
    std::vector<std::string> cur_states = all_states;
    if (char_state_.find(obs[0]) != char_state_.end())
    {
        cur_states = char_state_[obs[0]];
    }

    auto get_with_default = [](const auto &map, const std::string &key, double def_val)
    {
        auto it = map.find(key);
        return it == map.end() ? def_val : it->second;
    };

    for (const auto &y : cur_states)
    { // 初始化
        V[0][y] = get_with_default(prob_start_, y, MIN_FLOAT) + get_with_default(prob_emit_.at(y), obs[0], MIN_FLOAT);
        mem_path[0][y] = "";
    }

    for (size_t t = 1; t < obs.size(); ++t)
    {
        V.emplace_back();
        mem_path.emplace_back();

        std::vector<std::string> prev_states;
        for (const auto &kv : mem_path[t - 1])
        {
            if (!prob_trans_.at(kv.first).empty())
            {
                prev_states.push_back(kv.first);
            }
        }

        std::unordered_set<std::string> obs_states;
        for (const auto &x : prev_states)
        {
            for (const auto &y : prob_trans_.at(x))
            {
                obs_states.insert(y.first);
            }
        }

        if (obs_states.empty())
        {
            obs_states.insert(all_states.begin(), all_states.end());
        }

        for (const auto &y : obs_states)
        {
            double prob = MIN_INF;
            std::string state;
            for (const auto &y0 : prev_states)
            {
                double new_prob = V[t - 1][y0] + get_with_default(prob_trans_.at(y0), y, MIN_INF) +
                                  get_with_default(prob_emit_.at(y), obs[t], MIN_FLOAT);
                if (new_prob > prob)
                {
                    prob = new_prob;
                    state = y0;
                }
            }
            V[t][y] = prob;
            mem_path[t][y] = state;
        }
    }

    double prob = MIN_INF;
    std::string state = "x";
    for (const auto &kv : mem_path.back())
    {
        if (V.back().at(kv.first) > prob)
        {
            prob = V.back().at(kv.first);
            state = kv.first;
        }
    }

    std::vector<std::string> route(obs.size());
    for (int i = obs.size() - 1; i >= 0; --i)
    {
        route[i] = state;
        state = mem_path[i][state];
        if (state == "")
        {
            state = "x";
        }
    }

    return {prob, route};
}

std::vector<WordPosPair> WordSpliter::Cut(const std::string &sentence)
{
    std::vector<std::string> ss = SplitUtf8String(sentence);
    PLOG(TRACE, "wordspliter cut input:" + sentence);

    auto result = Viterbi(ss);
    PLOG(TRACE, "wordspliter cut viterbi:" + std::to_string(result.first) + "," + ConcatStrList(result.second, "|"));

    double prob = result.first;
    std::vector<std::string> pos_list = result.second;

    std::vector<WordPosPair> pairs;
    size_t begin = 0;
    size_t nexti = 0;

    for (size_t i = 0; i < ss.size(); ++i)
    {
        std::string pos = pos_list[i].substr(0, 1); // Assuming the position is the first character of the string.

        if (pos == "B")
        {
            begin = i;
        }
        else if (pos == "E")
        {
            std::string cur_word = "";
            for (int idx = begin; idx < i + 1; idx++)
            {
                cur_word += ss[idx];
            }
            pairs.push_back(WordPosPair(cur_word, pos_list[i].substr(3)));
            nexti = i + 1;
        }
        else if (pos == "S")
        {
            pairs.push_back(WordPosPair(ss[i], pos_list[i].substr(3)));
            nexti = i + 1;
        }

        // 不处理"中间(M)"或"结尾(E)"标记，因为它们已经在"B"或"S"中处理了
    }

    // 处理句子中最后的未分类部分
    if (nexti < ss.size())
    {
        std::string left_str = "";
        for (int i = nexti; i < ss.size(); i++)
        {
            left_str += ss[i];
        }
        pairs.push_back(WordPosPair(left_str, pos_list[nexti].substr(1)));
    }
    return pairs;
}

std::vector<WordPosPair> WordSpliter::CutDetail(const std::string &sentence)
{
    std::regex re_han_detail("([一-鿕]+)");
    std::regex re_skip_detail("([\\.0-9]+|[a-zA-Z0-9]+)");
    std::regex re_num("([\\.0-9]+)");
    std::regex re_eng("([a-zA-Z0-9]+)");

    // std::vector<Pair> words = Cut(sentence);
    std::vector<WordPosPair> result;
    std::sregex_token_iterator end;

    auto blocks = SplitChineseSentenceToParts(sentence);
    PLOG(TRACE, "wordspliter cutdetail blocks:" + ConcatStrList(blocks));

    for (auto &blk : blocks)
    {
        if (std::regex_match(blk, re_han_detail))
        {
            std::vector<WordPosPair> words = Cut(blk);
            result.insert(result.end(), words.begin(), words.end());
        }
        else
        {
            std::sregex_token_iterator tmp_iter(blk.begin(), blk.end(), re_skip_detail, -1);
            for (; tmp_iter != end; ++tmp_iter)
            {
                std::string x = *tmp_iter;
                if (!x.empty())
                {
                    if (std::regex_match(x, re_num))
                    {
                        result.push_back(WordPosPair(x, "m"));
                    }
                    else if (std::regex_match(x, re_eng))
                    {
                        result.push_back(WordPosPair(x, "eng"));
                    }
                    else
                    {
                        result.push_back(WordPosPair(x, "x"));
                    }
                }
            }
        }
    }

    return result;
}

// 计算当前句子对应的汉字的DAG
std::unordered_map<int, std::vector<int>> WordSpliter::GetDAG(const std::vector<std::string> &sentence)
{
    std::unordered_map<int, std::vector<int>> DAG;

    int N = sentence.size();
    for (int k = 0; k < N; ++k)
    {
        std::vector<int> tmplist;
        int i = k;
        std::string frag = HanSubstr(sentence, k, 1);
        bool has_find = word_freq_.find(frag) != word_freq_.end();
        while (i < N && word_freq_.find(frag) != word_freq_.end())
        {
            if (word_freq_[frag] > 0)
            {
                tmplist.push_back(i);
            }
            ++i;
            frag = HanSubstr(sentence, k, i - k + 1);
        }
        if (tmplist.empty())
        {
            tmplist.push_back(k);
        }
        DAG[k] = tmplist;
    }
    return DAG;
}

void WordSpliter::calc(const std::vector<std::string> &sentence,
                       const std::unordered_map<int, std::vector<int>> &DAG,
                       std::unordered_map<int, std::pair<double, int>> &route)
{
    int N = sentence.size();
    route[N] = std::make_pair(0, 0);
    double logtotal = std::log(total);
    for (int idx = N - 1; idx >= 0; --idx)
    {
        std::pair<double, int> max_pair = std::make_pair(std::numeric_limits<double>::lowest(), idx);
        for (int x : DAG.at(idx))
        {
            std::string word = HanSubstr(sentence, idx, x - idx + 1);
            double freq = std::log(word_freq_.find(word) != word_freq_.end() ? word_freq_[word] : 1);
            double prob = freq - logtotal + route[x + 1].first;
            if (prob > max_pair.first)
            {
                max_pair = std::make_pair(prob, x);
            }
        }
        route[idx] = max_pair;
    }
}

std::vector<std::string> WordSpliter::CutDAGNoHMM(const std::vector<std::string> &sentence, const std::string &raw_s)
{
    std::vector<std::string> result;
    auto DAG = GetDAG(sentence);
    for (auto &d : DAG)
    {
        auto f = d.first;
        auto s = d.second;
    }

    std::unordered_map<int, std::pair<double, int>> route;
    calc(sentence, DAG, route);
    std::vector<std::string> chars = split_utf8(raw_s);
    size_t N = chars.size();

    int x = 0;
    std::string buf;

    while (x < N)
    {
        PLOG(TRACE, "==========step=======================" + std::to_string(x));
        int y = route[x].second + 1;
        std::string l_word;
        for (size_t i = x; i < y; ++i)
        {
            l_word += chars[i];
        }
        PLOG(TRACE, "y:" + std::to_string(y));
        PLOG(TRACE, "l_word:" + l_word);

        if (IsEng(l_word) && l_word.size() == 1)
        {
            buf += l_word;
            x = y;
        }
        else
        {
            if (!buf.empty())
            {
                result.push_back(buf);
                buf = "";
            }
            result.push_back(l_word);
            x = y;
        }
    }

    if (!buf.empty())
    {
        result.push_back(buf);
        buf = "";
    }

    return result;
}

std::vector<WordPosPair> WordSpliter::CutDAG(const std::vector<std::string> &sentence, const std::string &raw_s)
{
    std::vector<WordPosPair> result;
    auto DAG = GetDAG(sentence);
    for (auto &d : DAG)
    {
        auto f = d.first;
        auto s = d.second;
    }

    std::unordered_map<int, std::pair<double, int>> route;
    calc(sentence, DAG, route);
    std::vector<std::string> chars = split_utf8(raw_s);
    size_t N = chars.size();

    int x = 0;
    std::string buf;

    while (x < N)
    {
        PLOG(TRACE, "=========step=======================" + std::to_string(x));
        int y = route[x].second + 1;
        std::string l_word;
        for (size_t i = x; i < y; ++i)
        {
            l_word += chars[i];
        }
        PLOG(TRACE, "y:" + std::to_string(y));
        PLOG(TRACE, "l_word:" + l_word);

        if (y - x == 1)
        {
            buf += l_word;
            PLOG(TRACE, "y-x==1, buf:" + buf);
        }
        else
        {
            PLOG(TRACE, "y-x!=1, buf:" + buf);
            if (!buf.empty())
            {
                if (split_utf8(buf).size() == 1)
                {
                    PLOG(TRACE, "dag_type0:" + buf);
                    result.emplace_back(buf, word_tag_[buf]);
                }
                else if (word_freq_.find(buf) == word_freq_.end() || word_freq_[buf] == 0)
                {
                    PLOG(TRACE, "dag_type1:" + buf);
                    auto recognized = CutDetail(buf);
                    PLOG(TRACE, "return1:" + std::to_string(recognized.size()));
                    result.insert(result.end(), recognized.begin(), recognized.end());
                }
                else
                {
                    PLOG(TRACE, "return2:" + buf);
                    result.emplace_back(buf, word_tag_[buf]);
                    for (char elem : buf)
                    {
                    }
                }
                buf.clear();
            }
            PLOG(TRACE, "return3:" + l_word);
            result.emplace_back(l_word, word_tag_[l_word]);
        }
        x = y;
    }

    if (!buf.empty())
    {
        auto buf_utf8_list = SplitUtf8String(buf);
        if (buf_utf8_list.size() == 1)
        {
            PLOG(TRACE, "return4:" + buf);
            result.emplace_back(buf, word_tag_[buf]);
        }
        else if (word_freq_.find(buf) == word_freq_.end())
        {
            auto recognized = CutDetail(buf);
            //            PLOG(TRACE, "return5:" + recognized[0].word);
            result.insert(result.end(), recognized.begin(), recognized.end());
        }
        else
        {
            PLOG(TRACE, "return6:" + buf);
            result.emplace_back(buf, word_tag_[buf]);
        }
    }
    //    PLOG(TRACE, "return:" + result[0].word);

    return result;
}

std::vector<WordPosPair> WordSpliter::Process(const std::string &seg, bool for_search)
{
    // 保存没有精细拆分的原始结果
    std::vector<WordPosPair> raw_result;
    // 保存精细拆分的结果
    std::vector<WordPosPair> fine_result;

    std::regex re_han_internal("([一-鿕a-zA-Z0-9+#&\\._]+)");
    std::regex re_skip_internal("(\\r\\n|\\s)");
    std::regex re_num("([\\.0-9]+)");
    std::regex re_eng("([a-zA-Z0-9]+)");

    auto parts = SplitChineseSentenceToParts(seg, false);
    // auto parts = split_han_internal(seg);

    PLOG(PDEBUG, "分词输入文本: " + seg);
    PLOG(PDEBUG, "标点分句结果: " + ConcatStrList(parts, "|"));

    //    std::vector<WordPosPair> raw_result;
    for (auto &part : parts)
    {
        PLOG(TRACE, DELIMITER_LINE);
        PLOG(TRACE, "分词开始处理|" + part + "|");
        // 如果包含汉字
        if (MatchHanInternal(part))
        {
            auto utf8_part = SplitUtf8String(part);
            auto fixed_part = ConcatStrList(utf8_part);
            PLOG(TRACE, "cutdag 分词输入文本：" + ConcatList(utf8_part, "|"));
            PLOG(TRACE, "cutdag 分词输入文本2：" + fixed_part);

            auto cur_raw_result = CutDAG(utf8_part, fixed_part);
            raw_result.insert(raw_result.end(), cur_raw_result.begin(), cur_raw_result.end());
            PLOG(TRACE, "cutdag 分词初步结果：" + ConcatList(raw_result, "|"));
        }
        else
        {
            auto tmp = SplitSkipInternal(part);
            for (auto &x : tmp)
            {
                if (MatchSkipInternal(x))
                {
                    raw_result.push_back(WordPosPair(x, "x"));
                }
                else
                {
                    for (auto &c : x)
                    {
                        std::string xx(1, c);
                        if (IsNum(xx))
                        {
                            raw_result.push_back(WordPosPair(xx, "m"));
                        }
                        else if (IsEng(xx))
                        {
                            raw_result.push_back(WordPosPair(xx, "eng"));
                        }
                        else
                        {
                            raw_result.push_back(WordPosPair(xx, "x"));
                        }
                    }
                }
            }
        }
    }

    if (for_search)
    {
        PLOG(PDEBUG, "分词for_search = true");
        for (auto &r : raw_result)
        {

            auto hanzi_list = SplitUtf8String(r.word);
            // 判断两个字的情况
            if (hanzi_list.size() > 2)
            {
                for (int i = 0; i < hanzi_list.size() - 1; i++)
                {
                    std::string cur_word = hanzi_list[i] + hanzi_list[i + 1];
                    bool is_word = word_freq_.find(cur_word) != word_freq_.end();
                    if (is_word)
                    {
                        // 注意：word_freq词频表中有一些存在但频率为0的词，不是常见的词，也需要过滤掉，如'男朋'
                        if (word_freq_[cur_word] > 0)
                        {
                            PLOG(PDEBUG, "找到新的词语：" + cur_word);
                            fine_result.push_back(WordPosPair(cur_word, r.flag));
                        }
                    }
                }
            }
            // 判断三个字的情况
            if (hanzi_list.size() > 3)
            {
                for (int i = 0; i < hanzi_list.size() - 2; i++)
                {
                    std::string cur_word = hanzi_list[i] + hanzi_list[i + 1] + hanzi_list[i + 2];
                    bool is_word = word_freq_.find(cur_word) != word_freq_.end();
                    if (is_word)
                    {
                        // 注意：word_freq词频表中有一些存在但频率为0的词，不是常见的词，也需要过滤掉
                        if (word_freq_[cur_word] > 0)
                        {
                            PLOG(PDEBUG, "找到新的词语：" + cur_word);
                            fine_result.push_back(WordPosPair(cur_word, r.flag));
                        }
                    }
                }
            }

            // 最后将原始的汉字词性对加入进去
            fine_result.push_back(r);
        }
    }
    else
    {
        fine_result = raw_result;
    }
    PLOG(PDEBUG, "分词最终结果：" + ConcatList(fine_result, "|"));

    return fine_result;
}

std::vector<std::string> WordSpliter::ProcessWoPos(const std::string &seg, bool for_search)
{
    std::vector<std::string> results;
    auto results_w_pos = Process(seg, for_search);
    for (auto &r : results_w_pos)
    {
        results.push_back(r.word);
    }
    return results;
}
