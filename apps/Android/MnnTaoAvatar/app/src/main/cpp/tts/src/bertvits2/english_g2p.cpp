#include "english_g2p.hpp"

EnglishG2P::EnglishG2P() {}

EnglishG2P::EnglishG2P(const std::string &local_resource_root)
{
    resource_root_ = local_resource_root;

    auto english_dict_json_path = resource_root_ + "common/text_processing_jsons/eng_dict.json";

    auto english_dict_bin_path = resource_root_ + "common/text_processing_jsons/eng_dict.bin";
    eng_dict_ = deserialize(english_dict_bin_path);

    // 创建映射
    int id = 0;
    for (auto &symbol : symbols)
    {
        _symbol_to_id[symbol] = id++;
    }
}

std::unordered_map<char, std::vector<std::string>> alphabet_dict = {
    {'A', {"EY1"}},
    {'B', {"B", "IY1"}},
    {'C', {"S", "IY1"}},
    {'D', {"D", "IY1"}},
    {'E', {"IY1"}},
    {'F', {"EH1", "F"}},
    {'G', {"JH", "IY1"}},
    {'H', {"EY1", "CH"}},
    {'I', {"AY1"}},
    {'J', {"JH", "EY1"}},
    {'K', {"K", "EY1"}},
    {'L', {"EH1", "L"}},
    {'M', {"EH1", "M"}},
    {'N', {"EH1", "N"}},
    {'O', {"OW1"}},
    {'P', {"P", "IY1"}},
    {'Q', {"K", "Y", "UW1"}},
    {'R', {"AA1", "R"}},
    {'S', {"EH1", "S"}},
    {'T', {"T", "IY1"}},
    {'U', {"Y", "UW1"}},
    {'V', {"V", "IY1"}},
    {'W', {"D", "AH1", "B", "AH1", "L", "Y", "UW1"}},
    {'X', {"EH1", "K", "S"}},
    {'Y', {"W", "AY1"}},
    {'Z', {"Z", "IY1"}},
};

// Define the custom lexicon
std::unordered_map<std::string, std::vector<std::vector<std::string>>> custom_lexicon = {
    // key using uppercase, assuming your keys are case-insensitive
    {"AI", {{"EY0"}, {"AY1"}}},
    {"IOT", {{"AY1"}, {"OW1"}, {"T", "IY0"}}},
    {"BILIBILI", {{"B", "IY1"}, {"L", "IY1"}, {"B", "IY1"}, {"L", "IY1"}}},
    {"BTOC", {{"B", "IY1"}, {"T", "UW1"}, {"S", "IY1"}}},
    {"ALIBABA", {{"EH1"}, {"L", "IY1"}, {"B", "AH1"}, {"B", "AH1"}}},
    {"TMALL", {{"T", "IY1"}, {"M", "AH1", "L"}}},
    {"CAINIAO", {{"CH", "AY1"}, {"N", "IY1", "OW1"}}}};

// Function to initialize phoneme sequences for "YYDS" using alphabet_dict
std::vector<std::vector<std::string>> getPhonemesForYYDS()
{
    std::vector<std::vector<std::string>> yyds;
    yyds.push_back(alphabet_dict['Y']);
    yyds.push_back(alphabet_dict['Y']);
    yyds.push_back(alphabet_dict['D']);
    yyds.push_back(alphabet_dict['S']);
    return yyds;
}

std::unordered_map<std::string, std::string> create_punctuation_mapping()
{
    const std::string from = "，。！？；：（）【】《》“”‘’";
    const std::string to = ",.!?;:()[]<>\"\"''";
    auto from_list = SplitUtf8String(from);
    auto to_list = SplitUtf8String(to);
    std::unordered_map<std::string, std::string> mapping;

    for (size_t i = 0; i < from.size(); ++i)
    {
        mapping[from_list[i]] = to_list[i];
    }

    return mapping;
}

void EnglishG2P::ParseEnglishDict(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);

    // 将json对象转换为std::map
    eng_dict_ = json_obj.get<english_dict>();
}

std::string remove_commas(const std::string &text)
{
    return std::regex_replace(text, std::regex(","), "");
}

std::string expand_dollars(const std::smatch &match)
{
    std::string dollars_part = match.str(1);
    return dollars_part + " dollars"; // 简化的示例
}

std::string replace_regex(const std::regex &re, const std::string &text,
                          std::function<std::string(const std::smatch &)> func)
{
    std::string result;
    std::string::const_iterator searchStart(text.cbegin());
    std::smatch matches;
    while (std::regex_search(searchStart, text.cend(), matches, re))
    {
        result += matches.prefix().str() + func(matches);
        searchStart = matches.suffix().first;
    }
    result += std::string(searchStart, text.cend());
    return result;
}

std::string normalize_numbers(const std::string &text)
{
    std::regex comma_number_re(R"(([0-9][0-9\,]+[0-9]))");
    std::regex dollars_re(R"(\$([0-9\.\,]*[0-9]+))");
    // ... 其他正则表达式定义

    std::string result = text;

    result = replace_regex(comma_number_re, result, [](const std::smatch &m)
                           { return remove_commas(m.str(1)); });

    result = replace_regex(dollars_re, result, [](const std::smatch &m)
                           { return expand_dollars(m); });

    return result;
}

std::string replace_punctuation(const std::string &text, const std::unordered_map<std::string, std::string> &punctuation_mapping)
{

    std::string modified_text = text;

    //  // 替换先定义的字符映射
    for (const auto &pair : punctuation_mapping)
    {
        auto word_splits = SplitUtf8String(modified_text);
        if (HanListContains(word_splits, pair.first))
        {
            StrReplaceAll(modified_text, pair.first, pair.second);
        }
    }

    return modified_text;
}

std::string text_normalize(const std::string &text)
{
    std::string result = text;

    std::unordered_map<std::string, std::string> rep_map = {
        {"：", ","},
        {"；", ","},
        {"，", ","},
        {"。", "."},
        {"！", "!"},
        {"？", "?"},
        {"\n", "."},
        {"·", ","},
        {"、", ","},
        {"...", "…"},
        {"(", ","},
        {")", ","},
        {"（", ","},
        {"）", ","},
    };
    result = replace_punctuation(result, rep_map);

    result = normalize_numbers(result);
    return result;
}

std::vector<std::string> split_camel_case(const std::string &input_str)
{
    std::regex regex_step1(R"(([^A-Z]+)([A-Z][a-z]))");
    std::string step1 = std::regex_replace(input_str, regex_step1, "$1 $2");

    std::regex regex_step2(R"(([a-z])([A-Z]))");
    std::string step2 = std::regex_replace(step1, regex_step2, "$1 $2");

    std::istringstream iss(step2);
    std::vector<std::string> results(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{});

    return results;
}

std::pair<std::string, int> refine_ph(const std::string &phn)
{
    int tone = 0;
    std::regex tone_regex("\\d$"); // Regular expression to match a digit at the end of the string

    // Check if the input string has a tone digit at the end
    if (std::regex_search(phn, tone_regex))
    {
        tone = phn.back() - '0' + 1;                               // Convert the last character to an integer and increment by 1
        return {ToLowercase(phn.substr(0, phn.size() - 1)), tone}; // Return the phoneme without the last character
    }
    return {ToLowercase(phn), tone}; // Return the original phoneme and tone=0
}

std::pair<std::vector<std::string>, std::vector<int>> refine_syllables(
    const std::vector<std::vector<std::string>> &syllables)
{
    std::vector<int> tones;
    std::vector<std::string> phonemes;

    for (const auto &phn_list : syllables)
    {
        for (const auto &phn : phn_list)
        {
            auto [refined_phn, tone] = refine_ph(phn);
            phonemes.push_back(refined_phn);
            tones.push_back(tone);
        }
    }

    return {phonemes, tones};
}

std::string post_replace_ph(std::string ph)
{
    static const std::unordered_map<std::string, std::string> rep_map = {
        {"：", ","},
        {"；", ","},
        {"，", ","},
        {"。", "."},
        {"！", "!"},
        {"？", "?"},
        {"\n", "."},
        {"·", ","},
        {"、", ","},
        {"...", "…"},
        {"v", "V"},
        {"(", ","},
        {")", ","},
        {"（", ","},
        {"）", ","},
    };

    // Check if the phoneme is in the replacement map
    auto it = rep_map.find(ph);
    if (it != rep_map.end())
    {
        ph = it->second; // Replace phoneme with its mapped value
    }

    // Check against the list of valid symbols
    if (!HanListContains(symbols, ph))
    {
        ph = "UNK"; // Set to "UNK" if the phoneme is not in the valid symbols
    }

    return ph;
}

bool is_uppercase_and_digit(const std::string &word)
{
    std::regex pattern(R"(^[A-Z0-9]+$)");
    return std::regex_match(word, pattern);
}

g2p_data EnglishG2P::G2P(const std::string &input_text)
{
    // Replace punctuation
    std::vector<std::string> punctuation = {"，", "。", "！", "？", "；", "：", "、", "（", "）", "【", "】",
                                            "《", "》", "“", "”", "‘", "’", ",", ".", "!", "?", ";",
                                            ":", "(", ")", "[", "]", "<", ">", "\"", "'"};

    std::string text = input_text;
    auto words_tmp = SplitEnSentenceToWords(text);

    // Split camel case and further process words
    std::vector<std::string> words;
    for (const auto &word : words_tmp)
    {
        auto split_words = split_camel_case(word);
        words.insert(words.end(), split_words.begin(), split_words.end());
    }

    std::vector<std::string> phones;
    std::vector<int> tones;
    std::vector<int> word2ph;

    // Main processing loop
    for (const auto &word : words)
    {
        if (HanListContains(punctuation, word))
        {
            phones.push_back(word);
            tones.push_back(0);
            word2ph.push_back(1);
        }
        else
        {
            // Convert word to uppercase for dictionary lookup
            std::string upper_word = word;
            std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

            if (eng_dict_.find(upper_word) != eng_dict_.end())
            {
                auto [phns, tns] = refine_syllables(eng_dict_[upper_word]);

                phones.insert(phones.end(), phns.begin(), phns.end());
                tones.insert(tones.end(), tns.begin(), tns.end());
                word2ph.push_back(phns.size());
            }
            else if (is_uppercase_and_digit(word))
            {
                std::vector<std::vector<std::string>> spell_list;
                for (char c : word)
                {
                    spell_list.push_back(alphabet_dict[c]);
                }
                auto [phns, tns] = refine_syllables(spell_list);

                phones.insert(phones.end(), phns.begin(), phns.end());
                tones.insert(tones.end(), tns.begin(), tns.end());
                word2ph.push_back(phns.size());
            }
            else
            {
                // 非全大写，也按单个字母来发音
                std::vector<std::vector<std::string>> spell_list;
                auto upper_word = ToUppercase(word);
                for (char c : upper_word)
                {
                    spell_list.push_back(alphabet_dict[c]);
                }
                auto [phns, tns] = refine_syllables(spell_list);

                phones.insert(phones.end(), phns.begin(), phns.end());
                tones.insert(tones.end(), tns.begin(), tns.end());
                word2ph.push_back(phns.size());
            }
        }
    }

    // Apply post replacements
    for (auto &phone : phones)
    {
        phone = post_replace_ph(phone);
    }

    // Add boundary markers
    phones.insert(phones.begin(), "_");
    phones.push_back("_");
    tones.insert(tones.begin(), 0);
    tones.push_back(0);
    word2ph.insert(word2ph.begin(), 1);
    word2ph.push_back(1);

    return {phones, tones, word2ph};
}

phone_data EnglishG2P::CleanedTextToSequence(const g2p_data &g2p_data_, const std::string &language)
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

    return std::make_tuple(int_phones, adjusted_tones, lang_ids, word2ph);
}

std::tuple<std::string, phone_data> EnglishG2P::Process(SentLangPair &sent_lang)
{
    auto text = text_normalize(sent_lang.sent);
    g2p_data g2p_data_ = G2P(text);
    auto res = CleanedTextToSequence(g2p_data_, sent_lang.lang);
    return std::make_tuple(text, res);
}
