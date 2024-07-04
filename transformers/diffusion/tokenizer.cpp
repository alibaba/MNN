
#include <fstream>
#include <iostream>
#include <utility>
#include "tokenizer.hpp"
#include "core/Macro.h"
#include <sstream>
#include <functional>
#include <codecvt>
#include <regex>
#include <set>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"
namespace diffusion {

bool BertTokenizer::load(const std::string& dictPath) {
    std::ifstream dictFile(dictPath + "/vocab.txt");
    if(!dictFile.is_open()) {
        MNN_ERROR("tokenize load error, vocab.txt not found in %s\n", dictPath.c_str());
        return false;
    }
    int index = 0;
    std::string word;
    while (dictFile >> word) {
        mVocabs.insert(std::make_pair<std::string, int>(std::move(word), index++));
    }
    mStartIdx = this->word_piece("[CLS]")[0];
    mEndIdx = this->word_piece("[SEP]")[0];
    return true;
}

std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    auto it = mVocabs.find(token);
    if (it != mVocabs.end()) {
        return {it->second};
    }
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;
        size_t match_pos = 0;
        for (int len = current.size(); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            if (!ids.empty()) {
                candidate = "##" + candidate;
            }
            auto it = mVocabs.find(candidate);
            if (it != mVocabs.end()) {
                match_id = it->second;
                match_pos = len;
                break;
            }
        }
        // [UNK]
        if (match_id == -1) {
            ids.push_back(100);
            break;
        }
        ids.push_back(match_id);
        // not first word, adding ## prefix
        current = current.substr(match_pos);
    }
    return ids;
}

    
std::vector<int> BertTokenizer::encode(const std::string& str, int maxlen) {
    std::vector<int> ids(maxlen * 2, 0);
    // uncond
    ids[0] = mStartIdx;
    ids[1] = mEndIdx;
    // ids
    int idx = maxlen;
    ids[idx++] = mStartIdx;

    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);
        // handle multi-byte UTF-8 characters
        if ((c & 0x80) != 0) {
            unsigned char mask = 0xE0; // 1110 0000 for 3-byte char
            if ((c & mask) == mask) {
                current_token = str.substr(i, 3);
                i += 3;
            } else {
                ++i;
                continue;
            }
        }
        // handle continuous sequence of letters and digits
        else if (std::isalnum(c)) {
            while (i < str.size() && std::isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += std::tolower(str[i]);
                ++i;
            }
        }
        // handle punctuation and symbols
        else if (std::ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // handle space, tab, enter
        else if (std::isspace(c)) {
            ++i;
            continue;
        }
        // handle any other single-byte characters
        else {
            current_token = str[i];
            ++i;
        }
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }

    for (auto token : tokens) {
        for (auto id : word_piece(token)) {
            ids[idx++] = id;
        }
    }
   
    ids[idx++] = mEndIdx;
    return ids;
}
    
bool CLIPTokenizer::load(const std::string& filePath) {
    bool res_0 = loadVocab(filePath + "/vocab.json");
    bool res_1 = loadMerges(filePath + "/merges.txt");
    return res_0 && res_1;
}
    
std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

std::string wstring_to_utf8(const std::wstring& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
}

bool CLIPTokenizer::loadVocab(const std::string& vocabFilePath) {
    FILE* fp = fopen(vocabFilePath.c_str(), "rb");
    if (!fp) {
        MNN_ERROR("File %s open failed.\n", vocabFilePath.c_str());
        return false;
    }

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    rapidjson::Document doc;
    doc.ParseStream(is);
    fclose(fp);

    if (doc.HasParseError()) {
        MNN_ERROR("JSON parse error: %s\n", rapidjson::GetParseError_En(doc.GetParseError()));
        return false;
    }

    if (!doc.IsObject()) {
        MNN_ERROR("JSON is not an object.\n");
        return false;
    }

    for (rapidjson::Value::ConstMemberIterator itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr) {
        const char* key = itr->name.GetString();
        
        if (itr->value.IsInt()) {
            int intValue = itr->value.GetInt();
            mVocabs[std::string(key)] = intValue;
//            std::cout << key << ": " << intValue << std::endl;
        } else {
            MNN_ERROR("Value for key: %s is not an integer.\n", key);
        }
    }
    
    auto _insert_range = [=](int start, int end) {
       for (int c = start; c <= end; c++) {
           b2u_.insert({uint8_t(c), wchar_t(c)});
       }
   };
    
    b2u_.clear();
    _insert_range(L'!', L'~');
    _insert_range(L'¡', L'¬');
    _insert_range(L'®', L'ÿ');

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u_.find(uint8_t(b)) == b2u_.end()) {
            b2u_.insert({uint8_t(b), wchar_t(256 + n)});
            n++;
        }
    }
    for (auto e : b2u_) {
        u2b_.insert({e.second, e.first});
    }
    
    return true;
}

bool CLIPTokenizer::loadMerges(const std::string& mergesFilePath) {
    std::ifstream file(mergesFilePath);
    std::string line;

    if (!file.is_open()) {
        MNN_ERROR("Failed to open merges file: %s\n", mergesFilePath.c_str());
        return false;
    }
    int count = 0;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t colonPos = line.find(' ');
        if (colonPos != std::string::npos) {
            std::string token_0 = line.substr(0, colonPos);
            std::string token_1 = line.substr(colonPos + 1);
            bpe_ranks_[std::make_pair(utf8_to_wstring(token_0), utf8_to_wstring(token_1))] = count++;
        }
    }
    return true;
}
    
void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();

    if (word.size() < 2) return;

    wchar_t previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        pairs->push_back({std::wstring(1, previous), std::wstring(1, word[i])});
        previous = word[i];
    }
}

void CLIPTokenizer::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
    std::set<int> merged;  // records indices in pairs that were merged.
    auto _left = [](int i, std::set<int>& merged) {
        for (int j = i - 1; j >= -1; j--) {
        if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    auto _right = [](int i, int cap, std::set<int>& merged) {
        for (int j = i + 1; j < cap; j++) {
        if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, &pairs);

    while (true) {
        int min_score = INT_MAX;
        int to_merge = -1;  // indices into pairs.

        for (int i = 0; i < pairs.size(); ++i) {
        if (merged.find(i) == merged.end()) {  // pair i is not merged.
            auto iter = bpe_ranks.find(pairs[i]);
            int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
            if (score < min_score) {
            min_score = score;
            to_merge = i;
            }
        }
        }

        if (to_merge == -1) break;

        merged.insert(to_merge);
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        int l = _left(to_merge, merged);
        if (l >= 0) pairs[l].second = merge_into;
        int r = _right(to_merge, pairs.size(), merged);
        if (r < pairs.size()) pairs[r].first = merge_into;
    }  // end while (true)

    if (merged.size() == pairs.size()) {
        result->push_back(token);

    } else {
        for (int i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {
                if (_left(i, merged) < 0) result->push_back(pairs[i].first);
                result->push_back(pairs[i].second);
            }
        }
    }
}

std::vector<int> CLIPTokenizer::encode(const std::string& text, int maxlen) {
    
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = text;
    std::vector<std::string> result;
    std::string token;
    std::smatch match;
    while (std::regex_search(input, match, re)) {
        token = match.str(0);
        input = match.suffix().str();
        
        std::wstring wtoken;
        for (char c : token) {
            wtoken.push_back(b2u_.at(uint8_t(c)));
        }
        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_ranks_, &bpe_tokens);

        for (auto ws : bpe_tokens) {
            result.push_back(wstring_to_utf8(ws));
        }
    }

    std::vector<int> ids(maxlen * 2, 0);
    // uncond
    ids[0] = mStartIdx;
    ids[1] = mEndIdx;
    // ids
    int idx = maxlen;
    ids[idx++] = mStartIdx;
    for (auto s : result) {
        ids[idx++] = mVocabs.at(s);
    }
    ids[idx++] = mEndIdx;

    return ids;
}

} // diffusion
