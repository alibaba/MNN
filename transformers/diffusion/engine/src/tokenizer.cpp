
#include <fstream>
#include <iostream>
#include <utility>
#include "tokenizer.hpp"
#include "core/Macro.h"
#include <sstream>
#include <ctype.h>
#include <functional>
#include <codecvt>
#include <regex>
#include <set>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

namespace MNN {
namespace DIFFUSION {

static bool ReadVarint(const char*& ptr, const char* end, uint64_t& value) {
    value = 0;
    int shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        value |= (uint64_t)(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return true;
        shift += 7;
    }
    return false;
}

T5Tokenizer::~T5Tokenizer() {
}

void T5Tokenizer::Trie::insert(const std::string& key, int id) {
    int node_idx = 0;
    for (char c : key) {
        if (list[node_idx].children.find(c) == list[node_idx].children.end()) {
            if (size >= list.size()) {
                list.resize(list.size() * 2);
            }
            list[node_idx].children[c] = size;
            node_idx = size++;
        } else {
            node_idx = list[node_idx].children[c];
        }
    }
    list[node_idx].id = id;
}

std::vector<std::pair<int, int>> T5Tokenizer::Trie::commonPrefixSearch(const std::string& str, int start) {
    std::vector<std::pair<int, int>> results;
    int node_idx = 0;
    for (int i = start; i < str.length(); ++i) {
        char c = str[i];
        if (list[node_idx].children.find(c) == list[node_idx].children.end()) {
            break;
        }
        node_idx = list[node_idx].children[c];
        if (list[node_idx].id != -1) {
            results.push_back({list[node_idx].id, i - start + 1});
        }
    }
    return results;
}

bool T5Tokenizer::load(const std::string& filePath) {
    std::string modelPath = filePath + "/spiece.model";
    std::ifstream input(modelPath, std::ios::binary);
    if (!input.is_open()) {
        MNN_ERROR("Failed to open %s\n", modelPath.c_str());
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const char* ptr = content.data();
    const char* end = content.data() + content.size();
    
    mPieces.clear();
    mTrie = Trie();
    
    while (ptr < end) {
        uint64_t tag;
        if (!ReadVarint(ptr, end, tag)) break;
        int field_num = tag >> 3;
        int wire_type = tag & 7;
        
        if (field_num == 1 && wire_type == 2) { // pieces
            uint64_t len;
            if (!ReadVarint(ptr, end, len)) break;
            const char* msg_end = ptr + len;
            
            std::string piece;
            float score = 0.0f;
            int type = 1;
            
            while (ptr < msg_end) {
                uint64_t sp_tag;
                if (!ReadVarint(ptr, end, sp_tag)) break;
                int sp_field = sp_tag >> 3;
                int sp_wire = sp_tag & 7;
                
                if (sp_field == 1 && sp_wire == 2) { // piece
                    uint64_t str_len;
                    if (!ReadVarint(ptr, end, str_len)) break;
                    piece = std::string(ptr, str_len);
                    ptr += str_len;
                } else if (sp_field == 2 && sp_wire == 5) { // score
                    uint32_t val;
                    memcpy(&val, ptr, 4);
                    ptr += 4;
                    memcpy(&score, &val, 4);
                } else if (sp_field == 3 && sp_wire == 0) { // type
                    uint64_t type_val;
                    ReadVarint(ptr, end, type_val);
                    type = (int)type_val;
                } else {
                    if (sp_wire == 0) { uint64_t tmp; ReadVarint(ptr, end, tmp); }
                    else if (sp_wire == 1) ptr += 8;
                    else if (sp_wire == 2) { uint64_t tmp; ReadVarint(ptr, end, tmp); ptr += tmp; }
                    else if (sp_wire == 5) ptr += 4;
                }
            }
            
            mPieces.push_back({piece, score});
            if (type == 1) { // NORMAL
                mTrie.insert(piece, mPieces.size() - 1);
            } else if (type == 3) { // CONTROL
                mTrie.insert(piece, mPieces.size() - 1);
            } else if (type == 2) { // UNKNOWN
                mUnkId = mPieces.size() - 1;
            }
            
        } else {
            if (wire_type == 0) { uint64_t tmp; ReadVarint(ptr, end, tmp); }
            else if (wire_type == 1) ptr += 8;
            else if (wire_type == 2) { uint64_t tmp; ReadVarint(ptr, end, tmp); ptr += tmp; }
            else if (wire_type == 5) ptr += 4;
        }
    }
    
    // Find EOS ID
    for (int i = 0; i < mPieces.size(); ++i) {
        if (mPieces[i].first == "</s>") {
            mEosId = i;
            break;
        }
    }
    
    return true;
}

std::vector<int> T5Tokenizer::encodeUnigram(const std::string& text) {
    struct Node {
        int id = -1;
        float score = -1e9;
        int prev_idx = -1;
        int length = 0;
    };
    
    std::vector<Node> lattice(text.length() + 1);
    lattice[0].score = 0.0f;
    
    for (int i = 0; i < text.length(); ++i) {
        if (lattice[i].score <= -1e9) continue;
        
        auto matches = mTrie.commonPrefixSearch(text, i);
        
        bool has_single = false;
        for (auto& m : matches) {
            int id = m.first;
            int len = m.second;
            if (len == 1) has_single = true;
            
            float score = lattice[i].score + mPieces[id].second;
            if (score > lattice[i + len].score) {
                lattice[i + len].score = score;
                lattice[i + len].id = id;
                lattice[i + len].prev_idx = i;
                lattice[i + len].length = len;
            }
        }
        
        if (!has_single) {
            float score = lattice[i].score - 10.0f;
            if (score > lattice[i + 1].score) {
                lattice[i + 1].score = score;
                lattice[i + 1].id = mUnkId;
                lattice[i + 1].prev_idx = i;
                lattice[i + 1].length = 1;
            }
        }
    }
    
    std::vector<int> ids;
    int idx = text.length();
    while (idx > 0) {
        int id = lattice[idx].id;
        if (id != -1) ids.push_back(id);
        idx = lattice[idx].prev_idx;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::vector<int> T5Tokenizer::encode(const std::string& sentence, int maxlen) {
    std::string normalized;
    for (char c : sentence) {
        if (c == ' ') normalized += "\xe2\x96\x81";
        else normalized += c;
    }
    if (normalized.empty() || normalized[0] != '\xe2\x96\x81') {
        normalized = "\xe2\x96\x81" + normalized;
    }
    
    std::vector<int> ids = encodeUnigram(normalized);
    ids.push_back(mEosId); 
    
    if (maxlen > 0) {
        if (ids.size() > maxlen) {
            ids.resize(maxlen);
            ids[maxlen - 1] = mEosId; 
        } else {
            while (ids.size() < maxlen) {
                ids.push_back(0);
            }
        }
    }
    return ids;
}
    
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
        else if (::isalnum(c)) {
            while (i < str.size() && std::isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += std::tolower(str[i]);
                ++i;
            }
        }
        // handle punctuation and symbols
        else if (::ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // handle space, tab, enter
        else if (::isspace(c)) {
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
    
    int max_token_limit = maxlen * 2 - 1; 

    for (auto token : tokens) {
        std::vector<int> sub_tokens = word_piece(token);

        bool out_of_bounds = false;
        for (auto id : sub_tokens) {
            if (idx >= max_token_limit) {
                out_of_bounds = true;
                break;
            }
            ids[idx++] = id;
        }
        if (out_of_bounds) {
            break;
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
            mVocabs[utf8_to_wstring(key)] = intValue;
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

void get_pairs(std::vector<std::wstring> word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();
    
    if (word.size() < 2) return;
    
    std::wstring previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        pairs->push_back({std::wstring(previous), std::wstring(word[i])});
        previous = word[i];
    }
}

// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
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
    
    std::vector<std::wstring> word;
    for (int i = 0; i < token.size() - 1; i++) {
        word.emplace_back(1, token[i]);
    }
    word.push_back(token.substr(token.size() - 1) + utf8_to_wstring("</w>"));
    
    
    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(word, &pairs);
    
    if (pairs.size() == 0) {
        result->push_back(token + utf8_to_wstring("</w>"));
        return;
    }
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
        result->push_back(token + utf8_to_wstring("</w>"));
        
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
    // Use static regex to avoid recompilation and potential stack/heap issues with repeated construction
    // Also replaced POSIX classes with explicit ranges to avoid potential locale issues
    static const std::regex re(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^ \t\n\r\f\va-zA-Z0-9]+)",
                   std::regex::icase);
    std::string input = text;
    std::vector<std::wstring> result;
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
            result.push_back(ws);
        }
    }
    
    std::vector<int> ids(maxlen * 2, 0);
    // uncond
    ids[0] = mStartIdx;
    ids[1] = mEndIdx;
    // ids
    int idx = maxlen;
    ids[idx++] = mStartIdx;
    int max_token_limit = maxlen * 2 - 1; 

    for (auto s : result) {
        if (idx >= max_token_limit) {
            break;
        }
        ids[idx++] = mVocabs.at(s);
    }
    ids[idx++] = mEndIdx;
    
    return ids;
}
}
} // diffusion
