//
//  tokenizer.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#include <MNN/MNNDefine.h>
#define MNN_OPEN_TIME_TRACE 1
#include <MNN/AutoTime.hpp>
#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <random>
#include <codecvt>
#include <regex>
#include <set>
#include <climits>
#include <cctype>
#include <cstring>
namespace MNN {
namespace Transformer {

// base64
static const int kBase64DecodeTable[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0-15
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 16-31
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, // 32-47 (+, /)
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, // 48-63 (0-9)
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, // 64-79 (A-O)
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, // 80-95 (P-Z)
    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, // 96-111 (a-o)
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1  // 112-127 (p-z)
};

static std::string base64_decode(const std::string& str) {
    if (str.empty()) return "";
    size_t in_len = str.size();
    std::string ret;
    ret.reserve(in_len * 3 / 4 + 2);

    int val = 0, valb = -8;
    for (unsigned char c : str) {
        if (c > 127) continue;
        int d = kBase64DecodeTable[c];
        if (d == -1) continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            ret.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return ret;
}

static inline size_t one_char_len(const char *src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

static inline void to_lower_case(std::string& str) {
    for (auto &c : str) {
        if (c >= 'A' && c <= 'Z') {
            c = tolower(static_cast<unsigned char>(c));
        }
    }
}

Tokenizer* Tokenizer::createTokenizer(const std::string& filename) {
    AUTOTIME;
    Tokenizer* tokenizer = nullptr;
    // check file
    std::ifstream tok_file(filename);
    if (!tok_file.good()) {
        printf("Failed: can't load tokenzier from: %s.\n", filename.c_str());
        return tokenizer;
    }
    // check tokenizer info
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int magic_number, tokenizer_type;
    line_str >> magic_number;
    if (magic_number != MAGIC_NUMBER) {
        printf("Failed: magic number is wrong from: %s.\n", filename.c_str());
        return tokenizer;
    }
    line_str >> tokenizer_type;
    // create tokenizer
    switch (tokenizer_type)
    {
        case SENTENCEPIECE:
            tokenizer = new Sentencepiece();
            break;
        case TIKTOIKEN:
            tokenizer = new Tiktoken();
            break;
        case BERT:
            tokenizer = new BertTokenizer();
            break;
        case HUGGINGFACE:
            tokenizer = new HuggingfaceTokenizer();
            break;
        default:
            return tokenizer;
    }
    // load special tokens
    tokenizer->load_special(tok_file);
    // load vocabs
    tokenizer->load_vocab(tok_file);
    tok_file.close();
    tokenizer->cache_special_tokens();
    return tokenizer;
}

bool Tokenizer::is_stop(int token) {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
}

bool Tokenizer::is_special(int token) {
    return std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end();
}

void Tokenizer::load_special(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int special_num, stop_num, prefix_num;
    line_str >> special_num >> stop_num >> prefix_num;
    std::getline(tok_file, line);
    std::istringstream specail_line(line);
    if (special_num) {
        // load special tokens
        special_tokens_.resize(special_num);
        for (int i = 0; i < special_num; i++) {
            specail_line >> special_tokens_[i];
        }
    }
    if (stop_num) {
        // load stop tokens
        stop_tokens_.resize(stop_num);
        for (int i = 0; i < stop_num; i++) {
            specail_line >> stop_tokens_[i];
        }
    }
    if (prefix_num) {
        // load prefix tokens
        prefix_tokens_.resize(prefix_num);
        for (int i = 0; i < prefix_num; i++) {
            specail_line >> prefix_tokens_[i];
        }
    }
}

void Tokenizer::cache_special_tokens() {
    special_tokens_cache_.clear();
    for (int id : special_tokens_) {
        std::string token_str = decode(id);
        if (!token_str.empty()) {
            special_tokens_cache_.emplace_back(token_str, id);
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& str) {
    std::vector<int> ids = prefix_tokens_;
    if (special_tokens_cache_.empty() && !special_tokens_.empty()) {
        cache_special_tokens();
    }

    if (!special_tokens_cache_.empty()) {
        std::string text = str;
        size_t start = 0;
        for (size_t i = 0; i < text.length(); ++i) {
            for (const auto& pair : special_tokens_cache_) {
                const std::string& token = pair.first;
                int special_id = pair.second;
                if (i + token.length() <= text.length() &&
                    strncmp(text.c_str() + i, token.c_str(), token.length()) == 0) {
                    if (i > start) {
                        encode(text.substr(start, i - start), ids);
                    }
                    ids.push_back(special_id);
                    start = i + token.length();
                    i = start - 1;
                    break;
                }
            }
        }
        if (start < text.length()) {
            encode(text.substr(start), ids);
        }
    } else {
        encode(str, ids);
    }
    return ids;
}

bool Sentencepiece::load_vocab(std::ifstream& tok_file) {
    AUTOTIME;
    std::string line;
    if (!std::getline(tok_file, line)) return false;
    int vocab_len = std::stoi(line);
    sentence_pieces_.resize(vocab_len);
    pieces_.reserve(vocab_len);
    reserved_id_map_.reserve(vocab_len);

    for (int index = 0; index < vocab_len; index++) {
        std::getline(tok_file, line);

        size_t first_space = line.find(' ');
        if (first_space == std::string::npos) continue;
        size_t second_space = line.find(' ', first_space + 1);
        if (second_space == std::string::npos) continue;

        std::string token = base64_decode(line.substr(0, first_space));
        float score = std::strtof(line.c_str() + first_space + 1, nullptr);
        int type = std::atoi(line.c_str() + second_space + 1);

        auto piece_type = static_cast<PieceType>(type);
        sentence_pieces_[index] = {token, score, piece_type};
        string_view_ token_sv(sentence_pieces_[index].piece);
        if (piece_type == PieceType::NORMAL) {
            pieces_.insert({token_sv, index});
        } else {
            reserved_id_map_.insert({token_sv, index});
            if (piece_type == PieceType::UNKNOWN) {
                unk_id_ = index;
            }
        }
    }
    return true;
}

int Sentencepiece::piece_to_id(string_view_ piece) const {
    auto it = reserved_id_map_.find(piece);
    if (it != reserved_id_map_.end()) {
        return it->second;
    }
    auto it2 = pieces_.find(piece);
    if (it2 != pieces_.end()) {
        return it2->second;
    }
    return unk_id_;
}

std::string Sentencepiece::byte_to_piece(unsigned char c) const {
    const int len = ::snprintf(nullptr, 0, "<0x%02X>", c);
    std::string s;
    s.resize(len);
    ::snprintf(&s[0], s.size() + 1, "<0x%02X>", c);
    return s;
}

// ref: https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
Sentencepiece::EncodeResult Sentencepiece::bpe_encode(string_view_ normalized, float alpha) {
    // util class begin
    struct SymbolPair {
        int left;     // left index of this pair
        int right;    // right index of this pair
        float score;  // score of this pair. large is better.
        size_t size;  // length of this piece
    };

    class SymbolPairComparator {
    public:
        const bool operator()(SymbolPair *h1, SymbolPair *h2) {
            return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
        }
    };

    struct Symbol {
        int prev;     // prev index of this symbol. -1 for BOS.
        int next;     // next index of tihs symbol. -1 for EOS.
        bool freeze = false;  // this symbol is never be merged.
        string_view_ piece;
    };
    // util class end

    using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
    Agenda agenda;
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    // Reverse merge rules. key: merged symbol, value: pair of original symbols.
    std::unordered_map<string_view_, std::pair<string_view_, string_view_>> rev_merge;
    // SymbolPair holder.
    std::vector<std::unique_ptr<SymbolPair>> symbol_pair_holder;
    // Lookup new symbol pair at [left, right] and inserts it to agenda.
    auto MaybeAddNewSymbolPair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) {
            return;
        }
        const string_view_ piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        const auto it = pieces_.find(piece);
        if (it == pieces_.end()) {
            return;
        }
        symbol_pair_holder.emplace_back(new SymbolPair);
        auto *h = symbol_pair_holder.back().get();
        h->left = left;
        h->right = right;
        h->score = get_score(it->second);
        h->size = piece.size();
        agenda.push(h);

        // Makes `rev_merge` for resegmentation.
        if (is_unused(it->second)) {
            rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
        }
    };
    // Splits the input into character sequence
    int index = 0;
    while (!normalized.empty()) {
        Symbol s;
        // const int mblen = matcher_->PrefixMatch(normalized, &s.freeze);
        int mblen = std::min<int>(normalized.size(), one_char_len(normalized.data()));
        s.piece = string_view_(normalized.data(), mblen);
        s.prev = index == 0 ? -1 : index - 1;
        normalized.remove_prefix(mblen);
        s.next = normalized.empty() ? -1 : index + 1;
        ++index;
        symbols.emplace_back(s);
    }

    if (symbols.empty()) {
        return {};
    }
    // Lookup all bigrams.
    for (size_t i = 1; i < symbols.size(); ++i) {
        MaybeAddNewSymbolPair(i - 1, i);
    }

    // BPE-dropout: https://arxiv.org/pdf/1910.13267.pdf
    // std::mt19937 *rand_gen = nullptr;
    std::mt19937 rand_gen;
    auto skip_merge = [&]() {
        if (alpha <= 0.0) return false;
        if (alpha >= 1.0) return true;
        // if (rand_gen == nullptr) rand_gen = random::GetRandomGenerator();
        std::uniform_real_distribution<> gen(0.0, 1.0);
        return gen(rand_gen) < alpha;
    };

    // Main loop.
    while (!agenda.empty()) {
        SymbolPair *top = agenda.top();
        agenda.pop();

        // `top` is no longer available.
        if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
            continue;
        }

        if (skip_merge()) continue;
        // Replaces symbols with `top` rule.
        symbols[top->left].piece = string_view_(
                                                symbols[top->left].piece.data(),
                                                symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // Updates prev/next pointers.
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = string_view_("");

        // Adds new symbol pairs which are newly added after symbol replacement.
        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    std::function<void(string_view_, EncodeResult*)> resegment;
    resegment = [this, &resegment, &rev_merge](string_view_ w, EncodeResult *output) -> void {
        const int id = piece_to_id(w);
        // std::cout << "piece: " << w << ", id = " << id << std::endl;
        if (id == -1 || !is_unused(id)) {
            output->emplace_back(w, id);
            return;
        }
        const auto p = rev_merge.find(w);
        if (p == rev_merge.end()) {
            // This block will never be called, as `rev_merge` stores all the
            // resegmentation info for unused id.
            output->emplace_back(w, id);
            return;
        }
        // Recursively resegment left and right symbols.
        resegment(p->second.first, output);
        resegment(p->second.second, output);
    };
    EncodeResult output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        resegment(symbols[index].piece, &output);
    }
    return output;
}

void Sentencepiece::encode(const std::string& str, std::vector<int>& ids) {
    // For SentencePiece, replace all spaces with ▁ and add ▁ prefix to the beginning
    std::string normalized_str = "▁";
    for (char c : str) {
        if (c == ' ') {
            normalized_str += "▁";
        } else {
            normalized_str += c;
        }
    }
    auto result = bpe_encode(normalized_str);
    size_t consumed = 0;
    for (const auto &p : result) {
        const string_view_ w = p.first;   // piece
        const int id = p.second;              // id
        const bool is_unk = (id == unk_id_);
        if (is_unk && byte_fall_back_) {
            // Decomposes an unknown piece into UTF-8 bytes
            for (int i = 0; i < w.size(); ++i) {
                // Create a byte piece
                const char b = w[i];
                const auto piece = byte_to_piece(b);
                auto sp_id = piece_to_id(piece);
                ids.push_back(sp_id);
            }
        } else {
            ids.push_back(id);
        }
    }
}

std::string Sentencepiece::decode(int id) {
    if (id < 0 || id >= static_cast<int>(sentence_pieces_.size())) {
        return "";
    }
    auto piece = sentence_pieces_[id].piece;
    int pos = piece.find("▁");
    if (pos != -1) {
        piece.replace(pos, pos + 3, " ");
    }
    return piece;
}

float Sentencepiece::get_score(int id) const {
    return sentence_pieces_[id].score;
}

bool Sentencepiece::is_unused(int id) const {
    return sentence_pieces_[id].type == PieceType::UNUSED;
}

bool Sentencepiece::is_control(int id) const {
    return sentence_pieces_[id].type == PieceType::CONTROL;
}

bool Tiktoken::load_vocab(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    // load vocab
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        auto token = base64_decode(line);
        encoder_.insert({token, i});
        decoder_[i] = token;
    }
    return true;
}

void Tiktoken::encode(const std::string& str, std::vector<int>& ids) {
    if (str.empty()) {
        return;
    }
    auto it = str.begin();
    while(it!=str.end()) {
        auto last_it = it;
        int token_id = encoder_.find(it, str.end());
        if (token_id>=0) { ids.push_back(token_id); }
        else {
            MNN_ERROR("Error: No encoding found for the sequence %s\n", std::string(last_it, it).c_str());
        }
    }
}

std::string Tiktoken::decode(int id) {
    if (id < 0 || id >= static_cast<int>(decoder_.size())) {
        return "";
    }
    return decoder_[id];
}

bool BertTokenizer::load_vocab(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    // load vocab
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        auto token = base64_decode(line);
        encoder_.insert({token, i});
        decoder_[i] = token;
    }
    return true;
}

std::string BertTokenizer::decode(int id) {
    if (id < 0 || id >= static_cast<int>(decoder_.size())) {
        return "";
    }
    return decoder_[id];
}

std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    // First check if the entire token exists in vocabulary
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
        return {it->second};
    }

    std::vector<int> ids;
    std::string current = token;
    bool is_first_piece = true;

    std::string candidate;
    candidate.reserve(token.size() + 2);

    while (!current.empty()) {
        int match_id = -1;
        size_t match_pos = 0;

        // Try to find the longest matching piece in vocabulary
        // Start from the full length and work backwards
        for (size_t len = current.size(); len > 0; --len) {
            candidate.clear();
            // Add ## prefix for sub-word pieces (not the first piece)
            if (!is_first_piece) {
                candidate.append("##");
            }
            candidate.append(current.data(), len);

            auto vocab_it = encoder_.find(candidate);
            if (vocab_it != encoder_.end()) {
                match_id = vocab_it->second;
                match_pos = len;
                break;
            }
        }

        // If no match found, use [UNK] token
        if (match_id == -1) {
            // Try to find [UNK] token, commonly has id 100 or 0
            auto unk_it = encoder_.find("[UNK]");
            if (unk_it != encoder_.end()) {
                ids.push_back(unk_it->second);
            } else {
                // Fallback to id 100 which is commonly used for [UNK]
                ids.push_back(100);
            }
            break;
        }

        ids.push_back(match_id);
        current = current.substr(match_pos);
        is_first_piece = false;  // Subsequent pieces need ## prefix
    }

    return ids;
}

void BertTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    // Use a simpler approach that matches Python tokenizer behavior more closely
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;

    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);

        // Handle UTF-8 multi-byte characters (including Chinese characters)
        if ((c & 0x80) != 0) {
            // Determine UTF-8 character length
            int char_len = 1;
            if ((c & 0xE0) == 0xC0) {
                char_len = 2;  // 2-byte character
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;  // 3-byte character (most Chinese characters)
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;  // 4-byte character
            }

            // Extract the complete UTF-8 character
            if (i + char_len <= str.size()) {
                current_token = str.substr(i, char_len);
                i += char_len;
            } else {
                // Invalid UTF-8 sequence, skip this byte
                ++i;
                continue;
            }
        }
        // Handle ASCII letters and digits - collect consecutive alphanumeric characters
        else if (isalnum(c)) {
            while (i < str.size() && isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += tolower(str[i]);
                ++i;
            }
        }
        // Handle punctuation and symbols - treat each as separate token
        else if (ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // Skip whitespace characters
        else if (isspace(c)) {
            ++i;
            continue;
        }
        // Handle any other single-byte characters
        else {
            current_token = str[i];
            ++i;
        }

        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }

    // Apply WordPiece algorithm to each token
    for (const auto& token : tokens) {
        std::vector<int> token_ids = word_piece(token);
        for (int id : token_ids) {
            ids.push_back(id);
        }
    }
}

std::wstring utf8_to_wstring(const char* str, size_t len) {
    if (len == 0) return std::wstring();

    std::wstring wstr;
    wstr.reserve(len);

    const char* p = str;
    const char* end = str + len;

    while (p < end) {
        unsigned char c = static_cast<unsigned char>(*p);

        if (c < 0x80) {
            wstr.push_back(static_cast<wchar_t>(c));
            ++p;
        } else if (c < 0xE0) {
            if (p + 1 < end) {
                wstr.push_back(static_cast<wchar_t>(
                    ((c & 0x1F) << 6) | (static_cast<unsigned char>(p[1]) & 0x3F)
                ));
            }
            p += 2;
        } else if (c < 0xF0) {
            if (p + 2 < end) {
                wstr.push_back(static_cast<wchar_t>(
                    ((c & 0x0F) << 12) |
                    ((static_cast<unsigned char>(p[1]) & 0x3F) << 6) |
                    (static_cast<unsigned char>(p[2]) & 0x3F)
                ));
            }
            p += 3;
        } else if (c < 0xF8) {
            if (p + 3 < end) {
                unsigned int cp = ((c & 0x07) << 18) |
                                  ((static_cast<unsigned char>(p[1]) & 0x3F) << 12) |
                                  ((static_cast<unsigned char>(p[2]) & 0x3F) << 6) |
                                  (static_cast<unsigned char>(p[3]) & 0x3F);

                if (sizeof(wchar_t) == 2) {
                    // Windows: Surrogate pairs for code points > 0xFFFF
                    if (cp > 0xFFFF) {
                        cp -= 0x10000;
                        wstr.push_back(static_cast<wchar_t>(0xD800 + (cp >> 10)));
                        wstr.push_back(static_cast<wchar_t>(0xDC00 + (cp & 0x3FF)));
                    } else {
                        wstr.push_back(static_cast<wchar_t>(cp));
                    }
                } else {
                    // Linux/macOS: Direct UTF-32
                    wstr.push_back(static_cast<wchar_t>(cp));
                }
            }
            p += 4;
        } else {
            ++p;
        }
    }
    return wstr;
}

std::string wstring_to_utf8(const std::wstring& str) {
    if (str.empty()) return std::string();
    std::string res;
    res.reserve(str.size() * 3);

    const wchar_t* p = str.data();
    const wchar_t* end = p + str.size();

    while (p < end) {
        unsigned int cp = static_cast<unsigned int>(*p);
        p++;
        if (sizeof(wchar_t) == 2) {
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (p < end) {
                    unsigned int low = static_cast<unsigned int>(*p);
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        p++;
                    }
                }
            }
        }
        if (cp < 0x80) {
            res.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            res.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            res.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            res.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            res.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            res.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            res.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            res.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            res.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            res.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    return res;
}

// Given a token as a UTF8 string, encode each byte into an wchar_t
void byte_encode_token(const std::string& token,
                       const std::unordered_map<uint8_t, wchar_t>& b2u,
                       std::wstring* result) {
    result->resize(0);
    for (char c : token) {
        wchar_t wc = b2u.at(uint8_t(c));
        result->push_back(wc);
    }
}

bool HuggingfaceTokenizer::load_vocab(std::ifstream& tok_file) {
    std::string line;
    line.reserve(256); // Reduce allocation during getline

    // 1. Get nums
    int vocab_len = 0;
    int merge_len = 0;
    if (std::getline(tok_file, line)) {
        std::istringstream line_str(line);
        line_str >> vocab_len >> merge_len;
    }

    // 2. Load vocab
    decoder_.resize(vocab_len);
    encoder_.reserve(vocab_len);

    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        // Move string to decoder to avoid copy, then use reference for encoder
        decoder_[i] = std::move(line);
        encoder_.emplace(decoder_[i], i);
    }

    // 3. Load merge_rules
    bpe_ranks_.reserve(merge_len);
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);

        size_t d = line.find(' ');
        if (d != std::string::npos) {
            // Use pointer-based conversion to avoid creating temporary substr strings
            bpe_ranks_.emplace(std::make_pair(
                utf8_to_wstring(line.data(), d),
                utf8_to_wstring(line.data() + d + 1, line.size() - d - 1)
            ), i);
        }
    }

    // 4. bytes_to_unicode initialization
    // Use a temporary local vector for O(1) access during construction
    std::vector<wchar_t> temp_map(256, 0);

    auto set_range = [&](int start, int end) {
        for (int c = start; c <= end; c++) {
            temp_map[c] = static_cast<wchar_t>(c);
        }
    };

    set_range(L'!', L'~');
    set_range(L'¡', L'¬');
    set_range(L'®', L'ÿ');

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (temp_map[b] == 0) {
            temp_map[b] = static_cast<wchar_t>(256 + n);
            n++;
        }
    }

    // Batch insert into member maps
    b2u_.clear();
    u2b_.clear();
    // Hint: Assuming typical map implementations, insertion order matters slightly,
    // but just bulk inserting is clean enough.
    for (int i = 0; i < 256; ++i) {
        uint8_t u8 = static_cast<uint8_t>(i);
        wchar_t wc = temp_map[i];
        b2u_.emplace(u8, wc);
        u2b_.emplace(wc, u8);
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

void HuggingfaceTokenizer::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
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

void HuggingfaceTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    /* original regex from tokenizer.json
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
     //    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
     */
    static const std::regex re("('s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n[:alpha:][:digit:]]?[[:alpha:]]+|[[:digit:]]| ?[^\\s[:alpha:][:digit:]]+[\r\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", std::regex_constants::icase);

    std::string input = str;
    std::vector<std::string> result;
    std::smatch match;

    std::string token;
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
    for (auto s : result) {
        ids.push_back(encoder_.at(s));
    }
}

std::string HuggingfaceTokenizer::decode(int id) {
    // printf("decode id = %d, %lu, %s#\n", id, decoder_.size(), decoder_.at(id).c_str());
    if (id < 0 || id >= static_cast<int>(decoder_.size())) {
        return "";
    }
    auto decode_utf8 = decoder_.at(id);
    std::wstring w = utf8_to_wstring(decode_utf8.data(), decode_utf8.size());
    std::string r;
    for (wchar_t c : w) {
        if (u2b_.find(c) != u2b_.end()) {
            r.push_back(char(u2b_.at(c)));
        }
    }
    return r;
}
}
}
