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
#ifdef LLM_USE_JINJA
#include "jinja.hpp"
#include "../ujson.hpp"
#endif
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
    // AUTOTIME;
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
    if (tokenizer_type == PIPELINE) {
        // .mtok binary format (PipelineTokenizer)
        auto* pt = new PipelineTokenizer();
        tokenizer = pt;
        tokenizer->load_special(tok_file);
        auto pos = tok_file.tellg();
        tok_file.close();
        std::ifstream bin_file(filename, std::ios::binary);
        bin_file.seekg(pos);
        pt->load_vocab_binary(bin_file);
        bin_file.close();
        tokenizer->cache_special_tokens();
        return tokenizer;
    }
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

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) result += decode(id);
    return result;
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

// Tokenizer chat template methods
void Tokenizer::set_chat_template(const std::string& tpl, const std::string& eos, const std::string& context) {
    chat_template_ = tpl;
    chat_template_eos_ = eos;
    chat_template_context_ = context;
}

std::string Tokenizer::apply_chat_template(const ChatMessages& messages, bool add_generation_prompt) const {
    if (chat_template_.empty()) {
        std::string result;
        for (const auto& m : messages) {
            result += m.second;
        }
        return result;
    }
#ifdef LLM_USE_JINJA
    jinja::json default_ctx = jinja::json::object();
    default_ctx["eos_token"] = chat_template_eos_;
    if (!chat_template_bos_.empty()) {
        default_ctx["bos_token"] = chat_template_bos_;
    }
    jinja::Template tpl(chat_template_, default_ctx);
    jinja::json msgs = jinja::json::array();
    for (const auto& m : messages) {
        if (m.first == "json") {
            auto parsed = jinja::json::parse(m.second);
            if (parsed.is_object()) {
                msgs.push_back(parsed);
                continue;
            }
        }
        jinja::json msg = jinja::json::object();
        msg["role"] = m.first;
        msg["content"] = m.second;
        msgs.push_back(msg);
    }
    jinja::json extra_ctx = jinja::json::object();
    if (!chat_template_context_.empty()) {
        auto parsed = jinja::json::parse(chat_template_context_);
        if (parsed.is_object()) {
            extra_ctx = parsed;
        }
    }
    return tpl.apply_chat_template(msgs, add_generation_prompt, jinja::json::array(), extra_ctx);
#else
    std::string result;
    for (const auto& m : messages) {
        result += m.second;
    }
    return result;
#endif
}

std::string Tokenizer::apply_chat_template(const std::string& user_content, const std::string& system_prompt) const {
    ChatMessages messages;
    if (!system_prompt.empty()) {
        messages.push_back({"system", system_prompt});
    }
    messages.push_back({"user", user_content});
    return apply_chat_template(messages, true);
}

// ==========================================
// PipelineTokenizer Implementation
// ==========================================

} // namespace Transformer
} // namespace MNN

#include "unicode.hpp"

namespace MNN {
namespace Transformer {

template<typename T, typename... Args>
static std::unique_ptr<T> make_unique_(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// ==========================================
// Utils
// ==========================================


static std::vector<std::string> create_bytes_char_map() {
    auto u2u = [](int cp) -> std::string {
        std::string out;
        if (cp <= 0x7F) out += (char)cp;
        else if (cp <= 0x7FF) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp <= 0xFFFF) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        return out;
    };
    std::vector<std::string> bs(256);
    std::unordered_map<unsigned char, std::string> temp_bs;
    for (int b = 33; b <= 126; ++b) temp_bs[(unsigned char)b] = u2u(b);
    for (int b = 161; b <= 172; ++b) temp_bs[(unsigned char)b] = u2u(b);
    for (int b = 174; b <= 255; ++b) temp_bs[(unsigned char)b] = u2u(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (temp_bs.find((unsigned char)b) == temp_bs.end()) temp_bs[(unsigned char)b] = u2u(256 + n++);
        bs[b] = temp_bs[(unsigned char)b];
    }
    return bs;
}


// ==========================================
// Normalizer Implementations
// ==========================================

class NFKCNormalizer : public Normalizer {
    std::vector<std::pair<uint32_t, std::string>> table_;
public:
    NFKCNormalizer() = default;
    NFKCNormalizer(std::vector<std::pair<uint32_t, std::string>>&& t) : table_(std::move(t)) {}
    std::string normalize(const std::string& text) const override {
        if (table_.empty()) return text;
        std::string out;
        out.reserve(text.size());
        const uint8_t* ptr = (const uint8_t*)text.data();
        size_t len = text.size(), i = 0;
        while (i < len) {
            int32_t cp;
            int r = Unicode::utf8_decode(ptr + i, len - i, &cp);
            // SentencePiece replaces control/format chars (Cc/Cf) with space,
            // except for common whitespace chars that are kept as-is
            if (cp != 0 && cp != '\n' && cp != '\r' && cp != '\t' &&
                ((cp >= 1 && cp <= 0x1F) || cp == 0x7F ||  // Cc: C0/C1 control
                 (cp >= 0x80 && cp <= 0x9F) ||              // Cc: C1 control
                 cp == 0x200D || cp == 0x200C ||             // Cf: ZWJ, ZWNJ
                 cp == 0x200B || cp == 0xFEFF ||             // Cf: ZWSP, BOM
                 (cp >= 0x200E && cp <= 0x200F) ||           // Cf: LRM, RLM
                 (cp >= 0x202A && cp <= 0x202E) ||           // Cf: bidi controls
                 (cp >= 0x2060 && cp <= 0x2069) ||           // Cf: word joiner etc.
                 (cp >= 0xFFF0 && cp <= 0xFFF8))) {          // Cf: specials
                out += ' ';
            } else {
                // binary search in table_
                size_t lo = 0, hi = table_.size();
                const std::string* found = nullptr;
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if ((uint32_t)cp > table_[mid].first) lo = mid + 1;
                    else if ((uint32_t)cp < table_[mid].first) hi = mid;
                    else { found = &table_[mid].second; break; }
                }
                if (found) {
                    out += *found;
                } else {
                    out.append((const char*)ptr + i, r);
                }
            }
            i += r;
        }
        return out;
    }
};

class PrependNormalizer : public Normalizer {
    std::string prepend_;
public:
    PrependNormalizer(const std::string& p) : prepend_(p) {}
    std::string normalize(const std::string& text) const override { return prepend_ + text; }
};

class StripNormalizer : public Normalizer {
    bool strip_left_;
    bool strip_right_;

public:
    StripNormalizer(bool left, bool right) : strip_left_(left), strip_right_(right) {}
    std::string normalize(const std::string& text) const override {
        if (text.empty()) {
            return text;
        }
        size_t begin = 0;
        size_t end = text.size();

        if (strip_left_) {
            while (begin < end) {
                int32_t cp = 0;
                int r = Unicode::utf8_decode((const uint8_t*)text.data() + begin, end - begin, &cp);
                if (r <= 0 || !Unicode::is_whitespace(cp)) {
                    break;
                }
                begin += (size_t)r;
            }
        }

        if (strip_right_) {
            while (begin < end) {
                size_t pos = end;
                while (pos > begin && (((unsigned char)text[pos - 1] & 0xC0) == 0x80)) {
                    pos--;
                }
                int32_t cp = 0;
                int r = Unicode::utf8_decode((const uint8_t*)text.data() + pos, end - pos, &cp);
                if (r <= 0 || !Unicode::is_whitespace(cp)) {
                    break;
                }
                end = pos;
            }
        }

        return text.substr(begin, end - begin);
    }
};

class ReplaceNormalizer : public Normalizer {
    std::string pattern_, content_;
public:
    ReplaceNormalizer(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    std::string normalize(const std::string& text) const override {
        if (pattern_.empty()) return text;
        std::string out = text;
        size_t pos = 0;
        while ((pos = out.find(pattern_, pos)) != std::string::npos) {
            out.replace(pos, pattern_.length(), content_);
            pos += content_.length();
        }
        return out;
    }
};

class SequenceNormalizer : public Normalizer {
    std::vector<std::shared_ptr<Normalizer>> normalizers_;
public:
    SequenceNormalizer(const std::vector<std::shared_ptr<Normalizer>>& n) : normalizers_(n) {}
    std::string normalize(const std::string& text) const override {
        std::string out = text;
        for (const auto& n : normalizers_) {
            if (!n) {
                MNN_ERROR("[Tokenizer] SequenceNormalizer got null normalizer, skipping.\n");
                continue;
            }
            out = n->normalize(out);
        }
        return out;
    }
};

class BertNormalizer : public Normalizer {
    bool clean_text_, handle_chinese_chars_, strip_accents_, lowercase_;
    std::vector<std::pair<uint32_t, std::string>> nfd_table_;
public:
    BertNormalizer(bool clean = true, bool chinese = true, bool accents = false, bool lower = true,
                   std::vector<std::pair<uint32_t, std::string>>&& nfd = std::vector<std::pair<uint32_t, std::string>>())
        : clean_text_(clean), handle_chinese_chars_(chinese), strip_accents_(accents), lowercase_(lower),
          nfd_table_(std::move(nfd)) {}

    std::string normalize(const std::string& text) const override {
        std::string out;
        const uint8_t* ptr = (const uint8_t*)text.c_str();
        size_t len = text.length(), i = 0;
        int32_t cp;
        while (i < len) {
            int r = Unicode::utf8_decode(ptr + i, len - i, &cp);
            if (r <= 0) { i++; continue; }
            std::string ch((const char*)ptr + i, r);

            if (clean_text_) {
                if (cp == '\t' || cp == '\n' || cp == '\r' || Unicode::get_category(cp) == Unicode::CAT_Zs) {
                    out += ' '; i += r; continue;
                }
                if (cp == 0 || cp == 0xFFFD || Unicode::get_category(cp) == Unicode::CAT_Cc) { i += r; continue; }
            }

            if (handle_chinese_chars_ && is_chinese_char(cp)) {
                out += ' '; out += ch; out += ' ';
                i += r; continue;
            }

            if (strip_accents_ && !nfd_table_.empty()) {
                std::string decomposed = nfd_lookup(ch);
                const uint8_t* dptr = (const uint8_t*)decomposed.data();
                size_t dlen = decomposed.size(), dj = 0;
                while (dj < dlen) {
                    int32_t dcp;
                    int dr = Unicode::utf8_decode(dptr + dj, dlen - dj, &dcp);
                    if (dr <= 0) break;
                    if (!Unicode::is_mark(dcp)) {
                        out.append((const char*)dptr + dj, dr);
                    }
                    dj += dr;
                }
                i += r; continue;
            }

            out += ch;
            i += r;
        }
        if (lowercase_) {
            std::string lower_out;
            ptr = (const uint8_t*)out.c_str();
            len = out.length(); i = 0;
            while (i < len) {
                int r = Unicode::utf8_decode(ptr + i, len - i, &cp);
                if (r <= 0) { lower_out += out[i++]; continue; }
                int32_t lc = Unicode::to_lower(cp);
                char buf[8]; int n = Unicode::utf8_encode(lc, buf);
                lower_out.append(buf, n);
                i += r;
            }
            return lower_out;
        }
        return out;
    }
private:
    std::string nfd_lookup(const std::string& ch) const {
        const uint8_t* ptr = (const uint8_t*)ch.data();
        size_t len = ch.size(), i = 0;
        std::string out;
        out.reserve(len * 2);
        while (i < len) {
            int32_t cp;
            int r = Unicode::utf8_decode(ptr + i, len - i, &cp);
            size_t lo = 0, hi = nfd_table_.size();
            const std::string* found = nullptr;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if ((uint32_t)cp > nfd_table_[mid].first) lo = mid + 1;
                else if ((uint32_t)cp < nfd_table_[mid].first) hi = mid;
                else { found = &nfd_table_[mid].second; break; }
            }
            if (found) {
                out += *found;
            } else {
                out.append((const char*)ptr + i, r);
            }
            i += r;
        }
        return out;
    }
    static bool is_chinese_char(int32_t cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
               (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
};

// ==========================================
// PreTokenizer Implementations
// ==========================================

class SequencePreTokenizer : public PreTokenizer {
public:
    std::vector<std::shared_ptr<PreTokenizer>> pts_;
    SequencePreTokenizer(const std::vector<std::shared_ptr<PreTokenizer>>& pts) : pts_(pts) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        for (const auto& pt : pts_) {
            if (!pt) {
                MNN_ERROR("[Tokenizer] SequencePreTokenizer got null component, skipping.\n");
                continue;
            }
            pt->pre_tokenize(pts);
        }
    }
};

class ByteLevelPreTokenizer : public PreTokenizer {
    bool use_regex_;
    std::string gpt2_pattern_;
public:
    ByteLevelPreTokenizer(bool use_regex = false) : use_regex_(use_regex),
        gpt2_pattern_("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+") {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        if (use_regex_) {
            std::vector<std::string> next_splits;
            for (const auto& s : pts.splits) {
                if (s.empty()) continue;
                auto tokens = Unicode::regex_scanner(s, gpt2_pattern_);
                next_splits.insert(next_splits.end(), tokens.begin(), tokens.end());
            }
            pts.splits = next_splits;
        }
        static auto byte_map = create_bytes_char_map();
        for (auto& s : pts.splits) {
            std::string out;
            for (unsigned char b : s) out += byte_map[b];
            s = out;
        }
    }
};

class DigitsPreTokenizer : public PreTokenizer {
    bool individual_digits_;
public:
    DigitsPreTokenizer(bool id) : individual_digits_(id) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> next_splits;
        for (const auto& s : pts.splits) {
            std::string current;
            for (size_t i = 0; i < s.length(); ) {
                int32_t cp;
                int clen = Unicode::utf8_decode((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (clen <= 0) break;
                std::string c = s.substr(i, clen);
                bool is_digit = (c.length() == 1 && c[0] >= '0' && c[0] <= '9');
                if (is_digit && individual_digits_) {
                    if (!current.empty()) { next_splits.push_back(current); current.clear(); }
                    next_splits.push_back(c);
                } else {
                    current += c;
                }
                i += clen;
            }
            if (!current.empty()) next_splits.push_back(current);
        }
        pts.splits = next_splits;
    }
};

class MetaspacePreTokenizer : public PreTokenizer {
public:
    std::string replacement_;
    bool add_prefix_space_;
    MetaspacePreTokenizer(const std::string& rep, bool aps) : replacement_(rep), add_prefix_space_(aps) {}
    void pre_tokenize(PreTokenizedString& pts) const override {
        for (auto& s : pts.splits) {
            if (add_prefix_space_ && !s.empty() && s[0] != ' ') {
                s = " " + s;
            }
            std::string out;
            for (size_t i = 0; i < s.size();) {
                int32_t cp;
                int clen = Unicode::utf8_decode((const uint8_t*)s.data() + i, s.size() - i, &cp);
                if (clen <= 0) break;
                std::string c = s.substr(i, clen);
                out += (c == " ") ? replacement_ : c;
                i += clen;
            }
            s = out;
        }
    }
};

class SplitPreTokenizer : public PreTokenizer {
public:
    std::string pattern_;
    bool invert_;
    std::string behavior_;

    SplitPreTokenizer(const std::string& pattern, bool invert, const std::string& behavior = "Isolated")
        : pattern_(pattern), invert_(invert), behavior_(behavior) {}

    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> new_splits;
        for (const auto& s : pts.splits) {
            if (s.empty()) continue;
            auto tokens = Unicode::regex_split(s, pattern_, invert_, behavior_);
            new_splits.insert(new_splits.end(), tokens.begin(), tokens.end());
        }
        pts.splits = new_splits;
    }
};

class BertPreTokenizer : public PreTokenizer {
public:
    void pre_tokenize(PreTokenizedString& pts) const override {
        std::vector<std::string> new_splits;
        for (const auto& s : pts.splits) {
            std::string current;
            const uint8_t* ptr = (const uint8_t*)s.c_str();
            size_t len = s.length(), i = 0;
            int32_t cp;
            while (i < len) {
                int r = Unicode::utf8_decode(ptr + i, len - i, &cp);
                if (r <= 0) { i++; continue; }
                std::string ch((const char*)ptr + i, r);
                bool ws = Unicode::is_whitespace(cp);
                bool punc = Unicode::is_punctuation(cp) ||
                            (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
                            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
                if (ws) {
                    if (!current.empty()) { new_splits.push_back(current); current.clear(); }
                } else if (punc) {
                    if (!current.empty()) { new_splits.push_back(current); current.clear(); }
                    new_splits.push_back(ch);
                } else {
                    current += ch;
                }
                i += r;
            }
            if (!current.empty()) new_splits.push_back(current);
        }
        pts.splits = new_splits;
    }
};

// ==========================================
// Model Implementations
// ==========================================

static inline uint64_t merge_key(int a, int b) {
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

struct StringRef {
    const char* ptr;
    uint16_t len;
    std::string str() const { return std::string(ptr, len); }
    bool operator==(const StringRef& o) const { return len == o.len && memcmp(ptr, o.ptr, len) == 0; }
};

struct VocabEntry {
    const char* str;
    uint16_t len;
    int id;
};

static inline bool vocab_entry_less(const VocabEntry& a, const VocabEntry& b) {
    int c = memcmp(a.str, b.str, std::min(a.len, b.len));
    return c < 0 || (c == 0 && a.len < b.len);
}

static inline int vocab_find(const std::vector<VocabEntry>& v, const char* s, uint16_t len) {
    VocabEntry key = {s, len, 0};
    auto it = std::lower_bound(v.begin(), v.end(), key, vocab_entry_less);
    if (it != v.end() && it->len == len && memcmp(it->str, s, len) == 0) return it->id;
    return -1;
}

class BPEModel : public TokenizerModel {
public:
    bool use_byte_level_;
    std::vector<StringRef> id_to_token_;
    std::vector<VocabEntry> sorted_vocab_;
    std::vector<std::pair<uint64_t, int>> sorted_merges_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    int find_merge(uint64_t key) const {
        auto it = std::lower_bound(sorted_merges_.begin(), sorted_merges_.end(), key,
            [](const std::pair<uint64_t, int>& p, uint64_t k) { return p.first < k; });
        if (it != sorted_merges_.end() && it->first == key) return it->second;
        return -1;
    }

    // Constructor: zero-copy, buf owned by PipelineTokenizer::binary_buf_
    BPEModel(std::vector<StringRef>&& id_to_token,
             std::vector<VocabEntry>&& sorted_vocab,
             std::vector<std::pair<uint64_t, int>>&& merges,
             bool use_byte_level)
        : id_to_token_(std::move(id_to_token)),
          sorted_vocab_(std::move(sorted_vocab)), sorted_merges_(std::move(merges)),
          use_byte_level_(use_byte_level) {}

    int token_to_id(const std::string& token) const {
        return vocab_find(sorted_vocab_, token.c_str(), (uint16_t)token.size());
    }
    int token_to_id(const char* s, uint16_t len) const {
        return vocab_find(sorted_vocab_, s, len);
    }
    std::string id_to_token(int id) const override {
        if (id >= 0 && id < (int)id_to_token_.size()) return id_to_token_[id].str();
        return "";
    }
    size_t vocab_size() const override { return id_to_token_.size(); }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty()) return {};
        auto cit = cache_.find(text);
        if (cit != cache_.end()) return cit->second;

        std::vector<int> out;
        if (use_byte_level_) {
            static auto byte_map = create_bytes_char_map();
            for (unsigned char b : text) {
                int id = token_to_id(byte_map[b]);
                if (id != -1) out.push_back(id);
            }
        } else {
            const uint8_t* ptr = (const uint8_t*)text.c_str();
            size_t len = text.length(), off = 0;
            int32_t cp;
            while (off < len) {
                int ret = Unicode::utf8_decode(ptr + off, len - off, &cp);
                if (cp == 0xFFFD && ret == 1) {
                    char buf[16]; snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off]);
                    int id = token_to_id(buf); if (id != -1) out.push_back(id);
                    off++; continue;
                }
                int id = token_to_id((const char*)ptr + off, (uint16_t)ret);
                if (id != -1) out.push_back(id);
                else {
                    for (int i = 0; i < ret; ++i) {
                        char buf[16]; snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ptr[off + i]);
                        int bid = token_to_id(buf); if (bid != -1) out.push_back(bid);
                    }
                }
                off += ret;
            }
        }
        while (out.size() > 1) {
            int best = -1, min_r = (int)1e9;
            for (size_t i = 0; i < out.size() - 1; ++i) {
                int rank = find_merge(merge_key(out[i], out[i + 1]));
                if (rank >= 0 && rank < min_r) { min_r = rank; best = (int)i; }
            }
            if (best == -1) break;
            std::string m = id_to_token(out[best]) + id_to_token(out[best + 1]);
            int nid = token_to_id(m); if (nid == -1) break;
            out[best] = nid; out.erase(out.begin() + best + 1);
        }
        cache_[text] = out;
        return out;
    }
};

class WordPieceModel : public TokenizerModel {
    std::string unk_token_;
    std::string continuing_subword_prefix_;
    int max_input_chars_per_word_;
    std::vector<StringRef> id_to_token_;
    std::vector<VocabEntry> sorted_vocab_;
    int unk_token_id_;
public:
    WordPieceModel(const std::string& unk = "[UNK]", const std::string& prefix = "##", int max_chars = 100)
        : unk_token_(unk), continuing_subword_prefix_(prefix), max_input_chars_per_word_(max_chars), unk_token_id_(-1) {}

    void load_direct_sorted(std::vector<StringRef>&& id_to_token, std::vector<VocabEntry>&& sorted_vocab) {
        id_to_token_ = std::move(id_to_token);
        sorted_vocab_ = std::move(sorted_vocab);
        unk_token_id_ = token_to_id(unk_token_);
    }

    int token_to_id(const std::string& token) const {
        return vocab_find(sorted_vocab_, token.c_str(), (uint16_t)token.size());
    }

    std::string id_to_token(int id) const override {
        if (id >= 0 && id < (int)id_to_token_.size()) return id_to_token_[id].str();
        return unk_token_;
    }

    size_t vocab_size() const override { return id_to_token_.size(); }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty()) return {};
        if ((int)text.length() > max_input_chars_per_word_) {
            return unk_token_id_ != -1 ? std::vector<int>{unk_token_id_} : std::vector<int>{};
        }
        std::vector<int> out;
        size_t start = 0;
        bool is_bad = false;

        while (start < text.length()) {
            size_t end = text.length();
            int cur_id = -1;

            while (end > start) {
                std::string substr = text.substr(start, end - start);
                if (start > 0) substr = continuing_subword_prefix_ + substr;
                int id = token_to_id(substr);
                if (id != -1 && id != unk_token_id_) {
                    cur_id = id;
                    break;
                }
                end--;
            }

            if (cur_id == -1) {
                is_bad = true;
                break;
            } else {
                out.push_back(cur_id);
                start = end;
            }
        }

        if (is_bad) return { unk_token_id_ };
        return out;
    }
};

class UnigramModel : public TokenizerModel {
    std::string unk_token_;
    int unk_token_id_;
    std::vector<StringRef> id_to_token_;
    std::vector<VocabEntry> sorted_vocab_;
    std::vector<double> scores_;
    bool byte_fallback_;
    size_t max_token_len_;

public:
    UnigramModel(int unk_id = 0, bool byte_fallback = false)
        : unk_token_id_(unk_id), byte_fallback_(byte_fallback), max_token_len_(0) {}

    void load_direct_sorted(std::vector<StringRef>&& id_to_token, std::vector<double>&& scores,
                            std::vector<VocabEntry>&& sorted_vocab) {
        id_to_token_ = std::move(id_to_token);
        scores_ = std::move(scores);
        sorted_vocab_ = std::move(sorted_vocab);
        int n = (int)id_to_token_.size();
        max_token_len_ = 0;
        for (int i = 0; i < n; i++) {
            if (id_to_token_[i].len > max_token_len_) max_token_len_ = id_to_token_[i].len;
            if (i == unk_token_id_) unk_token_ = id_to_token_[i].str();
        }
    }

    int token_to_id(const std::string& token) const {
        return vocab_find(sorted_vocab_, token.c_str(), (uint16_t)token.size());
    }

    std::string id_to_token(int id) const override {
        if (id >= 0 && id < (int)id_to_token_.size()) return id_to_token_[id].str();
        return unk_token_;
    }

    size_t vocab_size() const override { return id_to_token_.size(); }

    std::vector<int> tokenize(const std::string& text) const override {
        if (text.empty()) return {};

        size_t n = text.length();
        std::vector<double> best_scores(n + 1, -1e18);
        std::vector<int> best_ids(n + 1, -1);
        std::vector<size_t> best_prev_pos(n + 1, 0);

        best_scores[0] = 0.0;

        for (size_t i = 1; i <= n; ++i) {
            size_t start_len = (i > max_token_len_) ? (i - max_token_len_) : 0;
            for (size_t j = i - 1; j != (size_t)-1 && j >= start_len; --j) {
                if (best_scores[j] <= -1e17) continue;

                std::string sub = text.substr(j, i - j);
                int found_id = token_to_id(sub);

                int token_id = -1;
                double score = -1e18;

                if (found_id != -1 && found_id != unk_token_id_) {
                    token_id = found_id;
                    score = scores_[token_id];
                } else if (byte_fallback_ && (i - j) == 1) {
                    unsigned char b = (unsigned char)text[j];
                    char buf[16];
                    snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    int bf_id = token_to_id(buf);
                    if (bf_id != unk_token_id_) {
                        token_id = bf_id;
                        score = scores_[token_id];
                    } else {
                        token_id = unk_token_id_;
                        score = (unk_token_id_ < (int)scores_.size()) ? scores_[unk_token_id_] : -10.0;
                    }
                } else {
                    continue;
                }

                double new_score = best_scores[j] + score;
                if (new_score > best_scores[i] || best_scores[i] <= -1e17) {
                    best_scores[i] = new_score;
                    best_prev_pos[i] = j;
                    best_ids[i] = token_id;
                }
            }

            if (best_scores[i] <= -1e17) {
                int char_len = 1;
                for (int k = 1; k <= 4 && (int)i - k >= 0; ++k) {
                    unsigned char c = (unsigned char)text[i - k];
                    if ((c & 0xC0) != 0x80) {
                        int expected = 1;
                        if (c >= 0xF0) expected = 4;
                        else if (c >= 0xE0) expected = 3;
                        else if (c >= 0xC0) expected = 2;
                        if (expected == k) char_len = k;
                        break;
                    }
                }

                double prev_score = best_scores[i - char_len];
                if (prev_score > -1e17) {
                    double unk_score = (unk_token_id_ < (int)scores_.size()) ? scores_[unk_token_id_] : -10.0;
                    best_scores[i] = prev_score + unk_score;
                    best_prev_pos[i] = i - char_len;
                    best_ids[i] = unk_token_id_;
                }
            }
        }

        std::vector<int> out;
        if (best_scores[n] <= -1e17) return {};

        size_t cur = n;
        while (cur > 0) {
            int id = best_ids[cur];
            if (out.empty() || id != unk_token_id_ || out.back() != unk_token_id_) {
                out.push_back(id);
            }
            cur = best_prev_pos[cur];
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
};

// ==========================================
// Decoder Implementations
// ==========================================

class ReplaceDecoder : public TokenDecoder {
    std::string pattern_, content_;
public:
    ReplaceDecoder(const std::string& p, const std::string& c) : pattern_(p), content_(c) {}
    void decode_chain(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            size_t pos = 0;
            while ((pos = t.find(pattern_, pos)) != std::string::npos) {
                t.replace(pos, pattern_.length(), content_);
                pos += content_.length();
            }
        }
    }
};

class StripDecoder : public TokenDecoder {
    std::string content_;
    int start_, stop_;
public:
    StripDecoder(const std::string& c, int start, int stop) : content_(c), start_(start), stop_(stop) {}
    void decode_chain(std::vector<std::string>& tokens) const override {
        if (tokens.empty()) return;
        if (start_ > 0 && !tokens[0].empty() && tokens[0].find(content_) == 0) {
            tokens[0] = tokens[0].substr(content_.length());
        }
        if (stop_ > 0 && !tokens.back().empty()) {
            size_t pos = tokens.back().rfind(content_);
            if (pos != std::string::npos && pos + content_.length() == tokens.back().length()) {
                tokens.back() = tokens.back().substr(0, pos);
            }
        }
    }
};

class FuseDecoder : public TokenDecoder {
public:
    void decode_chain(std::vector<std::string>& tokens) const override {
        if (tokens.size() <= 1) return;
        std::string fused;
        for (const auto& t : tokens) fused += t;
        tokens = {fused};
    }
};

class ByteFallbackDecoder : public TokenDecoder {
public:
    void decode_chain(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            if (t.length() >= 3 && t.substr(0, 3) == "<0x") {
                int b; if (sscanf(t.c_str(), "<0x%02X>", &b) == 1) t = std::string(1, (char)b);
            }
        }
    }
};

class ByteLevelDecoder : public TokenDecoder {
public:
    void decode_chain(std::vector<std::string>& tokens) const override {
        static auto bm = []() {
            std::unordered_map<std::string, unsigned char> m;
            auto byte_vec = create_bytes_char_map();
            for (int i = 0; i < 256; ++i) {
                if (!byte_vec[i].empty()) m[byte_vec[i]] = (unsigned char)i;
            }
            return m;
        }();
        for (auto& t : tokens) {
            std::string out;
            for (size_t i = 0; i < t.length(); ) {
                const uint8_t* tp = (const uint8_t*)t.c_str();
                int32_t cp; int r = Unicode::utf8_decode(tp + i, t.length() - i, &cp);
                if (r > 0) {
                    std::string ch(t.substr(i, r)); auto it = bm.find(ch);
                    if (it != bm.end()) out += (char)it->second; else out += ch;
                    i += r;
                } else out += t[i++];
            }
            t = out;
        }
    }
};

class WordPieceDecoder : public TokenDecoder {
    std::string prefix_;
    bool cleanup_;
public:
    WordPieceDecoder(const std::string& prefix = "##", bool cleanup = true) : prefix_(prefix), cleanup_(cleanup) {}

    void decode_chain(std::vector<std::string>& tokens) const override {
        // HuggingFace WordPiece decoder: strip ## prefix, add space before non-suffix tokens
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].rfind(prefix_, 0) == 0) {
                tokens[i] = tokens[i].substr(prefix_.length());
            } else if (i > 0) {
                tokens[i] = " " + tokens[i];
            }
        }
        std::string out;
        for (const auto& t : tokens) out += t;
        tokens.clear();
        tokens.push_back(out);
    }
};

class MetaspaceDecoder : public TokenDecoder {
    std::string replacement_;
    bool add_prefix_space_;
public:
    MetaspaceDecoder(const std::string& rep = "\xe2\x96\x81", bool aps = true) : replacement_(rep), add_prefix_space_(aps) {}
    void decode_chain(std::vector<std::string>& tokens) const override {
        for (auto& t : tokens) {
            std::string out;
            size_t i = 0;
            while (i < t.length()) {
                if (t.substr(i, replacement_.length()) == replacement_) {
                    out += " ";
                    i += replacement_.length();
                } else {
                    out += t[i++];
                }
            }
            t = out;
        }
        if (add_prefix_space_ && !tokens.empty()) {
            if (!tokens[0].empty() && tokens[0][0] == ' ') {
                tokens[0] = tokens[0].substr(1);
            }
        }
    }
};

class SequenceDecoder : public TokenDecoder {
    std::vector<std::shared_ptr<TokenDecoder>> decoders_;
public:
    SequenceDecoder(const std::vector<std::shared_ptr<TokenDecoder>>& d) : decoders_(d) {}
    void decode_chain(std::vector<std::string>& tokens) const override {
        for (const auto& d : decoders_) {
            if (!d) {
                MNN_ERROR("[Tokenizer] SequenceDecoder got null component, skipping.\n");
                continue;
            }
            d->decode_chain(tokens);
        }
    }
};

// ==========================================
// PipelineTokenizer Methods
// ==========================================


PipelineTokenizer::PipelineTokenizer() = default;
PipelineTokenizer::~PipelineTokenizer() = default;

void PipelineTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    if (str.empty()) return;

    // Split by added tokens first
    std::vector<std::pair<std::string, bool>> units; // text, is_added_token
    size_t last = 0;
    while (last < str.length()) {
        if (added_token_strings_.empty()) {
            units.push_back({str.substr(last), false});
            break;
        }
        auto result = Unicode::multi_string_find(str, (int)last, added_token_strings_);
        int match_pos = result.first;
        int match_idx = result.second;
        if (match_pos >= 0 && match_idx >= 0) {
            const AddedToken& at = added_tokens_[match_idx];
            size_t prefix_start = last;
            size_t prefix_end = (size_t)match_pos;
            size_t next_start = (size_t)match_pos + at.content.size();

            if (at.lstrip) {
                while (prefix_end > prefix_start && isspace((unsigned char)str[prefix_end - 1])) prefix_end--;
            }
            if (at.rstrip) {
                while (next_start < str.length() && isspace((unsigned char)str[next_start])) next_start++;
            }

            if (prefix_end > prefix_start) units.push_back({str.substr(prefix_start, prefix_end - prefix_start), false});
            units.push_back({at.content, true});
            last = next_start;
        } else {
            units.push_back({str.substr(last), false});
            break;
        }
    }

    for (const auto& unit : units) {
        if (unit.second) {
            // Added token - find its id directly
            for (const auto& at : added_tokens_) {
                if (at.content == unit.first) { ids.push_back(at.id); break; }
            }
        } else {
            // Normal text - run through pipeline
            std::string normalized = normalizer_ ? normalizer_->normalize(unit.first) : unit.first;
            if (normalized.empty()) continue;
            PreTokenizedString pts; pts.splits.push_back(normalized);
            if (pre_tokenizer_) pre_tokenizer_->pre_tokenize(pts);
            for (const auto& s : pts.splits) {
                auto token_ids = model_->tokenize(s);
                ids.insert(ids.end(), token_ids.begin(), token_ids.end());
            }
        }
    }
}

static std::string str_replace_all(const std::string& s, const std::string& from, const std::string& to) {
    std::string out;
    size_t pos = 0;
    while (true) {
        size_t f = s.find(from, pos);
        if (f == std::string::npos) { out += s.substr(pos); break; }
        out += s.substr(pos, f - pos) + to;
        pos = f + from.size();
    }
    return out;
}

std::string PipelineTokenizer::decode(const std::vector<int>& ids) {
    if (!model_) return "";
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        // Skip special tokens (like [UNK], [CLS], [SEP], etc.)
        if (is_special(id)) continue;
        tokens.push_back(model_->id_to_token(id));
    }
    if (decoder_) {
        decoder_->decode_chain(tokens);
    }
    std::string result;
    for (const auto& t : tokens) result += t;
    // For WordPiece models with cleanup, remove spaces before . , ? !
    if (wordpiece_decode_) {
        result = str_replace_all(result, " .", ".");
        result = str_replace_all(result, " ?", "?");
        result = str_replace_all(result, " !", "!");
        result = str_replace_all(result, " ,", ",");
    }
    if (clean_up_spaces_) {
        result = str_replace_all(result, " ' ", "'");
        result = str_replace_all(result, " n't", "n't");
        result = str_replace_all(result, " 'm", "'m");
        result = str_replace_all(result, " 's", "'s");
        result = str_replace_all(result, " 've", "'ve");
        result = str_replace_all(result, " 're", "'re");
    }
    return result;
}

std::string PipelineTokenizer::decode(int id) {
    if (!model_) return "";
    std::string token = model_->id_to_token(id);
    // For single-token decode (streaming), apply Replace/ByteFallback/ByteLevel
    // but not Strip/Fuse which are meant for full-sequence decode.
    // Simple approach: replace ▁ with space (Metaspace/SentencePiece convention)
    // and ByteLevel decode manually.
    std::string result;
    const char* meta = "\xe2\x96\x81"; // ▁ U+2581
    size_t pos = 0;
    while (pos < token.size()) {
        if (pos + 3 <= token.size() && token[pos] == meta[0] && token[pos+1] == meta[1] && token[pos+2] == meta[2]) {
            result += ' ';
            pos += 3;
        } else if (pos + 6 <= token.size() && token.substr(pos, 3) == "<0x") {
            // ByteFallback: <0xHH> → byte
            int b;
            if (sscanf(token.c_str() + pos, "<0x%02X>", &b) == 1) {
                result += (char)b;
                pos += 6;
            } else {
                result += token[pos++];
            }
        } else {
            result += token[pos++];
        }
    }
    // If model uses WordPiece decoder: add space before non-## tokens
    if (wordpiece_decode_) {
        bool is_suffix = (!wordpiece_prefix_.empty() && result.find(wordpiece_prefix_) == 0);
        if (is_suffix) {
            result = result.substr(wordpiece_prefix_.size());
        } else {
            result = " " + result;
        }
        return result;
    }
    // If model has ByteLevel encoding, reverse byte map
    if (byte_level_) {
        static auto bm = []() {
            std::unordered_map<std::string, unsigned char> m;
            auto byte_vec = create_bytes_char_map();
            for (int i = 0; i < 256; ++i) {
                if (!byte_vec[i].empty()) m[byte_vec[i]] = (unsigned char)i;
            }
            return m;
        }();
        std::string out;
        const uint8_t* p = (const uint8_t*)result.data();
        size_t len = result.size(), i = 0;
        while (i < len) {
            int32_t cp; int r = Unicode::utf8_decode(p + i, len - i, &cp);
            std::string ch(result.substr(i, r));
            auto it = bm.find(ch);
            if (it != bm.end()) out += (char)it->second;
            else out += ch;
            i += r;
        }
        return out;
    }
    return result;
}

// ==========================================
// Fast parsing helpers
// ==========================================

struct StringView {
    const char* data;
    size_t len;
};

// Advance ptr past next '\n', return view of the line (excluding '\n')
static inline StringView fast_getline(const char*& ptr, const char* end) {
    const char* start = ptr;
    while (ptr < end && *ptr != '\n') ++ptr;
    StringView sv = {start, (size_t)(ptr - start)};
    if (ptr < end) ++ptr; // skip '\n'
    return sv;
}

// Fast integer parse from pointer, advance ptr past digits
static inline int fast_int(const char*& p) {
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); ++p; }
    return neg ? -val : val;
}

// Skip whitespace (spaces)
static inline void skip_spaces(const char*& p) {
    while (*p == ' ') ++p;
}

// Fast double parse using strtod, advance ptr
static inline double fast_double(const char*& p) {
    char* endp;
    double val = strtod(p, &endp);
    p = endp;
    return val;
}

// Read a base64 token from current position until space or end of line
static inline StringView read_token(const char*& p, const char* line_end) {
    const char* start = p;
    while (p < line_end && *p != ' ') ++p;
    return {start, (size_t)(p - start)};
}

bool PipelineTokenizer::load_vocab(std::ifstream& file) { return false; }


// ==========================================
// Binary read helpers
// ==========================================
static inline uint8_t read_u8(const char*& p) {
    uint8_t v = *(const uint8_t*)p;
    p += 1;
    return v;
}

static inline uint16_t read_u16(const char*& p) {
    uint16_t v;
    memcpy(&v, p, 2);
    p += 2;
    return v;
}

static inline uint32_t read_u32(const char*& p) {
    uint32_t v;
    memcpy(&v, p, 4);
    p += 4;
    return v;
}

static inline double read_f64(const char*& p) {
    double v;
    memcpy(&v, p, 8);
    p += 8;
    return v;
}

static inline std::string read_str(const char*& p) {
    uint16_t len = read_u16(p);
    std::string s(p, len);
    p += len;
    return s;
}

// Zero-copy: returns StringRef pointing into buffer
static inline StringRef read_str_ref(const char*& p) {
    uint16_t len = read_u16(p);
    StringRef r = {p, len};
    p += len;
    return r;
}

// ==========================================
// load_vocab_binary - binary .mtok format
// ==========================================
bool PipelineTokenizer::load_vocab_binary(std::ifstream& file) {
    // Fast file read: seek to get size, then read all at once
    auto cur_pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    size_t remaining = (size_t)(end_pos - cur_pos);
    file.seekg(cur_pos);
    binary_buf_.resize(remaining);
    file.read(&binary_buf_[0], remaining);
    const char* ptr = binary_buf_.c_str();

    // --- Normalizer ---
    auto read_norm_table = [](const char*& p) -> std::vector<std::pair<uint32_t, std::string>> {
        uint32_t count = read_u32(p);
        std::vector<std::pair<uint32_t, std::string>> table;
        table.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t cp = read_u32(p);
            uint16_t len = read_u16(p);
            table.push_back({cp, std::string(p, len)});
            p += len;
        }
        return table;
    };
    std::function<std::unique_ptr<Normalizer>(const char*&)> read_normalizer_bin;
    read_normalizer_bin = [&](const char*& p) -> std::unique_ptr<Normalizer> {
        uint8_t type = read_u8(p);
        switch (type) {
            case 0: return nullptr;
            case 1: return make_unique_<NFKCNormalizer>();  // old format, empty table
            case 2: { std::string s = read_str(p); return make_unique_<PrependNormalizer>(s); }
            case 3: { std::string pat = read_str(p); std::string content = read_str(p);
                       return make_unique_<ReplaceNormalizer>(pat, content); }
            case 4: {
                uint32_t count = read_u32(p);
                std::vector<std::shared_ptr<Normalizer>> norms;
                for (uint32_t i = 0; i < count; i++) {
                    auto child = read_normalizer_bin(p);
                    if (!child) {
                        MNN_ERROR("[Tokenizer] Skip null normalizer in sequence.\n");
                        continue;
                    }
                    norms.push_back(std::shared_ptr<Normalizer>(child.release()));
                }
                return make_unique_<SequenceNormalizer>(norms);
            }
            case 5: {
                uint8_t clean = read_u8(p);
                uint8_t chinese = read_u8(p);
                uint8_t strip_accents = read_u8(p);
                uint8_t lowercase = read_u8(p);
                return make_unique_<BertNormalizer>((bool)clean, (bool)chinese, (bool)strip_accents, (bool)lowercase);
            }
            case 6: {  // new NFKC with embedded data
                auto table = read_norm_table(p);
                return make_unique_<NFKCNormalizer>(std::move(table));
            }
            case 7: {  // new BertNormalizer with embedded NFD
                uint8_t clean = read_u8(p);
                uint8_t chinese = read_u8(p);
                uint8_t strip = read_u8(p);
                uint8_t lower = read_u8(p);
                std::vector<std::pair<uint32_t, std::string>> nfd;
                if (strip) nfd = read_norm_table(p);
                return make_unique_<BertNormalizer>((bool)clean, (bool)chinese, (bool)strip, (bool)lower, std::move(nfd));
            }
            case 8: { // Strip normalizer
                uint8_t strip_left = read_u8(p);
                uint8_t strip_right = read_u8(p);
                return make_unique_<StripNormalizer>((bool)strip_left, (bool)strip_right);
            }
            default: break;
        }
        return nullptr;
    };
    normalizer_ = read_normalizer_bin(ptr);

    // --- PreTokenizer ---
    std::function<std::unique_ptr<PreTokenizer>(const char*&)> read_pre_tokenizer_bin;
    read_pre_tokenizer_bin = [&](const char*& p) -> std::unique_ptr<PreTokenizer> {
        uint8_t type = read_u8(p);
        switch (type) {
            case 0: return nullptr;
            case 1: { uint8_t use_regex = read_u8(p); return make_unique_<ByteLevelPreTokenizer>((bool)use_regex); }
            case 2: { uint8_t individual = read_u8(p); return make_unique_<DigitsPreTokenizer>((bool)individual); }
            case 3: { std::string rep = read_str(p); uint8_t aps = read_u8(p);
                       return make_unique_<MetaspacePreTokenizer>(rep, (bool)aps); }
            case 4: { std::string pat = read_str(p); uint8_t invert = read_u8(p); uint8_t behavior = read_u8(p);
                       std::string beh = behavior == 0 ? "Isolated" : (behavior == 2 ? "MergedWithPrevious" : "Removed");
                       return make_unique_<SplitPreTokenizer>(pat, (bool)invert, beh); }
            case 5: return make_unique_<BertPreTokenizer>();
            case 6: {
                uint32_t count = read_u32(p);
                std::vector<std::shared_ptr<PreTokenizer>> pts;
                for (uint32_t i = 0; i < count; i++) {
                    auto child = read_pre_tokenizer_bin(p);
                    if (!child) {
                        MNN_ERROR("[Tokenizer] Skip null pre-tokenizer in sequence.\n");
                        continue;
                    }
                    pts.push_back(std::shared_ptr<PreTokenizer>(child.release()));
                }
                return make_unique_<SequencePreTokenizer>(pts);
            }
            default: break;
        }
        return nullptr;
    };
    pre_tokenizer_ = read_pre_tokenizer_bin(ptr);

    // --- Model --- (zero-copy: StringRef points into buf)
    {
        uint8_t type = read_u8(ptr);
        uint32_t vocab_size = read_u32(ptr);

        if (type == 0) { // BPE
            uint8_t byte_fallback = read_u8(ptr);
            uint8_t byte_level = read_u8(ptr);
            uint32_t merge_size = read_u32(ptr);
            (void)byte_fallback;

            // Vocab stored sorted by string: [len][bytes][uint32 id]
            std::vector<StringRef> id_to_token(vocab_size);
            std::vector<VocabEntry> sorted_vocab(vocab_size);
            for (uint32_t i = 0; i < vocab_size; i++) {
                StringRef sr = read_str_ref(ptr);
                int id = (int)read_u32(ptr);
                id_to_token[id] = sr;
                sorted_vocab[i] = {sr.ptr, sr.len, id};
            }
            std::vector<std::pair<uint64_t, int>> merges;
            merges.reserve(merge_size);
            for (uint32_t i = 0; i < merge_size; i++) {
                uint32_t id1 = read_u32(ptr);
                uint32_t id2 = read_u32(ptr);
                uint32_t rank = read_u32(ptr);
                merges.push_back({merge_key((int)id1, (int)id2), (int)rank});
            }
            model_ = make_unique_<BPEModel>(std::move(id_to_token),
                std::move(sorted_vocab), std::move(merges), (bool)byte_level);
        } else if (type == 1) { // WordPiece
            std::string unk_token = read_str(ptr);
            std::string prefix = read_str(ptr);
            uint32_t max_chars = read_u32(ptr);
            auto wp = make_unique_<WordPieceModel>(unk_token, prefix, (int)max_chars);

            std::vector<StringRef> id_to_token(vocab_size);
            std::vector<VocabEntry> sorted_vocab(vocab_size);
            for (uint32_t i = 0; i < vocab_size; i++) {
                StringRef sr = read_str_ref(ptr);
                int id = (int)read_u32(ptr);
                id_to_token[id] = sr;
                sorted_vocab[i] = {sr.ptr, sr.len, id};
            }
            wp->load_direct_sorted(std::move(id_to_token), std::move(sorted_vocab));
            model_ = std::move(wp);
        } else if (type == 2) { // Unigram
            uint32_t unk_id = read_u32(ptr);
            uint8_t byte_fallback = read_u8(ptr);
            auto ug = make_unique_<UnigramModel>((int)unk_id, (bool)byte_fallback);

            std::vector<StringRef> id_to_token(vocab_size);
            std::vector<double> scores(vocab_size);
            std::vector<VocabEntry> sorted_vocab(vocab_size);
            for (uint32_t i = 0; i < vocab_size; i++) {
                StringRef sr = read_str_ref(ptr);
                int id = (int)read_u32(ptr);
                double score = read_f64(ptr);
                id_to_token[id] = sr;
                scores[id] = score;
                sorted_vocab[i] = {sr.ptr, sr.len, id};
            }
            ug->load_direct_sorted(std::move(id_to_token), std::move(scores), std::move(sorted_vocab));
            model_ = std::move(ug);
        }
    }

    // --- Decoder ---
    std::function<std::unique_ptr<TokenDecoder>(const char*&)> read_decoder_bin;
    read_decoder_bin = [&](const char*& p) -> std::unique_ptr<TokenDecoder> {
        uint8_t type = read_u8(p);
        switch (type) {
            case 0: byte_level_ = true; return make_unique_<ByteLevelDecoder>();
            case 1: return make_unique_<ByteFallbackDecoder>();
            case 2: { std::string rep = read_str(p); uint8_t aps = read_u8(p);
                       return make_unique_<MetaspaceDecoder>(rep, (bool)aps); }
            case 3: { std::string pfx = read_str(p); uint8_t cleanup = read_u8(p);
                       wordpiece_decode_ = true; wordpiece_prefix_ = pfx;
                       return make_unique_<WordPieceDecoder>(pfx, (bool)cleanup); }
            case 4: return make_unique_<FuseDecoder>();
            case 5: { std::string pat = read_str(p); std::string content = read_str(p);
                       return make_unique_<ReplaceDecoder>(pat, content); }
            case 6: { std::string content = read_str(p); uint32_t start = read_u32(p); uint32_t stop = read_u32(p);
                       return make_unique_<StripDecoder>(content, (int)start, (int)stop); }
            case 7: {
                uint32_t count = read_u32(p);
                std::vector<std::shared_ptr<TokenDecoder>> decs;
                for (uint32_t i = 0; i < count; i++) {
                    auto child = read_decoder_bin(p);
                    if (!child) {
                        MNN_ERROR("[Tokenizer] Skip null decoder in sequence.\n");
                        continue;
                    }
                    decs.push_back(std::shared_ptr<TokenDecoder>(child.release()));
                }
                return make_unique_<SequenceDecoder>(decs);
            }
            default: break;
        }
        return nullptr;
    };
    decoder_ = read_decoder_bin(ptr);

    // --- Added Tokens ---
    {
        uint32_t count = read_u32(ptr);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = read_u32(ptr);
            uint8_t special = read_u8(ptr);
            uint8_t lstrip = read_u8(ptr);
            uint8_t rstrip = read_u8(ptr);
            std::string content = read_str(ptr);
            added_tokens_.push_back({(int)id, std::move(content), (bool)special, (bool)lstrip, (bool)rstrip});
        }
        added_token_strings_.clear();
        for (const auto& t : added_tokens_) {
            added_token_strings_.push_back(t.content);
        }
    }

    // --- Chat Template (optional) ---
    const char* buf_end = binary_buf_.c_str() + binary_buf_.size();
    if (ptr < buf_end) {
        uint32_t tpl_len = read_u32(ptr);
        if (tpl_len > 0) {
            chat_template_ = std::string(ptr, tpl_len);
            ptr += tpl_len;
        }
        if (ptr < buf_end) {
            uint16_t eos_len = read_u16(ptr);
            if (eos_len > 0) {
                chat_template_eos_ = std::string(ptr, eos_len);
                ptr += eos_len;
            }
        }
    }

    // --- Flags (optional) ---
    if (ptr < buf_end) {
        uint8_t flags = read_u8(ptr);
        clean_up_spaces_ = (flags & 0x01) != 0;
    }

    // --- BOS token (optional) ---
    if (ptr < buf_end) {
        uint16_t bos_len = read_u16(ptr);
        if (bos_len > 0) {
            chat_template_bos_ = std::string(ptr, bos_len);
            ptr += bos_len;
        }
    }

    return true;
}

} // namespace Transformer
} // namespace MNN
