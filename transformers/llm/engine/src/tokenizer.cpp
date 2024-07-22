//
//  tokenizer.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

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
namespace MNN {
namespace Transformer {

// base64
static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline size_t one_char_len(const char *src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

static std::string base64_decode(const std::string& str) {
    int in_len = str.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && ( str[in_] != '=') && is_base64(str[in_])) {
        char_array_4[i++] = str[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        for (j = 0; (j < i - 1); j++) {
            ret.push_back(char_array_3[j]);
        }
    }
    return ret;
}

static inline void to_lower_case(std::string& str) {
    for (auto &c : str) {
        if (c >= 'A' && c <= 'Z') {
            c = std::tolower(static_cast<unsigned char>(c));
        }
    }
}

Tokenizer* Tokenizer::createTokenizer(const std::string& filename) {
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
    printf("tokenizer_type = %d\n", tokenizer_type);
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

std::vector<int> Tokenizer::encode(const std::string& str) {
    std::vector<int> ids = prefix_tokens_;
    if (!special_tokens_.empty()) {
        std::string text = str;
        size_t start = 0;
        for (size_t i = 0; i < text.length(); ++i) {
            for (auto special_id : special_tokens_) {
                const auto& token = decode(special_id);
                if (token.empty()) continue;
                if (i + token.length() <= text.length() && text.substr(i, token.length()) == token) {
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
    std::string line, token;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    float score;
    int type;
    sentence_pieces_.resize(vocab_len);
    for (int index = 0; index < vocab_len; index++) {
        std::getline(tok_file, line);
        std::istringstream line_str(line);
        line_str >> token >> score >> type;
        token = base64_decode(token);
        auto piece_type = static_cast<PieceType>(type);
        SentencePiece piece = {token, score, piece_type};
        sentence_pieces_[index] = std::move(piece);
        if (piece_type == PieceType::NORMAL) {
            pieces_.insert({token, index});
        } else {
            reserved_id_map_.insert({token, index});
            if (piece_type == PieceType::UNKNOWN) {
                unk_id_ = index;
            }
        }
    }
    return true;
}

int Sentencepiece::piece_to_id(const std::string& piece) const {
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
        std::string piece_str(piece.to_string());
        const auto it = pieces_.find(piece_str);
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
        std::string w_str(w.to_string());
        const int id = piece_to_id(w_str);
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
    auto result = bpe_encode(str);
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
    size_t i = 0;
    while (i < str.size()) {
        bool found_pair = false;
        // Attempt to match the longest possible symbol
        size_t longest_match_len = 0;
        std::string longest_match;

        // Check substrings of decreasing length
        for (size_t len = str.size() - i; len > 0; --len) {
            std::string token = str.substr(i, len);
            auto it = encoder_.find(token);
            if (it != encoder_.end()) {
                if (len > longest_match_len) {
                    longest_match_len = len;
                    longest_match = it->first;
                }
            }
        }

        if (!longest_match.empty()) {
            ids.push_back(encoder_.at(longest_match));
            i += longest_match_len;
        } else {
            // If no matching symbol is found, this typically means an error in the encoding
            // or the input text contains characters that the encoder doesn't know how to handle
            std::cerr << "Error: No encoding found for the sequence starting at position " << i << std::endl;
            return;
        }
    }
}

std::string Tiktoken::decode(int id) {
    if (id >= decoder_.size()) {
        return "";
    }
    return decoder_[id];
}

std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
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
            auto it = encoder_.find(candidate);
            if (it != encoder_.end()) {
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

void BertTokenizer::encode(const std::string& str, std::vector<int>& ids) {
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
            ids.push_back(id);
        }
    }
}

std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}

std::string wstring_to_utf8(const std::wstring& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
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
    std::string line, token;
    // get nums
    int vocab_len, merge_len;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    line_str >> vocab_len >> merge_len;
    // load vocab
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        encoder_.insert({line, i});
        decoder_[i] = line;
    }
    // load merge_rule
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);
        int d = line.find(" ");
        bpe_ranks_.insert({{utf8_to_wstring(line.substr(0, d)),
            utf8_to_wstring(line.substr(d + 1))}, i});
    }
    // bytes_to_unicode
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
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = str;
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
    for (auto s : result) {
        ids.push_back(encoder_.at(s));
    }
}

std::string HuggingfaceTokenizer::decode(int id) {
    // printf("decode id = %d, %lu, %s#\n", id, decoder_.size(), decoder_.at(id).c_str());
    if (id >= decoder_.size()) {
        return "";
    }
    std::wstring w = utf8_to_wstring(decoder_.at(id));
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
