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

bool Sentencepiece::load(const std::string& filename) {
    std::ifstream tok_file(filename);
    std::string line, token;
    float score;
    int index = 0, type;
    while (std::getline(tok_file, line)) {
        std::istringstream line_str(line);
        line_str >> token >> score >> type;
        token = base64_decode(token);
        auto piece_type = static_cast<PieceType>(type);
        SentencePiece piece {token, score, piece_type};
        sentence_pieces_.emplace_back(std::move(piece));
        if (piece_type == PieceType::NORMAL) {
            pieces_.insert({token, index});
        } else {
            reserved_id_map_.insert({token, index});
            if (piece_type == PieceType::UNKNOWN) {
                unk_id_ = index;
            }
        }
        index++;
    }
    tok_file.close();
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
Sentencepiece::EncodeResult Sentencepiece::bpe_encode(std::string_view normalized, float alpha) {
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
        std::string_view piece;
    };
    // util class end

    using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
    Agenda agenda;
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    // Reverse merge rules. key: merged symbol, value: pair of original symbols.
    std::unordered_map<std::string_view, std::pair<std::string_view, std::string_view>> rev_merge;
    // SymbolPair holder.
    std::vector<std::unique_ptr<SymbolPair>> symbol_pair_holder;
    // Lookup new symbol pair at [left, right] and inserts it to agenda.
    auto MaybeAddNewSymbolPair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) {
            return;
        }
        const std::string_view piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        std::string piece_str(piece);
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
        s.piece = std::string_view(normalized.data(), mblen);
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
        symbols[top->left].piece = std::string_view(
            symbols[top->left].piece.data(),
            symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // Updates prev/next pointers.
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
        symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = std::string_view("");

        // Adds new symbol pairs which are newly added after symbol replacement.
        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    std::function<void(std::string_view, EncodeResult*)> resegment;
    resegment = [this, &resegment, &rev_merge](std::string_view w, EncodeResult *output) -> void {
        std::string w_str(w);
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

std::vector<int> Sentencepiece::encode(const std::string& str) {
    std::vector<int> ids;
    auto result = bpe_encode(str);
    size_t consumed = 0;
    for (const auto &p : result) {
        const std::string_view w = p.first;   // piece
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
    return ids;
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

const int CHARACTER_VOCABULARY_SIZE = 256;

bool Tiktoken::load(const std::string& filename) {
    std::ifstream tok_file(filename);
    int index = -1, start = 0, rough_count = 0;
    std::string token;
    while (tok_file >> token) {
        token = base64_decode(token);
        decoder_.push_back(token);
        rough_count += token.size();
    }
    tok_file.close();
    tokens_.resize(rough_count * CHARACTER_VOCABULARY_SIZE, -1);
    token_ids_.resize(rough_count * CHARACTER_VOCABULARY_SIZE, -1);
    for (int n = 0; n < decoder_.size(); n++) {
        token = decoder_[n];
        int root = 0;
        for (int i = 0; i < token.size(); i++) {
            unsigned char x = token[i];
            // record the token id at the parent of leaf node
            if (i == token.size() - 1) {
                token_ids_[root + x] = n;
            }
            // trace down a tree node.
            // insert a subtree when needed.
            if (tokens_[root + x] == -1) {
                start += CHARACTER_VOCABULARY_SIZE;
                tokens_[root + x] = start;
                root = start;
            } else {
                root = tokens_[root + x];
            }
        }
    }
    tokens_.resize(start + CHARACTER_VOCABULARY_SIZE);
    token_ids_.resize(start + CHARACTER_VOCABULARY_SIZE);
    return true;
}

// ref: https://github.com/youkaichao/fast_bpe_tokenizer
std::vector<int> Tiktoken::encode(const std::string& str) {
    std::vector<int> ids;
    if (str.empty()) {
        return ids;
    }
    int i = 0;
    int root = 0;
    int root_token_id = -1;
    int last_found_position = -1;
    int last_found_token_id = -1;
    while (i < str.size()) {
        unsigned char x = str[i];
        bool should_fall_back = false;
        if (tokens_[root + x] != -1) {
            root_token_id = token_ids_[root + x];
            root = tokens_[root + x];
            if (root_token_id != -1) {
                // a token ends at position i
                last_found_position = i;
                last_found_token_id = root_token_id;
            }
            i++;
            if (i == str.size()) {
                should_fall_back = true;
            }
        } else {
            // assert(last_found_position != -1);
            should_fall_back = true;
        }
        if (should_fall_back) {
            i = last_found_position + 1;
            ids.push_back(last_found_token_id);
            // start searching from the root again
            root = 0;
            root_token_id = -1;
            last_found_position = -1;
            last_found_token_id = -1;
        }
    }
    return ids;
}

std::string Tiktoken::decode(int id) {
    if (id >= decoder_.size()) {
        return "";
    }
    return decoder_[id];
}