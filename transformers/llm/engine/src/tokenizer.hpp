//
//  tokenizer.hpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#ifndef TOKENIZER_hpp
#define TOKENIZER_hpp

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>
// #include <string_view>
#include <cstring>
class string_view_ {
public:
    string_view_() : data_(nullptr), size_(0) {}
    string_view_(const char* data) : data_(data), size_(std::strlen(data)) {}
    string_view_(const char* data, std::size_t size) : data_(data), size_(size) {}
    string_view_(const std::string& str) : data_(str.data()), size_(str.size()) {}
    constexpr string_view_(const string_view_&) noexcept = default;
    string_view_& operator=(const string_view_&) noexcept = default;
    const char& operator[](size_t pos) const { return data_[pos]; }
    constexpr const char* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const { return size_ == 0; }
    std::string to_string() const { return std::string(data_, size_); }
    bool operator==(const string_view_& other) const noexcept {
        return size_ == other.size_ && strncmp(data_, other.data_, size_) == 0;
    }
    void remove_prefix(size_t n) {
        if (n < size_) {
            data_ += n;
            size_ -= n;
        } else {
            data_ = "";
            size_ = 0;
        }
    }
private:
    const char* data_;
    std::size_t size_ = 0;
};
// std::string_view impl in c++11 end

namespace std {
    template<>
    class hash<string_view_> {
    public:
        size_t operator()(const string_view_& sv) const {
            size_t result = 0;
            for (size_t i = 0; i < sv.size(); ++i) {
                result = (result * 31) + static_cast<size_t>(sv[i]);
            }
            return result;
        }
    };
}
namespace MNN {
namespace Transformer {
// std::string_view impl in c++11 start


class Trie {
public:
    struct TrieNode
    {
        std::unordered_map<char, int> children;
        int id = -1;
    };
private:
    std::vector<TrieNode> list;
    int size = 1;
    int getFree() {
        if (size<list.size()) { return size++; }
        else {
            list.resize(list.size()*2);
            return size++;
        }
    }
    void insert(int nid, int token_id, std::string::const_iterator it, std::string::const_iterator end) {
        auto& node = list[nid];
        if (it==end) {
            if (node.id==-1) { node.id=token_id; }
            return;
        }
        auto cid = node.children.find(*it);
        if (cid==node.children.end()) {
            int new_id = getFree();
            list[nid].children.insert({*it, new_id}); // access the node again even after reallocation!!!
            insert(new_id, token_id, it+1, end);
        } else{
            insert(cid->second, token_id, it+1, end);
        }
    }
    int find(int nid, int current_matched, std::string::const_iterator current_it, std::string::const_iterator& it, const std::string::const_iterator& end) {
        const auto& node = list[nid];
        if (node.id!=-1) {
            current_matched = node.id;
            current_it = it;
        }
        auto cid = node.children.find(*it);
        if (cid != node.children.end()) {
            return find(cid->second, current_matched, current_it, ++it, end);
        } else {
            if (node.id!=-1) { return node.id; }
            else { it = current_it; return current_matched;}
        }
    }
public:
    Trie(int initial_size=10000) {
        list.resize(initial_size); // init the allocate size
        size = 1; // root
    }
    void insert(std::pair<const std::string&, int> entry) {
        insert(0, entry.second, entry.first.begin(), entry.first.end());
    }
    int find(std::string::const_iterator& it, const std::string::const_iterator& end) {
        if (it==end) { return -1; }
        return find(0, -1, it+1, it, end);
    }
};


class Tokenizer {
public:
    static constexpr int MAGIC_NUMBER = 430;
    enum TokenizerType {
        SENTENCEPIECE = 0,
        TIKTOIKEN = 1,
        BERT = 2,
        HUGGINGFACE = 3
    };
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    static Tokenizer* createTokenizer(const std::string& filename);
    bool is_stop(int token);
    bool is_special(int token);
    std::vector<int> encode(const std::string& str);
    virtual std::string decode(int id) = 0;
protected:
    void cache_special_tokens();
    virtual void load_special(std::ifstream& file);
    virtual bool load_vocab(std::ifstream& file) = 0;
    virtual void encode(const std::string& str, std::vector<int>& ids) = 0;
    std::vector<int> special_tokens_;
    std::vector<int> stop_tokens_;
    std::vector<int> prefix_tokens_;
    std::vector<std::pair<std::string, int>> special_tokens_cache_;
private:
    std::string mTemplate;
};

class Sentencepiece : public Tokenizer {
public:
    Sentencepiece() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    enum ModelType {
        UNIGRAM = 1,
        BPE = 2,
        WORD = 3,
        CHAR = 4
    };
    enum PieceType {
        NORMAL = 1,
        UNKNOWN = 2,
        CONTROL = 3,
        USER_DEFINED = 4,
        UNUSED = 5,
        BYTE = 6
    };
    struct SentencePiece {
        std::string piece;
        float score;
        PieceType type = PieceType::NORMAL;
        SentencePiece() {}
        SentencePiece(const std::string& p, float s, PieceType t) : piece(p), score(s), type(t) {}
    };
    using EncodeResult = std::vector<std::pair<string_view_, int>>;
private:
    // model train type
    ModelType type_ = BPE;
    // byte fall back enable
    bool byte_fall_back_ = true;
    // unknown id.
    int unk_id_ = 0;
    // pieces from model
    std::vector<SentencePiece> sentence_pieces_;
    // piece -> id map for normal pieces
    std::unordered_map<string_view_, int> pieces_;
    // piece -> id map for control, unknown, and byte pieces
    std::unordered_map<string_view_, int> reserved_id_map_;
private:
    float get_score(int id) const;
    bool is_unused(int id) const;
    bool is_control(int id) const;
    int piece_to_id(string_view_ w) const;
    std::string byte_to_piece(unsigned char c) const;
    EncodeResult bpe_encode(string_view_ str, float alpha = 0.f);
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    Trie encoder_;
    std::vector<std::string> decoder_;
};

class BertTokenizer : public Tokenizer {
public:
    BertTokenizer() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
private:
    std::vector<int> word_piece(const std::string& token);
};

class HuggingfaceTokenizer : public Tokenizer {
struct hash_pair_wstring {
    size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
        auto hash1 = std::hash<std::wstring>{}(p.first);
        auto hash2 = std::hash<std::wstring>{}(p.second);
        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};
using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;
public:
    HuggingfaceTokenizer() = default;
    virtual std::string decode(int id) override;
protected:
    virtual bool load_vocab(std::ifstream& file) override;
    virtual void encode(const std::string& str, std::vector<int>& ids) override;
private:
    void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
    BPERanks bpe_ranks_;
    std::unordered_map<uint8_t, wchar_t> b2u_;
    std::unordered_map<wchar_t, uint8_t> u2b_;
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};
};
};

#endif // TOKENIZER_hpp
