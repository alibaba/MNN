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
#include <string_view>

class Tokenizer {
public:
    Tokenizer() = default;
    virtual bool load(const std::string& filename) = 0;
    virtual std::vector<int> encode(const std::string& str) = 0;
    virtual std::string decode(int id) = 0;
};

class Sentencepiece : public Tokenizer {
public:
    Sentencepiece() = default;
    virtual bool load(const std::string& filename) override;
    virtual std::vector<int> encode(const std::string& str) override;
    virtual std::string decode(int id) override;
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
    };
    using EncodeResult = std::vector<std::pair<std::string_view, int>>;
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
    std::unordered_map<std::string, int> pieces_;
    // piece -> id map for control, unknown, and byte pieces
    std::unordered_map<std::string, int> reserved_id_map_;
private:
    float get_score(int id) const;
    bool is_unused(int id) const;
    bool is_control(int id) const;
    int piece_to_id(const std::string& w) const;
    std::string byte_to_piece(unsigned char c) const;
    EncodeResult bpe_encode(std::string_view str, float alpha = 0.f);
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual bool load(const std::string& filename) override;
    virtual std::vector<int> encode(const std::string& str) override;
    virtual std::string decode(int id) override;
private:
    std::vector<std::string> decoder_;
    std::vector<int> tokens_;
    std::vector<int> token_ids_;
};

#endif // TOKENIZER_hpp