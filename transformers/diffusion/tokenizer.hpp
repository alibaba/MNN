#include <vector>
#include <string>
#include <unordered_map>

namespace diffusion {

class Tokenizer {
public:
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    virtual bool load(const std::string& filePath) = 0;
    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) = 0;
};
    
class BertTokenizer : public Tokenizer{
public:
    BertTokenizer() = default;
    virtual bool load(const std::string& filePath) override;
    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) override;
private:
    std::vector<int> word_piece(const std::string& token);
private:
    int mStartIdx, mEndIdx;
    std::unordered_map<std::string, int> mVocabs;
};

class CLIPTokenizer : public Tokenizer{
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
    CLIPTokenizer() = default;
    virtual bool load(const std::string& filePath) override;
    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) override;
    
private:
    bool loadVocab(const std::string& vocabFilePath);
    bool loadMerges(const std::string& mergesFilePath);

private:
    void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
    BPERanks bpe_ranks_;
    std::unordered_map<uint8_t, wchar_t> b2u_;
    std::unordered_map<wchar_t, uint8_t> u2b_;
    
    std::unordered_map<std::string, int> mVocabs;
    
    int mStartIdx = 49406;
    int mEndIdx = 49407;
};

} // diffusion
