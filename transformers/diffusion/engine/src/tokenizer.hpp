#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#ifndef MNN_DIFFUSION_TOKENIZER_HPP
#define MNN_DIFFUSION_TOKENIZER_HPP

namespace MNN {
namespace Transformer {
class Tokenizer;
}
namespace DIFFUSION {
    
class Tokenizer {
public:
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    virtual bool load(const std::string& filePath) = 0;
    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) = 0;
};

class MtokTokenizer : public Tokenizer {
public:
    enum class Style {
        kPair,
        kSingle,
    };

    MtokTokenizer(Style style, int bosId = -1, int eosId = -1);
    virtual ~MtokTokenizer();
    virtual bool load(const std::string& filePath) override;
    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) override;

private:
    std::vector<int> encodeSingle(const std::string& sentence, int maxlen) const;

private:
    Style mStyle;
    int mBosId;
    int mEosId;
    MNN::Transformer::Tokenizer* mTokenizer = nullptr;
};
}
} // diffusion
#endif
