#include <vector>
#include <string>
#include <unordered_map>

namespace diffusion {

class tokenizer {
public:
    tokenizer(std::string dictPath);
    ~tokenizer() {}
    int word(std::string word);
    std::vector<int> sentence(std::string sentence, int maxlen = 0);
private:
    int mStartIdx, mEndIdx;
    std::unordered_map<std::string, int> mWordDict;
};

} // diffusion