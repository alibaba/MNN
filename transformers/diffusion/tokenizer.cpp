
#include <fstream>
#include <iostream>
#include <utility>
#include "tokenizer.hpp"

namespace diffusion {

tokenizer::tokenizer(std::string dictPath) {
    std::ifstream dictFile(dictPath);
    int index = 0;
    std::string word;
    while (dictFile >> word) {
        mWordDict.insert(std::make_pair<std::string, int>(std::move(word), index++));
    }
    mStartIdx = this->word("[CLS]");
    mEndIdx = this->word("[SEP]");
}

int tokenizer::word(std::string word) {
    const auto& iter = mWordDict.find(word);
    if (iter != mWordDict.end()) {
        return iter->second;
    }
    return -1;
}

std::vector<int> tokenizer::sentence(std::string sentence, int maxlen) {
    std::vector<int> ids(maxlen * 2, 0);
    // uncond
    ids[0] = mStartIdx;
    ids[1] = mEndIdx;
    // ids
    int idx = maxlen;
    ids[idx++] = mStartIdx;
    for (size_t i = 0; i < sentence.length();) {
        int wordlen = 1;
        if ((sentence[i] & 0xf8) == 0xf0) {
            wordlen = 4;
        } else if ((sentence[i] & 0xf0) == 0xe0) {
            wordlen = 3;
        } else if ((sentence[i] & 0xe0) == 0xc0) {
            wordlen = 2;
        }    
        if ((i + wordlen) > sentence.length()) {
            wordlen = 1;
        }  
        std::string word = sentence.substr(i, wordlen);
        ids[idx++] = this->word(word);
        i += wordlen;
    }
    ids[idx++] = mEndIdx;
    return ids;
}

} // diffusion