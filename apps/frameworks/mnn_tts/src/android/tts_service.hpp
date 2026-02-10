#pragma once
#include <string>
#include <functional>
#include <memory>
#include <filesystem>
#include "mnn_tts_sdk.hpp"

namespace TaoAvatar {

class TTSService {
public:
    explicit TTSService(std::string language);
    bool LoadTtsResources(const char *resPath, const char* modelName, const char* cacheDir);
    std::vector<int16_t> Process(const std::string &text, int id);
    void SetIndex(int index);
    void SetSpeakerId(const std::string &speaker_id);
    virtual ~TTSService();
private:
    std::shared_ptr<MNNTTSSDK> tts_ = nullptr;
    int current_index_{0};
    std::string language_;
};

} // namespace TaoAvatar
