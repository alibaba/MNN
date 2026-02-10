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
    bool LoadTtsResources(const char *resPath, const char* modelName, 
                         const char* cacheDir, const std::string &paramsJson = "{}");  // 新增参数
    std::vector<int16_t> Process(const std::string &text, int id);
    void SetIndex(int index);
    void SetSpeakerId(const std::string &speaker_id);  // 动态设置音色（仅英文模式）
    virtual ~TTSService();
private:
    std::shared_ptr<MNNTTSSDK> tts_ = nullptr;
    int current_index_{0};
    std::string language_;
};

} // namespace TaoAvatar
