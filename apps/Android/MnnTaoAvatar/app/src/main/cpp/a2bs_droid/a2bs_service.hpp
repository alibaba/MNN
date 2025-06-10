#pragma once
#include "common/Common.hpp"
#include "common/file_utils.hpp"
#include "a2bs/audio_to_3dgs_blend_shape.hpp"

namespace TaoAvatar {

    class A2BSService {
    public:
        A2BSService();
        ~A2BSService();
        bool LoadA2bsResources(const char *res_path, const char *temp_path);
        void SaveResultDataToBinaryFile(const AudioToBlendShapeData& data, int index);
        AudioToBlendShapeData Process(int index, const int16_t *audio_buffer, size_t length, int sample_rate);
        AudioToBlendShapeData Process(int index, const AudioData &audio_data, int sample_rate);
        static A2BSService *GetActiveInstance();
        FLAMEOuput GetActiveFrame(int index, int& segment_index, int& sub_index);
        size_t GetTotalFrameNum();
    private:
        std::shared_ptr<AudioTo3DGSBlendShape> audio_to_bs_ = nullptr;
        std::map<int, AudioToBlendShapeData> bs_data_map_;
        static A2BSService *instance_;
    };

} // namespace TaoAvatar
