#include "a2bs_service.hpp"

#include <memory>
#include "a2bs/audio_to_3dgs_blend_shape.hpp"
#include "a2bs/a2bs_utils.hpp"
#include "mh_log.hpp"
#include "mh_config.hpp"

using clk = std::chrono::system_clock;

namespace TaoAvatar {
    A2BSService* A2BSService::instance_ = nullptr;
    A2BSService::A2BSService() {
        instance_ = this;
    }

    A2BSService::~A2BSService() {
        audio_to_bs_ = nullptr;
        if (instance_ == this) {
            instance_ = nullptr;
        }
    }

    bool A2BSService::LoadA2bsResources(const char *res_path, const char* temp_path) {
        if (!audio_to_bs_) {
            audio_to_bs_ = std::make_shared<AudioTo3DGSBlendShape>(res_path, temp_path, false);
        }
        if (!audio_to_bs_) {
            MH_ERROR("Failed to create A2BSService.");
            return false;
        }
        return true;
    }

    void A2BSService::SaveResultDataToBinaryFile(const AudioToBlendShapeData& data, int index) {
        std::string filename = "/data/data/com.taobao.meta.avatar/a2bs_" + std::to_string(index) + ".bin";
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(&data.frame_num), sizeof(size_t));
            for (const auto& expr : data.expr) {
                outfile.write(reinterpret_cast<const char*>(expr.data()), expr.size() * sizeof(float));
            }
            for (const auto& jaw : data.jaw_pose) {
                outfile.write(reinterpret_cast<const char*>(jaw.data()), jaw.size() * sizeof(float));
            }
            outfile.close();
        } else {
            MH_ERROR("error write file to : %s", filename.c_str());
        }
    }

    AudioToBlendShapeData A2BSService::Process(int index,
                                               const int16_t *audio_buffer,
                                               size_t length,
                                               int sample_rate) {

        if (!audio_to_bs_ || !audio_buffer || length == 0) {
            MH_ERROR("Failed to process audio data.");
            return {};
        }
        std::vector<float> float_audio_data;
        for (int i = 0; i < length; i++) {
            float tmp = float(audio_buffer[i]) / 32767.0f;
            float_audio_data.push_back(tmp);
        }
        const size_t chunk_size = sample_rate * 10;
        size_t num_chunks = (float_audio_data.size() + chunk_size - 1) / chunk_size;
        MH_DEBUG("callA2Bs Total Audio Len %zu, Chunk Size %zu, Num Chunks %zu \n", float_audio_data.size(), chunk_size, num_chunks);
        auto begin_time = clk::now();
        std::vector<FLAMEOuput> all_results;
        for (size_t i = 0; i < num_chunks; ++i) {
            MH_DEBUG("callA2Bs begin for %zu", i);
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, float_audio_data.size());
            std::vector<float> current_chunk(float_audio_data.begin() + start, float_audio_data.begin() + end);
            auto current_result = audio_to_bs_->ProcessFLAME(current_chunk, sample_rate);
            all_results.insert(all_results.end(), current_result.begin(), current_result.end());
            auto res = all_results;
            auto end_time = clk::now();
            std::chrono::duration<double> elapsed = end_time - begin_time;
            MH_DEBUG("===> Elapsed time: %f  seconds frames count: %zu", elapsed.count(), res.size());
            if (!res.empty()) {
                MH_DEBUG("===> first frame: %d", res[0].frame_id);
            }
        }
        AudioToBlendShapeData result_data;
        result_data.frame_num = all_results.size();
        for (const auto &result : all_results) {
            result_data.expr.push_back(result.expr);
            result_data.jaw_pose.push_back(result.jaw_pose);
        }
        bs_data_map_[index] = result_data;
        MH_DEBUG("callA2Bs total frame: %zu", GetTotalFrameNum());
#if DEBUG_SAVE_A2BS_DATA
        SaveResultDataToBinaryFile(result_data, index);
#endif
        return result_data;
    }

    AudioToBlendShapeData A2BSService::Process(int index, const AudioData &audio_data, int sample_rate) {
        if (!audio_to_bs_) {
            MH_ERROR("Failed to process audio data.");
            return {};
        }
        return Process(index, (int16_t *)audio_data.samples.data(), audio_data.samples.size(), sample_rate);
    }

    FLAMEOuput A2BSService::GetActiveFrame(int index, int& segment_index, int& sub_index) {
        FLAMEOuput flameOuput;
        for (int i = 0; i < bs_data_map_.size(); i++) {
            if (index < bs_data_map_[i].frame_num) {
                segment_index = i;
                sub_index = index;
                flameOuput.expr = bs_data_map_[i].expr[index];
                flameOuput.jaw_pose = bs_data_map_[i].jaw_pose[index];
                flameOuput.frame_id = bs_data_map_[i].frame_num;
                return flameOuput;
            } else {
                index -= bs_data_map_[i].frame_num;
            }
        }
        return flameOuput;
    }

    size_t A2BSService::GetTotalFrameNum() {
        int result = 0;
        for (auto i = 0; i < bs_data_map_.size(); i++) {
            if (bs_data_map_.count(i)) {
                result += bs_data_map_[i].frame_num;
            } else {
                break;
            }
        }
        return result;
    }


    A2BSService *A2BSService::GetActiveInstance() {
        return instance_;
    }

}