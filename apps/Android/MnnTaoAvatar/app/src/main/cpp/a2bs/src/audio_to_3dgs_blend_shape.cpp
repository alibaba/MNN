#include "a2bs/audio_to_3dgs_blend_shape.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

AudioTo3DGSBlendShape::AudioTo3DGSBlendShape(const std::string &local_resource_root,
                                             const std::string &mnn_mmap_dir,
                                             const bool do_verts2flame,
                                             const int ori_fps,
                                             const int out_fps,
                                             const int num_exp)
    : atf(local_resource_root, mnn_mmap_dir, do_verts2flame, ori_fps, out_fps, num_exp), _num_exp(num_exp) {
    local_resource_root_ = local_resource_root;

    auto start_time = clk::now();
    auto end_time = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end_time - start_time);
    PLOG(ATB_INFO, "A2BSService ParseInputsFromJson execution time: " + std::to_string(duration.count()) + " ms");
}

std::vector<FLAMEOuput> AudioTo3DGSBlendShape::ProcessFLAME(const std::vector<float> &raw_audio,
                                                       int sample_rate)
{
    auto t0 = clk::now();

    // 1. 先根据音频计算对应的Flame系数，得到[N, 53]的2维std::vector， N为对应的数据帧
    // N可以根据音频长度x20得到，因为一秒对应20帧的数据
    auto flame_bs_list = atf.Process(raw_audio, sample_rate);
    auto t1 = clk::now();

    int num_frames = flame_bs_list.size();

    std::vector<FLAMEOuput> result;

    for (int i = 0; i < num_frames; i ++) {
        auto flame_bs = flame_bs_list[i];
        std::vector<float> expression(flame_bs.begin(), flame_bs.begin() + _num_exp);
        std::vector<float> jaw_pose(flame_bs.end() - 3, flame_bs.end());
        FLAMEOuput tmp;
        tmp.frame_id = i;
        tmp.jaw_pose = jaw_pose;
        tmp.expr = expression;
        result.push_back(tmp);
    }

    auto t2 = clk::now();
    auto duration_total = std::chrono::duration_cast<ms>(t2 - t0);
    auto duration_atf = std::chrono::duration_cast<ms>(t1 - t0);
    auto duration_convert = duration_total - duration_atf;

    float audio_len_in_ms = float(raw_audio.size() * 1000) / float(sample_rate);
    float total_timecost_in_ms = (float)duration_total.count();
    float atf_timecost_in_ms = (float)duration_atf.count();
    float convert_timecost_in_ms = (float)duration_convert.count();
    float total_rtf = total_timecost_in_ms / audio_len_in_ms;
    float atf_rtf = atf_timecost_in_ms / audio_len_in_ms;
    float convert_rtf = convert_timecost_in_ms / audio_len_in_ms;

    PLOG(ATB_INFO, "Audio2BS timecost: " + std::to_string(total_timecost_in_ms) + "ms, audio_duration: " + std::to_string(audio_len_in_ms) + "ms, rtf:" + std::to_string(total_rtf) + "(" + std::to_string(atf_rtf) + "+" + std::to_string(convert_rtf) + ")");
    return result;
}