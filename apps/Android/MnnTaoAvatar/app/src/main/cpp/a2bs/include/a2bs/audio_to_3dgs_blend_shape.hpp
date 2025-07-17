#pragma once

#include <chrono>
#include "audio_to_flame_blend_shape.hpp"
#include "gs_body_converter.hpp"
#include "a2bs_utils.hpp"
#include "mnn_audio2bs_logger.hpp"

class AudioTo3DGSBlendShape {
public:
    AudioTo3DGSBlendShape(const std::string &local_resource_root,
                          const std::string &mnn_mmap_dir,
                          bool do_verts2flame,
                          int ori_fps=25,
                          int out_fps=20,
                          int num_exp=50);
    std::vector<FLAMEOuput> ProcessFLAME(const std::vector<float> &audio,
                                        int sample_rate);

private:
    std::string local_resource_root_;
    AudioToFlameBlendShape atf;
    int _num_exp;
    std::map<int, AudioToFlameBlendShape> _audio_to_blendshape_data;
};