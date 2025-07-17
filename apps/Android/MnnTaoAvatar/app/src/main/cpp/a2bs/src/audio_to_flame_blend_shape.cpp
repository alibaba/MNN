#include "a2bs/audio_to_flame_blend_shape.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include "common/mh_log.hpp"



AudioToFlameBlendShape::AudioToFlameBlendShape(const std::string &local_resource_root,
                                               const std::string &mnn_mmap_dir,
                                               const bool do_verts2flame,
                                               const int ori_fps,
                                               const int out_fps,
                                               const int num_exp):
                                               do_verts2flame_(do_verts2flame),
                                               ori_fps_(ori_fps),
                                               out_fps_(out_fps),
                                               _num_exp(num_exp) {
    local_resource_root_ = local_resource_root;

    // 配置默认全局Exector
    MNN::BackendConfig backend_config; // default backend config
                                       // 设置使用4线程+CPU
                                       // Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, backend_config, 4);
    executor_ = Executor::newExecutor(MNN_FORWARD_CPU, backend_config, 1);
    ExecutorScope scope(executor_);

    MNN::ScheduleConfig sConfig;
    BackendConfig cpuBackendConfig;
    cpuBackendConfig.precision = BackendConfig::Precision_High;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    sConfig.numThread = 1;
    sConfig.backendConfig = &cpuBackendConfig;
    sConfig.type = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig),
                                                    Executor::RuntimeManager::destroy);

    if (rtmgr == nullptr)
    {
        MNN_ERROR("Empty RuntimeManger\n");
    }
    rtmgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    rtmgr->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1);
    rtmgr_ = rtmgr;

    Module::Config md_config; // default module config
    md_config.rearrange = true;
    // 若 rtMgr 为 nullptr ，Module 会使用Executor的后端配置


    float mem_0, mem_1;
    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
//    PLOG(ATB_INFO, "mnn_feat_dir: " + mnn_mmap_dir);
//    rtmgr->setExternalPath(mnn_mmap_dir, Interpreter::EXTERNAL_FEATUREMAP_DIR);

    auto audio2verts_model_path = local_resource_root_ + "audio2verts.mnn";
    audio2verts_module.reset(Module::load(audio2verts_input_names, audio2verts_output_names, audio2verts_model_path.c_str(), rtmgr, &md_config));
    if (do_verts2flame_){
        auto verts2flame_model_path = local_resource_root_ + "verts2flame.mnn";
        verts2flame_module.reset(Module::load(verts2flame_input_names, verts2flame_output_names, verts2flame_model_path.c_str(), rtmgr, &md_config));
    }

    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
    MNN_PRINT("### audio_to_flame_blend_shape load memory increase : %f \n", mem_1 - mem_0);

}

std::vector<std::vector<float>> AudioToFlameBlendShape::Process(const std::vector<float> &raw_audio,
                                                                int sample_rate) {
    ExecutorScope scope(executor_);

    auto audio(raw_audio);
    std::vector<VARP> audio2verts_inputs(1);

    // 进行重采样，保证采样率为16000
    audio = resampleAudioData(audio, sample_rate, 16000);

    // 进行normalize
    audio = normalizeAudio(audio);

    int audio_length = audio.size();
    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
    audio2verts_inputs[0] = MNN::Express::_Input({1, audio_length}, NCHW, halide_type_of<float>());

    // 设置输入数据
    std::vector<float *> input_pointer = {audio2verts_inputs[0]->writeMap<float>()};
    for (int i = 0; i < audio_length; ++i)
    {
        input_pointer[0][i] = audio[i];
    }

    // 执行推理
    float mem_0, mem_1;
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
    std::vector<MNN::Express::VARP> outputs = audio2verts_module->onForward(audio2verts_inputs);
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
    MNN_PRINT("### audio2verts forward memory increase : %f \n", mem_1 - mem_0);

    // 获取audio2verts输出
    auto output0_ptr = outputs[0]->readMap<float>();
    int output0_size = outputs[0]->getInfo()->size;
    int flame_out_size = output0_size;
    float *tmp0_ptr = (float *)output0_ptr;

    if(do_verts2flame_){
        // 转换成verts2flame输入
        std::vector<VARP> verts2flame_inputs(1);
        int frame_num = output0_size / (5023 * 3) + 1;
        verts2flame_inputs[0] = MNN::Express::_Input({frame_num, 5023, 3}, NCHW, halide_type_of<float>());
        std::vector<float *> verts2flame_input_pointer = {verts2flame_inputs[0]->writeMap<float>()};

        for (int i = 0; i < output0_size; ++i)
        {
            verts2flame_input_pointer[0][i] = tmp0_ptr[i];
        }
        // 最后补一帧
        for (int i = output0_size; i < output0_size + 5023 * 3; ++i)
        {
            verts2flame_input_pointer[0][i] = tmp0_ptr[i - 5023 * 3];
        }


        // 获取verts2flame输出
        rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
        std::vector<MNN::Express::VARP> vert2flame_outputs = verts2flame_module->onForward(verts2flame_inputs);
        executor_->gc(MNN::Express::Executor::FULL);

        rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
        MNN_PRINT("### verts2flame forward memory increase : %f \n", mem_1 - mem_0);

        auto vert2flame_out_ptr = vert2flame_outputs[0]->readMap<float>();
        flame_out_size = vert2flame_outputs[0]->getInfo()->size;
        printf("===> audio2flame expr&jaw size: %d\n", flame_out_size);
        tmp0_ptr = (float *)vert2flame_out_ptr;
    }

    std::vector<float> coeffs;
    coeffs.reserve(flame_out_size);
    for (int i = 0; i < flame_out_size; ++i) {
        coeffs.push_back(tmp0_ptr[i]);
    }

    int coeff_num = _num_exp + 3;

    if(false == do_verts2flame_){
        // 直接预测flame系数，要额外补一帧flame系数
        for(int i = flame_out_size; i < flame_out_size + coeff_num; ++i){
            coeffs.push_back(tmp0_ptr[i - coeff_num]);
        }
    }

    int num_rows = coeffs.size() / coeff_num;

    // 将1维[N*103]的结果转换为2维的[N, 103]数据，方便将N从25fps采样到20FPS
    auto coeffs_2d = convert_to_2d(coeffs, num_rows, coeff_num);

    // 将25fps采样到20fps
    auto resample_coeffs_2d = resample_bs_params(coeffs_2d, static_cast<int>((static_cast<double>(coeffs_2d.size())) / ori_fps_ * out_fps_));

    return resample_coeffs_2d;
}
