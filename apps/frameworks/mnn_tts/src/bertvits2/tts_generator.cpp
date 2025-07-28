#include "tts_generator.hpp"

TTSGenerator::TTSGenerator() {}

TTSGenerator::TTSGenerator(const std::string &tts_generator_model_path, const std::string &mnn_mmap_dir)
{
    // 配置默认全局Exector
    MNN::BackendConfig backend_config; // default backend config
    executor_ = Executor::newExecutor(MNN_FORWARD_CPU, backend_config, 4);
    ExecutorScope scope(executor_);

    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    BackendConfig cpuBackendConfig;
    cpuBackendConfig.precision = BackendConfig::Precision_Normal;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    sConfig.numThread = 1;
    sConfig.backendConfig = &cpuBackendConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig),
                                                    Executor::RuntimeManager::destroy);

    if (rtmgr == nullptr)
    {
        MNN_ERROR("Empty RuntimeManger\n");
    }
    rtmgr_ = rtmgr;
    rtmgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    rtmgr->setCache(".tts_generator_cachefile");

    Module::Config mdconfig; // default module config
    mdconfig.rearrange = true;
    // 若 rtMgr 为 nullptr ，Module 会使用Executor的后端配置
    float mem_0, mem_1;
    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
    module_.reset(Module::load(input_names, output_names, tts_generator_model_path.c_str(), rtmgr, &mdconfig));
    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);

    PLOG(INFO, "tts 模型加载成功: " + tts_generator_model_path);
    MNN_PRINT("### tts load memory increase : %f \n", mem_1 - mem_0);
}

std::vector<int16_t> TTSGenerator::Process(const phone_data &g2p_data_, const std::vector<std::vector<float>> &cn_bert, const std::vector<std::vector<float>> &en_bert)
{
    ExecutorScope scope(executor_);
    std::vector<VARP> inputs(5);
    auto phones = std::get<0>(g2p_data_);
    auto tones = std::get<1>(g2p_data_);
    auto lang_ids = std::get<2>(g2p_data_);
    int token_num = phones.size();
    PLOG(PDEBUG, "generator phones(size=" + std::to_string(phones.size()) + "):" + ConcatIntList(phones, "|"));
    PLOG(PDEBUG, "generator tones(size=" + std::to_string(tones.size()) + "):" + ConcatIntList(tones, "|"));
    PLOG(PDEBUG, "generator lang_ids(size=" + std::to_string(lang_ids.size()) + "):" + ConcatIntList(lang_ids, "|"));

    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
    inputs[0] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[1] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[2] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[3] = MNN::Express::_Input({1, bert_feature_dim_, token_num}, NCHW, halide_type_of<float>());
    inputs[4] = MNN::Express::_Input({1, bert_feature_dim_, token_num}, NCHW, halide_type_of<float>());
    // inputs[5] = MNN::Express::_Input({1}, NCHW, halide_type_of<float>());

    // 设置输入数据，为了能够批量赋值，将shape一样的输入放到同一个指针列表里
    std::vector<int *> input_pointer0 = {inputs[0]->writeMap<int>(), inputs[1]->writeMap<int>(),
                                         inputs[2]->writeMap<int>()};

    std::vector<float *> input_pointer1 = {inputs[3]->writeMap<float>(), inputs[4]->writeMap<float>()};

    for (int i = 0; i < token_num; ++i)
    {
        input_pointer0[0][i] = phones[i];
        input_pointer0[1][i] = tones[i];
        input_pointer0[2][i] = lang_ids[i];
    }

    for (int i = 0; i < bert_feature_dim_; ++i)
    {
        for (int j = 0; j < token_num; j++)
        {
            input_pointer1[0][i * token_num + j] = cn_bert[j][i];
            //            input_pointer1[1][i * token_num + j] = en_bert[j][i];
            input_pointer1[1][i * token_num + j] = 0.0f;
        }
    }

    // 执行推理
    float mem_0, mem_1;
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);

    std::vector<MNN::Express::VARP> outputs = module_->onForward(inputs);
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
    executor_->gc(MNN::Express::Executor::FULL);
    MNN_PRINT("### tts forward memory increase : %f \n", mem_1 - mem_0);

    // 获取输出
    auto output_ptr = outputs[0]->readMap<float>();
    int output_size = outputs[0]->getInfo()->size;
    std::vector<float> ret(output_size);
    float *tmp_ptr = (float *)output_ptr;
    float out_sum = 0.0f;
    for (int i = 0; i < output_size; ++i)
    {
        ret[i] = tmp_ptr[i];
        out_sum += tmp_ptr[i];
    }

    PLOG(PDEBUG, "generator output_sum: " + std::to_string(out_sum));

    auto audio_data = ConvertAudioToInt16(ret);
    return audio_data;
}
