#include "piper/audio_generator.hpp"

AudioGenerator::AudioGenerator()
{
}

AudioGenerator::AudioGenerator(const std::string &model_path)
{
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

    Module::Config mdconfig; // default module config
    module_.reset(Module::load(input_names, output_names, model_path.c_str(), rtmgr, &mdconfig));
}

std::vector<float> AudioGenerator::Process(const std::vector<int> &input, int input_lengths, const std::vector<float> &scales)
{
    ExecutorScope scope(executor_);

    std::vector<VARP> inputs(3);

    int input_length = input.size();
    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW

    inputs[0] = MNN::Express::_Input({1, input_length}, NCHW, halide_type_of<int>());
    inputs[1] = MNN::Express::_Input({1}, NCHW, halide_type_of<int>());
    inputs[2] = MNN::Express::_Input({3}, NCHW, halide_type_of<float>());

    // 设置输入数据，为了能够批量赋值，将shape一样的输入放到同一个指针列表里
    std::vector<int *> input_pointer0 = {inputs[0]->writeMap<int>()};
    std::vector<int *> input_pointer1 = {inputs[1]->writeMap<int>()};
    std::vector<float *> input_pointer2 = {inputs[2]->writeMap<float>()};

    for (int i = 0; i < input_length; ++i)
    {
        input_pointer0[0][i] = input[i];
    }
    for (int i = 0; i < 1; ++i)
    {
        input_pointer1[0][i] = input_lengths;
    }
    for (int i = 0; i < 3; ++i)
    {
        input_pointer2[0][i] = scales[i];
    }

    // 执行推理
    std::vector<MNN::Express::VARP> outputs = module_->onForward(inputs);

    // 获取输出
    auto output_ptr = outputs[0]->readMap<float>();
    int output_size = outputs[0]->getInfo()->size;
    std::vector<float> ret(output_size);
    float *tmp_ptr = (float *)output_ptr;
    for (int i = 0; i < output_size; ++i)
    {
        ret[i] = tmp_ptr[i];
    }
    return ret;
}
