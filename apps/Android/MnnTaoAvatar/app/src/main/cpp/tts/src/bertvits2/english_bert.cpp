#include "english_bert.hpp"

EnglishBert::EnglishBert() {
}

EnglishBert::EnglishBert(const std::string& local_resource_root) {
    local_resource_root_ = local_resource_root;

    // 读取bert token json
    auto json_path = local_resource_root_ + "common/text_processing_jsons/en_bert_token.json";
    ParseBertTokenJsonFile(json_path);

    // bert_tokenizer_ = BertTokenizer();
    // 配置默认全局Exector
    MNN::BackendConfig backend_config;  // default backend config
    // 设置使用4线程+CPU
    Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, backend_config, 4);

    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig),
                                                    Executor::RuntimeManager::destroy);

    if (rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
    }
    rtmgr->setCache(".cachefile");

    Module::Config mdconfig;  // default module config
    // 若 rtMgr 为 nullptr ，Module 会使用Executor的后端配置
    auto mnn_model_path = local_resource_root_ + "common/mnn_models/english_bert.mnn";
    module.reset(Module::load(input_names, output_names, mnn_model_path.c_str(), rtmgr, &mdconfig));
    PLOG(INFO, "en_bert模型加载成功: " + mnn_model_path);
}

void EnglishBert::ParseBertTokenJsonFile(const std::string& json_path) {
    json json_obj;
    std::ifstream file(json_path);
    file >> json_obj;
    file.close();

    // 将json对象转换为std::map
    bert_token_ = json_obj.get<bert_token>();
}

std::vector<int> EnglishBert::ObtainBertTokens(const std::string& text) {
    std::vector<int> tokens = {1};
    // auto split_text = split_string(text, ' ');

    auto split_text = SplitEnSentenceToWords(text);
    for (auto& word : split_text) {
        auto tmp_word = word;
        if (IsAlphabet(word[0])) {
            tmp_word = "▁" + word;
        }
        auto cur_token = bert_token_[tmp_word];
        tokens.push_back(cur_token);
    }
    tokens.push_back(2);
    return tokens;
}

std::vector<std::vector<float>> EnglishBert::Process(const std::string& text, const std::vector<int>& word2ph) {
    auto input_ids = ObtainBertTokens(text);
    int token_num = input_ids.size();
    // std::vector<float> res(feature_size*token_num, 0.0);
    // return res;

    // auto input_ids = bert_tokenizer_.encode(text);

    //    int token_num = input_ids.size();
    std::vector<int> token_type_ids(token_num, 0);
    std::vector<int> attention_mask(token_num, 1);

    std::vector<VARP> inputs(3);
    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
    inputs[0] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[1] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[2] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());

    // 设置输入数据
    std::vector<int*> input_pointer = {inputs[0]->writeMap<int>(), inputs[1]->writeMap<int>(),
                                       inputs[2]->writeMap<int>()};
    for (int i = 0; i < token_num; ++i) {
        input_pointer[0][i] = input_ids[i];
        input_pointer[1][i] = token_type_ids[i];
        input_pointer[2][i] = attention_mask[i];
    }

    // 执行推理
    std::vector<MNN::Express::VARP> outputs = module->onForward(inputs);

    // 获取输出
    // std::vector<float> res(int(feature_size * token_num), 0);
    // auto output_ptr = outputs[0]->readMap<float>();
    // int output_size = outputs[0]->getInfo()->size;

    // float* tmp_ptr = (float*)output_ptr;
    // for (int i = 0; i < feature_size * token_num; ++i) {
    //     res[i] = tmp_ptr[i];
    // }

    std::vector<std::vector<float>> res; // [token_num, bert_feature_dim_]
    auto output_ptr = outputs[0]->readMap<float>();
    int output_size = outputs[0]->getInfo()->size;
    PLOG(PDEBUG, "en_bert 输出bert feature 维度:" + std::to_string(output_size));
    float *tmp_ptr = (float *)output_ptr;
    float bert_sum = 0.0f;
    for (int i = 0; i < token_num; ++i)
    {
        std::vector<float> cur_bert_feat;
        for (int j = 0; j < bert_feature_dim_; j++)
        {
            cur_bert_feat.push_back(tmp_ptr[i * bert_feature_dim_ + j]);
        }
        res.push_back(cur_bert_feat);
        // res[i] = tmp_ptr[i];
        bert_sum += tmp_ptr[i];
    }

    // 为了中文输入也能采用英文bert模型（为了避免引入额外的bert模型增加内存占用）
    // 但tokenizer方式不同，因此用中文的tokenizer得到的input_ids与word2ph不
    // 一定长度相等，例如MyName这种。为了避免DuplicateBlocks中的assert报错，这里
    // 将res和word2ph的长度保持一致
    if (res.size() < word2ph.size())
    {
        auto last_element = res[res.size() - 1];
        int res_size = res.size();
        for (int i = 0; i < word2ph.size() - res_size; i++)
        {
            res.push_back(last_element);
        }
    }
    else if (res.size() > word2ph.size())
    {
        res.resize(word2ph.size());
    }

    // 根据word2ph对原始的网络输出进行复制，如word2ph=[2, 1, 3]，则会对原始的【1024xN]的数据
    // 的第二个维度依次复制2次，1次和3次
    auto final_feat = DuplicateBlocks(res, word2ph, bert_feature_dim_);
    return final_feat;
}
