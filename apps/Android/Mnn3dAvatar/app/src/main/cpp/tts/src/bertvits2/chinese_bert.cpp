#include "chinese_bert.hpp"

ChineseBert::ChineseBert() {}

ChineseBert::ChineseBert(const std::string &local_resource_root, const std::string &mnn_mmap_dir)
{
    resource_root_ = local_resource_root;

    // 读取bert token json
    LoadBertTokenFromBin(resource_root_ + "common/text_processing_jsons/cn_bert_token.bin", bert_token_);

    MNN::BackendConfig backend_config; // default backend config
    // 设置使用1线程+CPU
    executor_ = Executor::newExecutor(MNN_FORWARD_CPU, backend_config, 1);
    ExecutorScope scope(executor_);

    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    BackendConfig cpuBackendConfig;
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    sConfig.numThread = 1;
    sConfig.backendConfig = &cpuBackendConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig),
                                                    Executor::RuntimeManager::destroy);

    if (rtmgr == nullptr)
    {
        MNN_ERROR("Empty RuntimeManger\n");
    }
//    rtmgr->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1);
    rtmgr->setCache(".cachefile");
//    rtmgr->setExternalPath(mnn_mmap_dir, Interpreter::EXTERNAL_WEIGHT_DIR);
//    rtmgr->setHint(MNN::Interpreter::MMAP_FILE_SIZE, 768);
//    rtmgr->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);

    Module::Config mdconfig; // default module config
    mdconfig.rearrange = true;

    auto mnn_model_path = resource_root_ + "common/mnn_models/chinese_bert.mnn";
    module_.reset(Module::load(input_names, output_names, mnn_model_path.c_str(), rtmgr, &mdconfig));
    PLOG(INFO, "bert模型加载成功: " + mnn_model_path);
}

void ChineseBert::ParseBertTokenJsonFile(const std::string &json_path)
{
    auto json_obj = LoadJson(json_path);

    // 将json对象转换为std::map
    bert_token_ = json_obj.get<bert_token>();
}

void ChineseBert::SaveBertTokenToBin(const std::string &filename, const bert_token &token)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Unable to open file for writing");

    // 写入字典大小
    size_t size = token.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (const auto &pair : token)
    {
        // 写入键的大小和内容
        size_t key_size = pair.first.size();
        ofs.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        ofs.write(pair.first.c_str(), key_size);

        // 写入值
        ofs.write(reinterpret_cast<const char *>(&pair.second), sizeof(pair.second));
    }

    ofs.close();
}

void ChineseBert::LoadBertTokenFromBin(const std::string &filename, bert_token &token)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Unable to open file for reading");

    // 读取字典大小
    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i)
    {
        // 读取键的大小
        size_t key_size;
        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::string key(key_size, '\0');
        ifs.read(&key[0], key_size); // 读取键

        // 读取值
        int value;
        ifs.read(reinterpret_cast<char *>(&value), sizeof(value));

        token[key] = value; // 填充到 bert_token
    }

    ifs.close();
}

std::vector<int> ChineseBert::ObtainBertTokens(const std::string &text, const std::string &lang)
{
    std::vector<int> tokens = {101};
    // 英文单词，直接用词表中的token_id
    std::vector<std::string> words;
    if (lang == "en")
    {
        words = SplitEnSentenceToWords(text);
        // token词表中所有英文单词都是小写的，所以这里需要将输入转换为小写，否则匹配不上
        for (auto &w : words)
        {
            w = ToLowercase(w);
        }
    }
    else
    {
        words = SplitUtf8String(text);
    }

    PLOG(PDEBUG, "bert words: " + ConcatStrList(words, "|"));

    for (auto &word : words)
    {
        auto cur_token = bert_token_[word];
        if (cur_token == 0)
        {
            cur_token = 100;
        }
        tokens.push_back(cur_token);
    }

    tokens.push_back(102);
    return tokens;
}

std::vector<std::vector<float>> ChineseBert::Process(const std::string &text, const std::vector<int> &word2ph, const std::string &lang)
{
    ExecutorScope scope(executor_);

    PLOG(PDEBUG, "cn_bert 输入文本: " + text);
    auto input_ids = ObtainBertTokens(text, lang);
    int token_num = input_ids.size();
    PLOG(PDEBUG, "cn_bert 输入tokens(token_num=" + std::to_string(token_num) + "):" + ConcatIntList(input_ids, "|"));
    PLOG(PDEBUG, "cn_bert 输入word2ph(size=" + std::to_string(word2ph.size()) + "):" + ConcatIntList(word2ph, "|"));

    std::vector<int> token_type_ids(token_num, 0);
    std::vector<int> attention_mask(token_num, 1);

    std::vector<VARP> inputs(3);
    // 对于 tensoflow 转换过来的模型用 NHWC ，由 onnx 转换过来的模型用 NCHW
    inputs[0] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[1] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());
    inputs[2] = MNN::Express::_Input({1, token_num}, NCHW, halide_type_of<int>());

    // 设置输入数据
    std::vector<int *> input_pointer = {inputs[0]->writeMap<int>(), inputs[1]->writeMap<int>(),
                                        inputs[2]->writeMap<int>()};
    for (int i = 0; i < token_num; ++i)
    {
        input_pointer[0][i] = input_ids[i];
        input_pointer[1][i] = attention_mask[i];
        input_pointer[2][i] = token_type_ids[i];
    }

    // 执行推理
    std::vector<MNN::Express::VARP> outputs = module_->onForward(inputs);

    // 获取输出
    std::vector<std::vector<float>> res; // [token_num, bert_feature_dim_]
    auto output_ptr = outputs[0]->readMap<float>();
    int output_size = outputs[0]->getInfo()->size;
    PLOG(PDEBUG, "cn_bert 输出bert feature 维度:" + std::to_string(output_size));
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
    PLOG(PDEBUG, "cn_bert output_sum: " + std::to_string(bert_sum));

    // 目前英文输入也采用中文bert模型（为了避免引入额外的英文bert模型增加内存占用）
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

    // 将bert特征根据word2ph进行复制，比如一个word对应2个phone，那么feature会复制2次
    auto final_feat = DuplicateBlocks(res, word2ph, bert_feature_dim_);
    return final_feat;
}
