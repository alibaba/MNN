#include "mnn_bertvits2_tts_impl.hpp"

MNNBertVits2TTSImpl::MNNBertVits2TTSImpl(const std::string &local_resource_root, const std::string &tts_generator_model_path, const std::string &mnn_mmap_dir) : cn_g2p_(local_resource_root)
{
    auto t0 = clk::now();
    PLOG(INFO, "resource_root: " + local_resource_root);

    resource_root_ = local_resource_root;

    // 初始化所有用到的内部处理对象
    text_preprocessor_ = TextPreprocessor();
    PLOG(INFO, "finish TextPreprocessor init.");
    auto t1 = clk::now();

    std::string cn_bert_model_path = resource_root_ + "common/mnn_models/chinese_bert.mnn";
    std::string cn_bert_cache_dir = GetOrCreateHashDirectory(mnn_mmap_dir, cn_bert_model_path);
    PLOG(INFO, "cn_bert cache_dir:" + cn_bert_cache_dir);

    cn_bert_model_ = ChineseBert(resource_root_, cn_bert_cache_dir);
    PLOG(INFO, "finish ChineseBert init.");
    auto t2 = clk::now();
    en_bert_model_= EnglishBert(local_resource_root),

    en_g2p_ = EnglishG2P(resource_root_);
    PLOG(INFO, "finish EnglishG2P init.");
    auto t3 = clk::now();

    std::string tts_generator_cache_dir = GetOrCreateHashDirectory(mnn_mmap_dir, tts_generator_model_path);
    PLOG(INFO, "tts_generator cache_dir:" + tts_generator_cache_dir);
    tts_generator_ = TTSGenerator(tts_generator_model_path, tts_generator_cache_dir);
    auto t4 = clk::now();
    PLOG(INFO, "TTS 初始化成功");

    auto d1 = std::chrono::duration_cast<ms>(t1 - t0);
    auto d2 = std::chrono::duration_cast<ms>(t2 - t1);
    auto d3 = std::chrono::duration_cast<ms>(t3 - t2);
    auto d4 = std::chrono::duration_cast<ms>(t4 - t3);
    PLOG(INFO, "TextPreprocessor timecost: " + std::to_string(d1.count()) + "ms");
    PLOG(INFO, "Load ChineseBert timecost: " + std::to_string(d2.count()) + "ms");
    PLOG(INFO, "Load EnglishG2p timecost: " + std::to_string(d3.count()) + "ms");
    PLOG(INFO, "Load TTSGenerator timecost: " + std::to_string(d4.count()) + "ms");
}

std::tuple<phone_data, std::vector<std::vector<float>>, std::vector<std::vector<float>>> MNNBertVits2TTSImpl::ExtractPhoneTextFeatures(const std::vector<SentLangPair> &word_list_by_lang)
{
    std::vector<std::vector<float>> cn_bert_list, en_bert_list;
    std::vector<std::vector<int>> phone_list, tone_list, langid_list, word2ph_list;

    for (int i = 0; i < word_list_by_lang.size(); i++)
    {
        auto sent_lang = word_list_by_lang[i];
        phone_data phone_data_;
        std::vector<std::vector<float>> cn_bert_feat;
        std::vector<std::vector<float>> en_bert_feat;

        size_t st = 0;
        size_t ed = 0;
        phone_data cur_phone_data;
        std::vector<int> phones;
        std::vector<int> tones;
        std::vector<int> lang_ids;
        std::vector<int> word2ph;
        if (sent_lang.lang == "zh")
        {

            auto t2 = clk::now();
            auto [norm_text, g2p_data] = cn_g2p_.Process(sent_lang);
            phones = std::get<0>(g2p_data);
            tones = std::get<1>(g2p_data);
            lang_ids = std::get<2>(g2p_data);
            word2ph = std::get<3>(g2p_data);

            auto t3 = clk::now();

            cn_bert_feat = cn_bert_model_.Process(norm_text, word2ph, "zh");
            // cn_bert_feat = en_bert_model_.Process(norm_text, word2ph);
            ed = phones.size();
            auto t4 = clk::now();

            auto d_g2p = std::chrono::duration_cast<ms>(t3 - t2);
            auto d_bert = std::chrono::duration_cast<ms>(t4 - t3);

            st = 0;
            ed = phones.size();
            int whole_size = ed;
            int state = 0;
            if (i == 0)
            {
                st = 0;
                ed -= 2;
                state = 0;
            }
            else if (i == word_list_by_lang.size() - 1)
            {
                st = 3;
                state = 1;
            }
            else
            {
                st = 3;
                ed -= 2;
                state = 2;
            }

            PLOG(PDEBUG, "slice st:" + std::to_string(st));
            PLOG(PDEBUG, "slice ed:" + std::to_string(ed));

            phone_list.push_back(VectorSlice(phones, st, ed));
            tone_list.push_back(VectorSlice(tones, st, ed));
            langid_list.push_back(VectorSlice(lang_ids, st, ed));
            word2ph_list.push_back(VectorSlice(word2ph, st, ed));

            cn_bert_feat = SliceBertFeat(cn_bert_feat, st, ed);

            cn_bert_list.insert(cn_bert_list.end(), cn_bert_feat.begin(), cn_bert_feat.end());

            en_bert_feat.resize(cn_bert_feat.size(), std::vector<float>(1024, 0.0));
            en_bert_list.insert(en_bert_list.end(), en_bert_feat.begin(), en_bert_feat.end());
        }
        else
        {
            auto [norm_text, g2p_data] = en_g2p_.Process(sent_lang);

            phones = std::get<0>(g2p_data);
            tones = std::get<1>(g2p_data);
            lang_ids = std::get<2>(g2p_data);
            word2ph = std::get<3>(g2p_data);

            en_bert_feat = en_bert_model_.Process(norm_text, word2ph);
            // en_bert_feat = cn_bert_model_.Process(norm_text, word2ph, "en");

            st = 0;
            ed = phones.size();

            int whole_size = ed;
            if (i == 0)
            {
                st = 0;
                ed -= 2;
            }
            else if (i == word_list_by_lang.size() - 1)
            {
                st = 3;
                ed -= 1;
            }
            else
            {
                st = 3;
                ed -= 2;
            }

            phone_list.push_back(VectorSlice(phones, st, ed));
            tone_list.push_back(VectorSlice(tones, st, ed));
            langid_list.push_back(VectorSlice(lang_ids, st, ed));
            word2ph_list.push_back(VectorSlice(word2ph, st, ed));

            en_bert_feat = SliceBertFeat(en_bert_feat, st, ed);
            en_bert_list.insert(en_bert_list.end(), en_bert_feat.begin(), en_bert_feat.end());

            cn_bert_feat.resize(en_bert_feat.size(), std::vector<float>(1024, 0.0));
            cn_bert_list.insert(cn_bert_list.end(), cn_bert_feat.begin(), cn_bert_feat.end());
        }
    }

    auto phone_data = FlattenVector(phone_list);
    auto tone_data = FlattenVector(tone_list);
    auto lang_id_data = FlattenVector(langid_list);
    auto word2ph_data = FlattenVector(word2ph_list);
    auto g2p_data = std::make_tuple(phone_data, tone_data, lang_id_data, word2ph_data);
    return std::make_tuple(g2p_data, cn_bert_list, en_bert_list);
}

std::tuple<int, Audio> MNNBertVits2TTSImpl::Process(const std::string& in_text)
{
    auto t0 = clk::now();

    PLOG(INFO, "TTS input:" + in_text);

    std::vector<Audio> audio_list;

    // 由于输入文本是const，而代码内部有些地方会对文本进行修改，这里重新创建一个可修改的变量
    std::string text(in_text);

    // 合并多行文件为单行
    text = MergeLines(text);
    PLOG(INFO, "TTS mergeLines:" + text);

    auto sentences = text_preprocessor_.Process(text);
    for (int i = 0; i < sentences.size(); i++)
    {
        PLOG(PDEBUG, "中英文切分后句子" + std::to_string(i) + ": " + ConcatList(sentences[i], "|"));
    }

    for (int i = 0; i < sentences.size(); i++)
    {
        auto words_list = sentences[i];
        PLOG(PDEBUG, "处理句子: " + ConcatList(words_list));
        auto [g2p_data, cn_bert_feat_data, en_bert_feat_data] = ExtractPhoneTextFeatures(words_list);
        Audio audio = tts_generator_.Process(g2p_data, cn_bert_feat_data, en_bert_feat_data);
        audio_list.push_back(audio);
    }

    auto t1 = clk::now();
    auto duration_total = std::chrono::duration_cast<ms>(t1 - t0);

    // 进行空音频增加、音频合并等后处理
    auto audio = ConcatAudio(audio_list, 0, sample_rate_);
    audio = PadAudioForAtb(audio, sample_rate_);
    audio = PadEmptyAudio(audio, sample_rate_, 0);

    float audio_len_in_ms = float(audio.size() * 1000) / float(sample_rate_);
    float timecost_in_ms = (float)duration_total.count();
    float rtf = timecost_in_ms / audio_len_in_ms;

    PLOG(INFO, "TTS timecost: " + std::to_string(timecost_in_ms) + "ms, audio_duration: " + std::to_string(audio_len_in_ms) + "ms, rtf:" + std::to_string(rtf));

    return std::make_tuple(sample_rate_, audio);
}