/**
 * @file word_spliter.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 分词算法
 *
 * 分词算法，基于jieba python版本代码重写
 * 为了避免每次初始化时加载json文件引入额外耗时，将该类设计为单例模式，只创建一次
 */
#ifndef _HEADER_MNN_TTS_SDK_WORD_SPLITER_H_
#define _HEADER_MNN_TTS_SDK_WORD_SPLITER_H_

#include "utils.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

const double MIN_FLOAT = std::numeric_limits<double>::lowest(); // 替代Python中的MIN_FLOAT
const double MIN_INF = MIN_FLOAT;                               // 替代Python中的MIN_INF

typedef std::unordered_map<std::string, int> word_freq_map;
typedef std::unordered_map<std::string, std::string> word_tag_map;
typedef std::unordered_map<std::string, std::vector<std::string>> char_state_map;
typedef std::unordered_map<std::string, std::unordered_map<std::string, double>> prob_emit_map;
typedef std::unordered_map<std::string, double> prob_start_map;
typedef std::unordered_map<std::string, std::unordered_map<std::string, double>> prob_trans_map;

class WordSpliter
{
public:
    static WordSpliter &GetInstance(const std::string &local_resource_root);

    // 对输入的分句进行分词，分句可能包含标点符号，如'你好!'
    // for_search为true，则对结果中所有前后缀为单词的部分进行进一步分词，用于推理引擎的词表建立索引
    // 如for_search为false时，'我的男朋友' 结果为[我', '的', '男朋友']，for_search为true时为['我', '的', '朋友', '男朋友']
    std::vector<WordPosPair> Process(const std::string &seg, bool for_search = false);

    // 对输入的分句进行分词，不过只返回分词后的词语，不返回词性
    std::vector<std::string> ProcessWoPos(const std::string &seg, bool for_search = false);

private:
    // 从json中读取中间参数
    void ParseWordFreq(const std::string &json_path);
    void ParseWordTag(const std::string &json_path);
    void ParseCharState(const std::string &json_path);
    void ParseProbEmit(const std::string &json_path);
    void ParseProbStart(const std::string &json_path);
    void ParseProbTrans(const std::string &json_path);

    // 中间参数写入bin/从bin读取
    void SaveWordFreqToBin(const std::string &filename, const word_freq_map &word_freq);
    void SaveWordTagToBin(const std::string &filename, const word_tag_map &word_tag);
    void SaveCharStateToBin(const std::string &filename, const char_state_map &char_state);
    void SaveProbEmitToBin(const std::string &filename, const prob_emit_map &prob_emit);
    void SaveProbStartToBin(const std::string &filename, const prob_start_map &prob_start);
    void SaveProbTransToBin(const std::string &filename, const prob_trans_map &prob_trans);
    void LoadWordFreqFromBin(const std::string &filename, word_freq_map &word_freq);
    void LoadWordTagFromBin(const std::string &filename, word_tag_map &word_tag);
    void LoadCharStateFromBin(const std::string &filename, char_state_map &char_state);
    void LoadProbEmitFromBin(const std::string &filename, prob_emit_map &prob_emit);
    void LoadProbStartFromBin(const std::string &filename, prob_start_map &prob_start);
    void LoadProbTransFromBin(const std::string &filename, prob_trans_map &prob_trans);

    void SaveAllToBin();
    void LoadAllFromBin();

    // HMM中间处理函数
    std::pair<double, std::vector<std::string>> Viterbi(const std::vector<std::string> &obs);
    std::vector<WordPosPair> Cut(const std::string &sentence);
    std::vector<WordPosPair> CutDetail(const std::string &sentence);
    std::vector<WordPosPair> CutDAG(const std::vector<std::string> &sentence, const std::string &raw_s);
    std::vector<std::string> CutDAGNoHMM(const std::vector<std::string> &sentence, const std::string &raw_s);
    // 计算当前句子对应的汉字的DAG
    std::unordered_map<int, std::vector<int>> GetDAG(const std::vector<std::string> &sentence);
    void calc(const std::vector<std::string> &sentence, const std::unordered_map<int, std::vector<int>> &DAG,
              std::unordered_map<int, std::pair<double, int>> &route);
    void ParseHotwordsCNFile(const std::string &hotwords_cn_json_path);

    // 根据词库内容，自动计算用户添加的词语的词频
    int SuggestFreq(const std::string &segment);

private:
    // 私有化构造函数
    WordSpliter(const std::string &local_resource_root);
    static std::unique_ptr<WordSpliter> instance;
    static std::mutex mtx;

private:
    // 资源文件根目录
    std::string resource_root_;

    // 保存HMM中间状态的变量
    word_freq_map word_freq_;
    word_tag_map word_tag_;
    char_state_map char_state_;
    prob_emit_map prob_emit_;
    prob_start_map prob_start_;
    prob_trans_map prob_trans_;
    int total = -1;
};
#endif
