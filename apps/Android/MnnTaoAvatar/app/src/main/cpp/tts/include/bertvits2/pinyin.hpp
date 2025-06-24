/**
 * @file pinyin.hpp
 * @author MNN Team
 * @date 2024-01-25
 * @version 1.0
 * @brief 文本转拼音
 *
 * 对文本进行分词，然后获取对应的拼音。实现参考pypinyin.lazy_pinyin，
 * 代码参见：https://github.com/mozillazg/python-pinyin/blob/master/pypinyin/core.py
 */

#ifndef _HEADER_MNN_TTS_SDK_PINYIN_H_
#define _HEADER_MNN_TTS_SDK_PINYIN_H_
#include "utils.hpp"
#include "word_spliter.hpp"

typedef std::string pinyin_type;
typedef std::vector<std::string> pinyin_list_type;
typedef std::vector<std::string> phrase_type;
typedef std::map<std::string, pinyin_type> pinyin_map;
typedef std::map<std::string, phrase_type> phrase_map;
typedef std::map<std::string, std::string> pinyin_to_symbol_map;

using json = nlohmann::json;

enum PinyinStyle
{
    //: 普通风格，不带声调。如： 中国 -> ``zhong guo``
    NORMAL = 0,
    //: 标准声调风格，拼音声调在韵母第一个字母上（默认风格）。如： 中国 -> ``zhōng guó``
    TONE = 1,
    //: 声调风格2，即拼音声调在各个韵母之后，用数字 [1-4] 进行表示。如： 中国 -> ``zho1ng guo2``
    TONE2 = 2,
    //: 声调风格3，即拼音声调在各个拼音之后，用数字 [1-4] 进行表示。如： 中国 -> ``zhong1 guo2``
    INITIALS = 3,
    //: 首字母风格，只返回拼音的首字母部分。如： 中国 -> ``z g``
    FIRST_LETTER = 4,
    //: 韵母风格，只返回各个拼音的韵母部分，不带声调。如： 中国 -> ``ong uo``
    FINALS = 5,
    //: 标准韵母风格，带声调，声调在韵母第一个字母上。如：中国 -> ``ōng uó``
    FINALS_TONE = 6,
    //: 韵母风格2，带声调，声调在各个韵母之后，用数字 [1-4] 进行表示。如： 中国 -> ``o1ng uo2``
    FINALS_TONE2 = 7,
    //: 韵母风格3，带声调，声调在各个拼音之后，用数字 [1-4] 进行表示。如： 中国 -> ``ong1 uo2``
    TONE3 = 8,
    //: 声母风格，只返回各个拼音的声母部分（注：有的拼音没有声母，详见 `//27`_）。如： 中国 -> ``zh g``
    FINALS_TONE3 = 9,
    //: 注音风格，带声调，阴平（第一声）不标。如： 中国 -> ``ㄓㄨㄥ ㄍㄨㄛˊ``
    BOPOMOFO = 10,
    //: 注音风格，仅首字母。如： 中国 -> ``ㄓ ㄍ``
    BOPOMOFO_FIRST = 11,
    //: 汉语拼音与俄语字母对照风格，声调在各个拼音之后，用数字 [1-4] 进行表示。如： 中国 -> ``чжун1 го2``
    CYRILLIC = 12,
    //: 汉语拼音与俄语字母对照风格，仅首字母。如： 中国 -> ``ч г``
    CYRILLIC_FIRST = 13,
    //: 威妥玛拼音/韦氏拼音/威式拼音风格，无声调
    WADEGILES = 14,

};

extern const std::vector<std::string> _INITIALS;
extern std::map<std::string, std::string> PHONETIC_SYMBOL_DICT;

/**
 * @brief 参考 pypinyin中的lazy_pinyin实现的简化版本
 *
 */
class Pinyin
{
public:
    Pinyin();
    /**
     * @brief 初始化文字转拼音的类，对本地的资源文件进行读取
     *
     * @param local_resouce_root 资源文件在本地的保存地址
     */
    Pinyin(const std::string &local_resource_root);

    /**
     * @brief 添加自定义词组，将标准词组中没有考虑的case加进去
     *
     * @param text 待检测字符串
     */
    bool AddCustomPhrases(const std::map<std::string, std::vector<std::string>> &map);

    /**
     * @brief 检查某个字符串是否为固定的词组
     *
     * @param text 待检测字符串
     */
    bool IsPhrase(const std::string &text);

    bool IsInPrefixSet(const std::string &text);

    /**
     * @brief 获取某个词组对应的拼音
     *
     * @param text 待检测字符串
     */
    phrase_type ObtainPinyinOfPhrase(const std::string &text);

    /**
     * @brief 检查某个字符串是否为单个汉字
     *
     * @param text 待检测字符串
     */
    bool IsSingleHan(std::string &text);

    /**
     * @brief 获取单个汉字对应的拼音
     *
     * @param text 待检测汉字
     */
    pinyin_type ObtainPinyinOfSingleHan(const std::string &text);

    /**
     * @brief 从词语构造prefix_set数据，用于后面的分词步骤
     *
     */
    void PrepareSegPrefixSet();

    /**
     * @brief 输入文本，返回对应的pinyin列表，目前只支持拼音声母+韵母带声调的结果
     *
     * @param text 输入文本，传入的文本已经是经过预处理的，可以保证无特殊字符，但还是会有英文和数字和标点
     * @return 输入文本对应的拼音列表
     */
    std::tuple<std::vector<std::string>, std::vector<std::string>> Process(const std::string &text);

private:
    /**
     * @brief 解析pinyin_dict.json中的数据，转换为类的对象
     *
     * @param pinyin_json_path pinyin_dict.json文件的的完整路径
     */
    void ParsePinyinJsonFile(const std::string &pinyin_json_path);

    /**
     * @brief 解析phrase_dict.json中的数据，转换为类的对象
     *
     * @param phrase_json_path phrase_dict.json文件的的完整路径
     */
    void ParsePhraseJsonFile(const std::string &phrase_json_path);

    /**
     * @brief 解析hotwords_cn.json中的定制化热词表，加入到pharse_map中
     *
     * @param hotwords_cn_json_path hotwords_cn.json文件的的完整路径
     */
    void ParseHotwordsCNFile(const std::string &hotwords_cn_json_path);

    // 增加音调
    std::vector<std::string> AdjustTones(const std::vector<std::string> &pinyin_list,
                                         const std::vector<bool> &is_pinyin_valid_list);

    /**
     * @brief 对整个句子进行切分，得到短语或者单个的字
     *
     * @param text 输入文本
     * @return 分好的词的列表
     */
    std::vector<std::string> SentenceSplit(const std::string &text);

    // 保存中间变量到二进制bin文件
    void SavePinyinMapToBin(const std::string &filename, const pinyin_map &pinyin_map);
    void SavePhraseMapToBin(const std::string &filename, const phrase_map &phrase_map);
    void SavePinyinToSymbolMapToBin(const std::string &filename, const pinyin_to_symbol_map &pinyin_to_symbol);

    // 从二进制bin文件读取中间变量
    void LoadPinyinMapFromBin(const std::string &filename, pinyin_map &pinyin_map);
    void LoadPhraseMapFromBin(const std::string &filename, phrase_map &phrase_map);
    void LoadPinyinToSymbolMapFromBin(const std::string &filename, pinyin_to_symbol_map &pinyin_to_symbol);

private:
    /**
     * @brief 数据在本地的保存根目录
     */
    std::string resource_root_;

    /**
     * @brief 保存汉字unicode-pinyin列表数据对的类成员
     */
    pinyin_map pinyin_map_;

    /**
     * @brief 保存词语-pinyin列表数据对的类成员
     */
    phrase_map phrase_map_;

    /**
     * @brief 保存词语或者词语的前缀集合
     */
    std::set<std::string> prefix_set_;
};
#endif // _HEADER_MNN_TTS_SDK_PINYIN_H_
