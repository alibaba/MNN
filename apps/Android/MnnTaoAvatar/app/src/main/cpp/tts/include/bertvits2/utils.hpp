/**
 * @file utils.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 通用的结构体和功能函数存放文件
 *
 * 包含TTS处理过程中各个模块的辅助函数
 */
#ifndef _HEADER_MNN_TTS_SDK_UTILS_H_
#define _HEADER_MNN_TTS_SDK_UTILS_H_

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <codecvt>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

#include "mnn_tts_logger.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

// 将通用的类型定义放置到这里
typedef std::tuple<std::vector<std::string>, std::vector<int>, std::vector<int>> g2p_data;
typedef std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> phone_data;
typedef std::string han;
typedef std::vector<han> han_list;
typedef std::unordered_map<std::string, std::vector<std::vector<std::string>>> english_dict;
typedef std::unordered_map<std::string, int> bert_token;

extern std::string simplified_characters;
extern std::string traditional_characters;
// Define a map with the measurement notations and their replacements.
extern const std::vector<std::vector<std::string>> measure_dict;
extern std::unordered_map<std::string, std::string> DIGITS;
extern std::vector<std::pair<int, std::string>> UNITS;
extern std::unordered_map<int, int> UNITS_POS_MAP;

extern std::unordered_map<std::string, int> language_tone_start_map;
extern std::unordered_map<std::string, int> language_id_map;
extern std::vector<std::string> symbols;

// 表示句子和对应语言的结构体，
struct SentLangPair
{
    std::string sent; // 第一个元素是对应的句子
    std::string lang; // 第二个元素是对应的语言种类，如cn, en等

    SentLangPair(const std::string &word, const std::string &lang) : sent(word), lang(lang) {}

    friend std::ostream &operator<<(std::ostream &os, const SentLangPair &p)
    {
        os << p.sent << "/" << p.lang;
        return os;
    }
};

// 表示中文词语和对应词性的数据对
struct WordPosPair
{
    std::string word;
    std::string flag;

    WordPosPair(const std::string &word, const std::string &flag) : word(word), flag(flag) {}

    friend std::ostream &operator<<(std::ostream &os, const WordPosPair &p)
    {
        os << p.word << "/" << p.flag;
        return os;
    }
};

// 替换字符串中的所有匹配项到新字符串，实现Python中str.replace的功能
std::string StrReplaceAll(std::string &str, const std::string &from, const std::string &to);

// 替换1,000，000形式为1000000形式
std::string ReplaceSpecificCommaNumbers(const std::string &text);

// 删除空行
std::string RemoveEmptyLines(const std::string &text);

// 句尾增加句号
std::string AddPeriodBeforeNewline(const std::string &text);

// 每次替换一个量词
std::string ReplaceOneMeasure(const std::string &text, const std::string &unit, const std::string &cn_word);

// 替换所有量词
std::string ReplaceMeasure(const std::string &sentence);

// 替换货币符号
std::string ReplaceCurrency(const std::string &sentence);

// 替换：为比，例如"9:1"替换为"9比1"
std::string ReplaceCompare(const std::string &sentence);

// 去除字符串和数字中间的-，如GPT-4替换为GPT杠4
std::string ReplaceTyperead(const std::string &sentence);

// 去除字符串和字符串中间的的-，如GPT-4替换为GPT杠4
std::string ReplaceTyperead2(const std::string &sentence);

// 通用的正则表达式替换函数，例如str_replace_using_re_sub("3ml", R"((\d+)\s?ml)", "$1 毫升")会将3ml替换为3 毫升
std::string ReReplace(const std::string &str, const std::string &pattern, const std::string &replacement);

std::vector<std::string> GetValue(const std::string &value_string, bool use_zero = true);
std::string VerbalizeCardinal(const std::string &value_string);

std::string VerbalizeDigit(const std::string &value_string, bool alt_one = false);

std::string Num2Str(const std::string &value_string);

std::string TimeNum2Str(const std::string &num_string);

// 利用传入的函数来对正则表达式匹配到的部分进行替换
std::string RePeplaceByFunc(const std::string &text, const std::regex &pattern,
                            std::function<std::string(const std::smatch &)> func);

// 正则表达式的替换函数
std::string ReplaceDate(const std::smatch &match);
std::string RegexReplaceDate(const std::string &sentence);
std::string replace_date2(const std::smatch &match);

std::string RegexReplaceDate2(const std::string &sentence);

// 将10:00:00～12:00:00的时间区间转换为十点到十二点
std::string ReplaceTime(const std::smatch &match);

// 匹配10:00:00～12:00:00的时间区间
std::string RegexReplaceTimeRange(const std::string &sentence);

// 匹配10:00:00这样的时间表示
std::string RegexReplaceTime(const std::string &sentence);
std::string replace_temperature(const std::smatch &match);
std::string RegexReplaceTemperature(const std::string &sentence);

std::string ReplaceFrac(const std::smatch &match);

std::string RegexReplaceFrac(const std::string &sentence);

std::string ReplacePercentage(const std::smatch &match);

std::string RegexReplacePercentage(const std::string &sentence);

std::string Phone2Str(const std::string &phone_string, bool mobile = true);

std::string ReplacePhoneNumber(const std::smatch &match);

std::string RegexReplaceMobile(const std::string &sentence);

std::string RegexReplacePhone(const std::string &sentence);

std::string RegexReplaceNationalUniformNumber(const std::string &sentence);

std::string ReplaceNumber(const std::smatch &match);

std::string ReplaceRange(const std::smatch &match);

std::string RegexReplaceRange(const std::string &sentence);

// C++ 版本的 replace_negative_num 函数
std::string ReplaceNegativeNum(const std::smatch &match);

std::string RegexReplaceNegativeNum(const std::string &sentence);
std::string RegexReplaceDecimalNum(const std::string &sentence);

std::string replace_positive_quantifier(const std::smatch &match);

std::string RegexReplacePositiveQuantifier(const std::string &sentence);

// 替换2+量词为两+量词
std::string RegexReplace2WithQuantifier(const std::string &sentence);

std::string ReplaceDefaultNum(const std::smatch &match);

std::string RegexReplaceDefaultNum(const std::string &sentence);

std::string RegexReplaceNumber(const std::string &sentence);

std::string PostReplace(const std::string &sentence);

// 返回 UTF-8 字符的长度
int Utf8CharLength(char ch);

// 将 UTF-8 编码的字符串拆分为单个的字符，返回字符列表，注意每个字符格式还是string
std::vector<std::string> SplitUtf8String(const std::string &utf8_str);
std::string HanSubstr(const std::vector<std::string> &hans, int begin, size_t size);
// 将 UTF-8 编码的字符串的第一个字符转换为 Unicode 码点
uint32_t Utf8ToUnicode(const std::string &utf8_str);

// 检查给定的 Unicode 码点是否是汉字
bool IsChinese(uint32_t unicode_char);
// 判断字符是否为英文字母
bool IsAlphabet(char32_t char_code);

// 判断字符是否为英文字母和中文以外的其他字符
bool IsOther(char32_t char_code);
// std::string traditional_to_simplified(const std::string& text, const std::unordered_map<char, char>& t2s_dict) {
std::string TraditionalToSimplified(const std::string &text);

// Function to convert simplified to traditional Chinese
std::string SimplifiedToTraditional(const std::string &text);

std::string FullWidthToHalfWidth(const std::string &sentence);

// 将字符串数组合并成一个字符串，中间用delimiter分割，用于方便地打印字符串数组内容
std::string ConcatStrList(const std::vector<std::string> &strs, const std::string &delimiter = "");

// 将int数组合并成一个字符串，中间用delimiter分割，用于方便地打印数组内容
std::string ConcatIntList(const std::vector<int> &ints, const std::string &delimiter = "");

// 将float数组合并成一个字符串，中间用delimiter分割，用于方便地打印数组内容
std::string ConcatFloatList(const std::vector<float> &fs, const std::string &delimiter = "");
std::string Concat2dFloatList(const std::vector<std::vector<float>> &fs_2d, const std::string &delimiter = "");

// 判断字符串中是否包含数字
bool ContainsNumber(const std::string &str);

std::vector<std::vector<float>> SliceBertFeat(const std::vector<std::vector<float>> &bert_feat, int st, int ed);

std::vector<int16_t> ConcatAudio(std::vector<std::vector<int16_t>> audio_list, float pad_length=0.2, int sample_rate=44100);

bool HanIsDigit(const std::string &word);
bool HanListContains(const std::vector<std::string> &vec, const std::string &value);
std::vector<std::string> SplitString(const std::string &str, char delimiter);
std::vector<int16_t> ConvertAudioToInt16(std::vector<float> audio);
std::string ToLowercase(const std::string &str);
std::string ToUppercase(const std::string &str);
std::vector<std::string> SplitEnSentenceToWords(const std::string &sentence);
std::vector<std::vector<float>> DuplicateBlocks(const std::vector<std::vector<float>> &v, const std::vector<int> &word2ph, int bert_feat_dim);

//
std::vector<int16_t> PadAudioForAtb(const std::vector<int16_t> &audio, int sample_rate);

// 在音频的开始和结束部分pad特定长度的空音频
std::vector<int16_t> PadEmptyAudio(const std::vector<int16_t> &raw_audio_data, int sample_rate,
                                   double audio_pad_duration = 0.0);

// 判断字符串是否全是空白字符
bool IsAllWhitespace(const std::string &str);

// 利用分隔符将中文句子拆分为不同的部分，例如("你好，再见", ",")->[你好，再见]
std::vector<std::string> SplitChineseSentence(const std::string &input, const std::string &delimiter = "。");

// 将句子拆分为中文和非中文部分拆分开的结果，例如"你好Hello!再见“-> [你好，Hello!, 再见]
std::vector<std::string> SplitChineseSentenceToParts(const std::string &input, bool merge_punc = true);
std::vector<std::string> MergePunctuation(const std::vector<std::string> &input);

// 删除开头和结尾的特定字符，支持中文
std::string Strip(const std::string &str, const std::string &chars);

// 读取json文件，返回json对象
json LoadJson(const std::string &josn_path);

// 判断输入中是否有汉字
bool MatchHan(const std::string &input);

// 判断输入中是否有汉字和字母数字
bool MatchHanInternal(const std::string &input);
std::vector<std::string> SplitHanInternal(const std::string &input);

bool MatchSkipInternal(const std::string &input);
std::vector<std::string> SplitSkipInternal(const std::string &input);

bool IsNum(const std::string &input);
bool IsEng(const std::string &input);

// 读取文本文件内容，每行以string保存，所有内容以vector of string 返回
std::vector<std::string> ReadTextFile(const std::string &file_path);

// 是否是中文句尾符号
bool IsChineseEnding(const std::string &text);

// 是否是英文句尾符号
bool IsEnglishEnding(const std::string &text);

bool IsChinesePunctuation(const std::string &text);
bool IsEnglishPunctuation(const std::string &text);
bool IsPunctuation(const std::string &text);

void Serialize(const std::unordered_map<std::string, std::vector<std::vector<std::string>>> &map, const std::string &filename);
std::unordered_map<std::string, std::vector<std::vector<std::string>>> deserialize(const std::string &filename);
void SaveJsonToBin(const std::string &filename, const json &j);

void LoadBinToMap(const std::string &filename, std::map<std::string, std::string> &m);

std::string MergeLines(const std::string &multiLineString);

// 计算文件内容的哈希值
std::string CalculateFileHash(const std::string &filePath);

// 创建子目录
std::string CreateSubDirectory(const std::string &parent_path, const std::string &hash);

std::string GetOrCreateHashDirectory(const std::string &parent_dir, const std::string &filePath);

template <typename T>
std::vector<T> FlattenVector(const std::vector<std::vector<T>> &vec2d, float pad_length=0.2, int sample_rate=44100)
{
    std::vector<T> flatVec;
    for (const auto &innerVec : vec2d)
    {
        flatVec.insert(flatVec.end(), innerVec.begin(), innerVec.end());
    }
    return flatVec;
}

template <typename T>
std::vector<T> Intersperse(const std::vector<T> &lst, T item)
{
    std::vector<T> result(lst.size() * 2 + 1, 0);
    for (int i = 1; i < result.size(); i += 2)
    {
        int orig_idx = int(i / 2);
        result[i] = lst[orig_idx];
    }
    return result;
}

template <typename T>
std::vector<T> VectorSlice(const std::vector<T> &vec, size_t start, size_t end)
{
    // 检查边界条件和有效性
    if (start >= vec.size())
        start = vec.size();
    if (end > vec.size())
        end = vec.size();
    if (start > end)
        start = end;

    return std::vector<T>(vec.begin() + start, vec.begin() + end);
}

// Slice2d 接受一个一维数组（代表二维数组）、行数、列数、起始列和结束列
// 它返回一个新的一维数组，代表切片后的二维数组
template <typename T>
std::vector<T> Slice2d(const std::vector<T> &array, size_t rows, size_t cols, size_t col_start, size_t col_end)
{
    // row=312, cols = 29, col_start = 0, col_end=27 -> 312x29 -> 312x27
    std::vector<T> result;
    int new_cols = col_end - col_start;
    result.reserve(rows * new_cols);

    int begin_idx = rows * col_start;
    int end_idx = rows * col_end;
    result.insert(result.end(), array.begin() + begin_idx, array.begin() + end_idx);

    std::vector<T> result1(rows * new_cols);
    // 原始的vector是行优先的，也就是312x29维度，后面推理需要的29x312
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < new_cols; ++col)
        {
            result1[row * new_cols + col] = result[col * rows + row];
        }
    }

    return result1;
}

template <typename T>
std::string ConcatList(const std::vector<T> &list, const std::string &delimiter = "|")
{
    std::stringstream ss;
    ss << delimiter;
    for (auto &elem : list)
    {
        ss << elem << delimiter;
    }
    return ss.str();
}
#endif
