/**
 * @file an_to_en.hpp
 * @author MNN Team
 * @date 2024-08-01
 * @version 1.0
 * @brief 阿拉伯数字转中文数字
 *
 * 将阿拉伯数字转换为中文的数字，如"123.5"-> "一百二十三点五"，基于Python 库an2cn实现
 * 实现过程比较直接，即先对阿拉伯数字字符串按照小数点切分，然后分别处理整数部分和小数部分
 */
#ifndef _HEADER_MNN_TTS_SDK_AN_TO_CN_H_
#define _HEADER_MNN_TTS_SDK_AN_TO_CN_H_

#include "utils.hpp"

class An2Cn
{
public:
    // 类初始化函数
    An2Cn();

    // 处理主入口
    std::string Process(const std::string &inputs);

private:
    // 根据小数点对字符串进行切分
    std::vector<std::string> SplitByDot(const std::string &str);

    // 检查输入是否是合法的阿拉伯数字（只包含阿拉伯0-9数字和小数点和负号)
    bool CheckInputsIsValid(const std::string &inputs);

    // 处理整数部分
    std::string IntegerConvert(const std::string &inputs);

    // 处理小数部分
    std::string DecimalConvert(const std::string &inputs);

    // 去掉开头的0
    std::string RemovePrefixZero(const std::string &str);

private:
    std::string valid_chars_ = "0123456789.-";
    std::map<int, std::string> NUMBER_LOW_AN2CN = {
        {0, "零"},
        {1, "一"},
        {2, "二"},
        {3, "三"},
        {4, "四"},
        {5, "五"},
        {6, "六"},
        {7, "七"},
        {8, "八"},
        {9, "九"}};

    std::vector<std::string> UNIT_LOW_ORDER_AN2CN =
        {
            "",
            "十",
            "百",
            "千",
            "万",
            "十",
            "百",
            "千",
            "亿",
            "十",
            "百",
            "千",
            "万",
            "十",
            "百",
            "千"};
};
#endif // _HEADER_MNN_TTS_SDK_AN_TO_CN_H_
