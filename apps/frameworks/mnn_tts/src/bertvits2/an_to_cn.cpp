#include "an_to_cn.hpp"

An2Cn::An2Cn()
{
}

bool An2Cn::CheckInputsIsValid(const std::string &inputs)
{
    bool is_valid = true;
    for (auto &c : inputs)
    {
        if (std::find(valid_chars_.begin(), valid_chars_.end(), c) == valid_chars_.end())
        {
            is_valid = false;
            break;
        }
    }
    return is_valid;
}

std::vector<std::string> An2Cn::SplitByDot(const std::string &str)
{
    std::istringstream stream(str);
    std::string segment;
    std::vector<std::string> segments;
    char dot;

    while (std::getline(stream, segment, '.'))
    {
        segments.push_back(segment);
    }

    return segments;
}

std::string An2Cn::Process(const std::string &raw_inputs)
{
    auto inputs = raw_inputs;
    // 先检查是否为正确的数字，不是则报错
    if (!CheckInputsIsValid(inputs))
    {
        throw std::invalid_argument("[tts_an2cn] input is invalid number!");
    }

    // 获取正负号
    std::string sign = "";
    if (inputs[0] == '-')
    {
        sign = "负";
        inputs = inputs.substr(1);
    }

    // 符号后面的数字部分
    std::string output = "";
    // 根据小数点进行切分
    auto split_result = SplitByDot(inputs);
    auto len_split_result = split_result.size();

    // 该数字不含小数点后位数
    if (len_split_result == 1)
    {
        auto integer_data = split_result[0];
        output = IntegerConvert(integer_data);
    }
    // 该数字包含小数点后位数
    else if (len_split_result == 2)
    {
        // 处理整数部分
        auto integer_data = split_result[0];
        auto output1 = IntegerConvert(integer_data);

        // 处理小数部分
        auto decimal_data = split_result[1];
        auto output2 = DecimalConvert(decimal_data);

        output = output1 + output2;
    }
    else
    {
        throw std::invalid_argument("[tts_an2cn] input type error, more than one dot! ");
    }
    return sign + output;
}

std::string An2Cn::IntegerConvert(const std::string &inputs)
{
    if (inputs.size() < 1)
    {
        return std::string("零");
    }

    auto str = RemovePrefixZero(inputs);
    auto size = str.size();
    if (size > UNIT_LOW_ORDER_AN2CN.size())
    {
        throw std::invalid_argument("[tts_an2cn] 整数部分超出数据范围!");
    }

    std::string output_an = "";
    for (int i = 0; i < size; i++)
    {
        auto d = str[i];
        if (d != '0')
        {
            output_an += NUMBER_LOW_AN2CN[d - '0'] + UNIT_LOW_ORDER_AN2CN[size - i - 1];
        }
        else
        {
            if (!(size - i - 1) % 4)
            {
                output_an += NUMBER_LOW_AN2CN[d - '0'] + UNIT_LOW_ORDER_AN2CN[size - i - 1];
            }

            auto utf8_output = SplitUtf8String(output_an);
            int utf8_output_size = utf8_output.size();
            if (i > 0 && utf8_output[utf8_output_size - 1] == "零")
            {
                output_an += NUMBER_LOW_AN2CN[d - '0'];
            }
        }
    }

    // 去掉中间的零
    output_an = StrReplaceAll(output_an, "零零", "零");
    output_an = StrReplaceAll(output_an, "零万", "万");
    output_an = StrReplaceAll(output_an, "零亿", "亿");
    output_an = StrReplaceAll(output_an, "亿万", "亿");

    // Trim leading zeros
    output_an = Strip(output_an, "零");

    // 解决「一十几」问题
    auto utf8_output = SplitUtf8String(output_an);
    std::string output_an1 = "";
    if (utf8_output.size() > 2 && utf8_output[0] == "一" && utf8_output[1] == "十")
    {
        for (int i = 1; i < utf8_output.size(); i++)
        {
            output_an1 += utf8_output[i];
        }
    }
    else
    {
        output_an1 = output_an;
    }

    // 整数部分为空，即 0 - 1 之间的小数，添加零
    if (output_an1 == "")
    {
        output_an1 = "零";
    }
    return output_an1;
}

std::string An2Cn::DecimalConvert(const std::string &decimal_data)
{
    auto inputs = decimal_data;
    auto len_decimal_data = inputs.size();
    if (len_decimal_data > 16)
    {
        PLOG(WARNING, "[tts_an2cn] 注意：小数部分长度超出16位限制，将自动截取前 16 位有效精度！");
        inputs = inputs.substr(0, 16);
    }

    std::string output_an = "";
    if (len_decimal_data > 0)
    {
        output_an = "点";
    }

    for (int i = 0; i < inputs.size(); i++)
    {
        output_an += NUMBER_LOW_AN2CN[inputs[i] - '0'];
    }
    return output_an;
}

std::string An2Cn::RemovePrefixZero(const std::string &str)
{
    long num = std::stol(str);
    std::ostringstream oss;
    oss << num;
    auto new_str = oss.str();
    return new_str;
}
