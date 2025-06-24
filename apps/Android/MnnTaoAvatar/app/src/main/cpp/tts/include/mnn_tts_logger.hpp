#pragma once

/**
 * @file mnn_tts_logger.hpp
 * @author PixelAI Team
 * @date 2024-08-01
 * @version 1.0
 * @brief Yet Another Logging System
 *
 *  PixelAITTS 日志记录器类，单例模式，可以直接使用PLOG函数打印日志
 */

#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <mutex>
#include <iomanip>

// 日志级别定义
enum PixelAITTSLogLevel
{
    TRACE,   // 打印每次调用的非常细节的内容，比如分词时每次调用DAG计算边的参数
    PDEBUG,  // 普通调试使用，打印关键步骤的*最终结果* (而不是中间每次计算的数值)，比如出现bug，定位是哪一个环节结果出错了
    INFO,    // 运行时默认显示的内容，比如模型路径，配置文件路径，关键的结果等
    WARNING, // 警告内容，提示不常见的case，比如输入文本为空
    ERROR,   // 提示错误，但可以正常运行程序
    CRITICAL // 程序无法跑，throw 报错，终止程序
};

//
class MNNITTSLogger
{
public:
    // 获取单例对象的入口函数
    static MNNITTSLogger &GetInstance();

    void SetLogLevel(PixelAITTSLogLevel level);

    // template 实现需要在头文件里面
    template <typename... Args>
    void Log(PixelAITTSLogLevel level, const char *file, int line, const std::string &func, const std::string &fmt, Args... args)
    {
        std::string project_id = "PixelAI_TTS";
        std::lock_guard<std::mutex> lock(mtx);
        if (level >= log_level_)
        {
            std::time_t now = std::time(nullptr);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S")
               << "[" + project_id + "]"
               << "[" << LevelToString(level) << "]"
               //<< "(" << file << ":" << line << " " << func << ") "
               //    << "(" << ":" << line << " " << func << ")"
               << fmt;
            (ss << ... << args); // C++17 variadic templates fold expression
            std::cout << ss.str() << std::endl;
        }
    }

private:
    std::string LevelToString(PixelAITTSLogLevel level);

    MNNITTSLogger() {}
    PixelAITTSLogLevel log_level_ = INFO;
    std::mutex mtx;
};

#define PLOG(level, ...) MNNITTSLogger::GetInstance().Log(level, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#define DELIMITER_LINE "==================="
