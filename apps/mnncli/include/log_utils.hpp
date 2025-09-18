//
// Created by MNN on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <memory>

namespace mnncli {

// Terminal color constants for colored logging
namespace Colors {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BOLD = "\033[1m";
}

// Log levels enum
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Log utility class for centralized logging
class LogUtils {
public:
    // Set global verbose mode
    static void setVerbose(bool verbose) {
        is_verbose_ = verbose;
    }
    
    // Check if verbose mode is enabled
    static bool isVerbose() {
        return is_verbose_;
    }
    
    // Log methods with different levels
    static void debug(const std::string& message, const std::string& tag = "");
    static void info(const std::string& message, const std::string& tag = "");
    static void warning(const std::string& message, const std::string& tag = "");
    static void error(const std::string& message, const std::string& tag = "");
    
    // Add formatted error method for printf-style formatting
    static void errorFormatted(const char* format, ...);
    
    // Overloaded error method for const char* (for backward compatibility)
    static void error(const char* message, const std::string& tag = "");
    
    // Conditional debug logging (only outputs when verbose is enabled)
    static void debugIfVerbose(const std::string& message) {
        if (is_verbose_) {
            fprintf(stderr, "[DEBUG] %s\n", message.c_str());
        }
    }
    
    // Template version for backward compatibility (deprecated)
    template<typename... Args>
    static void debugIfVerbose(const std::string& format, Args... args) {
        if (is_verbose_) {
            fprintf(stderr, "[DEBUG] ");
            fprintf(stderr, format.c_str(), args...);
            fprintf(stderr, "\n");
        }
    }
    
    // Format file size for logging
    static std::string formatFileSize(int64_t bytes);
    
    // Format progress percentage
    static std::string formatProgress(double progress);
    
    // Get current timestamp string
    static std::string getTimestamp();

private:
    static bool is_verbose_;
    static std::string formatMessage(LogLevel level, const std::string& message, const std::string& tag);
};

// Convenience macros for logging
#define LOG_DEBUG(...) mnncli::LogUtils::debugIfVerbose(__VA_ARGS__)
#define LOG_DEBUG_TAG(msg, tag) mnncli::LogUtils::debugIfVerbose(msg, tag)
#define LOG_INFO(msg) mnncli::LogUtils::info(msg)
#define LOG_WARNING(msg) mnncli::LogUtils::warning(msg)

// Smart LOG_ERROR macro that automatically chooses the right method based on argument count
#define LOG_ERROR_1(msg) mnncli::LogUtils::error(msg)
#define LOG_ERROR_2(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_3(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_4(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_5(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_6(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_7(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_8(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_9(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)
#define LOG_ERROR_10(msg, ...) mnncli::LogUtils::errorFormatted(msg, ##__VA_ARGS__)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, NAME, ...) NAME
#define LOG_ERROR(...) GET_MACRO(__VA_ARGS__, LOG_ERROR_10, LOG_ERROR_9, LOG_ERROR_8, LOG_ERROR_7, LOG_ERROR_6, LOG_ERROR_5, LOG_ERROR_4, LOG_ERROR_3, LOG_ERROR_2, LOG_ERROR_1)(__VA_ARGS__)

// Special macro for string concatenation cases
#define LOG_ERROR_STR(msg) mnncli::LogUtils::error(msg)

// Conditional debug macro that only compiles when verbose is enabled
#ifdef DEBUG
#define VERBOSE_LOG(msg) mnncli::LogUtils::debug(msg)
#define VERBOSE_LOG_TAG(msg, tag) mnncli::LogUtils::debug(msg, tag)
#else
#define VERBOSE_LOG(msg) do {} while(0)
#define VERBOSE_LOG_TAG(msg, tag) do {} while(0)
#endif

} // namespace mnncli
