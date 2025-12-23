//
// Created by MNN on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "log_utils.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdarg> // Required for va_list, va_start, va_end, va_copy
#include <vector>  // Required for std::vector

namespace mnn::downloader {

// Initialize static member
bool LogUtils::is_verbose_ = false;

// All other methods are now defined inline in the header.

// Implement the missing methods that are not inline
void LogUtils::Debug(const std::string& message, const std::string& tag) {
    if (is_verbose_) {
        std::string formatted = FormatMessage(LogLevel::DEBUG_LEVEL, message, tag);
        std::cout << formatted << std::endl;
    }
}

void LogUtils::Info(const std::string& message, const std::string& tag) {
    std::string formatted = FormatMessage(LogLevel::INFO, message, tag);
    std::cout << formatted << std::endl;
}

void LogUtils::Warning(const std::string& message, const std::string& tag) {
    std::string formatted = FormatMessage(LogLevel::WARNING, message, tag);
    std::cerr << formatted << std::endl;
}

void LogUtils::Error(const std::string& message, const std::string& tag) {
    std::string formatted = FormatMessage(LogLevel::ERROR, message, tag);
    std::cerr << formatted << std::endl;
}

void LogUtils::Error(const char* message, const std::string& tag) {
    std::string formatted = FormatMessage(LogLevel::ERROR, std::string(message), tag);
    std::cerr << formatted << std::endl;
}

void LogUtils::ErrorFormatted(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    // Calculate required buffer size
    va_list args_copy;
    va_copy(args_copy, args);
    int size = vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);
    
    if (size > 0) {
        // Allocate buffer and format the message
        std::vector<char> buffer(size + 1);
        vsnprintf(buffer.data(), buffer.size(), format, args);
        
        // Create formatted message and output
        std::string formatted = FormatMessage(LogLevel::ERROR, std::string(buffer.data()), "");
        std::cerr << formatted << std::endl;
    }
    
    va_end(args);
}

std::string LogUtils::FormatFileSize(int64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    char buffer[32];
    if (unit_index == 0) {
        snprintf(buffer, sizeof(buffer), "%.0f %s", size, units[unit_index]);
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    }
    return std::string(buffer);
}

std::string LogUtils::FormatProgress(double progress) {
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%.1f%%", progress * 100.0);
    return std::string(buffer);
}

std::string LogUtils::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string LogUtils::FormatMessage(LogLevel level, const std::string& message, const std::string& tag) {
    std::stringstream ss;
    
    // Add timestamp
    ss << "[" << GetTimestamp() << "] ";
    
    // Add level with color
    switch (level) {
        case LogLevel::DEBUG_LEVEL:
            ss << Colors::CYAN << "[DEBUG]" << Colors::RESET << " ";
            break;
        case LogLevel::INFO:
            ss << Colors::GREEN << "[INFO]" << Colors::RESET << " ";
            break;
        case LogLevel::WARNING:
            ss << Colors::YELLOW << "[WARNING]" << Colors::RESET << " ";
            break;
        case LogLevel::ERROR:
            ss << Colors::RED << "[ERROR]" << Colors::RESET << " ";
            break;
    }
    
    // Add tag - use [*] if no tag specified, otherwise [tag]
    if (tag.empty()) {
        ss << Colors::BLUE << "[*]" << Colors::RESET << " ";
    } else {
        ss << Colors::BLUE << "[" << tag << "]" << Colors::RESET << " ";
    }
    
    // Add message
    ss << message;
    
    return ss.str();
}

} // namespace mnn::downloader
