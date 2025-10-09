#include "logging.h"

#include <android/log.h>
#include <sys/system_properties.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static constexpr LogSeverity kDefaultLogLevel = LogSeverity::INFO;

// Provides runtime logging level
class LoggingOptions {
private:
    static const char* kPropertyKey;
    explicit LoggingOptions() {
        char propValue[PROP_VALUE_MAX] = "0";
        if (__system_property_get(kPropertyKey, propValue)) {
            mLogLevel = atoi(propValue);
        }
    }

public:
    static LoggingOptions& instance() {
        static LoggingOptions loggingOptions;
        return loggingOptions;
    }

    int getLogLevel() const { return mLogLevel; }

private:
    int mLogLevel = static_cast<int>(kDefaultLogLevel);
};
const char* LoggingOptions::kPropertyKey = "debug.llm.loglevel";

bool runtimeShouldLog(const LogSeverity logSeverity) {
    const auto& loggingOptions = LoggingOptions::instance();
    return (static_cast<int>(logSeverity) >= loggingOptions.getLogLevel());
}

class NullStream : public std::ostream {
public:
    NullStream() : std::ostream(&mNullBuffer) {}

private:
    class NullBuffer : public std::streambuf {
    public:
        int overflow(int c) override { return c; }
    } mNullBuffer;
};

// clang-format off
static constexpr android_LogPriority kAndroidLogPriorityMap[] = {
    ANDROID_LOG_DEBUG,
    ANDROID_LOG_INFO,
    ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR,
    ANDROID_LOG_FATAL
};
// clang-format on

StreamLogger::StreamLogger(const LogSeverity logSeverity, const char* tag, const char* file,
                           const size_t line)
    : kLogSeverity(logSeverity), kTag(tag), kFile(file), kLine(line) {}

StreamLogger::~StreamLogger() {
    // Append file path and line
    if (kLogSeverity >= LogSeverity::ERROR && kFile) {
        mMsgStream << " [" << kFile << ":" << kLine << "]";
    }

    const auto& msg = mMsgStream.str();
    getOutStream() << msg << std::endl;
    const auto androidLogPriotity = kAndroidLogPriorityMap[static_cast<size_t>(kLogSeverity)];
    __android_log_print(androidLogPriotity, kTag, "%s", msg.c_str());

    if (shouldAbort()) {
        abort();
    }
}

std::ostream& StreamLogger::stream() {
    return mMsgStream;
}

bool StreamLogger::shouldAbort() const {
    return kLogSeverity == LogSeverity::FATAL;
}

// clang-format off
std::ostream& StreamLogger::getOutStream() const {
    switch (kLogSeverity) {
        case LogSeverity::DEBUG : return getNullStream(); // Ignore stdout/stderr for debug log
        case LogSeverity::WARN  : return std::cerr << "WARN: ";
        case LogSeverity::ERROR : return std::cerr << "ERROR: ";
        case LogSeverity::FATAL : return std::cerr << "FATAL: ";
        default:
            return std::cout;
    }
}
// clang-format on

std::ostream& StreamLogger::getNullStream() {
    static NullStream nullStream;
    return nullStream;
}
