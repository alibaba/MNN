#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "cpp11_compat.h"

#define __DEFAULT_LOGTAG__ "llm_sdk"

// All log severity will log to logcat and stdout/stderr, except DEBUG will only log to logcat.
enum class LogSeverity {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class StreamLogger {
public:
    StreamLogger(const LogSeverity logSeverity, const char* tag, const char* file = nullptr,
                 const size_t line = 0);
    ~StreamLogger();
    std::ostream& stream();

private:
    bool shouldAbort() const;
    std::ostream& getOutStream() const;
    static std::ostream& getNullStream();

private:
    std::ostringstream mMsgStream;
    const LogSeverity kLogSeverity;
    const char* kTag;

    const char* kFile = nullptr;
    const size_t kLine = 0;
};

bool runtimeShouldLog(const LogSeverity logSeverity);

#ifdef NDEBUG // release build
static constexpr bool kEnableDChecks = false;
#else
static constexpr bool kEnableDChecks = true;
#endif

#define ASSERT_LOGGER StreamLogger(LogSeverity::FATAL, "ASSERT_FAILED", __FILE__, __LINE__).stream()

// clang-format off

#define CHECK(cond) \
    if (!(cond)) ASSERT_LOGGER << "Check failed: " #cond " "

#define CHECK_OP(LHS, RHS, OP) \
    if (!(LHS OP RHS)) ASSERT_LOGGER << "Check failed: " << #LHS " " #OP " " #RHS \
                                     << " (" #LHS "=" << LHS << ", " #RHS "=" << RHS << "). "

#define CHECK_EQ(LHS, RHS) CHECK_OP(LHS, RHS, ==)
#define CHECK_NE(LHS, RHS) CHECK_OP(LHS, RHS, !=)
#define CHECK_LT(LHS, RHS) CHECK_OP(LHS, RHS, <)
#define CHECK_LE(LHS, RHS) CHECK_OP(LHS, RHS, <=)
#define CHECK_GE(LHS, RHS) CHECK_OP(LHS, RHS, >=)
#define CHECK_GT(LHS, RHS) CHECK_OP(LHS, RHS, >)

#define DCHECK(cond)        do { if (kEnableDChecks) CHECK(cond); } while(0)
#define DCHECK_EQ(LHS, RHS) do { if (kEnableDChecks) CHECK_EQ(LHS, RHS); } while(0)
#define DCHECK_NE(LHS, RHS) do { if (kEnableDChecks) CHECK_NE(LHS, RHS); } while(0)
#define DCHECK_LT(LHS, RHS) do { if (kEnableDChecks) CHECK_LT(LHS, RHS); } while(0)
#define DCHECK_LE(LHS, RHS) do { if (kEnableDChecks) CHECK_LE(LHS, RHS); } while(0)
#define DCHECK_GE(LHS, RHS) do { if (kEnableDChecks) CHECK_GE(LHS, RHS); } while(0)
#define DCHECK_GT(LHS, RHS) do { if (kEnableDChecks) CHECK_GT(LHS, RHS); } while(0)


// Macro overloading. See https://stackoverflow.com/a/11763277
#define _GET_ARG2(_0, _1, _2, ...) _2

// NOTE: LOG_*BARE macros take in actual LogSeverity values.
// For example, LOG(DEBUG) is equivalent to LOG_BARE(LogSeverity::DEBUG).

// Expands the following:
// - LOG(SEVERITY,TAG) to LOG_WITH_TAG(SEVERITY,TAG)
// - LOG(SEVERITY)     to LOG_DEFAULT_TAG(SEVERITY)
#define LOG(...) _GET_ARG2(__VA_ARGS__, LOG_WITH_TAG, LOG_DEFAULT_TAG)(__VA_ARGS__)
#define LOG_BARE(...) _GET_ARG2(__VA_ARGS__, LOG_WITH_TAG_BARE, LOG_DEFAULT_TAG_BARE)(__VA_ARGS__)

// Base logging macro
#define LOG_IMPL(SEVERITY, TAG) \
    if (runtimeShouldLog(SEVERITY)) \
        StreamLogger(SEVERITY, TAG, __FILE__, __LINE__).stream()

// Logging with provided tag
#define LOG_WITH_TAG(SEVERITY, TAG) LOG_IMPL(LogSeverity::SEVERITY, TAG)
#define LOG_WITH_TAG_BARE(SEVERITY, TAG) LOG_IMPL(SEVERITY, TAG)

// Logging using default tag
#define LOG_DEFAULT_TAG(SEVERITY) LOG_WITH_TAG(SEVERITY, __DEFAULT_LOGTAG__)
#define LOG_DEFAULT_TAG_BARE(SEVERITY) LOG_WITH_TAG_BARE(SEVERITY, __DEFAULT_LOGTAG__)

// clang-format on

// Support vector in ostream
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    if (vec.empty()) {
        stream << "{}";
        return stream;
    }
    auto iter = vec.cbegin();
    auto insertElem = [&]() {
        if (std::is_convertible<T, std::string>::value
            || std::is_convertible<T, ::mtk::cpp11_compat::string_view>::value)
            stream << '"' << *iter++ << '"';
        else
            stream << *iter++;
    };
    stream << "{";
    insertElem();
    while (iter != vec.cend()) {
        stream << ", ";
        insertElem();
    }
    stream << "}";
    return stream;
}
