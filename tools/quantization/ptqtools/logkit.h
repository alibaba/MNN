//
//  logkit.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LOGKIT_H
#define LOGKIT_H

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#endif

class LogCheckError {
public:
    LogCheckError() : str(nullptr) {
    }
    explicit LogCheckError(const std::string& str_) : str(new std::string(str_)) {
    }
    ~LogCheckError() {
        if (str != nullptr)
            delete str;
    }
    operator bool() {
        return str != nullptr;
    }
    std::string* str;
};

#define DEFINE_CHECK_FUNC(name, op)                                                                                  \
    template <typename X, typename Y>                                                                                \
    inline LogCheckError LogCheck##name(const X& x, const Y& y) {                                                    \
        if (x op y)                                                                                                  \
            return LogCheckError();                                                                                  \
        std::ostringstream os;                                                                                       \
        os << " (" << x << " vs. " << y << ") "; /* CHECK_XX(x, y) requires x and y can be serialized to string. Use \
                                                         CHECK(x OP y) otherwise. NOLINT(*) */                       \
        return LogCheckError(os.str());                                                                              \
    }                                                                                                                \
    inline LogCheckError LogCheck##name(int x, int y) {                                                              \
        return LogCheck##name<int, int>(x, y);                                                                       \
    }

#define CHECK_BINARY_OP(name, op, x, y)                  \
    if (LogCheckError _check_err = LogCheck##name(x, y)) \
    LogMessageFatal(__FILE__, __LINE__).stream() << "Check failed: " << #x " " #op " " #y << *(_check_err.str)

DEFINE_CHECK_FUNC(_LT, <)
DEFINE_CHECK_FUNC(_GT, >)
DEFINE_CHECK_FUNC(_LE, <=)
DEFINE_CHECK_FUNC(_GE, >=)
DEFINE_CHECK_FUNC(_EQ, ==)
DEFINE_CHECK_FUNC(_NE, !=)

// Always-on checking
#define CHECK(x) \
    if (!(x))    \
    LogMessageFatal(__FILE__, __LINE__).stream() << "Check failed: " #x << " ==> "
#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x) \
    ((x) == NULL ? LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: " #x << ' ', (x) : (x)) // NOLINT(*)

#define DCHECK(x) CHECK(x)
#define DCHECK_LT(x, y) CHECK((x) < (y))
#define DCHECK_GT(x, y) CHECK((x) > (y))
#define DCHECK_LE(x, y) CHECK((x) <= (y))
#define DCHECK_GE(x, y) CHECK((x) >= (y))
#define DCHECK_EQ(x, y) CHECK((x) == (y))
#define DCHECK_NE(x, y) CHECK((x) != (y))

#define LOG_INFO LogMessage(__FILE__, __LINE__)

#define LOG_ERROR LOG_FATAL
#define LOG_WARNING LOG_INFO
#define LOG_FATAL LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

// Poor man version of VLOG
#define VLOG(x) LOG_INFO.stream()

#define LOG(severity) LOG_##severity.stream()
#define LG LOG_INFO.stream()
#define LOG_IF(severity, condition) !(condition) ? (void)0 : LogMessageVoidify() & LOG(severity)

#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)

// Poor man version of LOG_EVERY_N
#define LOG_EVERY_N(severity, n) LOG(severity)

class DateLogger {
public:
    DateLogger() {
#if defined(_MSC_VER)
        _tzset();
#endif
    }
    const char* HumanDate() {
#if defined(_MSC_VER)
        _strtime_s(buffer_, sizeof(buffer_));
#else
        time_t time_value = time(NULL);
        struct tm* pnow;
#if !defined(_WIN32)
        struct tm now;
        pnow = localtime_r(&time_value, &now);
#else
        pnow = localtime(&time_value); // NOLINT(*)
#endif
        snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif
        return buffer_;
    }

private:
    char buffer_[9];
};

class LogMessage {
public:
    LogMessage(const char* file, int line) : log_stream_(std::cout) {
#ifdef NDEBUG
        log_stream_ << "[" << pretty_date_.HumanDate() << "] "
                    << "@ " << line << ": ";
#else
        log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":" << line << ": ";
#endif
    }
    ~LogMessage() {
        log_stream_ << '\n';
    }
    std::ostream& stream() {
        return log_stream_;
    }

protected:
    std::ostream& log_stream_;

private:
    DateLogger pretty_date_;
    LogMessage(const LogMessage&);
    void operator=(const LogMessage&);
};

class LogMessageFatal {
public:
    LogMessageFatal(const char* file, int line) {
        log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":" << line << ": ";
    }
#if defined(_MSC_VER) && _MSC_VER < 1900
    ~LogMessageFatal() {
#else
    ~LogMessageFatal() noexcept(false) {
#endif
        std::cout << log_stream_.str();
        std::cout.flush();
    }
    std::ostringstream& stream() {
        return log_stream_;
    }

private:
    std::ostringstream log_stream_;
    DateLogger pretty_date_;
    LogMessageFatal(const LogMessageFatal&);
    void operator=(const LogMessageFatal&);
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
public:
    LogMessageVoidify() {
    }
    // This has to be an operator with a precedence lower than << but
    // higher than "?:". See its usage.
    void operator&(std::ostream&) {
    }
};

#endif // LOGKIT_H
