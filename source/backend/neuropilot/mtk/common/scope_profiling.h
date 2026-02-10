#pragma once

#include "logging.h"
#include "timer.h"

#include <functional>
#include <string>

#define LATENCY_LOGTAG __DEFAULT_LOGTAG__ "_latency"

// This class logs the creation and destruction of its object via RAII.
class ScopeLogger {
public:
    ScopeLogger(const LogSeverity logSeverity, const std::string& name, const bool logEnter = true,
                const bool logExit = true);

    ScopeLogger(const std::string& name, const bool logEnter = true, const bool logExit = true);

    ~ScopeLogger();

private:
    void log(const char* msg) const;

private:
    const LogSeverity kLogSeverity = LogSeverity::INFO;
    const std::string kName;
    const bool kLogEnter;
    const bool kLogExit;
};

// clang-format off

#define STRINGIFY_IMPL(X) #X
#define STRINGIFY(X) STRINGIFY_IMPL(X)
#define __LINE_STR__ STRINGIFY(__LINE__)

// clang-format on

#define FUNC_DESC std::string(__FUNCTION__) + " [" __FILE__ ":" __LINE_STR__ "]"

#define SCOPE_DESC(NAME) NAME " [" __FILE__ ":" __LINE_STR__ "]"

#define LOG_FUNC_ENTER ScopeLogger __scopeLogger_enter(LogSeverity::INFO, FUNC_DESC, true, false);
#define LOG_FUNC_EXIT  ScopeLogger __scopeLogger_exit(LogSeverity::INFO, FUNC_DESC, false, true);
#define LOG_FUNC_SCOPE ScopeLogger __scopeLogger(LogSeverity::INFO, FUNC_DESC, true, true);

#define DLOG_FUNC_ENTER ScopeLogger __scopeLogger_enter(LogSeverity::DEBUG, FUNC_DESC, true, false);
#define DLOG_FUNC_EXIT  ScopeLogger __scopeLogger_exit(LogSeverity::DEBUG, FUNC_DESC, false, true);
#define DLOG_FUNC_SCOPE ScopeLogger __scopeLogger(LogSeverity::DEBUG, FUNC_DESC, true, true);

// This class logs the time duration between the creation and destruction of its object via RAII.
// If the log severity is set to DEBUG but it is running in release build, it will not do anything.
template <typename Unit = Timer::Seconds>
class ScopeTimer {
public:
    ScopeTimer(const LogSeverity logSeverity, const std::string& name);

    ScopeTimer(const LogSeverity logSeverity, const std::function<void(double)>& callback);

    ScopeTimer(const std::string& name);

    ScopeTimer(const std::function<void(double)>& callback);

    ~ScopeTimer();

    double getElapsed();

private:
    void logElapsed(const double elapsedSec) const;

    bool isDebugMode() const;

private:
    const LogSeverity kLogSeverity = LogSeverity::INFO;
    const std::string kName;
    std::function<void(double)> mCallback;
    Timer mTimer;
};

#define LOG_SCOPE_LATENCY(NAME, UNIT) \
    ScopeTimer<Timer::UNIT> __scopeTimer(LogSeverity::INFO, SCOPE_DESC(NAME));

#define DLOG_SCOPE_LATENCY(NAME, UNIT) \
    ScopeTimer<Timer::UNIT> __scopeTimer(LogSeverity::DEBUG, SCOPE_DESC(NAME));

#define LOG_FUNC_LATENCY(UNIT) ScopeTimer<Timer::UNIT> __scopeTimer(LogSeverity::INFO, FUNC_DESC);

#define DLOG_FUNC_LATENCY(UNIT) ScopeTimer<Timer::UNIT> __scopeTimer(LogSeverity::DEBUG, FUNC_DESC);
