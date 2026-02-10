#include "scope_profiling.h"

#include "cpp11_compat.h"
#include "logging.h"
#include "timer.h"

#include <functional>
#include <string>

#ifdef NDEBUG // release build
static constexpr bool kIsDebugBuild = false;
#else
static constexpr bool kIsDebugBuild = true;
#endif

//===-------===//
// ScopeLogger
//===-------===//

ScopeLogger::ScopeLogger(const LogSeverity logSeverity, const std::string& name,
                         const bool logEnter, const bool logExit)
    : kLogSeverity(logSeverity), kName(name), kLogEnter(logEnter), kLogExit(logExit) {
    if (kLogEnter) {
        log("Entering");
    }
}

ScopeLogger::ScopeLogger(const std::string& name, const bool logEnter, const bool logExit)
    : ScopeLogger(LogSeverity::INFO, name, logEnter, logExit) {}

ScopeLogger::~ScopeLogger() {
    if (kLogExit) {
        log("Done");
    }
}

void ScopeLogger::log(const char* msg) const {
    LOG_BARE(kLogSeverity) << msg << ' ' << kName;
}

//===------===//
// ScopeTimer
//===------===//

template <typename Unit>
ScopeTimer<Unit>::ScopeTimer(const LogSeverity logSeverity, const std::string& name)
    : kLogSeverity(logSeverity), kName(name) {
    if (isDebugMode() && !kIsDebugBuild) {
        return; // Ignore timer if requires but not in debug build
    }
    mTimer.start();
}

template <typename Unit>
ScopeTimer<Unit>::ScopeTimer(const LogSeverity logSeverity,
                             const std::function<void(double)>& callback)
    : kLogSeverity(logSeverity), mCallback(callback) {
    if (isDebugMode() && !kIsDebugBuild) {
        return; // Ignore timer if requires but not in debug build
    }
    mTimer.start();
}

template <typename Unit>
ScopeTimer<Unit>::ScopeTimer(const std::string& name) : ScopeTimer(LogSeverity::INFO, name) {}

template <typename Unit>
ScopeTimer<Unit>::ScopeTimer(const std::function<void(double)>& callback)
    : ScopeTimer(LogSeverity::INFO, callback) {}

template <typename Unit>
ScopeTimer<Unit>::~ScopeTimer() {
    if (isDebugMode() && !kIsDebugBuild) {
        return; // Ignore timer if requires but not in debug build
    }
    const double elapsed = getElapsed();
    if (mCallback)
        mCallback(elapsed);
    else
        logElapsed(elapsed);
}

template <typename Unit>
double ScopeTimer<Unit>::getElapsed() {
    if (isDebugMode() && !kIsDebugBuild) {
        LOG(WARN) << "ScopeTimer in debug mode is disabled in release build.";
    }
    return mTimer.elapsed<Unit>();
}

namespace detail {
    template<typename Unit>
    const char* getUnitString() { return "sec"; }
    
    template<>
    const char* getUnitString<Timer::Nanoseconds>() { return "ns"; }
    
    template<>
    const char* getUnitString<Timer::Microseconds>() { return "us"; }
    
    template<>
    const char* getUnitString<Timer::Milliseconds>() { return "ms"; }
}

template <typename Unit>
void ScopeTimer<Unit>::logElapsed(const double elapsedSec) const {
    const char* unitStr = detail::getUnitString<Unit>();
    LOG_BARE(kLogSeverity, LATENCY_LOGTAG)
        << "[Latency: " << elapsedSec << ' ' << unitStr << "] " << kName;
}

template <typename Unit>
bool ScopeTimer<Unit>::isDebugMode() const {
    return kLogSeverity == LogSeverity::DEBUG;
}

template class ScopeTimer<Timer::Nanoseconds>;
template class ScopeTimer<Timer::Microseconds>;
template class ScopeTimer<Timer::Milliseconds>;
template class ScopeTimer<Timer::Seconds>;