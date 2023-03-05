//  CUDATools.hpp
//  MNN
//
//  Created by MNN on b'2022/09/05'.
//  Copyright © 2022, Alibaba Group Holding Limited
//

#ifndef CUDATools_hpp


/*

use nvprof by open the MACRO 'MNN_CUDA_PROFILE'.
 cmake .. -DMNN_CUDA_PROFILE=ON -DMNN_CUDA=ON

*/
#ifdef MNN_CUDA_PROFILE
    #include "nvToolsExt.h"
    #define NVTX_PUSH(...) nvtxRangePushA(__VA_ARGS__)
    #define NVTX_POP(...) nvtxRangePop(__VA_ARGS__)
#else
    #define NVTX_PUSH(...)
    #define NVTX_POP(...)
#endif


#ifdef MNN_CUDA_PROFILE
#include <stdio.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cxxabi.h>
#include <execinfo.h>

// print stack trace log
class MNNLogMessage {
public:
  MNNLogMessage(const char* file, int line, const char* function)
      :
        log_stream_(std::cout)
  {
    const char* pos = file;
    const char* onlyName = file;
    while (*pos != 0) {
      if (*pos == '/') {
        onlyName = pos + 1;
      }
      pos++;
    }
    log_stream_ << onlyName << ":"
                << line << ":"
                << function << ": ";
  }
  ~MNNLogMessage() {
    log_stream_ << '\n';
  }
  std::ostream& stream() {
    return log_stream_;
  }
  std::string GetCachedString() {
    return "";
  }

protected:
  std::ostream& log_stream_;

private:
  MNNLogMessage(const MNNLogMessage&);
  void operator=(const MNNLogMessage&);
};
#if !defined(MNN_BUILD_FOR_ANDROID)
inline std::string
Demangle(char const* msg_str, std::ostringstream& os) {
  using std::string;
  string msg(msg_str);
  size_t symbol_start = string::npos;
  size_t symbol_end = string::npos;
  if (((symbol_start = msg.find("_Z")) != string::npos) &&
      (symbol_end = msg.find_first_of(" +", symbol_start))) {
    string left_of_symbol(msg, 0, symbol_start);
    string symbol(msg, symbol_start, symbol_end - symbol_start);
    string right_of_symbol(msg, symbol_end);

    int status = 0;
    size_t length = string::npos;
    std::unique_ptr<char, void (*)(void* __ptr)> demangled_symbol = {
        abi::__cxa_demangle(symbol.c_str(), 0, &length, &status), &std::free};
    if (demangled_symbol && status == 0 && length > 0) {
      string symbol_str(demangled_symbol.get());
      os << left_of_symbol << symbol_str << right_of_symbol;
      return os.str();
    }
  }
  return string(msg_str);
}

// By default skip the first frame because
// that belongs to ~FatalLogMessage
inline std::string
StackTrace(size_t start_frame = 2, const size_t stack_size = 12) {
  using std::string;
  std::ostringstream stacktrace_os;
  std::vector<void*> stack(stack_size);
  int nframes = backtrace(stack.data(), static_cast<int>(stack_size));
  stacktrace_os << "Stack trace:\n";
  char** msgs = backtrace_symbols(stack.data(), nframes);
  if (msgs != nullptr) {
    for (int frameno = start_frame; frameno < nframes; ++frameno) {
      stacktrace_os << "  [bt] (" << frameno - start_frame << ") ";
      string msg = Demangle(msgs[frameno], stacktrace_os);
      stacktrace_os << "\n";
    }
  }
  free(msgs);
  string stack_trace = stacktrace_os.str();
  return stack_trace;
}


class TraceMNNLogMessage : public MNNLogMessage {
public:
  TraceMNNLogMessage(const char* file, int line, const char* function) : MNNLogMessage(file, line, function) {}
  ~TraceMNNLogMessage() {
    log_stream_ << "\n" << StackTrace(1);
  }
};

#define MNN_LOG_TRACE TraceMNNLogMessage(__FILE__, __LINE__, __FUNCTION__)

#else
#define MNN_LOG_TRACE MNNLogMessage(__FILE__, __LINE__, __FUNCTION__)
#endif

#define MNN_LOG_INFO MNNLogMessage(__FILE__, __LINE__, __FUNCTION__)
#define MNN_LOG(severity) MNN_LOG_##severity.stream()
#define MNN_LOG_IF(severity, condition) \
  !(condition) ? (void)0 : MNNLogMessageVoidify() & MNN_LOG(severity)

// deal with release/debug
// release mode
#if !defined(NDEBUG)
#define MNN_DLOG(severity) MNN_LOG(severity)
#define MNN_DLOG_IF(severity, condition) MNN_LOG_IF(severity, (condition))

// debug mode, all MNN_DLOG() code would be compiled as empty.
#else
#define MNN_DLOG(severity) MNN_LOG_IF(severity, false)
#define MNN_DLOG_IF(severity, condition) MNN_LOG_IF(severity, false)

#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class MNNLogMessageVoidify {
public:
  MNNLogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

#endif
#endif


