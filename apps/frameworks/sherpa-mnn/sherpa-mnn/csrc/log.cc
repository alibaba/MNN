// sherpa-mnn/csrc/log.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/log.h"

#ifdef SHERPA_ONNX_HAVE_EXECINFO_H
#include <execinfo.h>  // To get stack trace in error messages.
#ifdef SHERPA_ONNX_HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // SHERPA_ONNX_HAVE_CXXABI_H
#endif  // SHERPA_ONNX_HAVE_EXECINFO_H

#include <stdlib.h>

#include <ctime>
#include <iomanip>
#include <string>

namespace sherpa_mnn {

std::string GetDateTimeStr() {
  std::ostringstream os;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  os << std::put_time(&tm, "%F %T");  // yyyy-mm-dd hh:mm:ss
  return os.str();
}

static bool LocateSymbolRange(const std::string &trace_name, std::size_t *begin,
                              std::size_t *end) {
  // Find the first '_' with leading ' ' or '('.
  *begin = std::string::npos;
  for (std::size_t i = 1; i < trace_name.size(); ++i) {
    if (trace_name[i] != '_') {
      continue;
    }
    if (trace_name[i - 1] == ' ' || trace_name[i - 1] == '(') {
      *begin = i;
      break;
    }
  }
  if (*begin == std::string::npos) {
    return false;
  }
  *end = trace_name.find_first_of(" +", *begin);
  return *end != std::string::npos;
}

#ifdef SHERPA_ONNX_HAVE_EXECINFO_H
static std::string Demangle(const std::string &trace_name) {
#ifndef SHERPA_ONNX_HAVE_CXXABI_H
  return trace_name;
#else   // SHERPA_ONNX_HAVE_CXXABI_H
  // Try demangle the symbol. We are trying to support the following formats
  // produced by different platforms:
  //
  // Linux:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  //
  // Mac:
  //   0 server 0x000000010f67614d _ZNK5kaldi13MessageLogger10LogMessageEv + 813
  //
  // We want to extract the name e.g., '_ZN5kaldi13UnitTestErrorEv' and
  // demangle it info a readable name like kaldi::UnitTextError.
  std::size_t begin, end;
  if (!LocateSymbolRange(trace_name, &begin, &end)) {
    return trace_name;
  }
  std::string symbol = trace_name.substr(begin, end - begin);
  int status;
  char *demangled_name = abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
  if (status == 0 && demangled_name != nullptr) {
    symbol = demangled_name;
    free(demangled_name);
  }
  return trace_name.substr(0, begin) + symbol +
         trace_name.substr(end, std::string::npos);
#endif  // SHERPA_ONNX_HAVE_CXXABI_H
}
#endif  // SHERPA_ONNX_HAVE_EXECINFO_H

std::string GetStackTrace() {
  std::string ans;
#ifdef SHERPA_ONNX_HAVE_EXECINFO_H
  constexpr const std::size_t kMaxTraceSize = 50;
  constexpr const std::size_t kMaxTracePrint = 50;  // Must be even.
                                                    // Buffer for the trace.
  void *trace[kMaxTraceSize];
  // Get the trace.
  std::size_t size = backtrace(trace, kMaxTraceSize);
  // Get the trace symbols.
  char **trace_symbol = backtrace_symbols(trace, size);
  if (trace_symbol == nullptr) return ans;

  // Compose a human-readable backtrace string.
  ans += "[ Stack-Trace: ]\n";
  if (size <= kMaxTracePrint) {
    for (std::size_t i = 0; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
  } else {  // Print out first+last (e.g.) 5.
    for (std::size_t i = 0; i < kMaxTracePrint / 2; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    ans += ".\n.\n.\n";
    for (std::size_t i = size - kMaxTracePrint / 2; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    if (size == kMaxTraceSize)
      ans += ".\n.\n.\n";  // Stack was too long, probably a bug.
  }

  // We must free the array of pointers allocated by backtrace_symbols(),
  // but not the strings themselves.
  free(trace_symbol);
#endif  // SHERPA_ONNX_HAVE_EXECINFO_H
  return ans;
}

}  // namespace sherpa_mnn
