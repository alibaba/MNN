// sherpa-mnn/csrc/display.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_DISPLAY_H_
#define SHERPA_ONNX_CSRC_DISPLAY_H_
#include <stdio.h>

#include <string>

namespace sherpa_mnn {

class Display {
 public:
  explicit Display(int32_t max_word_per_line = 60)
      : max_word_per_line_(max_word_per_line) {}

  void Print(int32_t segment_id, const std::string &s) {
#ifdef _MSC_VER
    if (segment_id != -1) {
      fprintf(stderr, "%d:%s\n", segment_id, s.c_str());
    } else {
      fprintf(stderr, "%s\n", s.c_str());
    }
    return;
#endif
    if (last_segment_ == segment_id) {
      Clear();
    } else {
      if (last_segment_ != -1) {
        fprintf(stderr, "\n\r");
      }
      last_segment_ = segment_id;
      num_previous_lines_ = 0;
    }

    if (segment_id != -1) {
      fprintf(stderr, "\r%d:", segment_id);
    }

    int32_t i = 0;
    for (size_t n = 0; n < s.size();) {
      if (s[n] > 0 && s[n] < 0x7f) {
        fprintf(stderr, "%c", s[n]);
        ++n;
      } else {
        // Each Chinese character occupies 3 bytes for UTF-8 encoding.
        std::string tmp(s.begin() + n, s.begin() + n + 3);
        fprintf(stderr, "%s", tmp.data());
        n += 3;
      }

      ++i;
      if (i >= max_word_per_line_ && n + 1 < s.size() &&
          (s[n] == ' ' || s[n] < 0)) {
        fprintf(stderr, "\n\r ");
        ++num_previous_lines_;
        i = 0;
      }
    }
  }

 private:
  // Clear the output for the current segment
  void Clear() {
    ClearCurrentLine();
    while (num_previous_lines_ > 0) {
      GoUpOneLine();
      ClearCurrentLine();
      --num_previous_lines_;
    }
  }

  // Clear the current line
  void ClearCurrentLine() const { fprintf(stderr, "\33[2K\r"); }

  // Move the cursor to the previous line
  void GoUpOneLine() const { fprintf(stderr, "\033[1A\r"); }

 private:
  int32_t max_word_per_line_;
  int32_t num_previous_lines_ = 0;
  int32_t last_segment_ = -1;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_DISPLAY_H_
