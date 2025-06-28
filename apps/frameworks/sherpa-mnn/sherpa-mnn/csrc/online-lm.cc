// sherpa-mnn/csrc/online-lm.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-lm.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/online-rnn-lm.h"

namespace sherpa_mnn {

std::unique_ptr<OnlineLM> OnlineLM::Create(const OnlineLMConfig &config) {
  return std::make_unique<OnlineRnnLM>(config);
}

}  // namespace sherpa_mnn
