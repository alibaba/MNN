// sherpa-mnn/csrc/fst-utils.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FST_UTILS_H_
#define SHERPA_ONNX_CSRC_FST_UTILS_H_

#include <string>

#include "fst/fstlib.h"

namespace sherpa_mnn {

fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename);

}

#endif  // SHERPA_ONNX_CSRC_FST_UTILS_H_
