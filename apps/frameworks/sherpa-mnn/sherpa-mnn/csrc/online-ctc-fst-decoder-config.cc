// sherpa-mnn/csrc/online-ctc-fst-decoder-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-ctc-fst-decoder-config.h"

#include <sstream>
#include <string>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

std::string OnlineCtcFstDecoderConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineCtcFstDecoderConfig(";
  os << "graph=\"" << graph << "\", ";
  os << "max_active=" << max_active << ")";

  return os.str();
}

void OnlineCtcFstDecoderConfig::Register(ParseOptions *po) {
  po->Register("ctc-graph", &graph, "Path to H.fst, HL.fst, or HLG.fst");

  po->Register("ctc-max-active", &max_active,
               "Decoder max active states.  Larger->slower; more accurate");
}

bool OnlineCtcFstDecoderConfig::Validate() const {
  if (!graph.empty() && !FileExists(graph)) {
    SHERPA_ONNX_LOGE("graph: '%s' does not exist", graph.c_str());
    return false;
  }
  return true;
}

}  // namespace sherpa_mnn
