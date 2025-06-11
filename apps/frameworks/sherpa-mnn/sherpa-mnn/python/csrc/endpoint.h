// sherpa-mnn/csrc/endpoint.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_
#define SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_

#include "sherpa-mnn/python/csrc/sherpa-mnn.h"

namespace sherpa_mnn {

void PybindEndpoint(py::module *m);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_PYTHON_CSRC_ENDPOINT_H_
