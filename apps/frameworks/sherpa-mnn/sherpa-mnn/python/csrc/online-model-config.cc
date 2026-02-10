// sherpa-mnn/python/csrc/online-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-model-config.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/online-model-config.h"
#include "sherpa-mnn/csrc/online-transducer-model-config.h"
#include "sherpa-mnn/csrc/provider-config.h"
#include "sherpa-mnn/python/csrc/online-nemo-ctc-model-config.h"
#include "sherpa-mnn/python/csrc/online-paraformer-model-config.h"
#include "sherpa-mnn/python/csrc/online-transducer-model-config.h"
#include "sherpa-mnn/python/csrc/online-wenet-ctc-model-config.h"
#include "sherpa-mnn/python/csrc/online-zipformer2-ctc-model-config.h"
#include "sherpa-mnn/python/csrc/provider-config.h"

namespace sherpa_mnn {

void PybindOnlineModelConfig(py::module *m) {
  PybindOnlineTransducerModelConfig(m);
  PybindOnlineParaformerModelConfig(m);
  PybindOnlineWenetCtcModelConfig(m);
  PybindOnlineZipformer2CtcModelConfig(m);
  PybindOnlineNeMoCtcModelConfig(m);
  PybindProviderConfig(m);

  using PyClass = OnlineModelConfig;
  py::class_<PyClass>(*m, "OnlineModelConfig")
      .def(py::init<const OnlineTransducerModelConfig &,
                    const OnlineParaformerModelConfig &,
                    const OnlineWenetCtcModelConfig &,
                    const OnlineZipformer2CtcModelConfig &,
                    const OnlineNeMoCtcModelConfig &,
                    const ProviderConfig &,
                    const std::string &, int32_t, int32_t,
                    bool, const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("transducer") = OnlineTransducerModelConfig(),
           py::arg("paraformer") = OnlineParaformerModelConfig(),
           py::arg("wenet_ctc") = OnlineWenetCtcModelConfig(),
           py::arg("zipformer2_ctc") = OnlineZipformer2CtcModelConfig(),
           py::arg("nemo_ctc") = OnlineNeMoCtcModelConfig(),
           py::arg("provider_config") = ProviderConfig(),
           py::arg("tokens"), py::arg("num_threads"), py::arg("warm_up") = 0,
           py::arg("debug") = false, py::arg("model_type") = "",
           py::arg("modeling_unit") = "", py::arg("bpe_vocab") = "")
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("paraformer", &PyClass::paraformer)
      .def_readwrite("wenet_ctc", &PyClass::wenet_ctc)
      .def_readwrite("zipformer2_ctc", &PyClass::zipformer2_ctc)
      .def_readwrite("nemo_ctc", &PyClass::nemo_ctc)
      .def_readwrite("provider_config", &PyClass::provider_config)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("warm_up", &PyClass::warm_up)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("model_type", &PyClass::model_type)
      .def_readwrite("modeling_unit", &PyClass::modeling_unit)
      .def_readwrite("bpe_vocab", &PyClass::bpe_vocab)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}
}  // namespace sherpa_mnn
