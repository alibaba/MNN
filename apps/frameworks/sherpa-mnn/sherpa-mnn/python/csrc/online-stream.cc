// sherpa-mnn/python/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/online-stream.h"

#include <vector>

#include "sherpa-mnn/csrc/online-stream.h"

namespace sherpa_mnn {

constexpr const char *kAcceptWaveformUsage = R"(
Process audio samples.

Args:
  sample_rate:
    Sample rate of the input samples. If it is different from the one
    expected by the model, we will do resampling inside.
  waveform:
    A 1-D float32 tensor containing audio samples. It must be normalized
    to the range [-1, 1].
)";


constexpr const char *kGetFramesUsage = R"(
Get n frames starting from the given frame index.
(hint: intended for debugging, for comparing FBANK features across pipelines)

Args:
  frame_index:
    The starting frame index
  n:
    Number of frames to get.
Return:
  Return a 2-D tensor of shape (n, feature_dim).
  which is flattened into a 1-D vector (flattened in row major).
  Unflatten in python with:
    `features = np.reshape(arr, (n, feature_dim))`
)";

void PybindOnlineStream(py::module *m) {
  using PyClass = OnlineStream;
  py::class_<PyClass>(*m, "OnlineStream")
      .def(
          "accept_waveform",
          [](PyClass &self, float sample_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
          },
          py::arg("sample_rate"), py::arg("waveform"), kAcceptWaveformUsage,
          py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>())
      .def("get_frames", &PyClass::GetFrames,
           py::arg("frame_index"), py::arg("n"), kGetFramesUsage,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_mnn
