// sherpa-mnn/python/csrc/offline-speaker-diarization-result.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/offline-speaker-diarization-result.h"

#include "sherpa-mnn/csrc/offline-speaker-diarization-result.h"

namespace sherpa_mnn {

static void PybindOfflineSpeakerDiarizationSegment(py::module *m) {
  using PyClass = OfflineSpeakerDiarizationSegment;
  py::class_<PyClass>(*m, "OfflineSpeakerDiarizationSegment")
      .def_property_readonly("start", &PyClass::Start)
      .def_property_readonly("end", &PyClass::End)
      .def_property_readonly("duration", &PyClass::Duration)
      .def_property_readonly("speaker", &PyClass::Speaker)
      .def_property("text", &PyClass::Text, &PyClass::SetText)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineSpeakerDiarizationResult(py::module *m) {
  PybindOfflineSpeakerDiarizationSegment(m);
  using PyClass = OfflineSpeakerDiarizationResult;
  py::class_<PyClass>(*m, "OfflineSpeakerDiarizationResult")
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers)
      .def_property_readonly("num_segments", &PyClass::NumSegments)
      .def("sort_by_start_time", &PyClass::SortByStartTime)
      .def("sort_by_speaker", &PyClass::SortBySpeaker);
}

}  // namespace sherpa_mnn
