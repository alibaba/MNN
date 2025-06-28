// sherpa-mnn/csrc/offline-speaker-diarization-result.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sherpa_mnn {

class OfflineSpeakerDiarizationSegment {
 public:
  OfflineSpeakerDiarizationSegment(float start, float end, int32_t speaker,
                                   const std::string &text = {});

  // If the gap between the two segments is less than the given gap, then we
  // merge them and return a new segment. Otherwise, it returns null.
  std::optional<OfflineSpeakerDiarizationSegment> Merge(
      const OfflineSpeakerDiarizationSegment &other, float gap) const;

  float Start() const { return start_; }
  float End() const { return end_; }
  int32_t Speaker() const { return speaker_; }
  const std::string &Text() const { return text_; }
  float Duration() const { return end_ - start_; }

  void SetText(const std::string &text) { text_ = text; }

  std::string ToString() const;

 private:
  float start_;       // in seconds
  float end_;         // in seconds
  int32_t speaker_;   // ID of the speaker, starting from 0
  std::string text_;  // If not empty, it contains the speech recognition result
                      // of this segment
};

class OfflineSpeakerDiarizationResult {
 public:
  // Add a new segment
  void Add(const OfflineSpeakerDiarizationSegment &segment);

  // Number of distinct speakers contained in this object at this point
  int32_t NumSpeakers() const;

  int32_t NumSegments() const;

  // Return a list of segments sorted by segment.start time
  std::vector<OfflineSpeakerDiarizationSegment> SortByStartTime() const;

  // ans.size() == NumSpeakers().
  // ans[i] is for speaker_i and is sorted by start time
  std::vector<std::vector<OfflineSpeakerDiarizationSegment>> SortBySpeaker()
      const;

 private:
  std::vector<OfflineSpeakerDiarizationSegment> segments_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_RESULT_H_
