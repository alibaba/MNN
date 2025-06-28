// sherpa-mnn/csrc/offline-speaker-diarization-result.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speaker-diarization-result.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

OfflineSpeakerDiarizationSegment::OfflineSpeakerDiarizationSegment(
    float start, float end, int32_t speaker, const std::string &text /*= {}*/) {
  if (start > end) {
    SHERPA_ONNX_LOGE("start %.3f should be less than end %.3f", start, end);
    SHERPA_ONNX_EXIT(-1);
  }

  start_ = start;
  end_ = end;
  speaker_ = speaker;
  text_ = text;
}

std::optional<OfflineSpeakerDiarizationSegment>
OfflineSpeakerDiarizationSegment::Merge(
    const OfflineSpeakerDiarizationSegment &other, float gap) const {
  if (other.speaker_ != speaker_) {
    SHERPA_ONNX_LOGE(
        "The two segments should have the same speaker. this->speaker: %d, "
        "other.speaker: %d",
        speaker_, other.speaker_);
    return std::nullopt;
  }

  if (end_ < other.start_ && end_ + gap >= other.start_) {
    return OfflineSpeakerDiarizationSegment(start_, other.end_, speaker_);
  } else if (other.end_ < start_ && other.end_ + gap >= start_) {
    return OfflineSpeakerDiarizationSegment(other.start_, end_, speaker_);
  } else {
    return std::nullopt;
  }
}

std::string OfflineSpeakerDiarizationSegment::ToString() const {
  std::array<char, 128> s{};

  snprintf(s.data(), s.size(), "%.3f -- %.3f speaker_%02d", start_, end_,
           speaker_);

  std::ostringstream os;
  os << s.data();

  if (!text_.empty()) {
    os << " " << text_;
  }

  return os.str();
}

void OfflineSpeakerDiarizationResult::Add(
    const OfflineSpeakerDiarizationSegment &segment) {
  segments_.push_back(segment);
}

int32_t OfflineSpeakerDiarizationResult::NumSpeakers() const {
  std::unordered_set<int32_t> count;
  for (const auto &s : segments_) {
    count.insert(s.Speaker());
  }

  return count.size();
}

int32_t OfflineSpeakerDiarizationResult::NumSegments() const {
  return segments_.size();
}

// Return a list of segments sorted by segment.start time
std::vector<OfflineSpeakerDiarizationSegment>
OfflineSpeakerDiarizationResult::SortByStartTime() const {
  auto ans = segments_;
  std::sort(ans.begin(), ans.end(), [](const auto &a, const auto &b) {
    return (a.Start() < b.Start()) ||
           ((a.Start() == b.Start()) && (a.Speaker() < b.Speaker()));
  });

  return ans;
}

std::vector<std::vector<OfflineSpeakerDiarizationSegment>>
OfflineSpeakerDiarizationResult::SortBySpeaker() const {
  auto tmp = segments_;
  std::sort(tmp.begin(), tmp.end(), [](const auto &a, const auto &b) {
    return (a.Speaker() < b.Speaker()) ||
           ((a.Speaker() == b.Speaker()) && (a.Start() < b.Start()));
  });

  std::vector<std::vector<OfflineSpeakerDiarizationSegment>> ans(NumSpeakers());
  for (auto &s : tmp) {
    ans[s.Speaker()].push_back(std::move(s));
  }

  return ans;
}

}  // namespace sherpa_mnn
