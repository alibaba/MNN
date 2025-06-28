// sherpa-mnn/csrc/speaker-embedding-manager.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/speaker-embedding-manager.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "Eigen/Dense"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

using FloatMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class SpeakerEmbeddingManager::Impl {
 public:
  explicit Impl(int32_t dim) : dim_(dim) {}

  bool Add(const std::string &name, const float *p) {
    if (name2row_.count(name)) {
      // a speaker with the same name already exists
      return false;
    }

    embedding_matrix_.conservativeResize(embedding_matrix_.rows() + 1, dim_);

    std::copy(p, p + dim_, &embedding_matrix_.bottomRows(1)(0, 0));

    embedding_matrix_.bottomRows(1).normalize();  // inplace

    name2row_[name] = embedding_matrix_.rows() - 1;
    row2name_[embedding_matrix_.rows() - 1] = name;

    return true;
  }

  bool Add(const std::string &name,
           const std::vector<std::vector<float>> &embedding_list) {
    if (name2row_.count(name)) {
      // a speaker with the same name already exists
      return false;
    }

    if (embedding_list.empty()) {
      SHERPA_ONNX_LOGE("Empty list of embeddings");
      return false;
    }

    for (const auto &x : embedding_list) {
      if (static_cast<int32_t>(x.size()) != dim_) {
        SHERPA_ONNX_LOGE("Given dim: %d, expected dim: %d",
                         static_cast<int32_t>(x.size()), dim_);
        return false;
      }
    }

    // compute the average
    Eigen::RowVectorXf v = Eigen::Map<Eigen::RowVectorXf>(
        const_cast<float *>(embedding_list[0].data()), dim_);
    int32_t i = -1;
    for (const auto &x : embedding_list) {
      ++i;
      if (i == 0) {
        continue;
      }
      v += Eigen::Map<Eigen::RowVectorXf>(const_cast<float *>(x.data()), dim_);
    }

    // no need to compute the mean since we are going to normalize it anyway
    // v /= embedding_list.size();

    v.normalize();

    embedding_matrix_.conservativeResize(embedding_matrix_.rows() + 1, dim_);
    embedding_matrix_.bottomRows(1) = v;

    name2row_[name] = embedding_matrix_.rows() - 1;
    row2name_[embedding_matrix_.rows() - 1] = name;

    return true;
  }

  bool Remove(const std::string &name) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    int32_t num_rows = embedding_matrix_.rows();

    if (row_idx < num_rows - 1) {
      embedding_matrix_.block(row_idx, 0, num_rows - 1 - row_idx, dim_) =
          embedding_matrix_.bottomRows(num_rows - 1 - row_idx);
    }

    embedding_matrix_.conservativeResize(num_rows - 1, dim_);
    for (auto &p : name2row_) {
      if (p.second > row_idx) {
        p.second -= 1;
        row2name_[p.second] = p.first;
      }
    }

    name2row_.erase(name);
    row2name_.erase(num_rows - 1);

    return true;
  }

  std::string Search(const float *p, float threshold) {
    if (embedding_matrix_.rows() == 0) {
      return {};
    }

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    Eigen::VectorXf scores = embedding_matrix_ * v;

    Eigen::VectorXf::Index max_index = 0;
    float max_score = scores.maxCoeff(&max_index);
    if (max_score < threshold) {
      return {};
    }

    return row2name_.at(max_index);
  }

  std::vector<SpeakerMatch> GetBestMatches(const float *p, float threshold,
                                           int32_t n) {
    std::vector<SpeakerMatch> matches;

    if (embedding_matrix_.rows() == 0) {
      return matches;
    }

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    Eigen::VectorXf scores = embedding_matrix_ * v;

    std::vector<std::pair<float, int>> score_indices;
    for (int i = 0; i < scores.size(); ++i) {
      if (scores[i] >= threshold) {
        score_indices.emplace_back(scores[i], i);
      }
    }

    std::sort(score_indices.rbegin(), score_indices.rend(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    matches.reserve(score_indices.size());
    for (int i = 0; i < std::min(n, static_cast<int32_t>(score_indices.size()));
         ++i) {
      const auto &pair = score_indices[i];
      matches.push_back({row2name_.at(pair.second), pair.first});
    }

    return matches;
  }

  bool Verify(const std::string &name, const float *p, float threshold) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    float score = embedding_matrix_.row(row_idx) * v;

    if (score < threshold) {
      return false;
    }

    return true;
  }

  float Score(const std::string &name, const float *p) {
    if (!name2row_.count(name)) {
      // Setting a default value if the name is not found
      return -2.0;
    }

    int32_t row_idx = name2row_.at(name);

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    float score = embedding_matrix_.row(row_idx) * v;

    return score;
  }

  bool Contains(const std::string &name) const {
    return name2row_.count(name) > 0;
  }

  int32_t NumSpeakers() const { return embedding_matrix_.rows(); }

  int32_t Dim() const { return dim_; }

  std::vector<std::string> GetAllSpeakers() const {
    std::vector<std::string> all_speakers;
    all_speakers.reserve(name2row_.size());
    for (const auto &p : name2row_) {
      all_speakers.push_back(p.first);
    }

    std::sort(all_speakers.begin(), all_speakers.end());
    return all_speakers;
  }

 private:
  int32_t dim_;
  FloatMatrix embedding_matrix_;
  std::unordered_map<std::string, int32_t> name2row_;
  std::unordered_map<int32_t, std::string> row2name_;
};

SpeakerEmbeddingManager::SpeakerEmbeddingManager(int32_t dim)
    : impl_(std::make_unique<Impl>(dim)) {}

SpeakerEmbeddingManager::~SpeakerEmbeddingManager() = default;

bool SpeakerEmbeddingManager::Add(const std::string &name,
                                  const float *p) const {
  return impl_->Add(name, p);
}

bool SpeakerEmbeddingManager::Add(
    const std::string &name,
    const std::vector<std::vector<float>> &embedding_list) const {
  return impl_->Add(name, embedding_list);
}

bool SpeakerEmbeddingManager::Remove(const std::string &name) const {
  return impl_->Remove(name);
}

std::string SpeakerEmbeddingManager::Search(const float *p,
                                            float threshold) const {
  return impl_->Search(p, threshold);
}

std::vector<SpeakerMatch> SpeakerEmbeddingManager::GetBestMatches(
    const float *p, float threshold, int32_t n) const {
  return impl_->GetBestMatches(p, threshold, n);
}

bool SpeakerEmbeddingManager::Verify(const std::string &name, const float *p,
                                     float threshold) const {
  return impl_->Verify(name, p, threshold);
}

float SpeakerEmbeddingManager::Score(const std::string &name,
                                     const float *p) const {
  return impl_->Score(name, p);
}

int32_t SpeakerEmbeddingManager::NumSpeakers() const {
  return impl_->NumSpeakers();
}

int32_t SpeakerEmbeddingManager::Dim() const { return impl_->Dim(); }

bool SpeakerEmbeddingManager::Contains(const std::string &name) const {
  return impl_->Contains(name);
}

std::vector<std::string> SpeakerEmbeddingManager::GetAllSpeakers() const {
  return impl_->GetAllSpeakers();
}

}  // namespace sherpa_mnn
