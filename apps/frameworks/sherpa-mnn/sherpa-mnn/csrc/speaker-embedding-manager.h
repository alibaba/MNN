// sherpa-mnn/csrc/speaker-embedding-manager.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

struct SpeakerMatch {
  const std::string name;
  float score;
};

namespace sherpa_mnn {

class SpeakerEmbeddingManager {
 public:
  // @param dim Embedding dimension.
  explicit SpeakerEmbeddingManager(int32_t dim);
  ~SpeakerEmbeddingManager();

  /* Add the embedding and name of a speaker to the manager.
   *
   * @param name Name of the speaker
   * @param p Pointer to the embedding. Its length is `dim`.
   * @return Return true if added successfully. Return false if it failed.
   *         At present, the only reason for a failure is that there is already
   *         a speaker with the same `name`.
   */
  bool Add(const std::string &name, const float *p) const;

  /** Add a list of embeddings of a speaker.
   *
   * @param name Name of the speaker
   * @param embedding_list A list of embeddings. Each entry should be of size
   *                       `dim`. The average of the list is the final
   *                       embedding.
   * @return Return true if added successfully. Return false if it failed.
   *         At present, the only reason for a failure is that there is already
   *         a speaker with the same `name`.
   */
  bool Add(const std::string &name,
           const std::vector<std::vector<float>> &embedding_list) const;

  /* Remove a speaker by its name.
   *
   * @param name Name of the speaker to remove.
   * @return Return true if it is removed successfully. Return false
   *         if there is no such a speaker.
   */
  bool Remove(const std::string &name) const;

  /** It is for speaker identification.
   *
   * It computes the cosine similarity between and given embedding and all
   * other embeddings and find the embedding that has the largest score
   * and the score is above or equal to threshold. Return the speaker
   * name for the embedding if found; otherwise, it returns an empty string.
   *
   * @param p The input embedding.
   * @param threshold A value between 0 and 1.
   * @param If found, return the name of the speaker. Otherwise, return an
   *        empty string.
   */
  std::string Search(const float *p, float threshold) const;

  /**
   * It is for speaker identification.
   *
   * It computes the cosine similarity between a given embedding and all
   * other embeddings and finds the embeddings that have the largest scores
   * and the scores are above or equal to the threshold. Returns a vector of
   * SpeakerMatch structures containing the speaker names and scores for the
   * embeddings if found; otherwise, returns an empty vector.
   *
   * @param p A pointer to the input embedding.
   * @param threshold A value between 0 and 1.
   * @param n The number of top matches to return.
   * @return A vector of SpeakerMatch structures. If matches are found, the
   *         vector contains the names and scores of the speakers. Otherwise,
   *         it returns an empty vector.
   */
  std::vector<SpeakerMatch> GetBestMatches(const float *p, float threshold,
                                           int32_t n) const;

  /* Check whether the input embedding matches the embedding of the input
   * speaker.
   *
   * It is for speaker verification.
   *
   * @param name The target speaker name.
   * @param p The input embedding to check.
   * @param threshold A value between 0 and 1.
   * @return Return true if it matches. Otherwise, it returns false.
   */
  bool Verify(const std::string &name, const float *p, float threshold) const;

  float Score(const std::string &name, const float *p) const;

  // Return true if the given speaker already exists; return false otherwise.
  bool Contains(const std::string &name) const;

  int32_t NumSpeakers() const;

  int32_t Dim() const;

  // Return a list of speaker names
  std::vector<std::string> GetAllSpeakers() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_MANAGER_H_
