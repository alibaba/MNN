// sherpa-mnn/csrc/onnx-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#include "sherpa-mnn/csrc/MNNUtils.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

static std::string GetInputName(MNN::Express::Module *sess, size_t index,
                                MNNAllocator *allocator) {
    return sess->getInfo()->inputNames[index];
}

static std::string GetOutputName(MNN::Express::Module *sess, size_t index,
                                 MNNAllocator *allocator) {
    return sess->getInfo()->outputNames[index];
}

void GetInputNames(MNN::Express::Module *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr) {
  MNNAllocator* allocator;
  size_t node_count = sess->getInfo()->inputNames.size();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*input_names)[i] = GetInputName(sess, i, allocator);
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

void GetOutputNames(MNN::Express::Module *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr) {
  MNNAllocator* allocator;
  size_t node_count = sess->getInfo()->outputNames.size();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*output_names)[i] = GetOutputName(sess, i, allocator);
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

MNN::Express::VARP GetEncoderOutFrame(MNNAllocator *allocator, MNN::Express::VARP encoder_out,
                              int32_t t) {
  std::vector<int> encoder_out_shape =
      encoder_out->getInfo()->dim;

  auto batch_size = encoder_out_shape[0];
  auto num_frames = encoder_out_shape[1];
  assert(t < num_frames);

  auto encoder_out_dim = encoder_out_shape[2];

  auto offset = num_frames * encoder_out_dim;

  std::array<int, 2> shape{batch_size, encoder_out_dim};

  MNN::Express::VARP ans =
      MNNUtilsCreateTensor<float>(allocator, shape.data(), shape.size());

  float *dst = ans->writeMap<float>();
  const float *src = encoder_out->readMap<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(src + t * encoder_out_dim, src + (t + 1) * encoder_out_dim, dst);
    src += offset;
    dst += encoder_out_dim;
  }
  return ans;
}

void PrintModelMetadata(std::ostream &os, const MNNMeta &meta_data) {
  MNNAllocator* allocator;
  for (auto& iter : meta_data) {
    os << iter.first << "=" << iter.second <<"\n";
  }
}

MNN::Express::VARP Clone(MNNAllocator *allocator, MNN::Express::VARP v) {
  return MNN::Express::_Clone(v, true);
}

MNN::Express::VARP View(MNN::Express::VARP v) {
  return v;
}

float ComputeSum(MNN::Express::VARP v, int32_t n /*= -1*/) {
  std::vector<int> shape = v->getInfo()->dim;
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  const float *p = v->readMap<float>();

  return std::accumulate(p, p + size, 1.0f);
}

float ComputeMean(MNN::Express::VARP v, int32_t n /*= -1*/) {
  std::vector<int> shape = v->getInfo()->dim;
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));

  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  auto sum = ComputeSum(v, n);
  return sum / size;
}

void PrintShape(MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  std::ostringstream os;
  for (auto i : shape) {
    os << i << ", ";
  }
  os << "\n";
  fprintf(stderr, "%s", os.str().c_str());
}

template <typename T /*= float*/>
void Print1D(MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  const T *d = v->readMap<T>();
  std::ostringstream os;
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    os << d[i] << " ";
  }
  os << "\n";
  fprintf(stderr, "%s\n", os.str().c_str());
}

template void Print1D<int>(MNN::Express::VARP v);
template void Print1D<float>(MNN::Express::VARP v);

template <typename T /*= float*/>
void Print2D(MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  const T *d = v->readMap<T>();

  std::ostringstream os;
  for (int32_t r = 0; r != static_cast<int32_t>(shape[0]); ++r) {
    for (int32_t c = 0; c != static_cast<int32_t>(shape[1]); ++c, ++d) {
      os << *d << " ";
    }
    os << "\n";
  }
  fprintf(stderr, "%s\n", os.str().c_str());
}

template void Print2D<int>(MNN::Express::VARP v);
template void Print2D<float>(MNN::Express::VARP v);

void Print3D(MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  const float *d = v->readMap<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    fprintf(stderr, "---plane %d---\n", p);
    for (int32_t r = 0; r != static_cast<int32_t>(shape[1]); ++r) {
      for (int32_t c = 0; c != static_cast<int32_t>(shape[2]); ++c, ++d) {
        fprintf(stderr, "%.3f ", *d);
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

void Print4D(MNN::Express::VARP v) {
  std::vector<int> shape = v->getInfo()->dim;
  const float *d = v->readMap<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    fprintf(stderr, "---plane %d---\n", p);
    for (int32_t q = 0; q != static_cast<int32_t>(shape[1]); ++q) {
      fprintf(stderr, "---subplane %d---\n", q);
      for (int32_t r = 0; r != static_cast<int32_t>(shape[2]); ++r) {
        for (int32_t c = 0; c != static_cast<int32_t>(shape[3]); ++c, ++d) {
          fprintf(stderr, "%.3f ", *d);
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

MNN::Express::VARP Repeat(MNNAllocator *allocator, MNN::Express::VARP cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split) {
  std::vector<int> cur_encoder_out_shape =
      cur_encoder_out->getInfo()->dim;

  std::array<int, 2> ans_shape{hyps_num_split.back(),
                                   cur_encoder_out_shape[1]};

  MNN::Express::VARP ans = MNNUtilsCreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());

  const float *src = cur_encoder_out->readMap<float>();
  float *dst = ans->writeMap<float>();
  int32_t batch_size = static_cast<int32_t>(hyps_num_split.size()) - 1;
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t cur_stream_hyps_num = hyps_num_split[b + 1] - hyps_num_split[b];
    for (int32_t i = 0; i != cur_stream_hyps_num; ++i) {
      std::copy(src, src + cur_encoder_out_shape[1], dst);
      dst += cur_encoder_out_shape[1];
    }
    src += cur_encoder_out_shape[1];
  }
  return ans;
}

CopyableOrtValue::CopyableOrtValue(const CopyableOrtValue &other) {
  *this = other;
}

CopyableOrtValue &CopyableOrtValue::operator=(const CopyableOrtValue &other) {
  if (this == &other) {
    return *this;
  }
  if (nullptr != other.value.get()) {
    MNNAllocator* allocator;
    value = Clone(allocator, other.value);
  }
  return *this;
}

CopyableOrtValue::CopyableOrtValue(CopyableOrtValue &&other) noexcept {
  *this = std::move(other);
}

CopyableOrtValue &CopyableOrtValue::operator=(
    CopyableOrtValue &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  value = std::move(other.value);
  return *this;
}

std::vector<CopyableOrtValue> Convert(std::vector<MNN::Express::VARP> values) {
  std::vector<CopyableOrtValue> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v));
  }

  return ans;
}

std::vector<MNN::Express::VARP> Convert(std::vector<CopyableOrtValue> values) {
  std::vector<MNN::Express::VARP> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v.value));
  }

  return ans;
}

std::string LookupCustomModelMetaData(const MNNMeta &meta_data,
                                      const char *key,
                                      MNNAllocator *allocator) {
    auto iter = meta_data.find(key);
    if (iter == meta_data.end()) {
      return "";
    }
    return iter->second;
}

MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const void* data, size_t data_size, const int* shapedata,
  int shapeSize, halide_type_t type ) {
    std::vector<int> s(shapedata, shapedata+shapeSize);
    return MNN::Express::_Const(data, s, MNN::Express::NCHW, type);
}

MNN::Express::VARP MNNUtilsCreateTensor(MNNAllocator* allocator, const int* shapedata,
  int shapeSize, halide_type_t type) {
    std::vector<int> s(shapedata, shapedata+shapeSize);
    return MNN::Express::_Input(s, MNN::Express::NCHW, type);
  
}

}  // namespace sherpa_mnn
