// sherpa-mnn/csrc/wave-writer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/wave-writer.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {
namespace {

// see http://soundfile.sapp.org/doc/WaveFormat/
//
// Note: We assume little endian here
// TODO(fangjun): Support big endian
struct WaveHeader {
  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;    // a tag of this chunk
  int32_t subchunk2_size;  // size of subchunk2
};

}  // namespace

int WaveFileSize(int32_t n_samples) {
  return sizeof(WaveHeader) + n_samples * sizeof(int16_t);
}

void WriteWave(char *buffer, int32_t sampling_rate, const float *samples,
               int32_t n) {
  WaveHeader header{};
  header.chunk_id = 0x46464952;      // FFIR
  header.format = 0x45564157;        // EVAW
  header.subchunk1_id = 0x20746d66;  // "fmt "
  header.subchunk1_size = 16;        // 16 for PCM
  header.audio_format = 1;           // PCM =1

  int32_t num_channels = 1;
  int32_t bits_per_sample = 16;  // int16_t
  header.num_channels = num_channels;
  header.sample_rate = sampling_rate;
  header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
  header.block_align = num_channels * bits_per_sample / 8;
  header.bits_per_sample = bits_per_sample;
  header.subchunk2_id = 0x61746164;  // atad
  header.subchunk2_size = n * num_channels * bits_per_sample / 8;

  header.chunk_size = 36 + header.subchunk2_size;

  std::vector<int16_t> samples_int16(n);
  for (int32_t i = 0; i != n; ++i) {
    samples_int16[i] = samples[i] * 32676;
  }

  memcpy(buffer, &header, sizeof(WaveHeader));
  memcpy(buffer + sizeof(WaveHeader), samples_int16.data(),
         n * sizeof(int16_t));
}

bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples, int32_t n) {
  std::string buffer;
  buffer.resize(WaveFileSize(n));
  WriteWave(buffer.data(), sampling_rate, samples, n);
  std::ofstream os(filename, std::ios::binary);
  if (!os) {
    SHERPA_ONNX_LOGE("Failed to create %s", filename.c_str());
    return false;
  }
  os << buffer;
  if (!os) {
    SHERPA_ONNX_LOGE("Write %s failed", filename.c_str());
    return false;
  }
  return true;
}

}  // namespace sherpa_mnn
