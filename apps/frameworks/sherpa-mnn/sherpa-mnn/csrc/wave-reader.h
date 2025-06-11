// sherpa-mnn/csrc/wave-reader.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_WAVE_READER_H_
#define SHERPA_ONNX_CSRC_WAVE_READER_H_

#include <istream>
#include <string>
#include <vector>

namespace sherpa_mnn {

/** Read a wave file with expected sample rate.

    @param filename Path to a wave file. It MUST be single channel, 16-bit
                    PCM encoded.
    @param sampling_rate  On return, it contains the sampling rate of the file.
    @param is_ok On return it is true if the reading succeeded; false otherwise.

    @return Return wave samples normalized to the range [-1, 1).
 */
std::vector<float> ReadWave(const std::string &filename, int32_t *sampling_rate,
                            bool *is_ok);

std::vector<float> ReadWave(std::istream &is, int32_t *sampling_rate,
                            bool *is_ok);

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_WAVE_READER_H_
