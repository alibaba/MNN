//
//  audio_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "gtest/gtest.h"
#include "audio/audio.hpp"

#include <fstream>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <functional>

#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>

using namespace MNN;
using namespace Express;
using namespace AUDIO;

static bool nearly(float x, float y, float eps = 1e-3) {
    return abs(x - y) <= eps;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int>& dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}
static inline VARP _zeros(const std::vector<int>& dims) {
    std::vector<float> data(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>()), 0);
    return _Const(data.data(), dims, NCHW, halide_type_of<float>());
}

static void dump_impl(const float *signal, size_t size, int row = 0) {
if (row) {
int col = size / row;
printf("# %d, %d: [\n", row, col);
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("..., \n");
for (int i = row - 3; i < row; i++) {
for (int j = 0; j < 3; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("..., ");
for (int j = col - 3; j < col; j++) {
printf("%f, ", signal[i * col + j]);
}
printf("\n");
}
printf("]\n");
} else {
printf("# %lu: [", size);
for (int i = 0; i < 3; i++) {
printf("%f, ", signal[i]);
}
printf("..., ");
for (int i = size - 3; i < size; i++) {
printf("%f, ", signal[i]);
}
printf("]\n");
}
}

void dump_var(VARP var) {
auto dims    = var->getInfo()->dim;
bool isfloat = true;
printf("{\ndtype = ");
if (var->getInfo()->type == halide_type_of<float>()) {
printf("float");
isfloat = true;
} else if (var->getInfo()->type == halide_type_of<int>()) {
printf("int");
isfloat = false;
}
printf("\nformat = %d\n", var->getInfo()->order);
printf("\ndims = [");
for (int i = 0; i < dims.size(); i++) {
printf("%d ", dims[i]);
}
printf("]\n");

if (isfloat) {
if ((dims.size() > 2 && dims[1] > 1 && dims[2] > 1) || (dims.size() == 2 && dims[0] > 1 && dims[1] > 1)) {
int row = dims[dims.size() - 2];
dump_impl(var->readMap<float>(), var->getInfo()->size, row);
} else {
printf("data = [");
auto total = var->getInfo()->size;
if (total > 32) {
for (int i = 0; i < 5; i++) {
printf("%f ", var->readMap<float>()[i]);
}
printf("..., ");
for (int i = total - 5; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
} else {
for (int i = 0; i < total; i++) {
printf("%f ", var->readMap<float>()[i]);
}
}
printf("]\n}\n");
}
} else {
printf("data = [");
int size = var->getInfo()->size > 10 ? 10 : var->getInfo()->size;
for (int i = 0; i < size; i++) {
printf("%d ", var->readMap<int>()[i]);
}
printf("]\n}\n");
}
}

TEST(load, wav) {
    auto audio_data = load("audio.wav");
    auto sample = audio_data.first;
    int sample_rate = audio_data.second;
    auto size = sample->getInfo()->size;
    auto mean = _ReduceMean(sample)->readMap<float>()[0];
    bool res = size == 88747 && sample_rate == 16000 && nearly(mean, -0.000021);
    EXPECT_TRUE(res);
}

TEST(save, wav) {
    auto audio_data = load("audio.wav");
    auto sample = audio_data.first;
    int sample_rate = audio_data.second;
    bool res = save("audio_save.wav", sample, sample_rate);
    EXPECT_TRUE(res);
}

TEST(hamming_window, 256) {
    auto window = hamming_window(256);
    auto mean = _ReduceMean(window)->readMap<float>()[0];
    bool res = std::vector<int>({256}) == window->getInfo()->dim && nearly(mean, 0.538203);
    EXPECT_TRUE(res);
}

TEST(hann_window, 256) {
    auto window = hann_window(256);
    auto mean = _ReduceMean(window)->readMap<float>()[0];
    bool res = std::vector<int>({256}) == window->getInfo()->dim && nearly(mean, 0.498047);
    EXPECT_TRUE(res);
}

TEST(melscale_fbanks, 80_400) {
    MelscaleParams mel_params;
    mel_params.n_mels = 80;
    mel_params.n_fft = 400;
    mel_params.sample_rate = 16000;
    auto mel = melscale_fbanks(&mel_params);
    auto mean = _ReduceMean(mel)->readMap<float>()[0];
    bool res = std::vector<int>({80, 201}) == mel->getInfo()->dim && nearly(mean, 0.000124);
    EXPECT_TRUE(res);
}

TEST(spectrogram, 512) {
    auto audio_data = load("audio.wav");
    auto sample = audio_data.first;
    int sample_rate = audio_data.second;
    SpectrogramParams spec_params;
    spec_params.n_fft = 512;
    spec_params.window_type = HANNING;
    auto specgram = spectrogram(sample, &spec_params);
    auto mean = _ReduceMean(specgram)->readMap<float>()[0];
    bool res = std::vector<int>({345, 257}) == specgram->getInfo()->dim && nearly(mean, 2.862101);
    EXPECT_TRUE(res);
}

TEST(mel_spectrogram, 400) {
    auto audio_data = load("audio.wav");
    auto sample = audio_data.first;
    int sample_rate = audio_data.second;
    MelscaleParams mel_params;
    mel_params.n_mels = 80;
    mel_params.n_fft = 400;
    mel_params.sample_rate = sample_rate;
    SpectrogramParams spec_params;
    spec_params.n_fft = 400;
    spec_params.hop_length = 160;
    spec_params.center = true;
    auto mel = mel_spectrogram(sample, &mel_params, &spec_params);
    auto mean = _ReduceMean(mel)->readMap<float>()[0];
    bool res = std::vector<int>({555, 80}) == mel->getInfo()->dim && nearly(mean, 0.149213);
    EXPECT_TRUE(res);
}

TEST(fbank, default) {
    auto audio_data = load("audio.wav", 0, 9600);
    auto chunk = audio_data.first;
    int sample_rate = audio_data.second;
    auto feat = fbank(chunk);
    auto mean = _ReduceMean(feat)->readMap<float>()[0];
    bool res = std::vector<int>({492, 80}) == feat->getInfo()->dim && nearly(mean, -9.875551);
    EXPECT_TRUE(res);
}

TEST(whisper_fbank, default) {
    auto audio_data = load("audio.wav");
    auto sample = audio_data.first;
    int sample_rate = audio_data.second;
    auto feat = whisper_fbank(sample);
    auto mean = _ReduceMean(feat)->readMap<float>()[0];
    bool res = std::vector<int>({1, 128, 3000}) == feat->getInfo()->dim && nearly(mean, -0.451097);
    EXPECT_TRUE(res);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();
    auto instance = testing::UnitTest::GetInstance();
    printf("\nTEST_NAME_AUDIO_UNIT: Audio单元测试\nTEST_CASE_AMOUNT_AUDIO_UNIT: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":%d}\n",
           instance->failed_test_count(), instance->successful_test_count(), instance->skipped_test_count());
    return res;
}