//
//  qwen3_tts_demo.cpp
//
//  Qwen3-TTS command line demo built on the LLM engine Omni/Talker path.
//

#include "llm/llm.hpp"
#include "audio/audio.hpp"

#include <MNN/expr/ExprCreator.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace MNN::Transformer;
using namespace MNN::Express;

static std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static bool ensureDirectory(const std::string& path) {
    if (path.empty()) {
        return true;
    }
#ifdef _WIN32
    if (_mkdir(path.c_str()) == 0 || errno == EEXIST) {
        return true;
    }
#else
    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
#endif
    MNN_ERROR("failed to create dump_dir: %s\n", path.c_str());
    return false;
}

static bool dumpFloatVector(const std::string& path, const std::vector<float>& values) {
    std::ofstream os(path.c_str(), std::ios::binary);
    os.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    return os.good();
}

static std::vector<float> normalizeWaveform(const std::vector<float>& waveform, float targetPeak) {
    if (targetPeak <= 0.0f) {
        return waveform;
    }
    float peak = 0.0f;
    for (float value : waveform) {
        peak = std::max(peak, std::fabs(value));
    }
    if (peak <= 0.0f) {
        MNN_PRINT("normalize target_peak=%.6g original_peak=0 scale=1\n", targetPeak);
        return waveform;
    }
    const float scale = targetPeak / peak;
    std::vector<float> result(waveform.size());
    for (int i = 0; i < waveform.size(); ++i) {
        result[i] = std::max(-1.0f, std::min(1.0f, waveform[i] * scale));
    }
    MNN_PRINT("normalize target_peak=%.6g original_peak=%.6g scale=%.6g\n", targetPeak, peak, scale);
    return result;
}

static void printWaveformSummary(const char* name, const std::vector<float>& waveform, int sampleCount) {
    MNN_PRINT("%s shape=[1,%zu] size=%zu\n", name, waveform.size(), waveform.size());
    int n = std::min<int>(sampleCount, waveform.size());
    MNN_PRINT("%s first%d:", name, n);
    for (int i = 0; i < n; ++i) {
        MNN_PRINT(" %.8g", waveform[i]);
    }
    MNN_PRINT("\n");
}

static bool isOption(const char* arg) {
    return std::strncmp(arg, "--", 2) == 0;
}

static int runTextMode(const std::string& modelDir, const std::string& text, const std::string& language,
                       int maxFrames, const std::string& dumpDir, float normalizePeak) {
    std::unique_ptr<Llm> llm(Llm::createLLM(joinPath(modelDir, "config.json")));
    llm->set_config("{\"tmp_path\":\"tmp\",\"async\":false}");
    if (!llm->load()) {
        MNN_ERROR("Qwen3-TTS load failed\n");
        return 1;
    }

    std::vector<float> waveform;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool lastChunk) {
        if (ptr && size > 0) {
            waveform.insert(waveform.end(), ptr, ptr + size);
        }
        return true;
    });

    if (!llm->generateTTS(text, language, maxFrames)) {
        MNN_ERROR("Qwen3-TTS generation failed\n");
        return 1;
    }
    if (waveform.empty()) {
        MNN_ERROR("Qwen3-TTS generated empty waveform\n");
        return 1;
    }

    printWaveformSummary("waveform", waveform, 16);
    std::vector<float> waveformToSave = waveform;
    if (normalizePeak > 0.0f) {
        waveformToSave = normalizeWaveform(waveform, normalizePeak);
        printWaveformSummary("waveform_normalized", waveformToSave, 16);
    }

    if (!dumpDir.empty()) {
        if (!ensureDirectory(dumpDir)) {
            return 1;
        }
        dumpFloatVector(joinPath(dumpDir, "mnn_text_waveform.bin"), waveform);
        if (normalizePeak > 0.0f) {
            dumpFloatVector(joinPath(dumpDir, "mnn_text_waveform_normalized.bin"), waveformToSave);
        }
        auto waveformVar = _Const(waveformToSave.data(), {static_cast<int>(waveformToSave.size())}, NCHW,
                                  halide_type_of<float>());
        MNN::AUDIO::save(joinPath(dumpDir, "qwen3_tts_text.wav"), waveformVar, 24000);
        MNN_PRINT("saved wav: %s\n", joinPath(dumpDir, "qwen3_tts_text.wav").c_str());
    }
    const auto* context = llm->getContext();
    MNN_PRINT("Qwen3-TTS text C++ chain finished. frames=%d\n", context ? context->gen_seq_len : 0);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[2], "--text") != 0) {
        MNN_PRINT("Usage: %s <model_dir> --text <text> [max_frames] [dump_dir] [language] [--normalize [target_peak]]\n",
                  argv[0]);
        return 1;
    }
    std::string modelDir = argv[1];
    std::string text = argv[3];
    int maxFrames = 16;
    std::string dumpDir;
    std::string language = "auto";
    float normalizePeak = -1.0f;
    bool normalizeRequested = false;

    int index = 4;
    if (index < argc && !isOption(argv[index])) {
        maxFrames = std::atoi(argv[index++]);
    }
    if (index < argc && !isOption(argv[index])) {
        dumpDir = argv[index++];
    }
    if (index < argc && !isOption(argv[index])) {
        language = argv[index++];
    }
    while (index < argc) {
        if (std::strcmp(argv[index], "--normalize") != 0) {
            MNN_ERROR("unknown option: %s\n", argv[index]);
            return 1;
        }
        normalizeRequested = true;
        normalizePeak = 1.0f;
        ++index;
        if (index < argc && !isOption(argv[index])) {
            normalizePeak = static_cast<float>(std::atof(argv[index++]));
        }
    }
    if (maxFrames <= 0) {
        MNN_ERROR("max_frames must be positive\n");
        return 1;
    }
    if (normalizeRequested && (normalizePeak <= 0.0f || normalizePeak > 1.0f)) {
        MNN_ERROR("normalize target_peak must be in (0, 1]\n");
        return 1;
    }
    return runTextMode(modelDir, text, language, maxFrames, dumpDir, normalizePeak);
}
