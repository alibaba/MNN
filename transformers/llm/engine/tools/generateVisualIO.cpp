#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <sstream>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <limits>

using namespace MNN::Express;

static void saveInputOutputs(const Module::Info* info,
                             std::vector<VARP> inputs,
                             std::vector<VARP> outputs,
                             const std::string& outputDir,
                             const std::string& tag) {
    for (int i = 0; i < (int)info->inputNames.size() && i < (int)inputs.size(); ++i) {
        inputs[i].fix(VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i = 0; i < (int)info->outputNames.size() && i < (int)outputs.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, tag);
    MNNCreateDir(subDir.c_str());

    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    Variable::save(inputs, inputPath.c_str());
    Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

struct VisualSize {
    int width;
    int height;
};

static std::vector<VisualSize> parseVisualSizes(const std::string& sizesStr) {
    std::vector<VisualSize> sizes;
    if (sizesStr.empty()) {
        return sizes;
    }
    std::stringstream ss(sizesStr);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto xpos = item.find('x');
        if (xpos == std::string::npos) continue;
        int w = std::atoi(item.substr(0, xpos).c_str());
        int h = std::atoi(item.substr(xpos + 1).c_str());
        if (w > 0 && h > 0) {
            sizes.push_back({w, h});
        }
    }
    return sizes;
}

static bool generateVisualIO(const std::string& modelDir,
                              const std::string& outputDir,
                              int imgWidth, int imgHeight,
                              int numGridPerSide) {
    std::string modelPath = MNNFilePathConcat(modelDir, "visual.mnn");
    std::string weightPath = modelPath + ".weight";

    MNN::ScheduleConfig config;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(
        Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile(weightPath.c_str());

    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    std::shared_ptr<Module> net(
        Module::load({}, {}, modelPath.c_str(), rtmgr, &module_config),
        Module::destroy);
    if (nullptr == net.get()) {
        MNN_ERROR("Failed to load visual module from %s.\n", modelPath.c_str());
        return false;
    }

    const auto& inputNames = net->getInfo()->inputNames;
    bool isQwen3VL = inputNames.size() == 5 &&
                     inputNames[3] == "idx_tensor";
    const int patch_size = isQwen3VL ? 16 : 14;
    constexpr int temporal_patch_size = 2;
    constexpr int merge_size = 2;
    const int align_size = patch_size * merge_size;

    int alignedH = (int)(round(imgHeight / (float)align_size)) * align_size;
    int alignedW = (int)(round(imgWidth / (float)align_size)) * align_size;
    if (alignedH < align_size) alignedH = align_size;
    if (alignedW < align_size) alignedW = align_size;

    int temporal = 2;
    int channel = 3;
    int grid_t = temporal / temporal_patch_size;
    int grid_h = alignedH / patch_size;
    int grid_w = alignedW / patch_size;
    int seq_len = grid_t * grid_h * grid_w;
    int patch_dim = channel * temporal_patch_size * patch_size * patch_size;

    MNN_PRINT("Visual IO: imgSize=%dx%d aligned=%dx%d grid=%dx%d seq_len=%d isQwen3VL=%d\n",
              imgWidth, imgHeight, alignedW, alignedH, grid_w, grid_h, seq_len, isQwen3VL);

    VARP patches = _Input({seq_len, patch_dim}, NCHW, halide_type_of<float>());
    auto patchData = patches->writeMap<float>();
    for (int i = 0; i < seq_len * patch_dim; ++i) {
        patchData[i] = (float)(rand()) / RAND_MAX;
    }

    VARP position_ids = _Input({2, seq_len}, NCHW, halide_type_of<int>());
    auto hpos_ptr = position_ids->writeMap<int>();
    auto wpos_ptr = hpos_ptr + seq_len;
    const int wblock_size = merge_size * merge_size;
    const int hblock_size = wblock_size * grid_w / merge_size;
    for (int i = 0; i < grid_h; i++) {
        int h_idx = i / merge_size, h_off = i % merge_size;
        for (int j = 0; j < grid_w; j++) {
            int w_idx = j / merge_size, w_off = j % merge_size;
            int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
            if (index < seq_len) {
                hpos_ptr[index] = i;
                wpos_ptr[index] = j;
            }
        }
    }

    VARP attention_mask = _Input({1, seq_len, seq_len}, NCHW, halide_type_of<float>());
    ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));

    VARPS moduleInputs = {patches, position_ids, attention_mask};

    if (isQwen3VL) {
        int num_patches = grid_h * grid_w;
        std::vector<float> h_idxs(grid_h), w_idxs(grid_w);
        for (int i = 0; i < grid_h; ++i) {
            h_idxs[i] = (grid_h > 1) ?
                static_cast<float>(i) * (numGridPerSide - 1) / (grid_h - 1) : 0.0f;
        }
        for (int i = 0; i < grid_w; ++i) {
            w_idxs[i] = (grid_w > 1) ?
                static_cast<float>(i) * (numGridPerSide - 1) / (grid_w - 1) : 0.0f;
        }
        auto idx_tensor = _Input({4, num_patches}, NCHW, halide_type_of<int>());
        auto weight_tensor = _Input({4, num_patches}, NCHW, halide_type_of<float>());
        auto idx_ptr = idx_tensor->writeMap<int>();
        auto weight_ptr = weight_tensor->writeMap<float>();
        for (int i = 0; i < grid_h; ++i) {
            int h_floor = static_cast<int>(h_idxs[i]);
            int h_ceil = std::min(h_floor + 1, numGridPerSide - 1);
            float dh = h_idxs[i] - h_floor;
            for (int j = 0; j < grid_w; ++j) {
                int w_floor = static_cast<int>(w_idxs[j]);
                int w_ceil = std::min(w_floor + 1, numGridPerSide - 1);
                float dw = w_idxs[j] - w_floor;
                int idx = i * grid_w + j;
                idx_ptr[0 * num_patches + idx] = h_floor * numGridPerSide + w_floor;
                idx_ptr[1 * num_patches + idx] = h_floor * numGridPerSide + w_ceil;
                idx_ptr[2 * num_patches + idx] = h_ceil * numGridPerSide + w_floor;
                idx_ptr[3 * num_patches + idx] = h_ceil * numGridPerSide + w_floor;
                weight_ptr[0 * num_patches + idx] = (1.0f - dh) * (1.0f - dw);
                weight_ptr[1 * num_patches + idx] = (1.0f - dh) * dw;
                weight_ptr[2 * num_patches + idx] = dh * (1.0f - dw);
                weight_ptr[3 * num_patches + idx] = dh * dw;
            }
        }
        idx_tensor = _Reshape(idx_tensor, {4, grid_t, grid_h / merge_size, merge_size, grid_w / merge_size, merge_size});
        idx_tensor = _Permute(idx_tensor, {0, 1, 2, 4, 3, 5});
        idx_tensor = _Reshape(idx_tensor, {4, -1});
        weight_tensor = _Reshape(weight_tensor, {4, grid_t, grid_h / merge_size, merge_size, grid_w / merge_size, merge_size});
        weight_tensor = _Permute(weight_tensor, {0, 1, 2, 4, 3, 5});
        weight_tensor = _Reshape(weight_tensor, {4, -1});
        moduleInputs.push_back(idx_tensor);
        moduleInputs.push_back(weight_tensor);
    }

    auto outputs = net->onForward(moduleInputs);
    if (outputs.empty()) {
        MNN_ERROR("Failed to run visual forward for size %dx%d.\n", imgWidth, imgHeight);
        return false;
    }

    std::string tag = std::to_string(imgWidth) + "x" + std::to_string(imgHeight);
    saveInputOutputs(net->getInfo(), moduleInputs, outputs, outputDir, tag);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateVisualIO modelDir outputDir [sizes]\n");
        MNN_PRINT("  sizes: comma separated WxH pairs, e.g. 416x416,384x416\n");
        MNN_PRINT("  default size: 420x420\n");
        return 1;
    }

    srand(time(NULL));
    std::string modelDir = argv[1];
    std::string outputDir = argv[2];
    MNNCreateDir(outputDir.c_str());

    std::string llmConfigPath = MNNFilePathConcat(modelDir, "llm_config.json");
    int numGridPerSide = 48;
    {
        std::ifstream ifs(llmConfigPath);
        if (ifs.is_open()) {
            rapidjson::IStreamWrapper isw(ifs);
            rapidjson::Document doc;
            doc.ParseStream(isw);
            if (!doc.HasParseError() && doc.IsObject()) {
                if (doc.HasMember("num_grid_per_side") && doc["num_grid_per_side"].IsInt()) {
                    numGridPerSide = doc["num_grid_per_side"].GetInt();
                }
            }
        }
    }
    MNN_PRINT("num_grid_per_side = %d\n", numGridPerSide);

    std::vector<VisualSize> sizes;
    if (argc >= 4) {
        sizes = parseVisualSizes(argv[3]);
    }
    if (sizes.empty()) {
        sizes.push_back({420, 420});
    }

    for (auto& sz : sizes) {
        MNN_PRINT("Generating visual IO for size %dx%d ...\n", sz.width, sz.height);
        if (!generateVisualIO(modelDir, outputDir, sz.width, sz.height, numGridPerSide)) {
            MNN_ERROR("Failed for size %dx%d\n", sz.width, sz.height);
            return 1;
        }
    }

    MNN_PRINT("All visual IO generated successfully.\n");
    return 0;
}
