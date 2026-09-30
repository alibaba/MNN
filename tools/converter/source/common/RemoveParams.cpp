//
//  RemoveParams.cpp
//  MNNConverter
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonUtils.hpp"
#include <climits>
#include <cstdint>
#include <fstream>


template <typename T>
static void storeWeight(std::ofstream* fs, std::vector<T>& weight, std::vector<int64_t>& external, int64_t& offset, bool check = true) {
    if (weight.empty() && check) {
        return;
    }
    if (external.empty()) {
        external.push_back(offset);
    }
    int64_t size = weight.size() * sizeof(T);
    fs->write(reinterpret_cast<const char*>(weight.data()), size);
    weight.clear();
    std::vector<T> empty;
    weight.swap(empty);
    external.push_back(size);
    offset += size;
}

void RemoveAndStoreParam(std::unique_ptr<MNN::OpT>& op, std::ofstream* fs, int64_t& offset) {
    if (!op->externalPath.empty()) {
        return;
    }
    const auto opType = op->main.type;
    switch (opType) {
        case MNN::OpParameter_IndexedLinearParam:
        {
            forEachIndexedLinearWeight(op, [&](std::unique_ptr<MNN::OpT>& member) {
                auto expert = member->main.AsConvolution2D();
                if (!expert->quanParameter) {
                    // Keep the bias length slot even for a bias-free expert.
                    storeWeight<float>(fs, expert->weight, expert->external, offset, false);
                    storeWeight<float>(fs, expert->bias, expert->external, offset, false);
                } else {
                    RemoveAndStoreParam(member, fs, offset);
                }
            });
            break;
        }
        case MNN::OpParameter_Convolution2D:
        {
            if (op->inputIndexes.size() > 1) {
                break;
            }
            auto param = op->main.AsConvolution2D();
            if (param->quanParameter) {
                storeWeight<int8_t>(fs, param->quanParameter->buffer, param->external, offset, false);
                if (param->quanParameter->scaleStorage == MNN::ScaleStorageType_FP16) {
                    storeWeight<uint16_t>(fs, param->quanParameter->alphaFp16, param->external, offset, false);
                } else {
                    storeWeight<float>(fs, param->quanParameter->alpha, param->external, offset, false);
                }
                storeWeight<float>(fs, param->bias, param->external, offset, false);
                storeWeight<uint32_t>(fs, param->quanParameter->index, param->external, offset, false);
            } else {
                storeWeight<float>(fs, param->weight, param->external, offset);
                storeWeight<float>(fs, param->bias, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_Scale: {
            auto param = op->main.AsScale();
            storeWeight<float>(fs, param->scaleData, param->external, offset);
            if (!param->biasData.empty()) {
                storeWeight<float>(fs, param->biasData, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            if (!param->gamma.empty() && !param->beta.empty()) {
                storeWeight<float>(fs, param->gamma, param->external, offset);
                storeWeight<float>(fs, param->beta, param->external, offset);
            }
            break;
        }
        case MNN::OpParameter_Blob: {
            auto param = op->main.AsBlob();
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            if (totalSize <= 1024) {
                break;
            }
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    storeWeight<float>(fs, param->float32s, param->external, offset);
                    break;
               case MNN::DataType_DT_BFLOAT16:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT32:
                    storeWeight<int>(fs, param->int32s, param->external, offset);
                    break;
                case MNN::DataType_DT_UINT8:
                    storeWeight<uint8_t>(fs, param->uint8s, param->external, offset);
                    break;
                case MNN::DataType_DT_INT8:
                    storeWeight<int8_t>(fs, param->int8s, param->external, offset);
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
}

bool saveExternalData(std::unique_ptr<MNN::NetT>& netT, const std::string& extraFileName) {
    std::ofstream extraFile(extraFileName, std::ios::binary);
    if (!extraFile.is_open()) {
        return false;
    }
    int64_t offset = 0;
    for (auto& op : netT->oplists) {
        RemoveAndStoreParam(op, &extraFile, offset);
    }
    for (auto& subgraph : netT->subgraphs) {
        for (auto& op : subgraph->nodes) {
            RemoveAndStoreParam(op, &extraFile, offset);
        }
    }
    extraFile.close();
    return true;
}

template <typename T>
static void loadExternalData(MNN::FileLoader* fl, std::vector<T>& data, int64_t size) {
    if (0 == size) {
        return;
    }
    data.resize(size / sizeof(T));
    fl->read(reinterpret_cast<char*>(data.data()), size);
}

template <typename T>
static bool loadIndexedExternalData(MNN::FileLoader* fl, std::vector<T>& data, int64_t size) {
    if (size < 0 || size % sizeof(T) != 0 || (uint64_t)size > SIZE_MAX) {
        return false;
    }
    data.resize(size / sizeof(T));
    return size == 0 || fl->read(reinterpret_cast<char*>(data.data()), size);
}

bool validateIndexedLinearExternal(const MNN::OpT* op, const MNN::FileLoader* fl) {
    auto param = op->main.AsIndexedLinearParam();
    if (!param || param->linears.empty()) {
        return false;
    }
    std::ifstream source;
    int64_t fileSize = 0;
    for (const auto& expert : param->linears) {
        if (!expert) {
            return false;
        }
        const auto& external = expert->external;
        if (external.empty()) {
            continue;
        }
        if (external.size() < 3 || external.size() > 5 || (!expert->quanParameter && external.size() != 3) ||
            external[1] <= 0) {
            MNN_ERROR("IndexedLinear %s: invalid external weight layout\n", op->name.c_str());
            return false;
        }
        if (!source.is_open()) {
            auto path = op->externalPath.empty() ? (fl ? fl->path() : "") : op->externalPath;
            source.open(path.c_str(), std::ios::binary | std::ios::ate);
            if (!source || (fileSize = source.tellg()) < 0) {
                MNN_ERROR("IndexedLinear %s: cannot open external weights\n", op->name.c_str());
                return false;
            }
        }
        int64_t end = external[0];
        for (size_t i = 1; i < external.size(); ++i) {
            const auto size = external[i];
            size_t elementBytes = sizeof(float);
            if (expert->quanParameter) {
                if (i == 1) {
                    elementBytes = 1;
                } else if (i == 2 && expert->quanParameter->scaleStorage == MNN::ScaleStorageType_FP16) {
                    elementBytes = sizeof(uint16_t);
                }
            }
            if (end < 0 || size < 0 || size % elementBytes || end > INT64_MAX - size || end + size > fileSize) {
                MNN_ERROR("IndexedLinear %s: invalid external weight offset/size\n", op->name.c_str());
                return false;
            }
            end += size;
        }
    }
    return true;
}

bool loadExternalParam(std::unique_ptr<MNN::OpT>& op, MNN::FileLoader* fl) {
    std::unique_ptr<MNN::FileLoader> flp;
    if (!op->externalPath.empty()) {
        flp.reset(new MNN::FileLoader(op->externalPath.c_str()));
        fl = flp.get();
    }
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_IndexedLinear:
        {
            if (!validateIndexedLinearExternal(op.get(), fl)) {
                return false;
            }
            auto param = op->main.AsIndexedLinearParam();
            for (auto& expert : param->linears) {
                if (expert && expert->external.empty()) {
                    continue;
                }
                if (!fl || fl->offset(expert->external[0]) != 0 || !fl->valid()) {
                    MNN_ERROR("IndexedLinear %s: cannot open/seek external weights\n", op->name.c_str());
                    return false;
                }
                bool loaded = true;
                if (expert->quanParameter) {
                    auto& quant = expert->quanParameter;
                    loaded = loadIndexedExternalData<int8_t>(fl, quant->buffer, expert->external[1]);
                    if (quant->scaleStorage == MNN::ScaleStorageType_FP16) {
                        loaded = loaded && loadIndexedExternalData<uint16_t>(fl, quant->alphaFp16, expert->external[2]);
                    } else {
                        loaded = loaded && loadIndexedExternalData<float>(fl, quant->alpha, expert->external[2]);
                    }
                    if (expert->external.size() > 3) {
                        loaded = loaded && loadIndexedExternalData<float>(fl, expert->bias, expert->external[3]);
                    }
                    if (expert->external.size() > 4) {
                        loaded = loaded && loadIndexedExternalData<uint32_t>(fl, quant->index, expert->external[4]);
                    }
                } else {
                    loaded = loadIndexedExternalData<float>(fl, expert->weight, expert->external[1]) &&
                             loadIndexedExternalData<float>(fl, expert->bias, expert->external[2]);
                }
                if (!loaded) {
                    MNN_ERROR("IndexedLinear %s: cannot read external weights\n", op->name.c_str());
                    return false;
                }
                expert->external.clear();
            }
            break;
        }
        case MNN::OpType_Convolution:
        case MNN::OpType_Deconvolution:
        case MNN::OpType_ConvolutionDepthwise:
        {
            auto param = op->main.AsConvolution2D();
            if (param->external.size() != 3) {
                return true;
            }
            fl->offset(param->external[0]);
            if (param->quanParameter) {
                loadExternalData<int8_t>(fl, param->quanParameter->buffer, param->external[1]);
                loadExternalData<float>(fl, param->quanParameter->alpha, param->external[2]);
                if (param->external.size() > 3) {
                    loadExternalData<float>(fl, param->bias, param->external[3]);
                }
                if (param->external.size() > 4) {
                    loadExternalData<uint32_t>(fl, param->quanParameter->index, param->external[4]);
                }
            } else {
                loadExternalData<float>(fl, param->weight, param->external[1]);
                loadExternalData<float>(fl, param->bias, param->external[2]);
            }
            param->external.clear();
            break;
        }
        case MNN::OpType_Scale: {
            auto param = op->main.AsScale();
            break;
        }
        case MNN::OpType_LayerNorm: {
            auto param = op->main.AsLayerNorm();
            break;
        }
        case MNN::OpType_TrainableParam:
        case MNN::OpType_Const: {
            auto param = op->main.AsBlob();
            if (param->external.size() != 2) {
                return true;
            }
            size_t totalSize = 1;
            for (auto dim : param->dims) {
                totalSize *= dim;
            }
            fl->offset(param->external[0]);
            switch (param->dataType) {
                case MNN::DataType_DT_FLOAT:
                    loadExternalData<float>(fl, param->float32s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT32:
                    loadExternalData<int>(fl, param->int32s, param->external[1]);
                    break;
                case MNN::DataType_DT_UINT8:
                    loadExternalData<uint8_t>(fl, param->uint8s, param->external[1]);
                    break;
                case MNN::DataType_DT_INT8:
                    loadExternalData<int8_t>(fl, param->int8s, param->external[1]);
                    break;
                default:
                    break;
            }
            param->external.clear();
            break;
        }
        default:
            break;
    }
    op->externalPath.clear();
    return true;
}
