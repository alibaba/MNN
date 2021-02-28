#include "MNN_generated.h"
#include <MNN/MNNDefine.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <map>
#include <queue>
#include <set>
using namespace MNN;
//#define OPT_SHAPE_TRANSFORM
static bool reIndex(MNN::NetT* mNet) {
    std::map<int, int> usefulTensorIndexMap;
    std::vector<std::string> usefulTensorName;

    std::vector<bool> tensorValid(mNet->tensorName.size(), false);
    for (auto& op : mNet->oplists) {
        for (auto index : op->inputIndexes) {
            tensorValid[index] = true;
        }
        for (auto index : op->outputIndexes) {
            tensorValid[index] = true;
        }
    }

    for (int i = 0; i < tensorValid.size(); ++i) {
        if (tensorValid[i]) {
            usefulTensorIndexMap.insert(std::make_pair(i, usefulTensorName.size()));
            usefulTensorName.push_back(mNet->tensorName[i]);
        }
    }

    // Re index
    for (auto& op : mNet->oplists) {
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->inputIndexes[i]);
            op->inputIndexes[i] = iter->second;
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->outputIndexes[i]);
            op->outputIndexes[i] = iter->second;
        }
    }

    mNet->tensorName = usefulTensorName;
    for (auto iter = mNet->extraTensorDescribe.begin(); iter != mNet->extraTensorDescribe.end();) {
        auto index = (*iter)->index;
        if (usefulTensorIndexMap.find(index) == usefulTensorIndexMap.end()) {
            iter = mNet->extraTensorDescribe.erase(iter);
            continue;
        }
        (*iter)->index = usefulTensorIndexMap.find(index)->second;
        iter++;
    }
    // Check dup name and modify
    std::set<std::string> names;
    std::set<std::string> tensorNames;
    for (int i = 0; i < mNet->oplists.size(); ++i) {
        auto& op    = mNet->oplists[i];
        auto opName = op->name;
        if (opName.empty() || names.find(opName) != names.end()) {
            std::ostringstream defaultName;
            defaultName << EnumNameOpType(op->type);
            defaultName << i;
            op->name = defaultName.str();
            MNN_PRINT("%d op name is empty or dup, set to %s\n", i, op->name.c_str());
            opName = op->name;
        }
        names.insert(opName);
        for (auto output : op->outputIndexes) {
            auto origin = mNet->tensorName[output];
            if (origin.empty() || tensorNames.find(origin) != tensorNames.end()) {
                std::ostringstream defaultName;
                defaultName << output;
                origin                  = defaultName.str();
                mNet->tensorName[output] = origin;
            }
            tensorNames.insert(origin);
        }
    }
    return true;
}
static std::set<OpType> gShapeTransformType {
    OpType_BatchToSpaceND,
    OpType_Crop,
    OpType_DepthToSpace,
    OpType_ExpandDims,
    OpType_Flatten,
    OpType_Gather,
    OpType_GatherV2,
    OpType_GatherND,
    OpType_Padding,
    OpType_Permute,
    OpType_Reshape,
    OpType_Slice,
    OpType_SliceTf,
    OpType_StridedSlice,
    OpType_Squeeze,
    OpType_SpaceToDepth,
    OpType_SpaceToBatchND,
    OpType_Tile,
    OpType_Unsqueeze,
    OpType_ConvertTensor,
    OpType_Broastcast,
};

static std::set<OpType> gInt8Ops {
    OpType_ConvInt8,
    OpType_DepthwiseConvInt8,
    OpType_PoolInt8,
    OpType_EltwiseInt8
};


static void onlyTurnScaleToSingle(const std::vector<std::unique_ptr<OpT>>& sourceOplists) {
    for (int i=0; i<sourceOplists.size(); ++i) {
        auto op = sourceOplists[i].get();
        if (op->type == OpType_Int8ToFloat || op->type == OpType_FloatToInt8) {
            auto quanParam = op->main.AsQuantizedFloatParam();
            auto tensorScale = quanParam->tensorScale[0];
            quanParam->tensorScale = {tensorScale};
        }
    }
}

int main(int argc, const char* argv[]) {
    MNN_PRINT("Optimize quantize model for smaller and faster, only valid to run after MNN 1.1.2\n");
    MNN_PRINT("The tool is just for temporary usage, in laster version may be depercerate\n");
    if (argc < 3) {
        MNN_ERROR("Usage: ./quantized_model_optimize.out origin_quan.mnn optimized_quan.mnn\n");
        return 0;
    }
    auto srcFile = argv[1];
    auto dstFile = argv[2];
    FUNC_PRINT_ALL(srcFile, s);
    FUNC_PRINT_ALL(dstFile, s);

    std::unique_ptr<NetT> source;
    {
        std::ifstream sourceFile(srcFile);
        if (sourceFile.fail()) {
            MNN_ERROR("Can't open source file\n");
            return 0;
        }
        std::ostringstream tempOs;
        tempOs << sourceFile.rdbuf();
        auto tempS = tempOs.str();
        source = std::move(UnPackNet((void*)tempS.c_str()));
    }
    if (nullptr == source) {
        MNN_ERROR("Source net invalid\n");
        return 0;
    }
    
#ifdef OPT_SHAPE_TRANSFORM
    // Compute the quan info for model
    struct ValueMapInfo {
        float scale = 0.0f;
        float bias = 0.0f;
        float minValue = -128.0f;
        float maxValue = 127.0f;
        bool valid = false;
    };
    std::vector<ValueMapInfo> quanInfos(source->tensorName.size());
    int beforeConvert = 0;
    // First Load info from convert op
    for (int i=0; i<source->oplists.size(); ++i) {
        auto op = source->oplists[i].get();
        if (op->type == OpType_Int8ToFloat || op->type == OpType_FloatToInt8) {
            auto quanParam = op->main.AsQuantizedFloatParam();
            auto tensorScale = quanParam->tensorScale[0];
            beforeConvert++;
            if (op->type == OpType_FloatToInt8) {
                if (tensorScale > 0.000001f) {
                    tensorScale = 1.0f / tensorScale;
                }
            }
            for (auto index : op->inputIndexes) {
                auto& info = quanInfos[index];
                info.valid = true;
                info.scale = tensorScale;
            }
            for (auto index : op->outputIndexes) {
                auto& info = quanInfos[index];
                info.valid = true;
                info.scale = tensorScale;
            }
            continue;
        }
        if (OpType_EltwiseInt8 == op->type) {
            auto quanParameters = op->main.AsEltwiseInt8();
            quanInfos[op->inputIndexes[0]].valid = true;
            quanInfos[op->inputIndexes[0]].scale = quanParameters->inputQuan0->tensorScale[0];
            quanInfos[op->inputIndexes[1]].valid = true;
            quanInfos[op->inputIndexes[1]].scale = quanParameters->inputQuan1->tensorScale[0];
            quanInfos[op->outputIndexes[0]].valid = true;
            quanInfos[op->outputIndexes[0]].scale = 1.0f / quanParameters->outputQuan->tensorScale[0];
            continue;
        }
    }
    // Compute indirect quan infos by shape transform
    std::vector<ValueMapInfo> quanInfoIndirects = quanInfos;
    for (int i=0; i<source->oplists.size(); ++i) {
        auto op = source->oplists[i].get();
        if (op->type == OpType_Int8ToFloat || op->type == OpType_FloatToInt8) {
            auto& info = quanInfoIndirects[op->inputIndexes[0]];
            if (info.valid) {
                for (auto index : op->outputIndexes) {
                    quanInfoIndirects[index] = info;
                }
            }
        }
        if (gShapeTransformType.find(op->type) != gShapeTransformType.end()) {
            auto& info = quanInfoIndirects[op->inputIndexes[0]];
            if (info.valid) {
                for (auto index : op->outputIndexes) {
                    quanInfoIndirects[index] = info;
                }
            }
            continue;
        }
    }
    // Reset Quan op's parameter by new quanInfoIndirects info
    for (int i=0; i<source->oplists.size(); ++i) {
        auto op = source->oplists[i].get();
        if (OpType_ConvInt8 == op->type || OpType_DepthwiseConvInt8 == op->type) {
            auto quanParameters = op->main.AsConvolution2D()->symmetricQuan.get();
            if (quanInfoIndirects[op->inputIndexes[0]].scale != quanInfos[op->inputIndexes[0]].scale) {
                // s0 * A1 = F, s1 * A2 = F, C = f(A1) * p0 = f(A2) * s1 / s0 * p0
                auto adjustScale = quanInfoIndirects[op->inputIndexes[0]].scale / quanInfos[op->inputIndexes[0]].scale;
                for (auto& s : quanParameters->scale) {
                    s = s * adjustScale;
                }
            }
            MNN_ASSERT(quanInfos[op->outputIndexes[0]].scale == quanInfoIndirects[op->outputIndexes[0]].scale);
            continue;
        }
        if (OpType_EltwiseInt8 == op->type) {
            auto quanParameters = op->main.AsEltwiseInt8();
            for (auto& s : quanParameters->inputQuan0->scale) {
                s = quanInfoIndirects[op->inputIndexes[0]].scale;
            }
            for (auto& s : quanParameters->inputQuan1->scale) {
                s = quanInfoIndirects[op->inputIndexes[1]].scale;
            }
            for (auto& s : quanParameters->outputQuan->scale) {
                s = 1.0f / quanInfoIndirects[op->outputIndexes[0]].scale;
            }
        }
    }
    quanInfos = std::move(quanInfoIndirects);
    // Remove Int8ToFloat and Float2Int8
    std::queue<int> unusedIndexes;
    {
        std::map<int, int> indexMap;
        auto oplists = std::move(source->oplists);
        for (int i=0; i<oplists.size(); ++i) {
            auto op = oplists[i].get();
            if (op->type == OpType_FloatToInt8 || op->type == OpType_Int8ToFloat) {
                auto inputIndex = op->inputIndexes[0];
                auto outputIndex = op->outputIndexes[0];
                auto iter = indexMap.find(inputIndex);
                if (iter == indexMap.end()) {
                    indexMap.insert(std::make_pair(outputIndex, inputIndex));
                } else {
                    indexMap.insert(std::make_pair(outputIndex, iter->second));
                }
                continue;
            }
            for (int j=0; j<op->inputIndexes.size(); ++j) {
                auto iter = indexMap.find(op->inputIndexes[j]);
                if (iter != indexMap.end()) {
                    op->inputIndexes[j] = iter->second;
                }
            }
            source->oplists.emplace_back(std::move(oplists[i]));
        }
        for (auto& iter : indexMap) {
            unusedIndexes.push(iter.first);
        }
    }
    
    // Add Float2Int8 and Int8ToFloat Back
    int afterConvert = 0;
    {
        // 0: float, 1: int
        enum DataType {
            FLOAT = 0,
            INT8 = 1
        };
        std::vector<DataType> tensorType(source->tensorName.size(), FLOAT);
        std::map<int, int> indexMap;
        auto oplists = std::move(source->oplists);
        for (int opIndex = 0; opIndex < oplists.size(); ++opIndex) {
            auto op = oplists[opIndex].get();
            DataType dataType = FLOAT;
            if (gInt8Ops.find(op->type) != gInt8Ops.end()) {
                dataType = INT8;
            } else if (gShapeTransformType.find(op->type) != gShapeTransformType.end()) {
                dataType = tensorType[op->inputIndexes[0]];
            }
            for (int i = 0; i < op->outputIndexes.size(); ++i) {
                tensorType[op->outputIndexes[i]] = dataType;
            }
            for (int i = 0; i < op->inputIndexes.size(); ++i) {
                auto index = op->inputIndexes[i];
                if (tensorType[index] != dataType) {
                    auto replaceIter = indexMap.find(index);
                    if (replaceIter != indexMap.end()) {
                        op->inputIndexes[i] = replaceIter->second;
                    } else if (quanInfos[index].valid) {
                        afterConvert++;
                        // Create Op
                        // construct new op
                        std::unique_ptr<OpT> convertType(new MNN::OpT);
                        convertType->main.type = MNN::OpParameter_QuantizedFloatParam;
                        std::ostringstream opName;
                        opName << "Convert_" << index << "_" << (int)dataType;
                        convertType->name      = opName.str();
                        auto dequantizationParam         = new MNN::QuantizedFloatParamT;
                        convertType->main.value     = dequantizationParam;
                        if (dataType == FLOAT) {
                            convertType->type           = MNN::OpType_Int8ToFloat;
                            dequantizationParam->tensorScale = {quanInfos[index].scale};
                        } else {
                            convertType->type           = MNN::OpType_FloatToInt8;
                            if (quanInfos[index].scale > 0.0f) {
                                dequantizationParam->tensorScale = {1.0f / quanInfos[index].scale};
                            } else {
                                dequantizationParam->tensorScale = {0.0f};
                            }
                        }

                        convertType->inputIndexes = {index};
                        convertType->outputIndexes = {(int)source->tensorName.size()};
                        source->tensorName.push_back(convertType->name);

                        // reset current op's input index at i
                        op->inputIndexes[i] = convertType->outputIndexes[0];
                        indexMap[index] = convertType->outputIndexes[0];
                        source->oplists.emplace_back(std::move(convertType));
                    }
                }
            }
            source->oplists.emplace_back(std::move(oplists[opIndex]));
        }
    }
    MNN_PRINT("From %d Convert to %d Convert\n", beforeConvert, afterConvert);
    reIndex(source.get());
#else
    onlyTurnScaleToSingle(source->oplists);
    for (auto& subGraph : source->subgraphs) {
        onlyTurnScaleToSingle(subGraph->nodes);
    }
#endif
    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        auto len = MNN::Net::Pack(builderOutput, source.get());
        builderOutput.Finish(len);
        int sizeOutput    = builderOutput.GetSize();
        auto bufferOutput = builderOutput.GetBufferPointer();
        std::ofstream output(dstFile);
        output.write((const char*)bufferOutput, sizeOutput);
    }
    
    return 0;
}
