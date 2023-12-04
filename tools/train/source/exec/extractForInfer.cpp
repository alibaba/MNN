#include <fstream>
#include "MNN_generated.h"
#include "TrainInfo_generated.h"
#include "MNN/MNNDefine.h"
#include "MNN/Interpreter.hpp"
#include "rapidjson/document.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "core/OpCommonUtils.hpp"
using namespace MNN;
#define VECTOR_EXTRACT(FLATBUFFER_TYPE, CPP_TYPE, JSON_TYPE)\
case flatbuffers::ET_##FLATBUFFER_TYPE:\
{\
    std::vector<CPP_TYPE> data(array.Size());\
    for (int i=0; i<array.Size(); ++i) {\
        data[i] = array[i].JSON_TYPE();\
    }\
    indexes[pos].second = builder.CreateVector(data).Union();\
    break;\
}\

#define SCALAR_EXTRACT(FLATBUFFER_TYPE, CPP_TYPE, JSON_TYPE)\
case flatbuffers::ET_##FLATBUFFER_TYPE:\
{\
builder.AddElement(field, (CPP_TYPE)(iter->value.JSON_TYPE()), (CPP_TYPE)0);\
break;\
}
static flatbuffers::Offset<void> _writeJsonToFlatbuffer(const flatbuffers::TypeTable * table, flatbuffers::FlatBufferBuilder& builder, const rapidjson::GenericObject<false, rapidjson::GenericValue<rapidjson::UTF8<>>>& object) {
    std::vector<std::pair<int, flatbuffers::Offset<void>>> indexes;
    // Load union type for easy to use
    std::map<std::string, int> unionNames;
    for (int i=0; i<table->num_elems; ++i) {
        if (table->type_codes[i].sequence_ref == -1) {
            continue;
        }
        const flatbuffers::TypeTable *ref = table->type_refs[table->type_codes[i].sequence_ref]();
        if (ref->st == flatbuffers::ST_UNION) {
            unionNames.insert(std::make_pair(std::string(table->names[i]) + "_type", i));
        }
    }
    // Find index and cache
    std::map<int, int> unionTypes;
    for (auto iter = object.begin(); iter !=object.end(); iter++) {
        auto name = iter->name.GetString();
        int index = -1;
        for (int i=0; i<table->num_elems; ++i) {
            if (0 == ::strcmp(table->names[i], name)) {
                index = i;
                break;
            }
        }
        auto uiter = unionNames.find(name);
        if (uiter != unionNames.end()) {
            // Find union type id
            auto value = iter->value.GetString();
            int typePos = -1;
            auto unionIndex = uiter->second;
            auto ref = table->type_refs[table->type_codes[unionIndex].sequence_ref]();
            for (int j=0; j<ref->num_elems; ++j) {
                if (0 == ::strcmp(ref->names[j], value)) {
                    typePos = j;
                    break;
                }
            }
            if (-1 == typePos) {
                MNN_ERROR("Can't find union type\n");
                continue;
            }
            if (typePos > 0) {
                // First is None
                unionTypes.insert(std::make_pair(unionIndex, typePos-1));
            }
        }
        if (index == -1) {
            MNN_PRINT("Invalid: %s, Skip it\n", name);
        }
        indexes.emplace_back(std::make_pair(index, 0));
    }

    // resolve single object
    int pos = 0;
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto code = table->type_codes[index];
        if (code.is_vector) {
            continue;
        }
        if (code.sequence_ref != -1 && code.base_type == flatbuffers::ET_SEQUENCE) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            if (ref->st == flatbuffers::ST_TABLE) {
                indexes[pos].second = _writeJsonToFlatbuffer(ref, builder, iter->value.GetObject());
            } else if (ref->st == flatbuffers::ST_UNION) {
                auto unionInd = unionTypes.find(index)->second;
                ref = ref->type_refs[unionInd]();
                indexes[pos].second = _writeJsonToFlatbuffer(ref, builder, iter->value.GetObject());
            }
        }
    }

    // Resolve Vector and String
    pos = 0;
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto code = table->type_codes[index];
        if (!code.is_vector) {
            if (code.base_type == flatbuffers::ET_STRING) {
                indexes[pos].second = builder.CreateString(iter->value.GetString()).Union();
            }
            continue;
        }
        auto array = iter->value.GetArray();
        if (code.sequence_ref != -1) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            std::vector<flatbuffers::Offset<void>> offsets(array.Size());
            for (int i=0; i<array.Size(); ++i) {
                offsets[i] = _writeJsonToFlatbuffer(ref, builder, array[i].GetObject());
            }
            indexes[pos].second = builder.CreateVector(offsets.data(), offsets.size()).Union();
            continue;
        }
        switch (code.base_type) {
                VECTOR_EXTRACT(BOOL, bool, GetBool);
                VECTOR_EXTRACT(CHAR, char, GetInt);
                VECTOR_EXTRACT(UCHAR, uint8_t, GetInt);
                VECTOR_EXTRACT(SHORT, int16_t, GetInt);
                VECTOR_EXTRACT(USHORT, uint16_t, GetInt);
                VECTOR_EXTRACT(INT, int, GetInt);
                VECTOR_EXTRACT(UINT, uint32_t, GetUint);
                VECTOR_EXTRACT(LONG, int64_t, GetInt64);
                VECTOR_EXTRACT(ULONG, uint64_t, GetUint64);
                VECTOR_EXTRACT(FLOAT, float, GetFloat);
                VECTOR_EXTRACT(DOUBLE, double, GetDouble);
            case flatbuffers::ET_STRING:
            {
                std::vector<std::string> data(array.Size());
                for (int i=0; i<array.Size(); ++i) {
                    data[i] = array[i].GetString();
                }
                indexes[pos].second = builder.CreateVectorOfStrings(data).Union();
                break;
            }
            default:
                break;
        }
    }

    // Resolve Others
    pos = 0;
    auto start = builder.StartTable();
    for (auto iter = object.begin(); iter !=object.end(); iter++, pos++) {
        int index = indexes[pos].first;
        if (-1 == index) {
            continue;
        }
        auto field = 4 + index * 2;
        if (indexes[pos].second.o != 0) {
            builder.AddOffset(field, indexes[pos].second);
            continue;
        }
        auto code = table->type_codes[index];
        if (code.sequence_ref != -1) {
            const flatbuffers::TypeTable *ref = table->type_refs[code.sequence_ref]();
            int value = -1;
            if (ref->st == flatbuffers::ST_UNION || ref->st == flatbuffers::ST_ENUM) {
                auto type = iter->value.GetString();
                for (int i=0; i<ref->num_elems; ++i) {
                    if (0 == ::strcmp(type, ref->names[i])) {
                        if (nullptr == ref->values) {
                            value = i;
                        } else {
                            value = ref->values[i];
                        }
                    }
                }
                switch (code.base_type) {
                    case flatbuffers::ET_UTYPE:
                    case flatbuffers::ET_UINT:
                        builder.AddElement(field, (uint32_t)value, (uint32_t)0);
                        break;
                    case flatbuffers::ET_INT:
                        builder.AddElement(field, (int32_t)value, (int32_t)-1);
                        break;
                    case flatbuffers::ET_UCHAR:
                        builder.AddElement(field, (uint8_t)value, (uint8_t)0);
                        break;
                    case flatbuffers::ET_CHAR:
                        builder.AddElement(field, (int8_t)value, (int8_t)0);
                        break;
                    default:
                        break;
                }
                continue;
            }
        }
        switch (code.base_type) {
                SCALAR_EXTRACT(BOOL, bool, GetBool);
                SCALAR_EXTRACT(CHAR, char, GetInt);
                SCALAR_EXTRACT(UCHAR, uint8_t, GetInt);
                SCALAR_EXTRACT(SHORT, int16_t, GetInt);
                SCALAR_EXTRACT(USHORT, uint16_t, GetInt);
                SCALAR_EXTRACT(INT, int, GetInt);
                SCALAR_EXTRACT(UINT, uint32_t, GetUint);
                SCALAR_EXTRACT(LONG, int64_t, GetInt64);
                SCALAR_EXTRACT(ULONG, uint64_t, GetUint64);
                SCALAR_EXTRACT(FLOAT, float, GetFloat);
                SCALAR_EXTRACT(DOUBLE, double, GetDouble);
            default:
                break;
        }
    }
    return builder.EndTable(start);
}

static void* _getBlobPtr(const MNN::Blob* b) {
    void* result = nullptr;
    switch (b->dataType()) {
        case DataType_DT_FLOAT:
            result = (void*)b->float32s()->Data();
            break;
        case DataType_DT_INT32:
            result = (void*)b->int32s()->Data();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            result = (void*)b->uint8s()->Data();
            break;
        case DataType_DT_INT8:
            result = (void*)b->int8s()->Data();
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return result;
}
static size_t _getBlobSize(const MNN::Blob* srcblob) {
    MNN::Tensor _tmpTensor;
    _tmpTensor.setType(srcblob->dataType());
    auto size = _tmpTensor.getType().bytes();
    if (nullptr != srcblob->dims()) {
        for (int j=0; j<srcblob->dims()->size(); ++j) {
            auto len = srcblob->dims()->data()[j];
            if (1 == j && srcblob->dataFormat() == MNN_DATA_FORMAT_NC4HW4) {
                len = UP_DIV(len, 4) * 4;
            }
            size *= len;
        }
    }
    return size;
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_ERROR("Usage: ./extractForInfer src_train.mnn dst_infer.mnn revert.json\n");
    }
    // TODO: Support Extern Weight/Bias
    std::shared_ptr<MNN::Interpreter> train(MNN::Interpreter::createFromFile(argv[1]));
    if (nullptr == train.get()) {
        MNN_ERROR("Open train.mnn error\n");
        return 0;
    }
    std::shared_ptr<MNN::Interpreter> infer(MNN::Interpreter::createFromFile(argv[2]));
    if (nullptr == infer.get()) {
        MNN_ERROR("Open train.mnn error\n");
        return 0;
    }
    auto trainMNN = flatbuffers::GetRoot<MNN::Net>(train->getModelBuffer().first);
    if (nullptr == trainMNN->oplists()) {
        MNN_ERROR("Train mnn file error\n");
        return 0;
    }
    auto inferMNN = flatbuffers::GetRoot<MNN::Net>(infer->getModelBuffer().first);
    if (nullptr == inferMNN->oplists()) {
        MNN_ERROR("Train mnn file error\n");
        return 0;
    }
    flatbuffers::FlatBufferBuilder configBuilder;
    {
        rapidjson::Document document;
        std::ifstream fileNames(argv[3]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        auto object = document.GetObject();
        configBuilder.ForceDefaults(true);
        auto table = MNNTrain::TrainInfoTypeTable();
        auto offset = _writeJsonToFlatbuffer(table, configBuilder, object);
        configBuilder.Finish(offset);
    }
    auto config = flatbuffers::GetRoot<MNNTrain::TrainInfo>(configBuilder.GetBufferPointer());
    
    // Find All Trainable from train.mnn
    std::map<std::string, const MNN::Blob*> trainables;
    for (int i=0; i<trainMNN->oplists()->size(); ++i) {
        auto op = trainMNN->oplists()->GetAs<MNN::Op>(i);
        if (MNN::OpType_TrainableParam == op->type()) {
            if (nullptr != op->main_as_Blob() && nullptr != op->name()) {
                trainables.insert(std::make_pair(op->name()->str(), op->main_as_Blob()));
            }
        }
    }
    // Update Raw Trainables
    if (nullptr != config->trainables()) {
        for (int i=0; i<config->trainables()->size(); ++i) {
            auto kv = config->trainables()->GetAs<MNNTrain::KV>(i);
            if (nullptr == kv->key() || nullptr == kv->value()) {
                continue;
            }
            auto key = kv->key()->str();
            auto value = kv->value()->str();
            auto updateBlobIter = trainables.find(value);
            if (updateBlobIter == trainables.end()) {
                MNN_ERROR("Can't find %s from train.mnn\n", value.c_str());
                continue;
            }
            auto srcblob = updateBlobIter->second;
            auto src = _getBlobPtr(srcblob);
            auto size = _getBlobSize(srcblob);
            // Find Op from infer mnn
            for (int opIndex=0; opIndex < inferMNN->oplists()->size(); ++opIndex) {
                auto op = inferMNN->oplists()->GetAs<MNN::Op>(opIndex);
                if (nullptr == op->name() || nullptr == op->main_as_Blob()) {
                    continue;
                }
                if (op->name()->str() == key) {
                    // Update
                    FUNC_PRINT_ALL(op->name()->c_str(), s);
                    auto dst = _getBlobPtr(op->main_as_Blob());
                    ::memcpy(dst, src, size);
                    break;
                }
            }
        }
    }
    // Update Convolution
    if (nullptr != config->convolutions()) {
        for (int i=0; i<config->convolutions()->size(); ++i) {
            auto kv = config->convolutions()->GetAs<MNNTrain::OpInfo>(i);
            if (nullptr == kv->op()) {
                continue;
            }
            auto key = kv->op()->str();
            // Find Convolution
            for (int opIndex=0; opIndex < inferMNN->oplists()->size(); ++opIndex) {
                auto op = inferMNN->oplists()->GetAs<MNN::Op>(opIndex);
                if (nullptr == op->name() || nullptr == op->main_as_Convolution2D()) {
                    continue;
                }
                if (op->name()->str() == key) {
                    auto convolutionParameter = op->main_as_Convolution2D();
                    if (nullptr == convolutionParameter->weight() || nullptr == convolutionParameter->bias()) {
                        MNN_ERROR("%s Convolution's weight is compressed, can't update\n", key.c_str());
                        continue;
                    }
                    // Update
                    do {
                        if (nullptr == kv->weight()) {
                            break;
                        }
                        auto updateBlobIter = trainables.find(kv->weight()->str());
                        if (updateBlobIter == trainables.end()) {
                            MNN_ERROR("Can't find %s from train.mnn\n", kv->weight()->c_str());
                            break;
                        }
                        auto srcblob = updateBlobIter->second;
                        auto src = _getBlobPtr(srcblob);
                        auto size = _getBlobSize(srcblob);
                        ::memcpy((void*)convolutionParameter->weight()->data(), src, size);
                    } while(false);
                    do {
                        if (nullptr == kv->bias()) {
                            break;
                        }
                        auto updateBlobIter = trainables.find(kv->bias()->str());
                        if (updateBlobIter == trainables.end()) {
                            MNN_ERROR("Can't find %s from train.mnn\n", kv->bias()->c_str());
                            break;
                        }
                        auto srcblob = updateBlobIter->second;
                        auto src = _getBlobPtr(srcblob);
                        auto size = _getBlobSize(srcblob);
                        ::memcpy((void*)convolutionParameter->bias()->data(), src, size);
                    } while(false);
                }
            }
        }
    }
    std::ofstream outputOs(argv[2]);
    outputOs.write((const char*)infer->getModelBuffer().first, infer->getModelBuffer().second);
    return 0;
}
