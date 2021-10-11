//
//  MNNRevert2Buffer.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <map>
#include "MNN_generated.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include <string.h>
#include <MNN/MNNDefine.h>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

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
}\

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

int main(int argc, const char** argv) {
    if (argc <= 2) {
        printf("Usage: ./MNNRevert2Buffer.out XXX.json XXX.mnn\n");
        return 0;
    }
    rapidjson::Document document;
    {
        std::ifstream fileNames(argv[1]);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
    }
    auto object = document.GetObject();
    flatbuffers::FlatBufferBuilder builder;
    builder.ForceDefaults(true);
    auto table = MNN::NetTypeTable();
    auto offset = _writeJsonToFlatbuffer(table, builder, object);
    builder.Finish(offset);
    std::ofstream outputOs(argv[2]);
    outputOs.write((char*)builder.GetBufferPointer(), builder.GetSize());
    return 0;
}
