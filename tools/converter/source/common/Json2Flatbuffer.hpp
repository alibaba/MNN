#ifndef JSON2FLATBUFFER_HPP
#define JSON2FLATBUFFER_HPP
#include "rapidjson/document.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

namespace MNN {
class Json2Flatbuffer {
public:
    static std::string writeFlatbufferToJson(const uint8_t* buffer, const flatbuffers::TypeTable* table);
    static flatbuffers::Offset<void> writeJsonToFlatbuffer(const flatbuffers::TypeTable * table, flatbuffers::FlatBufferBuilder& builder, const rapidjson::GenericObject<false, rapidjson::GenericValue<rapidjson::UTF8<>>>& object);
};
};

#endif
