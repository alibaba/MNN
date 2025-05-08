//
//  llmconfig.hpp
//
//  Created by MNN on 2024/07/19.
//  ZhaodeWang
//

#include "rapidjson/document.h"
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "llmconfig.hpp"

namespace MNN {
namespace Transformer {

bool merge_json(rapidjson::Value& destination, const rapidjson::Value& source,
                rapidjson::Document::AllocatorType& allocator) {
    if (!source.IsObject() || !destination.IsObject()) {
        return false;
    }

    for (auto it = source.MemberBegin(); it != source.MemberEnd(); ++it) {
        const char* key = it->name.GetString();
        if (destination.HasMember(key)) {
            if (destination[key].IsObject() && it->value.IsObject()) {
                // Recursively merge the two JSON objects
                merge_json(destination[key], it->value, allocator);
            } else {
                // Overwrite the value in the destination
                destination[key].CopyFrom(it->value, allocator);
            }
        } else {
            // Add the value to the destination
            rapidjson::Value newKey(key, allocator);
            rapidjson::Value newValue;
            newValue.CopyFrom(it->value, allocator);
            destination.AddMember(newKey, newValue, allocator);
        }
    }
    return true;
}

} // Transformer
} // MNN

