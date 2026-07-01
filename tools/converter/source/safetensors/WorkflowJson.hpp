#ifndef WorkflowJson_hpp
#define WorkflowJson_hpp

#include <string>

#include <rapidjson/document.h>

namespace MNN {
namespace SafeTensors {
namespace WorkflowJson {

inline const rapidjson::Value* _findMember(const rapidjson::Value& obj, const char* key) {
    if (!obj.IsObject() || nullptr == key) {
        return nullptr;
    }
    auto it = obj.FindMember(key);
    if (it == obj.MemberEnd()) {
        return nullptr;
    }
    return &it->value;
}

inline std::string getString(const rapidjson::Value& obj, const char* key, const std::string& defaultValue = "") {
    auto v = _findMember(obj, key);
    if (nullptr == v || !v->IsString()) {
        return defaultValue;
    }
    return v->GetString();
}

inline bool getBool(const rapidjson::Value& obj, const char* key, bool defaultValue = false) {
    auto v = _findMember(obj, key);
    if (nullptr == v) {
        return defaultValue;
    }
    return v->GetBool();
}
inline int getInt(const rapidjson::Value& obj, const char* key, int defaultValue = 0) {
    auto v = _findMember(obj, key);
    if (nullptr == v || !v->IsInt()) {
        return defaultValue;
    }
    return v->GetInt();
}

inline float getFloat(const rapidjson::Value& obj, const char* key, float defaultValue = 0.0f) {
    auto v = _findMember(obj, key);
    if (nullptr == v) {
        return defaultValue;
    }
    if (v->IsFloat()) {
        return v->GetFloat();
    }
    if (v->IsDouble()) {
        return static_cast<float>(v->GetDouble());
    }
    return defaultValue;
}

inline const rapidjson::Value* getArray(const rapidjson::Value& obj, const char* key) {
    auto v = _findMember(obj, key);
    if (nullptr == v || !v->IsArray()) {
        return nullptr;
    }
    return v;
}

inline bool firstArrayStringEquals(const rapidjson::Value& obj, const char* key, const char* expected) {
    if (nullptr == expected) {
        return false;
    }
    auto v = getArray(obj, key);
    if (nullptr == v || v->Empty()) {
        return false;
    }
    auto& first = (*v)[0];
    if (!first.IsString()) {
        return false;
    }
    return first.GetString() == std::string(expected);
}

inline bool arrayStringContains(const rapidjson::Value& obj, const char* key, const char* expected) {
    if (nullptr == expected) {
        return false;
    }
    auto v = getArray(obj, key);
    if (nullptr == v) {
        return false;
    }
    const std::string target(expected);
    for (auto& item : v->GetArray()) {
        if (item.IsString() && item.GetString() == target) {
            return true;
        }
    }
    return false;
}

inline const rapidjson::Value* findFirstBlockByType(const rapidjson::Value& model, const char* type) {
    if (nullptr == type) {
        return nullptr;
    }
    auto blocks = getArray(model, "blocks");
    if (nullptr == blocks) {
        return nullptr;
    }
    for (auto& item : blocks->GetArray()) {
        if (!item.IsObject()) {
            continue;
        }
        auto t = _findMember(item, "type");
        if (nullptr != t && t->IsString() && t->GetString() == std::string(type)) {
            return &item;
        }
    }
    return nullptr;
}

} // namespace WorkflowJson
} // namespace SafeTensors
} // namespace MNN

#endif
