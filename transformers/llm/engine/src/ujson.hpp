#ifndef UJSON_HPP
#define UJSON_HPP

#include <string>
#include <vector>
#include <map>
#include <initializer_list>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <cstdint>

// Backend selection: rapidjson by default
#ifndef UJSON_USE_RAPIDJSON
#define UJSON_USE_RAPIDJSON
#endif
#ifdef UJSON_USE_RAPIDJSON
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#else
#include <nlohmann/json.hpp>
#endif

namespace ujson {

#ifdef UJSON_USE_RAPIDJSON

class json;

/**
 * @brief Internal helper for type extraction from RapidJSON-based ujson bridge.
 */
template<typename T, typename Enable = void>
struct json_getter {
    static T get(const json& j);
};

/**
 * @brief RapidJSON implementation of the universal JSON wrapper.
 */
class json {
public:
    // Constructors
    json() : m_doc(std::make_shared<rapidjson::Document>()), m_val(m_doc.get()) {
        m_val->SetNull();
    }

    json(const json& other) : m_doc(other.m_doc), m_val(other.m_val) {}
    json(json&& other) : m_doc(std::move(other.m_doc)), m_val(other.m_val) {}

    // Implicit conversions from basic types
    json(std::nullptr_t) : json() {}
    json(bool b) : json() { m_val->SetBool(b); }

    template<typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
    json(T i) : json() {
        if (std::is_signed<T>::value) m_val->SetInt64(static_cast<int64_t>(i));
        else m_val->SetUint64(static_cast<uint64_t>(i));
    }

    json(float f) : json() { m_val->SetDouble(static_cast<double>(f)); }
    json(double d) : json() { m_val->SetDouble(d); }
    json(const std::string& s) : json() {
        m_val->SetString(s.c_str(), static_cast<rapidjson::SizeType>(s.length()), m_doc->GetAllocator());
    }
    json(const char* s) : json() {
        m_val->SetString(s, static_cast<rapidjson::SizeType>(strlen(s)), m_doc->GetAllocator());
    }

    // Initializer list support
    json(std::initializer_list<json> init) : json() {
        bool is_obj = true;
        for (const auto& v : init) {
            if (!v.is_array() || v.size() != 2 || !v[0].is_string()) {
                is_obj = false;
                break;
            }
        }
        if (is_obj && init.size() > 0) {
            m_val->SetObject();
            for (const auto& v : init) {
                rapidjson::Value k;
                std::string ks = v[0].get<std::string>();
                k.SetString(ks.c_str(), static_cast<rapidjson::SizeType>(ks.length()), m_doc->GetAllocator());
                rapidjson::Value val_copy;
                val_copy.CopyFrom(*(v.m_val), m_doc->GetAllocator());
                m_val->AddMember(k, val_copy, m_doc->GetAllocator());
            }
        } else {
            m_val->SetArray();
            for (const auto& v : init) {
                rapidjson::Value val_copy;
                val_copy.CopyFrom(*(v.m_val), m_doc->GetAllocator());
                m_val->PushBack(val_copy, m_doc->GetAllocator());
            }
        }
    }

    // Assignment updates the pointed-to value
    json& operator=(const json& other) {
        if (this == &other) return *this;
        if (m_val == other.m_val) return *this;
        if (!m_val || !other.m_val) return *this;
        m_val->CopyFrom(*(other.m_val), m_doc->GetAllocator());
        return *this;
    }

    // Type checks
    bool is_null() const { return m_val->IsNull(); }
    bool is_boolean() const { return m_val->IsBool(); }
    bool is_number() const { return m_val->IsNumber(); }
    bool is_number_integer() const { return m_val->IsInt() || m_val->IsInt64(); }
    bool is_number_float() const { return m_val->IsDouble(); }
    bool is_number_unsigned() const { return m_val->IsUint() || m_val->IsUint64(); }
    bool is_string() const { return m_val->IsString(); }
    bool is_array() const { return m_val->IsArray(); }
    bool is_object() const { return m_val->IsObject(); }

    // Accessors
    template<typename T>
    T get() const {
        if (is_null()) return T();
        return json_getter<T>::get(*this);
    }

    operator std::string() const { return get<std::string>(); }

    // operator[] for reading/writing
    json operator[](const std::string& key) const {
        if (is_object() && m_val->HasMember(key.c_str())) {
            return json(m_doc, const_cast<rapidjson::Value*>(&((*m_val)[key.c_str()])));
        }
        return json(nullptr, nullptr);
    }

    json operator[](const std::string& key) {
        if (is_null()) {
            m_val->SetObject();
        } else if (!is_object()) {
            m_val->SetObject();
        }
        if (!m_val->HasMember(key.c_str())) {
            rapidjson::Value k;
            k.SetString(key.c_str(), static_cast<rapidjson::SizeType>(key.length()), m_doc->GetAllocator());
            m_val->AddMember(k, rapidjson::Value(rapidjson::kNullType), m_doc->GetAllocator());
        }
        return json(m_doc, &((*m_val)[key.c_str()]));
    }

    json operator[](size_t idx) const {
        if (is_array() && idx < m_val->Size()) {
            return json(m_doc, const_cast<rapidjson::Value*>(&((*m_val)[static_cast<rapidjson::SizeType>(idx)])));
        }
        return json(nullptr, nullptr);
    }

    json operator[](size_t idx) {
        if (!is_array()) m_val->SetArray();
        while (m_val->Size() <= idx) {
            m_val->PushBack(rapidjson::Value().SetNull(), m_doc->GetAllocator());
        }
        return json(m_doc, &((*m_val)[static_cast<rapidjson::SizeType>(idx)]));
    }

    json at(const std::string& key) const {
        if (!is_object() || !m_val->HasMember(key.c_str())) return json(nullptr, nullptr);
        return json(m_doc, const_cast<rapidjson::Value*>(&((*m_val)[key.c_str()])));
    }

    json at(const std::string& key) {
        if (!is_object() || !m_val->HasMember(key.c_str())) return json(nullptr, nullptr);
        return json(m_doc, &((*m_val)[key.c_str()]));
    }

    template<typename T>
    T value(const std::string& key, const T& default_value) const {
        if (is_object() && m_val->HasMember(key.c_str())) {
            return (*this)[key].get<T>();
        }
        return default_value;
    }

    std::string value(const std::string& key, const char* default_value) const {
        if (is_object() && m_val->HasMember(key.c_str())) {
            return (*this)[key].get<std::string>();
        }
        return std::string(default_value);
    }

    bool contains(const std::string& key) const { return is_object() && m_val->HasMember(key.c_str()); }
    size_t count(const std::string& key) const { return contains(key) ? 1 : 0; }
    size_t size() const {
        if (is_array()) return (size_t)m_val->Size();
        if (is_object()) return (size_t)m_val->MemberCount();
        return 0;
    }
    bool empty() const { return size() == 0; }

    // Iteration support
    class iterator {
        friend class json;
        friend class const_iterator;
        std::shared_ptr<rapidjson::Document> m_doc;
        rapidjson::Value::MemberIterator m_mit;
        rapidjson::Value::ValueIterator m_vit;
        bool m_is_obj;
    public:
        iterator(std::shared_ptr<rapidjson::Document> doc, rapidjson::Value::MemberIterator it) : m_doc(doc), m_mit(it), m_is_obj(true) {}
        iterator(std::shared_ptr<rapidjson::Document> doc, rapidjson::Value::ValueIterator it) : m_doc(doc), m_vit(it), m_is_obj(false) {}
        bool operator!=(const iterator& other) const { return m_is_obj ? m_mit != other.m_mit : m_vit != other.m_vit; }
        bool operator==(const iterator& other) const { return !(*this != other); }
        iterator& operator++() { if (m_is_obj) ++m_mit; else ++m_vit; return *this; }
        std::string key() const { return m_mit->name.GetString(); }
        json value() const { return json(m_doc, &(m_mit->value)); }
        json operator*() { return m_is_obj ? json(m_doc, &(m_mit->value)) : json(m_doc, &(*m_vit)); }
    };

    class const_iterator {
        friend class json;
        std::shared_ptr<rapidjson::Document> m_doc;
        rapidjson::Value::ConstMemberIterator m_mit;
        rapidjson::Value::ConstValueIterator m_vit;
        bool m_is_obj;
    public:
        const_iterator(std::shared_ptr<rapidjson::Document> doc, rapidjson::Value::ConstMemberIterator it) : m_doc(doc), m_mit(it), m_is_obj(true) {}
        const_iterator(std::shared_ptr<rapidjson::Document> doc, rapidjson::Value::ConstValueIterator it) : m_doc(doc), m_vit(it), m_is_obj(false) {}
        const_iterator(const iterator& other) : m_doc(other.m_doc), m_mit(other.m_mit), m_vit(other.m_vit), m_is_obj(other.m_is_obj) {}
        bool operator!=(const const_iterator& other) const { return m_is_obj ? m_mit != other.m_mit : m_vit != other.m_vit; }
        bool operator==(const const_iterator& other) const { return !(*this != other); }
        bool operator!=(const iterator& other) const { return m_is_obj ? m_mit != other.m_mit : m_vit != other.m_vit; }
        bool operator==(const iterator& other) const { return !(*this != other); }
        const_iterator& operator++() { if (m_is_obj) ++m_mit; else ++m_vit; return *this; }
        std::string key() const { return m_mit->name.GetString(); }
        json value() const { return json(m_doc, const_cast<rapidjson::Value*>(&(m_mit->value))); }
        const json operator*() const { return m_is_obj ? json(m_doc, const_cast<rapidjson::Value*>(&(m_mit->value))) : json(m_doc, const_cast<rapidjson::Value*>(&(*m_vit))); }
    };

    iterator begin() {
        if (is_object()) return iterator(m_doc, m_val->MemberBegin());
        if (is_array()) return iterator(m_doc, m_val->Begin());
        return iterator(m_doc, rapidjson::Value::ValueIterator(nullptr));
    }
    iterator end() {
        if (is_object()) return iterator(m_doc, m_val->MemberEnd());
        if (is_array()) return iterator(m_doc, m_val->End());
        return iterator(m_doc, rapidjson::Value::ValueIterator(nullptr));
    }
    const_iterator begin() const {
        if (is_object()) return const_iterator(m_doc, m_val->MemberBegin());
        if (is_array()) return const_iterator(m_doc, m_val->Begin());
        return const_iterator(m_doc, rapidjson::Value::ConstValueIterator(nullptr));
    }
    const_iterator end() const {
        if (is_object()) return const_iterator(m_doc, m_val->MemberEnd());
        if (is_array()) return const_iterator(m_doc, m_val->End());
        return const_iterator(m_doc, rapidjson::Value::ConstValueIterator(nullptr));
    }

    iterator find(const std::string& key) {
        if (is_object()) {
            auto it = m_val->FindMember(key.c_str());
            if (it != m_val->MemberEnd()) return iterator(m_doc, it);
        }
        return end();
    }

    const_iterator find(const std::string& key) const {
        if (is_object()) {
            auto it = m_val->FindMember(key.c_str());
            if (it != m_val->MemberEnd()) return const_iterator(m_doc, it);
        }
        return end();
    }

    // Modification
    void push_back(const json& val) {
        if (!is_array()) m_val->SetArray();
        rapidjson::Value v;
        v.CopyFrom(*(val.m_val), m_doc->GetAllocator());
        m_val->PushBack(v, m_doc->GetAllocator());
    }
    void erase(const std::string& key) { if (is_object()) m_val->RemoveMember(key.c_str()); }
    void clear() {
        if (is_array()) m_val->Clear();
        else if (is_object()) m_val->RemoveAllMembers();
    }

    void merge(const json& source) {
        if (!is_object() || !source.is_object()) return;
        for (auto it = source.begin(); it != source.end(); ++it) {
            auto key = it.key();
            if (contains(key) && (*this)[key].is_object() && it.value().is_object()) {
                (*this)[key].merge(it.value());
            } else {
                (*this)[key] = it.value();
            }
        }
    }

    // Comparison helpers
    static int compare(const rapidjson::Value& lhs, const rapidjson::Value& rhs) {
        if (lhs.GetType() != rhs.GetType()) return lhs.GetType() < rhs.GetType() ? -1 : 1;
        switch (lhs.GetType()) {
            case rapidjson::kNullType: return 0;
            case rapidjson::kFalseType:
            case rapidjson::kTrueType: return (lhs.GetBool() == rhs.GetBool()) ? 0 : (lhs.GetBool() ? 1 : -1);
            case rapidjson::kNumberType: {
                if (lhs.IsDouble() || rhs.IsDouble()) {
                    double l = lhs.IsDouble() ? lhs.GetDouble() : (lhs.IsInt64() ? (double)lhs.GetInt64() : (double)lhs.GetUint64());
                    double r = rhs.IsDouble() ? rhs.GetDouble() : (rhs.IsInt64() ? (double)rhs.GetInt64() : (double)rhs.GetUint64());
                    return (l < r) ? -1 : (l > r ? 1 : 0);
                }
                int64_t l = lhs.IsInt64() ? lhs.GetInt64() : (int64_t)lhs.GetUint64();
                int64_t r = rhs.IsInt64() ? rhs.GetInt64() : (int64_t)rhs.GetUint64();
                return (l < r) ? -1 : (l > r ? 1 : 0);
            }
            case rapidjson::kStringType: return strcmp(lhs.GetString(), rhs.GetString());
            case rapidjson::kArrayType: {
                if (lhs.Size() != rhs.Size()) return lhs.Size() < rhs.Size() ? -1 : 1;
                for (rapidjson::SizeType i = 0; i < lhs.Size(); i++) {
                    int c = compare(lhs[i], rhs[i]);
                    if (c != 0) return c;
                }
                return 0;
            }
            case rapidjson::kObjectType: {
                if (lhs.MemberCount() != rhs.MemberCount()) return lhs.MemberCount() < rhs.MemberCount() ? -1 : 1;
                return 0;
            }
            default: return 0;
        }
    }

    // Comparison operators
    bool operator==(const json& other) const { return compare(*m_val, *(other.m_val)) == 0; }
    bool operator!=(const json& other) const { return !(*this == other); }
    bool operator<(const json& other) const { return compare(*m_val, *(other.m_val)) < 0; }
    bool operator>(const json& other) const { return other < *this; }
    bool operator<=(const json& other) const { return !(*this > other); }
    bool operator>=(const json& other) const { return !(*this < other); }

    // Static utilities
    static json parse(const std::string& s) {
        auto doc = std::make_shared<rapidjson::Document>();
        if (doc->Parse(s.c_str()).HasParseError()) {
            return json();
        }
        return json(doc, doc.get());
    }

    static json parse_insitu(char* buffer) {
        auto doc = std::make_shared<rapidjson::Document>();
        if (doc->ParseInsitu(buffer).HasParseError()) {
            return json();
        }
        return json(doc, doc.get());
    }

    static json object() {
        json j;
        j.m_val->SetObject();
        return j;
    }
    static json array() {
        json j;
        j.m_val->SetArray();
        return j;
    }

    std::string dump(int indent = -1) const {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        m_val->Accept(writer);
        return buffer.GetString();
    }

    template<typename T, typename Enable> friend struct json_getter;

private:
    // Internal constructor
    json(std::shared_ptr<rapidjson::Document> doc, rapidjson::Value* val) : m_doc(doc), m_val(val) {
        if (!m_val) {
            if (!m_doc) m_doc = std::make_shared<rapidjson::Document>();
            m_doc->SetNull();
            m_val = m_doc.get();
        }
    }

    std::shared_ptr<rapidjson::Document> m_doc;
    rapidjson::Value* m_val;
};

// Getter specializations
template<> struct json_getter<bool> { static bool get(const json& j) { return j.m_val->IsBool() ? j.m_val->GetBool() : false; } };

template<typename T>
struct json_getter<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type> {
    static T get(const json& j) {
        if (!j.m_val->IsNumber()) return 0;
        if (std::is_signed<T>::value) return static_cast<T>(j.m_val->GetInt64());
        return static_cast<T>(j.m_val->GetUint64());
    }
};
template<> struct json_getter<float> { static float get(const json& j) { return j.m_val->IsNumber() ? static_cast<float>(j.m_val->GetDouble()) : 0.0f; } };
template<> struct json_getter<double> { static double get(const json& j) { return j.m_val->IsNumber() ? j.m_val->GetDouble() : 0.0; } };
template<> struct json_getter<std::string> { static std::string get(const json& j) { return j.m_val->IsString() ? std::string(j.m_val->GetString(), j.m_val->GetStringLength()) : ""; } };

template<typename T>
struct json_getter<std::vector<T>> {
    static std::vector<T> get(const json& j) {
        std::vector<T> res;
        if (j.is_array()) {
            res.reserve(j.size());
            for (auto it = j.begin(); it != j.end(); ++it) {
                res.push_back((*it).template get<T>());
            }
        }
        return res;
    }
};

#else // !UJSON_USE_RAPIDJSON -> nlohmann/json implementation (view-based)

class json {
public:
    using nlohmann_json = nlohmann::json;

    // Constructors
    json() : m_doc(std::make_shared<nlohmann_json>()), m_val(m_doc.get()) {}
    json(const nlohmann_json& j) : m_doc(std::make_shared<nlohmann_json>(j)), m_val(m_doc.get()) {}
    json(nlohmann_json&& j) : m_doc(std::make_shared<nlohmann_json>(std::move(j))), m_val(m_doc.get()) {}

    json(const json& other) : m_doc(other.m_doc), m_val(other.m_val) {}
    json(json&& other) : m_doc(std::move(other.m_doc)), m_val(other.m_val) {}

    json(std::nullptr_t) : json() { *m_val = nullptr; }
    json(bool b) : json() { *m_val = b; }

    template<typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
    json(T i) : json() {
        if (std::is_signed<T>::value) *m_val = static_cast<int64_t>(i);
        else *m_val = static_cast<uint64_t>(i);
    }

    json(float f) : json() { *m_val = f; }
    json(double d) : json() { *m_val = d; }
    json(const std::string& s) : json() { *m_val = s; }
    json(const char* s) : json() { *m_val = s; }

    json(std::initializer_list<json> init) : json() {
        bool is_obj = true;
        for (const auto& v : init) {
            if (!v.is_array() || v.size() != 2 || !v[0].is_string()) {
                is_obj = false;
                break;
            }
        }
        if (is_obj && init.size() > 0) {
            *m_val = nlohmann_json::object();
            for (const auto& v : init) {
                (*m_val)[v[0].get<std::string>()] = *(v.m_val);
            }
        } else {
            *m_val = nlohmann_json::array();
            for (const auto& v : init) {
                m_val->push_back(*(v.m_val));
            }
        }
    }

    // Assignment updates the pointed-to value
    json& operator=(const json& other) {
        if (this == &other) return *this;
        if (m_val == other.m_val) return *this;
        *m_val = *(other.m_val);
        return *this;
    }

    // Type checks
    bool is_null() const { return m_val->is_null(); }
    bool is_boolean() const { return m_val->is_boolean(); }
    bool is_number() const { return m_val->is_number(); }
    bool is_number_integer() const { return m_val->is_number_integer(); }
    bool is_number_float() const { return m_val->is_number_float(); }
    bool is_number_unsigned() const { return m_val->is_number_unsigned(); }
    bool is_string() const { return m_val->is_string(); }
    bool is_array() const { return m_val->is_array(); }
    bool is_object() const { return m_val->is_object(); }

    template<typename T>
    T get() const { return m_val->get<T>(); }

    operator std::string() const { return m_val->get<std::string>(); }

    json operator[](const std::string& key) const {
        if (is_object() && m_val->contains(key)) {
            return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(key))));
        }
        return json(nullptr, nullptr);
    }

    json operator[](const std::string& key) {
        if (!is_object()) *m_val = nlohmann_json::object();
        return json(m_doc, &((*m_val)[key]));
    }

    json operator[](size_t idx) const {
        if (is_array() && idx < m_val->size()) {
            return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(idx))));
        }
        return json(nullptr, nullptr);
    }

    json operator[](size_t idx) {
        if (!is_array()) *m_val = nlohmann_json::array();
        return json(m_doc, &((*m_val)[idx]));
    }

    json at(const std::string& key) const { return json(m_doc, const_cast<nlohmann_json*>(&(m_val->at(key)))); }
    json at(const std::string& key) { return json(m_doc, &(m_val->at(key))); }

    template<typename T>
    T value(const std::string& key, const T& default_value) const {
        return m_val->value(key, default_value);
    }

    std::string value(const std::string& key, const char* default_value) const {
        return m_val->value(key, std::string(default_value));
    }

    bool contains(const std::string& key) const { return m_val->contains(key); }
    size_t count(const std::string& key) const { return m_val->count(key); }
    size_t size() const { return m_val->size(); }
    bool empty() const { return m_val->empty(); }

    class iterator {
        friend class json;
        friend class const_iterator;
        std::shared_ptr<nlohmann_json> m_doc;
        nlohmann_json::iterator m_it;
    public:
        iterator(std::shared_ptr<nlohmann_json> doc, nlohmann_json::iterator it) : m_doc(doc), m_it(it) {}
        bool operator!=(const iterator& other) const { return m_it != other.m_it; }
        bool operator==(const iterator& other) const { return m_it == other.m_it; }
        iterator& operator++() { ++m_it; return *this; }
        std::string key() const { return m_it.key(); }
        json value() const { return json(m_doc, &(*m_it)); }
        json operator*() { return json(m_doc, &(*m_it)); }
    };

    class const_iterator {
        friend class json;
        std::shared_ptr<nlohmann_json> m_doc;
        nlohmann_json::const_iterator m_it;
    public:
        const_iterator(std::shared_ptr<nlohmann_json> doc, nlohmann_json::const_iterator it) : m_doc(doc), m_it(it) {}
        const_iterator(const iterator& other) : m_doc(other.m_doc), m_it(other.m_it) {}
        bool operator!=(const const_iterator& other) const { return m_it != other.m_it; }
        bool operator==(const const_iterator& other) const { return m_it == other.m_it; }
        bool operator!=(const iterator& other) const { return m_it != other.m_it; }
        bool operator==(const iterator& other) const { return m_it == other.m_it; }
        const_iterator& operator++() { ++m_it; return *this; }
        std::string key() const { return m_it.key(); }
        json value() const { return json(m_doc, const_cast<nlohmann_json*>(&(*m_it))); }
        const json operator*() const { return json(m_doc, const_cast<nlohmann_json*>(&(*m_it))); }
    };

    iterator begin() { return iterator(m_doc, m_val->begin()); }
    iterator end() { return iterator(m_doc, m_val->end()); }
    const_iterator begin() const { return const_iterator(m_doc, m_val->begin()); }
    const_iterator end() const { return const_iterator(m_doc, m_val->end()); }

    iterator find(const std::string& key) { return iterator(m_doc, m_val->find(key)); }
    const_iterator find(const std::string& key) const { return const_iterator(m_doc, m_val->find(key)); }

    void push_back(const json& val) { m_val->push_back(*(val.m_val)); }
    void erase(const std::string& key) { m_val->erase(key); }
    void clear() { m_val->clear(); }

    void merge(const json& source) {
        if (!is_object() || !source.is_object()) return;
        for (auto it = source.begin(); it != source.end(); ++it) {
            auto key = it.key();
            if (contains(key) && (*this)[key].is_object() && it.value().is_object()) {
                (*this)[key].merge(it.value());
            } else {
                (*this)[key] = it.value();
            }
        }
    }

    bool operator==(const json& other) const { return *m_val == *(other.m_val); }
    bool operator!=(const json& other) const { return *m_val != *(other.m_val); }
    bool operator<(const json& other) const { return *m_val < *(other.m_val); }
    bool operator>(const json& other) const { return *m_val > *(other.m_val); }
    bool operator<=(const json& other) const { return *m_val <= *(other.m_val); }
    bool operator>=(const json& other) const { return *m_val >= *(other.m_val); }

    static json parse(const std::string& s) {
        try {
            auto doc = std::make_shared<nlohmann_json>(nlohmann_json::parse(s));
            return json(doc, doc.get());
        } catch (...) { return json(); }
    }
    static json object() {
        json j;
        *(j.m_val) = nlohmann_json::object();
        return j;
    }
    static json array() {
        json j;
        *(j.m_val) = nlohmann_json::array();
        return j;
    }

    std::string dump(int indent = -1) const { return m_val->dump(indent); }

    const nlohmann_json& raw() const { return *m_val; }
    nlohmann_json& raw() { return *m_val; }

private:
    // Internal constructor
    json(std::shared_ptr<nlohmann_json> doc, nlohmann_json* val) : m_doc(doc), m_val(val) {
        if (!m_val) {
            static nlohmann_json null_val = nullptr;
            m_val = &null_val;
            m_doc = std::make_shared<nlohmann_json>(nullptr);
        }
    }

    std::shared_ptr<nlohmann_json> m_doc;
    nlohmann_json* m_val;
};

#endif // UJSON_USE_RAPIDJSON

} // namespace ujson

#endif // UJSON_HPP