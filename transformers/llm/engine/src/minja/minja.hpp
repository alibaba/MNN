/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h" // For GetParseError_En

#include <MNN/MNNDefine.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

static void _printlog(const std::string& i) {
    MNN_PRINT("%s\n", i.c_str());
}


namespace minja {
enum ObjectType {
    JSON_NULL = 0,
    JSON_INT64 = 1,
    JSON_DOUBLE = 2,
    JSON_STRING = 3,
    JSON_BOOL = 4,
};
class json {
public:
    ObjectType mType;
    std::string mString;
    int64_t mInt;
    double mDouble;
    bool mBool;
    json() {
        mType = JSON_NULL;
    }
    json(bool v) {
        mBool = v;
        mType = JSON_BOOL;
    }
    json(int64_t v) {
        mInt = v;
        mType = JSON_INT64;
    }
    json(double v) {
        mDouble = v;
        mType = JSON_DOUBLE;
    }
    json(std::string v) {
        mString = v;
        mType = JSON_STRING;
    }
    json(const char* c, size_t len) {
        mString.assign(c, len);
        mType = JSON_STRING;
    }
    json(const json& right) {
        mType = right.mType;
        mInt = right.mInt;
        mBool = right.mBool;
        mDouble = right.mDouble;
        mString = right.mString;
    }
    bool operator==(const json& right) const {
        if (mType != right.mType) {
            return false;
        }
        switch (mType) {
            case JSON_STRING:
                return mString == right.mString;
            case JSON_INT64:
                return mInt == right.mInt;
            case JSON_DOUBLE:
                return mDouble == right.mDouble;
            case JSON_BOOL:
                return mBool == right.mBool;
            default:
                break;
        }
        return true;
    }
    bool is_string() const {
        return mType == JSON_STRING;
    }
    bool is_null() const { return mType == JSON_NULL; }
    bool is_boolean() const { return mType == JSON_BOOL; }
    bool is_number_integer() const { return mType == JSON_INT64; }
    bool is_number_float() const { return mType == JSON_DOUBLE; }
    bool is_number() const { return mType == JSON_DOUBLE || mType == JSON_INT64; }
    bool empty() const {
        return mString.empty();
    }
    bool get(bool& ) const {
        return mBool;
    }
    std::string get(std::string& ) const {
        return mString;
    }
    int64_t get(int64_t& ) const {
        return mInt;
    }
    int get(int& ) const {
        return mInt;
    }
    float get(float&) const {
        return mDouble;
    }
    double get(double& ) const {
        return mDouble;
    }

    std::string dump() const {
        switch (mType) {
            case JSON_STRING:
                return mString;
            case JSON_INT64:
                return std::to_string(mInt);
            case JSON_DOUBLE:
                return std::to_string(mDouble);
            case JSON_BOOL:
                return mBool ? "True" : "False";
            default:
                break;
        }
        return "null";
    }
};

class Context;

struct Options {
    bool trim_blocks;  // removes the first newline after a block
    bool lstrip_blocks;  // removes leading whitespace on the line of the block
    bool keep_trailing_newline;  // don't remove last newline
};

struct ArgumentsValue;

inline std::string normalize_newlines(const std::string & s) {
#ifdef _WIN32
  static const std::regex nl_regex("\r\n");
  return std::regex_replace(s, nl_regex, "\n");
#else
  return s;
#endif
}

/* Values that behave roughly like in Python. */
class Value : public std::enable_shared_from_this<Value> {
public:
  using CallableType = std::function<Value(const std::shared_ptr<Context> &, ArgumentsValue &)>;
  using FilterType = std::function<Value(const std::shared_ptr<Context> &, ArgumentsValue &)>;

private:
  using ObjectType = std::map<std::string, Value>;  // Only contains primitive keys
  using ArrayType = std::vector<Value>;

  std::shared_ptr<ArrayType> array_;
  std::shared_ptr<ObjectType> object_;
  std::shared_ptr<CallableType> callable_;
  json primitive_;

  Value(const std::shared_ptr<ArrayType> & array) : array_(array) {}
  Value(const std::shared_ptr<ObjectType> & object) : object_(object) {}
  Value(const std::shared_ptr<CallableType> & callable) : object_(std::make_shared<ObjectType>()), callable_(callable) {}

  /* Python-style string repr */
  static void dump_string(const std::string& s, std::ostringstream & out, char string_quote = '\'') {
    if (string_quote == '"' || s.find('\'') != std::string::npos) {
      out << s;
      return;
    }
    // Reuse json dump, just changing string quotes
    out << string_quote;
    for (size_t i = 1, n = s.size() - 1; i < n; ++i) {
      if (s[i] == '\\' && s[i + 1] == '"') {
        out << '"';
        i++;
      } else if (s[i] == string_quote) {
        out << '\\' << string_quote;
      } else {
        out << s[i];
      }
    }
    out << string_quote;
  }
  void dump(std::ostringstream & out, int indent = -1, int level = 0, bool to_json = false) const {
    auto print_indent = [&](int level) {
      if (indent > 0) {
          out << "\n";
          for (int i = 0, n = level * indent; i < n; ++i) out << ' ';
      }
    };
    auto print_sub_sep = [&]() {
      out << ',';
      if (indent < 0) out << ' ';
      else print_indent(level + 1);
    };

    auto string_quote = to_json ? '"' : '\'';

    if (is_null()) out << "null";
    else if (array_) {
      out << "[";
      print_indent(level + 1);
      for (size_t i = 0; i < array_->size(); ++i) {
        if (i) print_sub_sep();
        (*array_)[i].dump(out, indent, level + 1, to_json);
      }
      print_indent(level);
      out << "]";
    } else if (object_) {
      out << "{";
      print_indent(level + 1);
      for (auto begin = object_->begin(), it = begin; it != object_->end(); ++it) {
        if (it != begin) print_sub_sep();
        dump_string(it->first, out, string_quote);
        out << ": ";
        it->second.dump(out, indent, level + 1, to_json);
      }
      print_indent(level);
      out << "}";
    } else if (callable_) {
      _printlog("Cannot dump callable to JSON");
    } else if (is_boolean() && !to_json) {
      out << (this->to_bool() ? "True" : "False");
    } else if (is_string() && !to_json) {
      dump_string(primitive_.mString, out, string_quote);
    } else {
      out << primitive_.dump();
    }
  }

public:
  Value() {}
  Value(const bool& v) : primitive_(v) {}
  Value(const int64_t & v) : primitive_(v) {}
  Value(const double& v) : primitive_(v) {}
  Value(const std::nullptr_t &) {}
  Value(const std::string & v) : primitive_(v) {}
  Value(const char * v) : primitive_(std::string(v)) {}
    Value(json& json) : primitive_(json) {
        // Do nothing
    }

  Value(const rapidjson::Value & v) {
    if (v.IsObject()) {
      auto object = std::make_shared<ObjectType>();
      for (auto& it : v.GetObject()) {
        (*object)[it.name.GetString()] = it.value;
      }
      object_ = std::move(object);
    } else if (v.IsArray()) {
      auto array = std::make_shared<ArrayType>();
      for (const auto& item : v.GetArray()) {
        array->push_back(Value(item));
      }
      array_ = array;
    } else {
        if (v.IsFloat() || v.IsDouble()) {
            primitive_ = json(v.GetDouble());
        } else if (v.IsInt() || v.IsInt64()) {
            primitive_ = json(v.GetInt64());
        } else if (v.IsBool()) {
            primitive_ = json(v.GetBool());
        } else if (v.IsString()) {
            primitive_ = json(v.GetString(), v.GetStringLength());
        }
    }
  }

  std::vector<Value> keys() {
    if (!object_) _printlog("Value is not an object: " + dump());
    std::vector<Value> res;
    for (const auto& item : *object_) {
      res.push_back(item.first);
    }
    return res;
  }

  size_t size() const {
    if (is_object()) return object_->size();
    if (is_array()) return array_->size();
    if (is_string()) return primitive_.mString.size();
    _printlog("Value is not an array or object: " + dump());
    return 0;
  }

  static Value array(const std::vector<Value> values = {}) {
    auto array = std::make_shared<ArrayType>();
    for (const auto& item : values) {
      array->push_back(item);
    }
    return Value(array);
  }
  static Value object(const std::shared_ptr<ObjectType> object = std::make_shared<ObjectType>()) {
    return Value(object);
  }
  static Value callable(const CallableType & callable) {
    return Value(std::make_shared<CallableType>(callable));
  }

  void insert(size_t index, const Value& v) {
    if (!array_)
      _printlog("Value is not an array: " + dump());
    array_->insert(array_->begin() + index, v);
  }
  void push_back(const Value& v) {
    if (!array_)
      _printlog("Value is not an array: " + dump());
    array_->push_back(v);
  }
  Value pop(const Value& index) {
    if (is_array()) {
      if (array_->empty())
        _printlog("pop from empty list");
      if (index.is_null()) {
        auto ret = array_->back();
        array_->pop_back();
        return ret;
      } else if (!index.is_number_integer()) {
        _printlog("pop index must be an integer: " + index.dump());
      } else {
        auto i = index.get<int>();
        if (i < 0 || i >= static_cast<int>(array_->size()))
          _printlog("pop index out of range: " + index.dump());
        auto it = array_->begin() + (i < 0 ? array_->size() + i : i);
        auto ret = *it;
        array_->erase(it);
        return ret;
      }
    } else if (is_object()) {
      if (!index.is_hashable())
        _printlog("Unhashable type: " + index.dump());
      auto it = object_->find(index.primitive_.dump());
      if (it == object_->end())
        _printlog("Key not found: " + index.dump());
      auto ret = it->second;
      object_->erase(it);
      return ret;
    } else {
      _printlog("Value is not an array or object: " + dump());
    }
    return Value();
  }
  Value get(const Value& key) {
    if (array_) {
      if (!key.is_number_integer()) {
        return Value();
      }
      auto index = key.get<int>();
      return array_->at(index < 0 ? array_->size() + index : index);
    } else if (object_) {
      if (!key.is_hashable()) _printlog("Unhashable type: " + dump());
      auto it = object_->find(key.primitive_.dump());
      if (it == object_->end()) return Value();
      return it->second;
    }
    return Value();
  }
  void set(const std::string& key, const Value& value) {
      if (!object_) {
          _printlog("Value is not an object: " + dump());
          return;
      }
    (*object_)[key] = value;
  }
  Value call(const std::shared_ptr<Context> & context, ArgumentsValue & args) const {
      if (!callable_) {
//          _printlog("Value is not callable: " + dump());
          return Value();
      }
    return (*callable_)(context, args);
  }

  bool is_object() const { return !!object_; }
  bool is_array() const { return !!array_; }
  bool is_callable() const { return !!callable_; }
  bool is_null() const { return !object_ && !array_ && primitive_.is_null() && !callable_; }
  bool is_boolean() const { return primitive_.is_boolean(); }
  bool is_number_integer() const { return primitive_.is_number_integer(); }
  bool is_number_float() const { return primitive_.is_number_float(); }
  bool is_number() const { return primitive_.is_number(); }
  bool is_string() const { return primitive_.is_string(); }
  bool is_iterable() const { return is_array() || is_object() || is_string(); }

  bool is_primitive() const { return !array_ && !object_ && !callable_; }
  bool is_hashable() const { return is_primitive(); }

  bool empty() const {
    if (is_null())
      _printlog("Undefined value or reference");
    if (is_string()) return primitive_.empty();
    if (is_array()) return array_->empty();
    if (is_object()) return object_->empty();
    return false;
  }

  void for_each(const std::function<void(Value &)> & callback) const {
    if (is_null())
      _printlog("Undefined value or reference");
    if (array_) {
      for (auto& item : *array_) {
        callback(item);
      }
    } else if (object_) {
      for (auto & item : *object_) {
        Value key(item.first);
        callback(key);
      }
    } else if (is_string()) {
      for (char c : primitive_.mString) {
        auto val = Value(std::string(1, c));
        callback(val);
      }
    } else {
      _printlog("Value is not iterable: " + dump());
    }
  }

  bool to_bool() const {
    if (is_null()) return false;
    if (is_boolean()) return get<bool>();
    if (is_number()) return get<double>() != 0;
    if (is_string()) return !get<std::string>().empty();
    if (is_array()) return !empty();
    return true;
  }

  int64_t to_int() const {
    if (is_null()) return 0;
    if (is_boolean()) return get<bool>() ? 1 : 0;
    if (is_number()) return static_cast<int64_t>(get<double>());
    if (is_string()) {
        return std::stol(get<std::string>());
    }
    return 0;
  }

  bool operator<(const Value & other) const {
    if (is_null()) {
      _printlog("Undefined value or reference");
      return false;
    }
    if (is_number() && other.is_number()) return get<double>() < other.get<double>();
    if (is_string() && other.is_string()) return get<std::string>() < other.get<std::string>();
    _printlog("Cannot compare values: " + dump() + " < " + other.dump());
    return false;
  }
  bool operator>=(const Value & other) const { return !(*this < other); }

  bool operator>(const Value & other) const {
    if (is_null()) {
      _printlog("Undefined value or reference");
      return false;
    }
    if (is_number() && other.is_number()) return get<double>() > other.get<double>();
    if (is_string() && other.is_string()) return get<std::string>() > other.get<std::string>();
    _printlog("Cannot compare values: " + dump() + " > " + other.dump());
    return false;
  }
  bool operator<=(const Value & other) const { return !(*this > other); }

  bool operator==(const Value & other) const {
    if (callable_ || other.callable_) {
      if (callable_.get() != other.callable_.get()) return false;
    }
    if (array_) {
      if (!other.array_) return false;
      if (array_->size() != other.array_->size()) return false;
      for (size_t i = 0; i < array_->size(); ++i) {
        if (!(*array_)[i].to_bool() || !(*other.array_)[i].to_bool() || (*array_)[i] != (*other.array_)[i]) return false;
      }
      return true;
    } else if (object_) {
      if (!other.object_) return false;
      if (object_->size() != other.object_->size()) return false;
      for (const auto& item : *object_) {
        if (!item.second.to_bool() || !other.object_->count(item.first) || item.second != other.object_->at(item.first)) return false;
      }
      return true;
    } else {
      return primitive_ == other.primitive_;
    }
  }
  bool operator!=(const Value & other) const { return !(*this == other); }

  bool contains(const char * key) const { return contains(std::string(key)); }
  bool contains(const std::string & key) const {
    if (array_) {
      return false;
    } else if (object_) {
      return object_->find(key) != object_->end();
    } else {
      _printlog("contains can only be called on arrays and objects: " + dump());
    }
    return false;
  }
  bool contains(const Value & value) const {
    if (is_null())
      _printlog("Undefined value or reference");
    if (array_) {
      for (const auto& item : *array_) {
        if (item.to_bool() && item == value) return true;
      }
      return false;
    } else if (object_) {
      if (!value.is_hashable()) _printlog("Unhashable type: " + value.dump());
      return object_->find(value.primitive_.dump()) != object_->end();
    } else {
      _printlog("contains can only be called on arrays and objects: " + dump());
    }
    return false;
  }
  void erase(size_t index) {
    if (!array_) _printlog("Value is not an array: " + dump());
    array_->erase(array_->begin() + index);
  }
  void erase(const std::string & key) {
    if (!object_) _printlog("Value is not an object: " + dump());
    object_->erase(key);
  }
  const Value& at(const Value & index) const {
    return const_cast<Value*>(this)->at(index);
  }
  Value& at(const Value & index) {
    if (!index.is_hashable()) {
        _printlog("Unhashable type: " + dump());
    }
    if (is_array()) return array_->at(index.get<int>());
    if (is_object()) return object_->at(index.primitive_.dump());
    _printlog("Value is not an array or object: " + dump());
    return object_->at(index.primitive_.dump());
  }
  const Value& at(size_t index) const {
    return const_cast<Value*>(this)->at(index);
  }
  Value& at(size_t index) {
    if (is_null()) {
      _printlog("Undefined value or reference");
    }
      if (is_array()) {
          return array_->at(index);
      }
      if (is_object()) {
          return object_->at(std::to_string(index));
      }
    _printlog("Value is not an array or object: " + dump());
    return array_->at(index);
  }

  template <typename T>
  T get(const std::string & key, T default_value) const {
    if (!contains(key)) return default_value;
    return at(key).get<T>();
  }

  template <typename T>
  T get() const {
      T d;
      return primitive_.get(d);
  }

  std::string dump(int indent=-1, bool to_json=false) const {
    std::ostringstream out;
    dump(out, indent, 0, to_json);
    return out.str();
  }

  Value operator-() const {
      if (is_number_integer())
        return -get<int64_t>();
      else
        return -get<double>();
  }
  std::string to_str() const {
    if (is_string()) return get<std::string>();
    if (is_number_integer()) return std::to_string(get<int64_t>());
    if (is_number_float()) return std::to_string(get<double>());
    if (is_boolean()) return get<bool>() ? "True" : "False";
    if (is_null()) return "None";
    return dump();
  }
  Value operator+(const Value& rhs) const {
      if (is_string() || rhs.is_string()) {
        return to_str() + rhs.to_str();
      } else if (is_number_integer() && rhs.is_number_integer()) {
        return get<int64_t>() + rhs.get<int64_t>();
      } else if (is_array() && rhs.is_array()) {
        auto res = Value::array();
        for (const auto& item : *array_) res.push_back(item);
        for (const auto& item : *rhs.array_) res.push_back(item);
        return res;
      } else {
        return get<double>() + rhs.get<double>();
      }
  }
  Value operator-(const Value& rhs) const {
      if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() - rhs.get<int64_t>();
      else
        return get<double>() - rhs.get<double>();
  }
  Value operator*(const Value& rhs) const {
      if (is_string() && rhs.is_number_integer()) {
        std::ostringstream out;
        for (int64_t i = 0, n = rhs.get<int64_t>(); i < n; ++i) {
          out << to_str();
        }
        return out.str();
      }
      else if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() * rhs.get<int64_t>();
      else
        return get<double>() * rhs.get<double>();
  }
  Value operator/(const Value& rhs) const {
      if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() / rhs.get<int64_t>();
      else
        return get<double>() / rhs.get<double>();
  }
  Value operator%(const Value& rhs) const {
    return get<int64_t>() % rhs.get<int64_t>();
  }
};

struct ArgumentsValue {
  std::vector<Value> args;
  std::vector<std::pair<std::string, Value>> kwargs;

  bool has_named(const std::string & name) {
    for (const auto & p : kwargs) {
      if (p.first == name) return true;
    }
    return false;
  }

  Value get_named(const std::string & name) {
      for (const auto & p : kwargs) {
          if (p.first == name) {
              return p.second;
          }
      }
    return Value();
  }

  bool empty() {
    return args.empty() && kwargs.empty();
  }

  void expectArgs(const std::string & method_name, const std::pair<size_t, size_t> & pos_count, const std::pair<size_t, size_t> & kw_count) {
    if (args.size() < pos_count.first || args.size() > pos_count.second || kwargs.size() < kw_count.first || kwargs.size() > kw_count.second) {
      std::ostringstream out;
      out << method_name << " must have between " << pos_count.first << " and " << pos_count.second << " positional arguments and between " << kw_count.first << " and " << kw_count.second << " keyword arguments";
      _printlog(out.str());
    }
  }
};

} // namespace minja

namespace std {
  template <>
  struct hash<minja::Value> {
    size_t operator()(const minja::Value & v) const {
      if (!v.is_hashable())
        _printlog("Unsupported type for hashing: " + v.dump());
      return std::hash<std::string>()(v.dump());
    }
  };
} // namespace std

namespace minja {

static std::string error_location_suffix(const std::string & source, size_t pos) {
  auto get_line = [&](size_t line) {
    auto start = source.begin();
    for (size_t i = 1; i < line; ++i) {
      start = std::find(start, source.end(), '\n') + 1;
    }
    auto end = std::find(start, source.end(), '\n');
    return std::string(start, end);
  };
  auto start = source.begin();
  auto end = source.end();
  auto it = start + pos;
  auto line = std::count(start, it, '\n') + 1;
  auto max_line = std::count(start, end, '\n') + 1;
  auto col = pos - std::string(start, it).rfind('\n');
  std::ostringstream out;
  out << " at row " << line << ", column " << col << ":\n";
  if (line > 1) out << get_line(line - 1) << "\n";
  out << get_line(line) << "\n";
  out << std::string(col - 1, ' ') << "^\n";
  if (line < max_line) out << get_line(line + 1) << "\n";

  return out.str();
}

class Context : public std::enable_shared_from_this<Context> {
  protected:
    Value values_;
    std::shared_ptr<Context> parent_;
  public:
    Context(Value && values, const std::shared_ptr<Context> & parent = nullptr) : values_(std::move(values)), parent_(parent) {
        if (!values_.is_object()) _printlog("Context values must be an object: " + values_.dump());
    }
    virtual ~Context() {}

    static std::shared_ptr<Context> builtins();
    static std::shared_ptr<Context> make(Value && values, const std::shared_ptr<Context> & parent = builtins());

    std::vector<Value> keys() {
        return values_.keys();
    }
    virtual Value get(const Value & key) {
        if (values_.contains(key)) return values_.at(key);
        if (parent_) return parent_->get(key);
        return Value();
    }
    virtual Value & at(const Value & key) {
        if (values_.contains(key)) return values_.at(key);
        if (parent_) return parent_->at(key);
        _printlog("Undefined variable: " + key.dump());
        return values_.at(key);
    }
    virtual bool contains(const Value & key) {
        if (values_.contains(key)) return true;
        if (parent_) return parent_->contains(key);
        return false;
    }
    virtual void set(const std::string & key, const Value & value) {
        values_.set(key, value);
    }
};

struct Location {
    std::shared_ptr<std::string> source;
    size_t pos;
};

class Expression {
protected:
    virtual Value do_evaluate(const std::shared_ptr<Context> & context) const = 0;
public:
    enum Type {
        Type_Variable = 0,
        Type_If,
        Type_Liter,
        Type_Array,
        Type_Dict,
        Type_Slice,
        Type_Subscript,
        Type_Unary,
        Type_Binary,
        Type_MethodCall,
        Type_Call,
        Type_Filter,
    };
    using Parameters = std::vector<std::pair<std::string, std::shared_ptr<Expression>>>;

    Location location;
    const int mType;

    Expression(const Location & location, int type) : location(location), mType(type) {}
    virtual ~Expression() = default;

    Value evaluate(const std::shared_ptr<Context> & context) const {
            return do_evaluate(context);
    }
};

class VariableExpr : public Expression {
    std::string name;
public:
    VariableExpr(const Location & loc, const std::string& n)
      : Expression(loc, Expression::Type_Variable), name(n) {}
    std::string get_name() const { return name; }
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!context->contains(name)) {
            return Value();
        }
        return context->at(name);
    }
};

static void destructuring_assign(const std::vector<std::string> & var_names, const std::shared_ptr<Context> & context, Value& item) {
  if (var_names.size() == 1) {
      context->set(var_names[0], item);
  } else {
      if (!item.is_array() || item.size() != var_names.size()) {
          _printlog("Mismatched number of variables and items in destructuring assignment");
      }
      for (size_t i = 0; i < var_names.size(); ++i) {
          context->set(var_names[i], item.at(i));
      }
  }
}
enum class LoopControlType { Normal, Break, Continue};

class TemplateNode {
    Location location_;
protected:
    virtual LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const = 0;

public:
    TemplateNode(const Location & location) : location_(location) {}
    LoopControlType render(std::ostringstream & out, const std::shared_ptr<Context> & context) const {
        return do_render(out, context);
    }
    const Location & location() const { return location_; }
    virtual ~TemplateNode() = default;
    std::string render(const std::shared_ptr<Context> & context) const {
        std::ostringstream out;
        render(out, context);
        return out.str();
    }
};

class SequenceNode : public TemplateNode {
    std::vector<std::shared_ptr<TemplateNode>> children;
public:
    SequenceNode(const Location & loc, std::vector<std::shared_ptr<TemplateNode>> && c)
      : TemplateNode(loc), children(std::move(c)) {}
    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
        for (const auto& child : children) {
            auto type = child->render(out, context);
            if (LoopControlType::Normal != type) {
                return type;
            }
        }
        return LoopControlType::Normal;
    }
};

class TextNode : public TemplateNode {
    std::string text;
public:
    TextNode(const Location & loc, const std::string& t) : TemplateNode(loc), text(t) {}
    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> &) const override {
        out << text;
        return LoopControlType::Normal;
    }
};

class ExpressionNode : public TemplateNode {
    std::shared_ptr<Expression> expr;
public:
    ExpressionNode(const Location & loc, std::shared_ptr<Expression> && e) : TemplateNode(loc), expr(std::move(e)) {}
    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      if (!expr) _printlog("ExpressionNode.expr is null");
      auto result = expr->evaluate(context);
      if (result.is_string()) {
          out << result.get<std::string>();
      } else if (result.is_boolean()) {
          out << (result.get<bool>() ? "True" : "False");
      } else if (!result.is_null()) {
          out << result.dump();
      }
        return LoopControlType::Normal;
  }
};

class IfNode : public TemplateNode {
    std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> cascade;
public:
    IfNode(const Location & loc, std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> && c)
        : TemplateNode(loc), cascade(std::move(c)) {}
    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      for (const auto& branch : cascade) {
          auto enter_branch = true;
          if (branch.first) {
            enter_branch = branch.first->evaluate(context).to_bool();
          }
          if (enter_branch) {
            if (!branch.second) _printlog("IfNode.cascade.second is null");
              return branch.second->render(out, context);
          }
      }
        return LoopControlType::Normal;
    }
};

class LoopControlNode : public TemplateNode {
    LoopControlType control_type_;
  public:
    LoopControlNode(const Location & loc, LoopControlType control_type) : TemplateNode(loc), control_type_(control_type) {}
    LoopControlType do_render(std::ostringstream &, const std::shared_ptr<Context> &) const override {
        return control_type_;
    }
};

class ForNode : public TemplateNode {
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> iterable;
    std::shared_ptr<Expression> condition;
    std::shared_ptr<TemplateNode> body;
    bool recursive;
    std::shared_ptr<TemplateNode> else_body;
public:
    ForNode(const Location & loc, std::vector<std::string> && var_names, std::shared_ptr<Expression> && iterable,
      std::shared_ptr<Expression> && condition, std::shared_ptr<TemplateNode> && body, bool recursive, std::shared_ptr<TemplateNode> && else_body)
            : TemplateNode(loc), var_names(var_names), iterable(std::move(iterable)), condition(std::move(condition)), body(std::move(body)), recursive(recursive), else_body(std::move(else_body)) {}

    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      // https://jinja.palletsprojects.com/en/3.0.x/templates/#for
      if (!iterable) _printlog("ForNode.iterable is null");
      if (!body) _printlog("ForNode.body is null");

      auto iterable_value = iterable->evaluate(context);
      Value::CallableType loop_function;

      std::function<LoopControlType(Value&)> visit = [&](Value& iter) {
          auto filtered_items = Value::array();
          if (!iter.is_null()) {
            if (!iterable_value.is_iterable()) {
              _printlog("For loop iterable must be iterable: " + iterable_value.dump());
            }
            iterable_value.for_each([&](Value & item) {
                destructuring_assign(var_names, context, item);
                if (!condition || condition->evaluate(context).to_bool()) {
                  filtered_items.push_back(item);
                }
            });
          }
          if (filtered_items.empty()) {
            if (else_body) {
              auto loopcode = else_body->render(out, context);
                if (loopcode != LoopControlType::Normal) {
                    return loopcode;
                }
            }
          } else {
              auto loop = recursive ? Value::callable(loop_function) : Value::object();
              loop.set("length", (int64_t) filtered_items.size());

              size_t cycle_index = 0;
              loop.set("cycle", Value::callable([&](const std::shared_ptr<Context> &, ArgumentsValue & args) {
                  if (args.args.empty() || !args.kwargs.empty()) {
                      _printlog("cycle() expects at least 1 positional argument and no named arg");
                  }
                  auto item = args.args[cycle_index];
                  cycle_index = (cycle_index + 1) % args.args.size();
                  return item;
              }));
              auto loop_context = Context::make(Value::object(), context);
              loop_context->set("loop", loop);
              for (size_t i = 0, n = filtered_items.size(); i < n; ++i) {
                  auto & item = filtered_items.at(i);
                  destructuring_assign(var_names, loop_context, item);
                  loop.set("index", (int64_t) i + 1);
                  loop.set("index0", (int64_t) i);
                  loop.set("revindex", (int64_t) (n - i));
                  loop.set("revindex0", (int64_t) (n - i - 1));
                  loop.set("length", (int64_t) n);
                  loop.set("first", i == 0);
                  loop.set("last", i == (n - 1));
                  loop.set("previtem", i > 0 ? filtered_items.at(i - 1) : Value());
                  loop.set("nextitem", i < n - 1 ? filtered_items.at(i + 1) : Value());
                  auto control_type = body->render(out, loop_context);
                  if (control_type == LoopControlType::Break) break;
                  if (control_type == LoopControlType::Continue) continue;
              }
          }
          return LoopControlType::Normal;
      };

      if (recursive) {
        loop_function = [&](const std::shared_ptr<Context> &, ArgumentsValue & args) {
            if (args.args.size() != 1 || !args.kwargs.empty() || !args.args[0].is_array()) {
                _printlog("loop() expects exactly 1 positional iterable argument");
            }
            auto & items = args.args[0];
            auto code = visit(items);
            return Value();
        };
      }

      return visit(iterable_value);
  }
};

class MacroNode : public TemplateNode {
    std::shared_ptr<VariableExpr> name;
    Expression::Parameters params;
    std::shared_ptr<TemplateNode> body;
    std::unordered_map<std::string, size_t> named_param_positions;
public:
    MacroNode(const Location & loc, std::shared_ptr<VariableExpr> && n, Expression::Parameters && p, std::shared_ptr<TemplateNode> && b)
        : TemplateNode(loc), name(std::move(n)), params(std::move(p)), body(std::move(b)) {
        for (size_t i = 0; i < params.size(); ++i) {
          const auto & name = params[i].first;
          if (!name.empty()) {
            named_param_positions[name] = i;
          }
        }
    }
    LoopControlType do_render(std::ostringstream &, const std::shared_ptr<Context> & macro_context) const override {
        if (!name) _printlog("MacroNode.name is null");
        if (!body) _printlog("MacroNode.body is null");
        auto callable = Value::callable([&](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
            auto call_context = macro_context;
            std::vector<bool> param_set(params.size(), false);
            for (size_t i = 0, n = args.args.size(); i < n; i++) {
                auto & arg = args.args[i];
                if (i >= params.size()) _printlog("Too many positional arguments for macro " + name->get_name());
                param_set[i] = true;
                auto & param_name = params[i].first;
                call_context->set(param_name, arg);
            }
            for (auto& iter : args.kwargs) {
                auto& arg_name = iter.first;
                auto& value = iter.second;
                auto it = named_param_positions.find(arg_name);
                if (it == named_param_positions.end()) _printlog("Unknown parameter name for macro " + name->get_name() + ": " + arg_name);

                call_context->set(arg_name, value);
                param_set[it->second] = true;
            }
            // Set default values for parameters that were not passed
            for (size_t i = 0, n = params.size(); i < n; i++) {
                if (!param_set[i] && params[i].second != nullptr) {
                    auto val = params[i].second->evaluate(context);
                    call_context->set(params[i].first, val);
                }
            }
            return body->render(call_context);
        });
        macro_context->set(name->get_name(), callable);
        return LoopControlType::Normal;
    }
};

class FilterNode : public TemplateNode {
    std::shared_ptr<Expression> filter;
    std::shared_ptr<TemplateNode> body;

public:
    FilterNode(const Location & loc, std::shared_ptr<Expression> && f, std::shared_ptr<TemplateNode> && b)
        : TemplateNode(loc), filter(std::move(f)), body(std::move(b)) {}

    LoopControlType do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
        if (!filter) _printlog("FilterNode.filter is null");
        if (!body) _printlog("FilterNode.body is null");
        auto filter_value = filter->evaluate(context);
        if (!filter_value.is_callable()) {
            _printlog("Filter must be a callable: " + filter_value.dump());
        }
        std::string rendered_body = body->render(context);

        ArgumentsValue filter_args = {{Value(rendered_body)}, {}};
        auto result = filter_value.call(context, filter_args);
        out << result.to_str();
        return LoopControlType::Normal;
    }
};

class SetNode : public TemplateNode {
    std::string ns;
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> value;
public:
    SetNode(const Location & loc, const std::string & ns, const std::vector<std::string> & vns, std::shared_ptr<Expression> && v)
        : TemplateNode(loc), ns(ns), var_names(vns), value(std::move(v)) {}
    LoopControlType do_render(std::ostringstream &, const std::shared_ptr<Context> & context) const override {
      if (!value) _printlog("SetNode.value is null");
      if (!ns.empty()) {
        if (var_names.size() != 1) {
          _printlog("Namespaced set only supports a single variable name");
        }
        auto & name = var_names[0];
        auto ns_value = context->get(ns);
        if (!ns_value.is_object()) _printlog("Namespace '" + ns + "' is not an object");
        ns_value.set(name, this->value->evaluate(context));
      } else {
        auto val = value->evaluate(context);
        destructuring_assign(var_names, context, val);
      }
        return LoopControlType::Normal;

    }
};

class SetTemplateNode : public TemplateNode {
    std::string name;
    std::shared_ptr<TemplateNode> template_value;
public:
    SetTemplateNode(const Location & loc, const std::string & name, std::shared_ptr<TemplateNode> && tv)
        : TemplateNode(loc), name(name), template_value(std::move(tv)) {}
    LoopControlType do_render(std::ostringstream &, const std::shared_ptr<Context> & context) const override {
      if (!template_value) _printlog("SetTemplateNode.template_value is null");
      Value value { template_value->render(context) };
      context->set(name, value);
        return LoopControlType::Normal;

    }
};

class IfExpr : public Expression {
    std::shared_ptr<Expression> condition;
    std::shared_ptr<Expression> then_expr;
    std::shared_ptr<Expression> else_expr;
public:
    IfExpr(const Location & loc, std::shared_ptr<Expression> && c, std::shared_ptr<Expression> && t, std::shared_ptr<Expression> && e)
        : Expression(loc, Expression::Type_If), condition(std::move(c)), then_expr(std::move(t)), else_expr(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
      if (!condition) _printlog("IfExpr.condition is null");
      if (!then_expr) _printlog("IfExpr.then_expr is null");
      if (condition->evaluate(context).to_bool()) {
        return then_expr->evaluate(context);
      }
      if (else_expr) {
        return else_expr->evaluate(context);
      }
      return nullptr;
    }
};

class LiteralExpr : public Expression {
    Value value;
public:
    LiteralExpr(const Location & loc, const Value& v)
      : Expression(loc, Expression::Type_Liter), value(v) {}
    Value do_evaluate(const std::shared_ptr<Context> &) const override { return value; }
};

class ArrayExpr : public Expression {
    std::vector<std::shared_ptr<Expression>> elements;
public:
    ArrayExpr(const Location & loc, std::vector<std::shared_ptr<Expression>> && e)
      : Expression(loc, Expression::Type_Array), elements(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        auto result = Value::array();
        for (const auto& e : elements) {
            if (!e) _printlog("Array element is null");
            result.push_back(e->evaluate(context));
        }
        return result;
    }
};

class DictExpr : public Expression {
    std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> elements;
public:
    DictExpr(const Location & loc, std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> && e)
      : Expression(loc, Expression::Type_Dict), elements(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        auto result = Value::object();
        for (const auto& iter : elements) {
            const auto& key = iter.first;
            const auto& value = iter.second;
            if (!key) _printlog("Dict key is null");
            if (!value) _printlog("Dict value is null");
            result.set(key->evaluate(context).to_str(), value->evaluate(context));
        }
        return result;
    }
};

class SliceExpr : public Expression {
public:
    std::shared_ptr<Expression> start, end, step;
    SliceExpr(const Location & loc, std::shared_ptr<Expression> && s, std::shared_ptr<Expression> && e, std::shared_ptr<Expression> && st = nullptr)
      : Expression(loc, Expression::Type_Slice), start(std::move(s)), end(std::move(e)), step(std::move(st)) {}

    Value do_evaluate(const std::shared_ptr<Context> &) const override {
        _printlog("SliceExpr not implemented");
        return Value();
    }
};

class SubscriptExpr : public Expression {
    std::shared_ptr<Expression> base;
    std::shared_ptr<Expression> index;
public:
    SubscriptExpr(const Location & loc, std::shared_ptr<Expression> && b, std::shared_ptr<Expression> && i)
        : Expression(loc, Expression::Type_Subscript), base(std::move(b)), index(std::move(i)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        auto target_value = base->evaluate(context);
        if (index->mType == Expression::Type_Slice){
            auto slice = (SliceExpr*)(index.get());
            bool reverse = slice->step && slice->step->evaluate(context).get<int64_t>() == -1;
            if (slice->step && !reverse) {
              MNN_ERROR("Slicing with step other than -1 is not supported");
            }

            int64_t start = slice->start ? slice->start->evaluate(context).get<int64_t>() : (reverse ? target_value.size() - 1 : 0);
            int64_t end = slice->end ? slice->end->evaluate(context).get<int64_t>() : (reverse ? -1 : target_value.size());

            size_t len = target_value.size();

            if (slice->start && start < 0) {
              start = (int64_t)len + start;
            }
            if (slice->end && end < 0) {
              end = (int64_t)len + end;
            }
            if (target_value.is_string()) {
              std::string s = target_value.get<std::string>();

              std::string result_str;
              if (reverse) {
                for (int64_t i = start; i > end; --i) {
                  if (i >= 0 && i < (int64_t)len) {
                    result_str += s[i];
                  } else if (i < 0) {
                    break;
                  }
                }
              } else {
                result_str = s.substr(start, end - start);
              }
              return result_str;

            } else if (target_value.is_array()) {
              auto result = Value::array();
              if (reverse) {
                for (int64_t i = start; i > end; --i) {
                  if (i >= 0 && i < (int64_t)len) {
                    result.push_back(target_value.at(i));
                  } else if (i < 0) {
                    break;
                  }
                }
              } else {
                for (auto i = start; i < end; ++i) {
                  result.push_back(target_value.at(i));
                }
              }
              return result;
            } else {
                if(target_value.is_null()) {
                    MNN_ERROR("Cannot subscript null\n");
                } else {
                    MNN_ERROR("Subscripting only supported on arrays and strings\n");
                }
            }
        } else {
          auto index_value = index->evaluate(context);
          if (target_value.is_null()) {
            if (base->mType == Expression::Type_Variable) {
                auto t = (VariableExpr*)(base.get());
              _printlog("'" + t->get_name() + "' is " + (context->contains(t->get_name()) ? "null" : "not defined"));
            }
            _printlog("Trying to access property '" +  index_value.dump() + "' on null!");
          }
          return target_value.get(index_value);
        }
        return Value();
    }
};

class UnaryOpExpr : public Expression {
public:
    enum class Op { Plus, Minus, LogicalNot, Expansion, ExpansionDict };
    std::shared_ptr<Expression> expr;
    Op op;
    UnaryOpExpr(const Location & loc, std::shared_ptr<Expression> && e, Op o)
      : Expression(loc, Expression::Type_Unary), expr(std::move(e)), op(o) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!expr) _printlog("UnaryOpExpr.expr is null");
        auto e = expr->evaluate(context);
        switch (op) {
            case Op::Plus: return e;
            case Op::Minus: return -e;
            case Op::LogicalNot: return !e.to_bool();
            case Op::Expansion:
            case Op::ExpansionDict:
                _printlog("Expansion operator is only supported in function calls and collections");

        }
        _printlog("Unknown unary operator");
        return Value();
    }
};

class BinaryOpExpr : public Expression {
public:
    enum class Op { StrConcat, Add, Sub, Mul, MulMul, Div, DivDiv, Mod, Eq, Ne, Lt, Gt, Le, Ge, And, Or, In, NotIn, Is, IsNot };
private:
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    Op op;
public:
    BinaryOpExpr(const Location & loc, std::shared_ptr<Expression> && l, std::shared_ptr<Expression> && r, Op o)
        : Expression(loc, Expression::Type_Binary), left(std::move(l)), right(std::move(r)), op(o) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!left) _printlog("BinaryOpExpr.left is null");
        if (!right) _printlog("BinaryOpExpr.right is null");
        auto l = left->evaluate(context);

        auto do_eval = [&](const Value & l) -> Value {
          if (op == Op::Is || op == Op::IsNot) {
            auto t = (VariableExpr*)(right.get());
              if (right->mType != Expression::Type_Variable) {
                  _printlog("Right side of 'is' operator must be a variable");
              }

            auto eval = [&]() {
              const auto & name = t->get_name();
              if (name == "none") return l.is_null();
              if (name == "boolean") return l.is_boolean();
              if (name == "integer") return l.is_number_integer();
              if (name == "float") return l.is_number_float();
              if (name == "number") return l.is_number();
              if (name == "string") return l.is_string();
              if (name == "mapping") return l.is_object();
              if (name == "iterable") return l.is_iterable();
              if (name == "sequence") return l.is_array();
              if (name == "defined") return !l.is_null();
              if (name == "false") return !l.get<bool>();
              if (name == "true") return l.get<bool>();
              _printlog("Unknown type for 'is' operator: " + name);
              return false;
            };
            auto value = eval();
            return Value(op == Op::Is ? value : !value);
          }

          if (op == Op::And) {
            if (!l.to_bool()) return Value(false);
            return right->evaluate(context).to_bool();
          } else if (op == Op::Or) {
            if (l.to_bool()) return l;
            return right->evaluate(context);
          }

          auto r = right->evaluate(context);
          switch (op) {
              case Op::StrConcat: return l.to_str() + r.to_str();
              case Op::Add:       return l + r;
              case Op::Sub:       return l - r;
              case Op::Mul:       return l * r;
              case Op::Div:       return l / r;
              case Op::MulMul:    return std::pow(l.get<double>(), r.get<double>());
              case Op::DivDiv:    return l.get<int64_t>() / r.get<int64_t>();
              case Op::Mod:       return l.get<int64_t>() % r.get<int64_t>();
              case Op::Eq:        return l == r;
              case Op::Ne:        return l != r;
              case Op::Lt:        return l < r;
              case Op::Gt:        return l > r;
              case Op::Le:        return l <= r;
              case Op::Ge:        return l >= r;
              case Op::In:        return (r.is_array() || r.is_object()) && r.contains(l);
              case Op::NotIn:     return !(r.is_array() && r.contains(l));
              default:            break;
          }
          _printlog("Unknown binary operator");
          return false;
        };

        if (l.is_callable()) {
          return Value::callable([l, do_eval](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
            auto ll = l.call(context, args);
            return do_eval(ll); //args[0].second);
          });
        } else {
          return do_eval(l);
        }
    }
};

struct ArgumentsExpression {
    std::vector<std::shared_ptr<Expression>> args;
    std::vector<std::pair<std::string, std::shared_ptr<Expression>>> kwargs;

    ArgumentsValue evaluate(const std::shared_ptr<Context> & context) const {
        ArgumentsValue vargs;
        for (const auto& arg : this->args) {
            if (arg->mType == Expression::Type_Unary) {
                auto un_expr = (UnaryOpExpr*)(arg.get());
                if (un_expr->op == UnaryOpExpr::Op::Expansion) {
                    auto array = un_expr->expr->evaluate(context);
                    if (!array.is_array()) {
                        _printlog("Expansion operator only supported on arrays");
                    }
                    array.for_each([&](Value & value) {
                        vargs.args.push_back(value);
                    });
                    continue;
                } else if (un_expr->op == UnaryOpExpr::Op::ExpansionDict) {
                    auto dict = un_expr->expr->evaluate(context);
                    if (!dict.is_object()) {
                        _printlog("ExpansionDict operator only supported on objects");
                    }
                    dict.for_each([&](const Value & key) {
                        vargs.kwargs.push_back({key.get<std::string>(), dict.at(key)});
                    });
                    continue;
                }
            }
            vargs.args.push_back(arg->evaluate(context));
        }
        for (const auto& iter : this->kwargs) {
            const auto& name = iter.first;
            const auto& value = iter.second;
            vargs.kwargs.push_back({name, value->evaluate(context)});
        }
        return vargs;
    }
};

static std::string strip(const std::string & s, const std::string & chars = "", bool left = true, bool right = true) {
  auto charset = chars.empty() ? " \t\n\r" : chars;
  auto start = left ? s.find_first_not_of(charset) : 0;
  if (start == std::string::npos) return "";
  auto end = right ? s.find_last_not_of(charset) : s.size() - 1;
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> split(const std::string & s, const std::string & sep) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = s.find(sep);
  while (end != std::string::npos) {
    result.push_back(s.substr(start, end - start));
    start = end + sep.length();
    end = s.find(sep, start);
  }
  result.push_back(s.substr(start));
  return result;
}

static std::string capitalize(const std::string & s) {
  if (s.empty()) return s;
  auto result = s;
  result[0] = std::toupper(result[0]);
  return result;
}

static std::string html_escape(const std::string & s) {
  std::string result;
  result.reserve(s.size());
  for (const auto & c : s) {
    switch (c) {
      case '&': result += "&amp;"; break;
      case '<': result += "&lt;"; break;
      case '>': result += "&gt;"; break;
      case '"': result += "&#34;"; break;
      case '\'': result += "&apos;"; break;
      default: result += c; break;
    }
  }
  return result;
}

class MethodCallExpr : public Expression {
    std::shared_ptr<Expression> object;
    std::shared_ptr<VariableExpr> method;
    ArgumentsExpression args;
public:
    MethodCallExpr(const Location & loc, std::shared_ptr<Expression> && obj, std::shared_ptr<VariableExpr> && m, ArgumentsExpression && a)
        : Expression(loc, Expression::Type_MethodCall), object(std::move(obj)), method(std::move(m)), args(std::move(a)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!object) _printlog("MethodCallExpr.object is null");
        if (!method) _printlog("MethodCallExpr.method is null");
        auto obj = object->evaluate(context);
        auto vargs = args.evaluate(context);
        if (obj.is_null()) {
           // _printlog("Trying to call method '" + method->get_name() + "' on null");
            return Value();
        }
        if (obj.is_array()) {
          if (method->get_name() == "append") {
              vargs.expectArgs("append method", {1, 1}, {0, 0});
              obj.push_back(vargs.args[0]);
              return Value();
          } else if (method->get_name() == "pop") {
              vargs.expectArgs("pop method", {0, 1}, {0, 0});
              return obj.pop(vargs.args.empty() ? Value() : vargs.args[0]);
          } else if (method->get_name() == "insert") {
              vargs.expectArgs("insert method", {2, 2}, {0, 0});
              auto index = vargs.args[0].get<int64_t>();
              if (index < 0 || index > (int64_t) obj.size()) _printlog("Index out of range for insert method");
              obj.insert(index, vargs.args[1]);
              return Value();
          }
        } else if (obj.is_object()) {
          if (method->get_name() == "items") {
            vargs.expectArgs("items method", {0, 0}, {0, 0});
            auto result = Value::array();
            for (const auto& key : obj.keys()) {
              result.push_back(Value::array({key, obj.at(key)}));
            }
            return result;
          } else if (method->get_name() == "pop") {
            vargs.expectArgs("pop method", {1, 1}, {0, 0});
            return obj.pop(vargs.args[0]);
          } else if (method->get_name() == "get") {
            vargs.expectArgs("get method", {1, 2}, {0, 0});
            auto key = vargs.args[0];
            if (vargs.args.size() == 1) {
              return obj.contains(key) ? obj.at(key) : Value();
            } else {
              return obj.contains(key) ? obj.at(key) : vargs.args[1];
            }
          } else if (obj.contains(method->get_name())) {
            auto callable = obj.at(method->get_name());
            if (!callable.is_callable()) {
              _printlog("Property '" + method->get_name() + "' is not callable");
            }
            return callable.call(context, vargs);
          }
        } else if (obj.is_string()) {
          auto str = obj.get<std::string>();
          if (method->get_name() == "strip") {
            vargs.expectArgs("strip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars));
          } else if (method->get_name() == "lstrip") {
            vargs.expectArgs("lstrip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars, /* left= */ true, /* right= */ false));
          } else if (method->get_name() == "rstrip") {
            vargs.expectArgs("rstrip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars, /* left= */ false, /* right= */ true));
          } else if (method->get_name() == "split") {
            vargs.expectArgs("split method", {1, 1}, {0, 0});
            auto sep = vargs.args[0].get<std::string>();
            auto parts = split(str, sep);
            Value result = Value::array();
            for (const auto& part : parts) {
              result.push_back(Value(part));
            }
            return result;
          } else if (method->get_name() == "capitalize") {
            vargs.expectArgs("capitalize method", {0, 0}, {0, 0});
            return Value(capitalize(str));
          } else if (method->get_name() == "endswith") {
            vargs.expectArgs("endswith method", {1, 1}, {0, 0});
            auto suffix = vargs.args[0].get<std::string>();
            return suffix.length() <= str.length() && std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
          } else if (method->get_name() == "startswith") {
            vargs.expectArgs("startswith method", {1, 1}, {0, 0});
            auto prefix = vargs.args[0].get<std::string>();
            return prefix.length() <= str.length() && std::equal(prefix.begin(), prefix.end(), str.begin());
          } else if (method->get_name() == "title") {
            vargs.expectArgs("title method", {0, 0}, {0, 0});
            auto res = str;
            for (size_t i = 0, n = res.size(); i < n; ++i) {
              if (i == 0 || std::isspace(res[i - 1])) res[i] = std::toupper(res[i]);
              else res[i] = std::tolower(res[i]);
            }
            return res;
          }
        }
        // _printlog("Unknown method: " + method->get_name());
        return Value();
    }
};

class CallExpr : public Expression {
public:
    std::shared_ptr<Expression> object;
    ArgumentsExpression args;
    CallExpr(const Location & loc, std::shared_ptr<Expression> && obj, ArgumentsExpression && a)
        : Expression(loc, Expression::Type_Call), object(std::move(obj)), args(std::move(a)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!object) {
            _printlog("CallExpr.object is null");
            return Value();
        }
        auto obj = object->evaluate(context);
        if (!obj.is_callable()) {
            //_printlog("Object is not callable: " + obj.dump(2));
            return Value();
        }
        auto vargs = args.evaluate(context);
        return obj.call(context, vargs);
    }
};

class FilterExpr : public Expression {
    std::vector<std::shared_ptr<Expression>> parts;
public:
    FilterExpr(const Location & loc, std::vector<std::shared_ptr<Expression>> && p)
      : Expression(loc, Expression::Type_Filter), parts(std::move(p)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        Value result;
        bool first = true;
        for (const auto& part : parts) {
          if (!part) _printlog("FilterExpr.part is null");
          if (first) {
            first = false;
            result = part->evaluate(context);
          } else {
              if (part->mType == Expression::Type_Call) {
                  auto ce = (CallExpr*)(part.get());
              auto target = ce->object->evaluate(context);
              ArgumentsValue args = ce->args.evaluate(context);
              args.args.insert(args.args.begin(), result);
              result = target.call(context, args);
            } else {
              auto callable = part->evaluate(context);
              ArgumentsValue args;
              args.args.insert(args.args.begin(), result);
              result = callable.call(context, args);
            }
          }
        }
        return result;
    }

    void prepend(std::shared_ptr<Expression> && e) {
        parts.insert(parts.begin(), std::move(e));
    }
};

static Value simple_function(const std::string & fn_name, const std::vector<std::string> & params, const std::function<Value(const std::shared_ptr<Context> &, Value & args)> & fn) {
  std::map<std::string, size_t> named_positions;
  for (size_t i = 0, n = params.size(); i < n; i++) named_positions[params[i]] = i;

  return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) -> Value {
    auto args_obj = Value::object();
    std::vector<bool> provided_args(params.size());
    for (size_t i = 0, n = args.args.size(); i < n; i++) {
      auto & arg = args.args[i];
      if (i < params.size()) {
        args_obj.set(params[i], arg);
        provided_args[i] = true;
      } else {
        _printlog("Too many positional params for " + fn_name);
      }
    }
      for (auto & iter : args.kwargs) {
          auto& name = iter.first;
          auto& value = iter.second;
      auto named_pos_it = named_positions.find(name);
      if (named_pos_it == named_positions.end()) {
        _printlog("Unknown argument " + name + " for function " + fn_name);
      }
      provided_args[named_pos_it->second] = true;
      args_obj.set(name, value);
    }
    return fn(context, args_obj);
  });
}

inline std::shared_ptr<Context> Context::make(Value && values, const std::shared_ptr<Context> & parent) {
    AUTOTIME;
  return std::make_shared<Context>(values.is_null() ? Value::object() : std::move(values), parent);
}
std::shared_ptr<TemplateNode> parse(const std::string& template_str, const Options & options);
}  // namespace minja
