// SPDX-License-Identifier: MIT
// Copyright 2023 - Present, Syoyo Fujita.
// Inspired from:
// https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

#ifdef __ANDROID__
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
#include <android/asset_manager.h>
#endif

#ifdef SAFETENSORS_CPP_IMPLEMENTATION
AAssetManager *asset_manager = nullptr;
#else
extern AAssetManager *asset_manager;
#endif
#endif


namespace safetensors {

constexpr size_t kMaxDim =
    8;  // must be equal to SAFETENSORS_C_MAX_DIM in `safetensors-c.h`

enum dtype {
  kBOOL,
  kUINT8,
  kINT8,
  kINT16,
  kUINT16,
  kFLOAT16,
  kBFLOAT16,
  kINT32,
  kUINT32,
  kFLOAT32,
  kFLOAT64,
  kINT64,
  kUINT64,
};

namespace minijson {

// Simple C++ implementation of Python's OrderedDict like dictonary
// (preserves key insertion order)
// Modified for JSON:
// - No duplicated key allowed

template <typename T>
class ordered_dict {
 public:
  bool at(const size_t idx, T *dst) const {
    if (idx >= _keys.size()) {
      return false;
    }

    if (!_m.count(_keys[idx])) {
      // This should not happen though.
      return false;
    }

    (*dst) = _m.at(_keys[idx]);

    return true;
  }

  bool count(const std::string &key) const { return _m.count(key); }

  void insert(const std::string &key, const T &value) {
    if (_m.count(key)) {
      // overwrite existing value
    } else {
      _keys.push_back(key);
    }

    _m[key] = value;
  }

  void insert(const std::string &key, T &&value) {
    if (_m.count(key)) {
      // overwrite existing value
    } else {
      _keys.push_back(key);
    }

    _m[key] = std::move(value);
  }

  bool at(const std::string &key, T *dst) const {
    if (!_m.count(key)) {
      // This should not happen though.
      return false;
    }

    (*dst) = _m.at(key);

    return true;
  }

  const std::vector<std::string> &keys() const { return _keys; }

  size_t size() const { return _m.size(); }

  bool erase(const std::string &key) {
    // simple linear search
    for (size_t i = 0; i < _keys.size(); i++) {
      if (_keys[i] == key) {
        _keys.erase(_keys.begin() + i);
        _m.erase(key);
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<std::string> _keys;
  std::map<std::string, T> _m;
};

} // namespace minijson

template <typename T>
using ordered_dict = minijson::ordered_dict<T>;

struct tensor_t {
  safetensors::dtype dtype;
  std::vector<size_t> shape;
  std::array<size_t, 2> data_offsets;
};

struct safetensors_t {
  // we need ordered dict(preserves the order of key insertion)
  // as done in Python's OrderedDict, since JSON data may not be sorted by its key string.
  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;
  std::vector<uint8_t> storage;  // empty when mmap'ed
  size_t header_size{0};         // JSON size

  bool mmaped{false};

  //
  // Following members are set when mmaped.
  //
  const uint8_t *mmap_addr{nullptr};
  size_t mmap_size{0};
  const uint8_t *databuffer_addr{nullptr};  // [mmap_addr + header_size + 8]
  size_t databuffer_size{0};                // mmap_size - header_size - 8
  // opaque pointer to safetensors_file and safetensors_mmap
  void *st_file{nullptr};
  void *st_mmap{nullptr};

  ~safetensors_t();
};

//
// Load safetensors from file.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool load_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err);

//
// Load safetensors data from memory.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] addr Memory address of safetensors data.
// @param[in] nbytes The size in bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
//
bool load_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err);

//
// Load safetensors with memory mapping(i.e. zero-copy).
// databuffer is not copied to `safetensors_t` object, thus the app must hold
// file during `safetensor_t` object is live.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err);

//
// Load safetensors from mmaped region.
// databuffer is not copied to `safetensors_t` object, thus the app must not
// free/unmap `addr` during `safetensor_t` object is live.
//
// @param[in] addr mmaped memory address of safetensors data.
// @param[in] nbytes mmap bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_memory(const uint8_t *arr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err);

//
// Save safetensors to file.
//
// @param[in] st safetensors data.
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_file(const safetensors_t &st, const std::string &filename,
                  std::string *warn, std::string *err);

//
// Save safetensors to memory.
//
// @param[in] st safetensors data.
// @param[out] data_out Serialized safetensor data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_memory(const std::string &filename, std::vector<uint8_t> *data_out,
                    std::string *warn, std::string *err);

//
// Utility functions
//

// Returns shape[0] * shape[1] * ...
// Empty Tensor(any shape[i] is 0) returns 0.
// Zero-rank tensor([]) return 1.
size_t get_shape_size(const tensor_t &t);

// Returns dtype size in bytes.
size_t get_dtype_bytes(const safetensors::dtype dtype);
std::string get_dtype_str(const safetensors::dtype dtype);

// Validate data_offsets of all tensors in safetensors_t.
bool validate_data_offsets(const safetensors_t &st, std::string &err);

uint16_t float_to_bfloat16(float x);
float bfloat16_to_float(uint16_t x);

uint16_t float_to_fp16(float x);
float fp16_to_float(uint16_t x);

}  // namespace safetensors

#if defined(SAFETENSORS_CPP_IMPLEMENTATION)

#include <cstring>
#include <fstream>
#include <memory>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>  // for _fseeki64
#include <windows.h>
#endif

#if !defined(MINIJSON_IMPLEMENTATION)
#define MINIJSON_IMPLEMENTATION
#endif

// minijson: https://github.com/syoyo/minijson

/*
 * JSON parser: C++ oriented JSON parser.
 */

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

//#define __MINIJSON_LIBERAL

// We recommended to use simdjson from_chars.
// Using strtod() is a fallback
#if defined(MINIJSON_USE_STRTOD)
// Use stdlib's strtod
#include <cstring>
#else

namespace minijson {
namespace simdjson {
namespace internal {

double from_chars(const char *first) noexcept;
double from_chars(const char *first, const char *end) noexcept;

char *to_chars(char *first, const char *last, double value);

}  // namespace internal
}  // namespace simdjson
}  // namesapce minijson

#endif

namespace minijson {

namespace detail {

double from_chars(const char *p);
const char *my_strchr(const char *p, int ch);

}  // namespace detail

namespace detail {

//
// Usage:
//  - set_input()
//  - scan_string()
//    - success: use `token_buffer` string
//    - error: use `error_message`
//
struct string_parser {
  // input string must be UTF-8
  void set_input(const std::string &s) { _input = s; }

  bool scan_string();

  void reset() {
    if (_input.size()) {
      current = _input[0];
    } else {
      current = '\0';
    }
    curr_idx = 0;
    token_buffer.clear();
  }

  // fetch next token.
  unsigned char get() {
    if ((curr_idx + 1) < _input.size()) {
      curr_idx++;
      current = _input[curr_idx];
      return current;
    }
    current = '\0';
    return current;
  }

  bool eof() {
    if (_input.empty()) {
      return true;
    }

    if (curr_idx >= _input.size()) {
      return true;
    }

    return false;
  }

  void add(const unsigned char c) { token_buffer += c; }

  void add(const int i) {
    // use lower 8bit
    token_buffer += static_cast<unsigned char>(i & 0xff);
  }

  int get_codepoint();

  bool next_byte_in_range(const std::initializer_list<int> ranges);

  std::string error_message;
  std::string token_buffer;  // output

  unsigned char current{'\0'};
  size_t curr_idx{0};
  std::string _input;
};

}  // namespace detail

typedef enum {
  unknown_type,
  null_type,
  boolean_type,
  number_type,
  string_type,
  array_type,
  object_type,
} type;

typedef enum {
  no_error,
  undefined_error,
  invalid_token_error,
  unknown_type_error,
  memory_allocation_error,
  corrupted_json_error,
  duplicated_key_error,
} error;

class value;

typedef bool boolean;
typedef double number;
typedef std::string string;
typedef safetensors::ordered_dict<value> object;
typedef std::vector<value> array;
typedef struct {
} null_t;

// null_t null;

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<null_t> {
  static constexpr uint32_t type_id() { return 0; }
};

template <>
struct TypeTraits<boolean> {
  static constexpr uint32_t type_id() { return 1; }
};

template <>
struct TypeTraits<number> {
  static constexpr uint32_t type_id() { return 2; }
};

template <>
struct TypeTraits<string> {
  static constexpr uint32_t type_id() { return 3; }
};

template <>
struct TypeTraits<object> {
  static constexpr uint32_t type_id() { return 4; }
};

template <>
struct TypeTraits<array> {
  static constexpr uint32_t type_id() { return 5; }
};

class value {
 private:
  type t;
  union {
    null_t n;
    boolean b;
    number d;
    std::string *s;
    array *a;
    object *o;
  } u;

  void _free_u() {
    if (t == string_type) {
      delete this->u.s;
      this->u.s = nullptr;
    }
    if (t == array_type) {
      delete this->u.a;
      this->u.a = nullptr;
    }
    if (t == object_type) {
      delete this->u.o;
      this->u.o = nullptr;
    }
  }

 public:
  value() : t(unknown_type), u() {}
  value(null_t n) : t(null_type), u() { u.n = n; }
  value(boolean b) : t(boolean_type), u() { u.b = b; }
  value(number d) : t(boolean_type), u() { u.d = d; }
  value(const char *s) : t(string_type), u() { u.s = new std::string(s); }
  value(const std::string &s) : t(string_type), u() {
    u.s = new std::string(s);
  }
  value(const array &a) : t(array_type), u() { u.a = new array(a); }
  value(const object &o) : t(object_type), u() { u.o = new object(o); }
  value(const value &v) : t(v.t), u() {
    if (t == array_type) {
      u.a = new array();
      *u.a = *v.u.a;
    } else if (t == object_type) {
      u.o = new object();
      *u.o = *v.u.o;
    } else if (t == string_type) {
      u.s = new std::string();
      *u.s = *v.u.s;
    } else
      u.d = v.u.d;
  }
  ~value() { _free_u(); }

  template <typename T>
  bool is() const {
    if (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id() &&
        t == null_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id() &&
        t == boolean_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<number>::type_id() &&
        t == number_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id() &&
        t == string_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<array>::type_id() &&
        t == array_type)
      return true;
    if (TypeTraits<T>::type_id() == TypeTraits<object>::type_id() &&
        t == object_type)
      return true;
    return false;
  }

  template <typename T>
  const T *as() const {
    if ((t == array_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
      return reinterpret_cast<const T *>(u.a);
    }

    if ((t == object_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
      return reinterpret_cast<const T *>(u.o);
    }

    if ((t == string_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id())) {
      return reinterpret_cast<const T *>(u.s);
    }

    if ((t == null_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
      return reinterpret_cast<const T *>(&u.n);
    }

    if ((t == boolean_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
      return reinterpret_cast<const T *>(&u.b);
    }

    if ((t == number_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
      return reinterpret_cast<const T *>(&u.d);
    }

    return nullptr;
  }

  template <typename T>
  T *as() {
    if ((t == array_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
      return reinterpret_cast<T *>(u.a);
    }

    if ((t == object_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
      return reinterpret_cast<T *>(u.o);
    }

    if ((t == string_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<string>::type_id())) {
      return reinterpret_cast<T *>(u.s);
    }

    if ((t == null_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
      return reinterpret_cast<T *>(&u.n);
    }

    if ((t == boolean_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
      return reinterpret_cast<T *>(&u.b);
    }

    if ((t == number_type) &&
        (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
      return reinterpret_cast<T *>(&u.d);
    }

    return nullptr;
  }

  null_t &operator=(null_t &n) {
    t = null_type;
    u.n = n;
    return u.n;
  }
  boolean &operator=(boolean b) {
    t = boolean_type;
    u.b = b;
    return u.b;
  }
  number &operator=(number d) {
    t = number_type;
    u.d = d;
    return u.d;
  }
  const std::string &operator=(const char *s) {
    _free_u();
    t = string_type;
    u.s = new std::string(s);
    return *u.s;
  }
  const std::string &operator=(const std::string &s) {
    _free_u();
    t = string_type;
    u.s = new std::string(s);
    return *u.s;
  }
  const object &operator=(const object &o) {
    _free_u();
    t = object_type;
    u.o = new object(o);
    return *u.o;
  }
  const array &operator=(const array &a) {
    _free_u();
    t = array_type;
    u.a = new array(a);
    return *u.a;
  }
  const value &operator=(const value &v) {
    _free_u();
    t = v.t;
    if (t == array_type) {
      u.a = new array(*v.u.a);
    } else if (t == object_type) {
      u.o = new object(*v.u.o);
    } else if (t == string_type) {
      u.s = new std::string(*v.u.s);
    } else
      u.d = v.u.d;
    return *this;
  }

  std::string type_name() const {
    if (t == array_type) {
      return "array";
    }

    if (t == object_type) {
      return "object";
    }

    if (t == string_type) {
      return "string";
    }

    if (t == null_type) {
      return "null";
    }

    if (t == boolean_type) {
      return "boolean";
    }

    if (t == number_type) {
      return "number";
    }

    return "[[invalid]]";
  }

  std::string str(const char *p) const {
    std::stringstream ss;
    ss << '"';
    while (*p) {
      if (*p == '\n') {
        ss << "\\n";
      } else if (*p == '\r') {
        ss << "\\r";
      } else if (*p == '\t') {
        ss << "\\t";
      } else if (detail::my_strchr("\"", *p)) {
        ss << "\\" << *p;
      } else {
        ss << *p;
      }
      p++;
    }
    ss << '"';
    return ss.str();
  }

  std::string str() const {
    std::stringstream ss;
    if (t == unknown_type) {
      ss << "undefined";
    } else if (t == null_type) {
      ss << "null";
    } else if (t == boolean_type) {
      ss << (u.b ? "true" : "false");
    } else if (t == number_type) {
      ss << double(u.d);
    } else if (t == string_type) {
      ss << str(u.s->c_str());
    } else if (const array *pa = as<array>()) {
      array::const_iterator i;
      ss << "[";
      // array a = get<array>();
      for (i = pa->begin(); i != pa->end(); i++) {
        if (i != pa->begin()) ss << ", ";
        ss << i->str();
      }
      ss << "]";
    } else if (auto po = as<object>()) {
      // object::const_iterator i;
      ss << "{";
      // object o = get<object>();
      for (size_t i = 0; i < po->size(); i++) {
        if (i > 0) ss << ", ";
        ss << "\"" << po->keys()[i] << "\"";

        value v;
        if (po->at(i, &v)) {
          ss << ": " << v.str();
        } else {
          // TODO: report error
          ss << ": null";
        }
      }
      ss << "}";
    }
    return ss.str();
  }
};

#define MINIJSON_SKIP(i)                           \
  while (*i && detail::my_strchr("\r\n \t", *i)) { \
    i++;                                           \
  }

template <typename Iter>
inline error parse_object(Iter &i, value &v) {
  object o;
  i++;
  MINIJSON_SKIP(i)
  if (!(*i)) {
    return corrupted_json_error;
  }
  if (*i != '\x7d') {
    while (*i) {
      value vk, vv;
      error e = parse_string(i, vk);
      if (e != no_error) return e;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i != ':') return invalid_token_error;
      i++;
      e = parse_any(i, vv);
      if (e != no_error) return e;

      auto ps = vk.as<std::string>();
      if (!ps) {
        return unknown_type_error;
      }

      if (o.count(*ps)) {
        return duplicated_key_error;
      }
      o.insert(*ps, vv);

      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i == '\x7d') break;
      if (*i != ',') return invalid_token_error;
      i++;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
#ifdef __MINIJSON_LIBERAL
      if (*i == '\x7d') break;
#endif
    }
  }
  v = value(o);
  i++;
  return no_error;
}

template <typename Iter>
inline error parse_array(Iter &i, value &v) {
  array a;
  i++;
  MINIJSON_SKIP(i)
  if (!(*i)) {
    return corrupted_json_error;
  }
  if (*i != ']') {
    while (*i) {
      value va;
      error e = parse_any(i, va);
      if (e != no_error) return e;
      a.push_back(va);
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
      if (*i == ']') break;
      if (*i != ',') return invalid_token_error;
      i++;
      MINIJSON_SKIP(i)
      if (!(*i)) {
        return corrupted_json_error;
      }
#ifdef __MINIJSON_LIBERAL
      if (*i == '\x7d') break;
#endif
    }
  }
  v = value(a);
  i++;
  return no_error;
}

template <typename Iter>
inline error parse_null(Iter &i, value &v) {
  Iter p = i;
  if (*i == 'n' && *(i + 1) == 'u' && *(i + 2) == 'l' && *(i + 3) == 'l') {
    i += 4;
    v = null_t();
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return undefined_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_boolean(Iter &i, value &v) {
  Iter p = i;
  if (*i == 't' && *(i + 1) == 'r' && *(i + 2) == 'u' && *(i + 3) == 'e') {
    i += 4;
    v = static_cast<boolean>(true);
  } else if (*i == 'f' && *(i + 1) == 'a' && *(i + 2) == 'l' &&
             *(i + 3) == 's' && *(i + 4) == 'e') {
    i += 5;
    v = static_cast<boolean>(false);
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return undefined_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_number(Iter &i, value &v) {
  Iter p = i;

  if (*i == '-') {
    i++;
  }

#define MINIJSON_IS_NUM(x) ('0' <= x && x <= '9')
#define MINIJSON_IS_ALNUM(x) \
  (('0' <= x && x <= '9') || ('a' <= x && x <= 'f') || ('A' <= x && x <= 'F'))
  if (*i == '0' && *(i + 1) == 'x' && MINIJSON_IS_ALNUM(*(i + 2))) {
    i += 3;
    while (MINIJSON_IS_ALNUM(*i)) i++;
    v = static_cast<number>(detail::from_chars(p));
  } else {
    while (MINIJSON_IS_NUM(*i)) i++;
    if (*i == '.') {
      i++;
      if (!MINIJSON_IS_NUM(*i)) {
        i = p;
        return invalid_token_error;
      }
      while (MINIJSON_IS_NUM(*i)) i++;
    }
    if (*i == 'e') {
      i++;
      if (!MINIJSON_IS_NUM(*i)) {
        i = p;
        return invalid_token_error;
      }
      while (MINIJSON_IS_NUM(*i)) i++;
    }
    v = static_cast<number>(detail::from_chars(p));
  }
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }
  return no_error;
}

template <typename Iter>
inline error parse_string(Iter &i, value &v) {
  if (*i != '"') return invalid_token_error;

  Iter s = i;
  char t = *i++;  // = '"'
  Iter p = i;

#if 0
  std::stringstream ss;
  while (*i && *i != t) {
    if (*i == '\\' && *(i + 1)) {
      i++;
      if (*i == 'n')
        ss << "\n";
      else if (*i == 'r')
        ss << "\r";
      else if (*i == 't')
        ss << "\t";
      else
        ss << *i;
    } else {
      ss << *i;
    }
    i++;
  }
#else
  // read until '"'
  while (*i && *i != t) {
    if (*i == '\\' && *(i + 1)) {
      i++;
    }
    i++;
  }

#endif
  if (!*i) return invalid_token_error;
  if (i < p) {
    return corrupted_json_error;
  }

#if 0
  v = std::string(p, size_t(i - p));

  i++;
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }

#else

  i++;
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }

  // include first and last '"' char
  std::string buf(s, size_t(i - s));

  detail::string_parser str_parser;
  str_parser.set_input(buf);

  if (!str_parser.scan_string()) {
    // TODO: error message
    // str_parser.error_message;
    return invalid_token_error;
  } else {
    v = str_parser.token_buffer;
  }

#endif

  return no_error;
}

template <typename Iter>
inline error parse_any(Iter &i, value &v) {
  MINIJSON_SKIP(i)
  if (*i == '\x7b') return parse_object(i, v);
  if (*i == '[') return parse_array(i, v);
  if (*i == 't' || *i == 'f') return parse_boolean(i, v);
  if (*i == 'n') return parse_null(i, v);
  if ((*i == '-') || ('0' <= *i && *i <= '9')) return parse_number(i, v);
  if (*i == '"') return parse_string(i, v);
  return invalid_token_error;
}

template <typename Iter>
inline error parse(Iter &i, value &v) {
  return parse_any(i, v);
}

#undef MINIJSON_SKIP

inline const char *errstr(error e) {
  const char *s = "unknown error";
  switch (e) {
    case no_error: {
      s = "no error";
      break;
    }
    case undefined_error: {
      s = "undefined";
      break;
    }
    case invalid_token_error: {
      s = "invalid token";
      break;
    }
    case unknown_type_error: {
      s = "unknown type";
      break;
    }
    case memory_allocation_error: {
      s = "memory allocation error";
      break;
    }
    case corrupted_json_error: {
      s = "input is corrupted";
      break;
    }
    case duplicated_key_error: {
      s = "duplicated key found";
      break;
    }
      // default: return "unknown error";
  }

  return s;
}

}  // namespace minijson

#if defined(MINIJSON_IMPLEMENTATION)

namespace minijson {

namespace detail {

// clang-format off
//
// From json.hpp ---------------------------------------------------------
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT

#if 1
    #define JSON_HEDLEY_UNLIKELY(cond) (cond)
    #define JSON_HEDLEY_LIKELY(cond) (cond)

    /*!
    @brief get codepoint from 4 hex characters following `\u`

    For input "\u c1 c2 c3 c4" the codepoint is:
      (c1 * 0x1000) + (c2 * 0x0100) + (c3 * 0x0010) + c4
    = (c1 << 12) + (c2 << 8) + (c3 << 4) + (c4 << 0)

    Furthermore, the possible characters '0'..'9', 'A'..'F', and 'a'..'f'
    must be converted to the integers 0x0..0x9, 0xA..0xF, 0xA..0xF, resp. The
    conversion is done by subtracting the offset (0x30, 0x37, and 0x57)
    between the ASCII value of the character and the desired integer value.

    @return codepoint (0x0000..0xFFFF) or -1 in case of an error (e.g. EOF or
            non-hex character)
    */
    int string_parser::get_codepoint() 
    {
        // this function only makes sense after reading `\u`
        //JSON_ASSERT(current == 'u');
        if (current != 'u') {
          return -1;
        }
        int codepoint = 0;

        const auto factors = { 12u, 8u, 4u, 0u };
        for (const auto factor : factors)
        {
            get();

            if (current >= '0' && current <= '9')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x30u) << factor);
            }
            else if (current >= 'A' && current <= 'F')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x37u) << factor);
            }
            else if (current >= 'a' && current <= 'f')
            {
                codepoint += static_cast<int>((static_cast<unsigned int>(current) - 0x57u) << factor);
            }
            else
            {
                return -1;
            }
        }

        if (0x0000 <= codepoint && codepoint <= 0xFFFF) {
        } else {
          return -1;
        }
        return codepoint;
    }

    /*!
    @brief check if the next byte(s) are inside a given range

    Adds the current byte and, for each passed range, reads a new byte and
    checks if it is inside the range. If a violation was detected, set up an
    error message and return false. Otherwise, return true.

    @param[in] ranges  list of integers; interpreted as list of pairs of
                       inclusive lower and upper bound, respectively

    @pre The passed list @a ranges must have 2, 4, or 6 elements; that is,
         1, 2, or 3 pairs. This precondition is enforced by an assertion.

    @return true if and only if no range violation was detected
    */
    bool string_parser::next_byte_in_range(const std::initializer_list<int> ranges)
    {
        if (ranges.size() == 2 || ranges.size() == 4 || ranges.size() == 6) {
        } else {
          return false;
        }

        add(current);

        for (auto range = ranges.begin(); range != ranges.end(); ++range)
        {
            get();
            if (JSON_HEDLEY_LIKELY(*range <= current && current <= *(++range))) // NOLINT(bugprone-inc-dec-in-conditions)
            {
                add(current);
            }
            else
            {
                error_message = "invalid string: ill-formed UTF-8 byte";
                return false;
            }
        }

        return true;
    }
    /*!
    @brief scan a string literal

    This function scans a string according to Sect. 7 of RFC 8259. While
    scanning, bytes are escaped and copied into buffer token_buffer. Then the
    function returns successfully, token_buffer is *not* null-terminated (as it
    may contain \0 bytes), and token_buffer.size() is the number of bytes in the
    string.

    @return true if string could be successfully scanned,
            false otherwise

    @note In case of errors, variable error_message contains a textual
          description.
    */
    bool string_parser::scan_string()
    {
        // reset token_buffer (ignore opening quote)
        reset();

        // we entered the function by reading an open quote
        //JSON_ASSERT(current == '\"');
        if (current != '\"') {
            error_message = "first character must be '\"'";
            return false;
        }


        while (!eof())
        {

            // get next character
            switch (get())
            {

                // closing quote
                case '\"':
                {
                    return true;
                }

                // escapes
                case '\\':
                {
                    switch (get())
                    {
                        // quotation mark
                        case '\"':
                            add('\"');
                            break;
                        // reverse solidus
                        case '\\':
                            add('\\');
                            break;
                        // solidus
                        case '/':
                            add('/');
                            break;
                        // backspace
                        case 'b':
                            add('\b');
                            break;
                        // form feed
                        case 'f':
                            add('\f');
                            break;
                        // line feed
                        case 'n':
                            add('\n');
                            break;
                        // carriage return
                        case 'r':
                            add('\r');
                            break;
                        // tab
                        case 't':
                            add('\t');
                            break;

                        // unicode escapes
                        case 'u':
                        {
                            const int codepoint1 = get_codepoint();
                            int codepoint = codepoint1; // start with codepoint1

                            if (JSON_HEDLEY_UNLIKELY(codepoint1 == -1))
                            {
                                error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                return false;
                            }

                            // check if code point is a high surrogate
                            if (0xD800 <= codepoint1 && codepoint1 <= 0xDBFF)
                            {
                                // expect next \uxxxx entry
                                if (JSON_HEDLEY_LIKELY(get() == '\\' && get() == 'u'))
                                {
                                    const int codepoint2 = get_codepoint();

                                    if (JSON_HEDLEY_UNLIKELY(codepoint2 == -1))
                                    {
                                        error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                        return false;
                                    }

                                    // check if codepoint2 is a low surrogate
                                    if (JSON_HEDLEY_LIKELY(0xDC00 <= codepoint2 && codepoint2 <= 0xDFFF))
                                    {
                                        // overwrite codepoint
                                        codepoint = static_cast<int>(
                                                        // high surrogate occupies the most significant 22 bits
                                                        (static_cast<unsigned int>(codepoint1) << 10u)
                                                        // low surrogate occupies the least significant 15 bits
                                                        + static_cast<unsigned int>(codepoint2)
                                                        // there is still the 0xD800, 0xDC00 and 0x10000 noise
                                                        // in the result, so we have to subtract with:
                                                        // (0xD800 << 10) + DC00 - 0x10000 = 0x35FDC00
                                                        - 0x35FDC00u);
                                    }
                                    else
                                    {
                                        error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                        return false;
                                    }
                                }
                                else
                                {
                                    error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                    return false;
                                }
                            }
                            else
                            {
                                if (JSON_HEDLEY_UNLIKELY(0xDC00 <= codepoint1 && codepoint1 <= 0xDFFF))
                                {
                                    error_message = "invalid string: surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF";
                                    return false;
                                }
                            }

                            // result of the above calculation yields a proper codepoint
                            //JSON_ASSERT(0x00 <= codepoint && codepoint <= 0x10FFFF);
                            if (0x00 <= codepoint && codepoint <= 0x10FFFF) {
                            } else {
                                error_message = "invalid string: invalid codepoint";
                                return false;
                            }

                            // translate codepoint into bytes
                            if (codepoint < 0x80)
                            {
                                // 1-byte characters: 0xxxxxxx (ASCII)
                                add(static_cast<int>(codepoint));
                            }
                            else if (codepoint <= 0x7FF)
                            {
                                // 2-byte characters: 110xxxxx 10xxxxxx
                                add(static_cast<int>(0xC0u | (static_cast<unsigned int>(codepoint) >> 6u)));
                                add(static_cast<int>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }
                            else if (codepoint <= 0xFFFF)
                            {
                                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
                                add(static_cast<int>(0xE0u | (static_cast<unsigned int>(codepoint) >> 12u)));
                                add(static_cast<int>(0x80u | ((static_cast<unsigned int>(codepoint) >> 6u) & 0x3Fu)));
                                add(static_cast<int>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }
                            else
                            {
                                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                                add(static_cast<int>(0xF0u | (static_cast<unsigned int>(codepoint) >> 18u)));
                                add(static_cast<int>(0x80u | ((static_cast<unsigned int>(codepoint) >> 12u) & 0x3Fu)));
                                add(static_cast<int>(0x80u | ((static_cast<unsigned int>(codepoint) >> 6u) & 0x3Fu)));
                                add(static_cast<int>(0x80u | (static_cast<unsigned int>(codepoint) & 0x3Fu)));
                            }

                            break;
                        }

                        // other characters after escape
                        default:
                            error_message = "invalid string: forbidden character after backslash";
                            return false;
                    }

                    break;
                }

                // invalid control characters
                case 0x00:
                {
                    error_message = "invalid string: control character U+0000 (NUL) must be escaped to \\u0000";
                    return false;
                }

                case 0x01:
                {
                    error_message = "invalid string: control character U+0001 (SOH) must be escaped to \\u0001";
                    return false;
                }

                case 0x02:
                {
                    error_message = "invalid string: control character U+0002 (STX) must be escaped to \\u0002";
                    return false;
                }

                case 0x03:
                {
                    error_message = "invalid string: control character U+0003 (ETX) must be escaped to \\u0003";
                    return false;
                }

                case 0x04:
                {
                    error_message = "invalid string: control character U+0004 (EOT) must be escaped to \\u0004";
                    return false;
                }

                case 0x05:
                {
                    error_message = "invalid string: control character U+0005 (ENQ) must be escaped to \\u0005";
                    return false;
                }

                case 0x06:
                {
                    error_message = "invalid string: control character U+0006 (ACK) must be escaped to \\u0006";
                    return false;
                }

                case 0x07:
                {
                    error_message = "invalid string: control character U+0007 (BEL) must be escaped to \\u0007";
                    return false;
                }

                case 0x08:
                {
                    error_message = "invalid string: control character U+0008 (BS) must be escaped to \\u0008 or \\b";
                    return false;
                }

                case 0x09:
                {
                    error_message = "invalid string: control character U+0009 (HT) must be escaped to \\u0009 or \\t";
                    return false;
                }

                case 0x0A:
                {
                    error_message = "invalid string: control character U+000A (LF) must be escaped to \\u000A or \\n";
                    return false;
                }

                case 0x0B:
                {
                    error_message = "invalid string: control character U+000B (VT) must be escaped to \\u000B";
                    return false;
                }

                case 0x0C:
                {
                    error_message = "invalid string: control character U+000C (FF) must be escaped to \\u000C or \\f";
                    return false;
                }

                case 0x0D:
                {
                    error_message = "invalid string: control character U+000D (CR) must be escaped to \\u000D or \\r";
                    return false;
                }

                case 0x0E:
                {
                    error_message = "invalid string: control character U+000E (SO) must be escaped to \\u000E";
                    return false;
                }

                case 0x0F:
                {
                    error_message = "invalid string: control character U+000F (SI) must be escaped to \\u000F";
                    return false;
                }

                case 0x10:
                {
                    error_message = "invalid string: control character U+0010 (DLE) must be escaped to \\u0010";
                    return false;
                }

                case 0x11:
                {
                    error_message = "invalid string: control character U+0011 (DC1) must be escaped to \\u0011";
                    return false;
                }

                case 0x12:
                {
                    error_message = "invalid string: control character U+0012 (DC2) must be escaped to \\u0012";
                    return false;
                }

                case 0x13:
                {
                    error_message = "invalid string: control character U+0013 (DC3) must be escaped to \\u0013";
                    return false;
                }

                case 0x14:
                {
                    error_message = "invalid string: control character U+0014 (DC4) must be escaped to \\u0014";
                    return false;
                }

                case 0x15:
                {
                    error_message = "invalid string: control character U+0015 (NAK) must be escaped to \\u0015";
                    return false;
                }

                case 0x16:
                {
                    error_message = "invalid string: control character U+0016 (SYN) must be escaped to \\u0016";
                    return false;
                }

                case 0x17:
                {
                    error_message = "invalid string: control character U+0017 (ETB) must be escaped to \\u0017";
                    return false;
                }

                case 0x18:
                {
                    error_message = "invalid string: control character U+0018 (CAN) must be escaped to \\u0018";
                    return false;
                }

                case 0x19:
                {
                    error_message = "invalid string: control character U+0019 (EM) must be escaped to \\u0019";
                    return false;
                }

                case 0x1A:
                {
                    error_message = "invalid string: control character U+001A (SUB) must be escaped to \\u001A";
                    return false;
                }

                case 0x1B:
                {
                    error_message = "invalid string: control character U+001B (ESC) must be escaped to \\u001B";
                    return false;
                }

                case 0x1C:
                {
                    error_message = "invalid string: control character U+001C (FS) must be escaped to \\u001C";
                    return false;
                }

                case 0x1D:
                {
                    error_message = "invalid string: control character U+001D (GS) must be escaped to \\u001D";
                    return false;
                }

                case 0x1E:
                {
                    error_message = "invalid string: control character U+001E (RS) must be escaped to \\u001E";
                    return false;
                }

                case 0x1F:
                {
                    error_message = "invalid string: control character U+001F (US) must be escaped to \\u001F";
                    return false;
                }

                // U+0020..U+007F (except U+0022 (quote) and U+005C (backspace))
                case 0x20:
                case 0x21:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x37:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x47:
                case 0x48:
                case 0x49:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x4D:
                case 0x4E:
                case 0x4F:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                case 0x56:
                case 0x57:
                case 0x58:
                case 0x59:
                case 0x5A:
                case 0x5B:
                case 0x5D:
                case 0x5E:
                case 0x5F:
                case 0x60:
                case 0x61:
                case 0x62:
                case 0x63:
                case 0x64:
                case 0x65:
                case 0x66:
                case 0x67:
                case 0x68:
                case 0x69:
                case 0x6A:
                case 0x6B:
                case 0x6C:
                case 0x6D:
                case 0x6E:
                case 0x6F:
                case 0x70:
                case 0x71:
                case 0x72:
                case 0x73:
                case 0x74:
                case 0x75:
                case 0x76:
                case 0x77:
                case 0x78:
                case 0x79:
                case 0x7A:
                case 0x7B:
                case 0x7C:
                case 0x7D:
                case 0x7E:
                case 0x7F:
                {
                    add(current);
                    break;
                }

                // U+0080..U+07FF: bytes C2..DF 80..BF
                case 0xC2:
                case 0xC3:
                case 0xC4:
                case 0xC5:
                case 0xC6:
                case 0xC7:
                case 0xC8:
                case 0xC9:
                case 0xCA:
                case 0xCB:
                case 0xCC:
                case 0xCD:
                case 0xCE:
                case 0xCF:
                case 0xD0:
                case 0xD1:
                case 0xD2:
                case 0xD3:
                case 0xD4:
                case 0xD5:
                case 0xD6:
                case 0xD7:
                case 0xD8:
                case 0xD9:
                case 0xDA:
                case 0xDB:
                case 0xDC:
                case 0xDD:
                case 0xDE:
                case 0xDF:
                {
                    if (JSON_HEDLEY_UNLIKELY(!next_byte_in_range({0x80, 0xBF})))
                    {
                        return false;
                    }
                    break;
                }

                // U+0800..U+0FFF: bytes E0 A0..BF 80..BF
                case 0xE0:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0xA0, 0xBF, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // U+1000..U+CFFF: bytes E1..EC 80..BF 80..BF
                // U+E000..U+FFFF: bytes EE..EF 80..BF 80..BF
                case 0xE1:
                case 0xE2:
                case 0xE3:
                case 0xE4:
                case 0xE5:
                case 0xE6:
                case 0xE7:
                case 0xE8:
                case 0xE9:
                case 0xEA:
                case 0xEB:
                case 0xEC:
                case 0xEE:
                case 0xEF:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // U+D000..U+D7FF: bytes ED 80..9F 80..BF
                case 0xED:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0x9F, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // U+10000..U+3FFFF F0 90..BF 80..BF 80..BF
                case 0xF0:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x90, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // U+40000..U+FFFFF F1..F3 80..BF 80..BF 80..BF
                case 0xF1:
                case 0xF2:
                case 0xF3:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // U+100000..U+10FFFF F4 80..8F 80..BF 80..BF
                case 0xF4:
                {
                    if (JSON_HEDLEY_UNLIKELY(!(next_byte_in_range({0x80, 0x8F, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return false;
                    }
                    break;
                }

                // remaining bytes (80..C1 and F5..FF) are ill-formed
                default:
                {
                    error_message = "invalid string: ill-formed UTF-8 byte";
                    return false;
                }
            }
        }

        error_message = "invalid string: missing closing quote";
        return false;
    }
#endif
// end json.hpp
// clang-format on

}  // namespace detail

namespace detail {

double from_chars(const char *p) {
#if defined(MINIJSON_USE_STRTOD)
  return strtod(p, nullptr);
#else
  return simdjson::internal::from_chars(p);
#endif
}

const char *my_strchr(const char *p, int ch) {
  char c;

  constexpr uint64_t kMaxCount = 1024ull * 1024ull;  // up to 1M chars

  uint64_t cnt{0};

  c = ch;
  for (;; ++p, cnt++) {
    if (cnt > kMaxCount) {
      return nullptr;
    }

    if (*p == c) {
      return (p);
    }
    if (*p == '\0') {
      return (nullptr);
    }
  }
}

}  // namespace detail
}  // namespace minijson

#if !defined(MINIJSON_USE_STRTOD)

#include <cstring>
#include <limits>

namespace minijson {
namespace simdjson {
namespace internal {

/**
 * The code in the internal::from_chars function is meant to handle the
 *floating-point number parsing when we have more than 19 digits in the decimal
 *mantissa. This should only be seen in adversarial scenarios: we do not expect
 *production systems to even produce such floating-point numbers.
 *
 * The parser is based on work by Nigel Tao (at
 *https://github.com/google/wuffs/) who credits Ken Thompson for the design (via
 *a reference to the Go source code). See
 * https://github.com/google/wuffs/blob/aa46859ea40c72516deffa1b146121952d6dfd3b/internal/cgen/base/floatconv-submodule-data.c
 * https://github.com/google/wuffs/blob/46cd8105f47ca07ae2ba8e6a7818ef9c0df6c152/internal/cgen/base/floatconv-submodule-code.c
 * It is probably not very fast but it is a fallback that should almost never be
 * called in real life. Google Wuffs is published under APL 2.0.
 **/

namespace {
constexpr uint32_t max_digits = 768;
constexpr int32_t decimal_point_range = 2047;
}  // namespace

struct adjusted_mantissa {
  uint64_t mantissa;
  int power2;
  adjusted_mantissa() : mantissa(0), power2(0) {}
};

struct decimal {
  uint32_t num_digits;
  int32_t decimal_point;
  bool negative;
  bool truncated;
  uint8_t digits[max_digits];
};

template <typename T>
struct binary_format {
  static constexpr int mantissa_explicit_bits();
  static constexpr int minimum_exponent();
  static constexpr int infinite_power();
  static constexpr int sign_index();
};

template <>
constexpr int binary_format<double>::mantissa_explicit_bits() {
  return 52;
}

template <>
constexpr int binary_format<double>::minimum_exponent() {
  return -1023;
}
template <>
constexpr int binary_format<double>::infinite_power() {
  return 0x7FF;
}

template <>
constexpr int binary_format<double>::sign_index() {
  return 63;
}

inline bool is_integer(char c) noexcept { return (c >= '0' && c <= '9'); }

// This should always succeed since it follows a call to parse_number.
static decimal parse_decimal(const char *&p) noexcept {
  decimal answer;
  answer.num_digits = 0;
  answer.decimal_point = 0;
  answer.truncated = false;
  answer.negative = (*p == '-');
  if ((*p == '-') || (*p == '+')) {
    ++p;
  }

  while (*p == '0') {
    ++p;
  }
  while (is_integer(*p)) {
    if (answer.num_digits < max_digits) {
      answer.digits[answer.num_digits] = uint8_t(*p - '0');
    }
    answer.num_digits++;
    ++p;
  }
  if (*p == '.') {
    ++p;
    const char *first_after_period = p;
    // if we have not yet encountered a zero, we have to skip it as well
    if (answer.num_digits == 0) {
      // skip zeros
      while (*p == '0') {
        ++p;
      }
    }
    while (is_integer(*p)) {
      if (answer.num_digits < max_digits) {
        answer.digits[answer.num_digits] = uint8_t(*p - '0');
      }
      answer.num_digits++;
      ++p;
    }
    answer.decimal_point = int32_t(first_after_period - p);
  }
  if (answer.num_digits > 0) {
    const char *preverse = p - 1;
    int32_t trailing_zeros = 0;
    while ((*preverse == '0') || (*preverse == '.')) {
      if (*preverse == '0') {
        trailing_zeros++;
      }
      --preverse;
    }
    answer.decimal_point += int32_t(answer.num_digits);
    answer.num_digits -= uint32_t(trailing_zeros);
  }
  if (answer.num_digits > max_digits) {
    answer.num_digits = max_digits;
    answer.truncated = true;
  }
  if (('e' == *p) || ('E' == *p)) {
    ++p;
    bool neg_exp = false;
    if ('-' == *p) {
      neg_exp = true;
      ++p;
    } else if ('+' == *p) {
      ++p;
    }
    int32_t exp_number = 0;  // exponential part
    while (is_integer(*p)) {
      uint8_t digit = uint8_t(*p - '0');
      if (exp_number < 0x10000) {
        exp_number = 10 * exp_number + digit;
      }
      ++p;
    }
    answer.decimal_point += (neg_exp ? -exp_number : exp_number);
  }
  return answer;
}

// This should always succeed since it follows a call to parse_number.
// Will not read at or beyond the "end" pointer.
static decimal parse_decimal(const char *&p, const char *end) noexcept {
  decimal answer;
  answer.num_digits = 0;
  answer.decimal_point = 0;
  answer.truncated = false;
  if (p == end) {
    return answer;
  }  // should never happen
  answer.negative = (*p == '-');
  if ((*p == '-') || (*p == '+')) {
    ++p;
  }

  while ((p != end) && (*p == '0')) {
    ++p;
  }
  while ((p != end) && is_integer(*p)) {
    if (answer.num_digits < max_digits) {
      answer.digits[answer.num_digits] = uint8_t(*p - '0');
    }
    answer.num_digits++;
    ++p;
  }
  if ((p != end) && (*p == '.')) {
    ++p;
    if (p == end) {
      return answer;
    }  // should never happen
    const char *first_after_period = p;
    // if we have not yet encountered a zero, we have to skip it as well
    if (answer.num_digits == 0) {
      // skip zeros
      while (*p == '0') {
        ++p;
      }
    }
    while ((p != end) && is_integer(*p)) {
      if (answer.num_digits < max_digits) {
        answer.digits[answer.num_digits] = uint8_t(*p - '0');
      }
      answer.num_digits++;
      ++p;
    }
    answer.decimal_point = int32_t(first_after_period - p);
  }
  if (answer.num_digits > 0) {
    const char *preverse = p - 1;
    int32_t trailing_zeros = 0;
    while ((*preverse == '0') || (*preverse == '.')) {
      if (*preverse == '0') {
        trailing_zeros++;
      }
      --preverse;
    }
    answer.decimal_point += int32_t(answer.num_digits);
    answer.num_digits -= uint32_t(trailing_zeros);
  }
  if (answer.num_digits > max_digits) {
    answer.num_digits = max_digits;
    answer.truncated = true;
  }
  if ((p != end) && (('e' == *p) || ('E' == *p))) {
    ++p;
    if (p == end) {
      return answer;
    }  // should never happen
    bool neg_exp = false;
    if ('-' == *p) {
      neg_exp = true;
      ++p;
    } else if ('+' == *p) {
      ++p;
    }
    int32_t exp_number = 0;  // exponential part
    while ((p != end) && is_integer(*p)) {
      uint8_t digit = uint8_t(*p - '0');
      if (exp_number < 0x10000) {
        exp_number = 10 * exp_number + digit;
      }
      ++p;
    }
    answer.decimal_point += (neg_exp ? -exp_number : exp_number);
  }
  return answer;
}

namespace {

// remove all final zeroes
inline void trim(decimal &h) {
  while ((h.num_digits > 0) && (h.digits[h.num_digits - 1] == 0)) {
    h.num_digits--;
  }
}

uint32_t number_of_digits_decimal_left_shift(decimal &h, uint32_t shift) {
  shift &= 63;
  const static uint16_t number_of_digits_decimal_left_shift_table[65] = {
      0x0000, 0x0800, 0x0801, 0x0803, 0x1006, 0x1009, 0x100D, 0x1812, 0x1817,
      0x181D, 0x2024, 0x202B, 0x2033, 0x203C, 0x2846, 0x2850, 0x285B, 0x3067,
      0x3073, 0x3080, 0x388E, 0x389C, 0x38AB, 0x38BB, 0x40CC, 0x40DD, 0x40EF,
      0x4902, 0x4915, 0x4929, 0x513E, 0x5153, 0x5169, 0x5180, 0x5998, 0x59B0,
      0x59C9, 0x61E3, 0x61FD, 0x6218, 0x6A34, 0x6A50, 0x6A6D, 0x6A8B, 0x72AA,
      0x72C9, 0x72E9, 0x7B0A, 0x7B2B, 0x7B4D, 0x8370, 0x8393, 0x83B7, 0x83DC,
      0x8C02, 0x8C28, 0x8C4F, 0x9477, 0x949F, 0x94C8, 0x9CF2, 0x051C, 0x051C,
      0x051C, 0x051C,
  };
  uint32_t x_a = number_of_digits_decimal_left_shift_table[shift];
  uint32_t x_b = number_of_digits_decimal_left_shift_table[shift + 1];
  uint32_t num_new_digits = x_a >> 11;
  uint32_t pow5_a = 0x7FF & x_a;
  uint32_t pow5_b = 0x7FF & x_b;
  const static uint8_t
      number_of_digits_decimal_left_shift_table_powers_of_5[0x051C] = {
          5, 2, 5, 1, 2, 5, 6, 2, 5, 3, 1, 2, 5, 1, 5, 6, 2, 5, 7, 8, 1, 2, 5,
          3, 9, 0, 6, 2, 5, 1, 9, 5, 3, 1, 2, 5, 9, 7, 6, 5, 6, 2, 5, 4, 8, 8,
          2, 8, 1, 2, 5, 2, 4, 4, 1, 4, 0, 6, 2, 5, 1, 2, 2, 0, 7, 0, 3, 1, 2,
          5, 6, 1, 0, 3, 5, 1, 5, 6, 2, 5, 3, 0, 5, 1, 7, 5, 7, 8, 1, 2, 5, 1,
          5, 2, 5, 8, 7, 8, 9, 0, 6, 2, 5, 7, 6, 2, 9, 3, 9, 4, 5, 3, 1, 2, 5,
          3, 8, 1, 4, 6, 9, 7, 2, 6, 5, 6, 2, 5, 1, 9, 0, 7, 3, 4, 8, 6, 3, 2,
          8, 1, 2, 5, 9, 5, 3, 6, 7, 4, 3, 1, 6, 4, 0, 6, 2, 5, 4, 7, 6, 8, 3,
          7, 1, 5, 8, 2, 0, 3, 1, 2, 5, 2, 3, 8, 4, 1, 8, 5, 7, 9, 1, 0, 1, 5,
          6, 2, 5, 1, 1, 9, 2, 0, 9, 2, 8, 9, 5, 5, 0, 7, 8, 1, 2, 5, 5, 9, 6,
          0, 4, 6, 4, 4, 7, 7, 5, 3, 9, 0, 6, 2, 5, 2, 9, 8, 0, 2, 3, 2, 2, 3,
          8, 7, 6, 9, 5, 3, 1, 2, 5, 1, 4, 9, 0, 1, 1, 6, 1, 1, 9, 3, 8, 4, 7,
          6, 5, 6, 2, 5, 7, 4, 5, 0, 5, 8, 0, 5, 9, 6, 9, 2, 3, 8, 2, 8, 1, 2,
          5, 3, 7, 2, 5, 2, 9, 0, 2, 9, 8, 4, 6, 1, 9, 1, 4, 0, 6, 2, 5, 1, 8,
          6, 2, 6, 4, 5, 1, 4, 9, 2, 3, 0, 9, 5, 7, 0, 3, 1, 2, 5, 9, 3, 1, 3,
          2, 2, 5, 7, 4, 6, 1, 5, 4, 7, 8, 5, 1, 5, 6, 2, 5, 4, 6, 5, 6, 6, 1,
          2, 8, 7, 3, 0, 7, 7, 3, 9, 2, 5, 7, 8, 1, 2, 5, 2, 3, 2, 8, 3, 0, 6,
          4, 3, 6, 5, 3, 8, 6, 9, 6, 2, 8, 9, 0, 6, 2, 5, 1, 1, 6, 4, 1, 5, 3,
          2, 1, 8, 2, 6, 9, 3, 4, 8, 1, 4, 4, 5, 3, 1, 2, 5, 5, 8, 2, 0, 7, 6,
          6, 0, 9, 1, 3, 4, 6, 7, 4, 0, 7, 2, 2, 6, 5, 6, 2, 5, 2, 9, 1, 0, 3,
          8, 3, 0, 4, 5, 6, 7, 3, 3, 7, 0, 3, 6, 1, 3, 2, 8, 1, 2, 5, 1, 4, 5,
          5, 1, 9, 1, 5, 2, 2, 8, 3, 6, 6, 8, 5, 1, 8, 0, 6, 6, 4, 0, 6, 2, 5,
          7, 2, 7, 5, 9, 5, 7, 6, 1, 4, 1, 8, 3, 4, 2, 5, 9, 0, 3, 3, 2, 0, 3,
          1, 2, 5, 3, 6, 3, 7, 9, 7, 8, 8, 0, 7, 0, 9, 1, 7, 1, 2, 9, 5, 1, 6,
          6, 0, 1, 5, 6, 2, 5, 1, 8, 1, 8, 9, 8, 9, 4, 0, 3, 5, 4, 5, 8, 5, 6,
          4, 7, 5, 8, 3, 0, 0, 7, 8, 1, 2, 5, 9, 0, 9, 4, 9, 4, 7, 0, 1, 7, 7,
          2, 9, 2, 8, 2, 3, 7, 9, 1, 5, 0, 3, 9, 0, 6, 2, 5, 4, 5, 4, 7, 4, 7,
          3, 5, 0, 8, 8, 6, 4, 6, 4, 1, 1, 8, 9, 5, 7, 5, 1, 9, 5, 3, 1, 2, 5,
          2, 2, 7, 3, 7, 3, 6, 7, 5, 4, 4, 3, 2, 3, 2, 0, 5, 9, 4, 7, 8, 7, 5,
          9, 7, 6, 5, 6, 2, 5, 1, 1, 3, 6, 8, 6, 8, 3, 7, 7, 2, 1, 6, 1, 6, 0,
          2, 9, 7, 3, 9, 3, 7, 9, 8, 8, 2, 8, 1, 2, 5, 5, 6, 8, 4, 3, 4, 1, 8,
          8, 6, 0, 8, 0, 8, 0, 1, 4, 8, 6, 9, 6, 8, 9, 9, 4, 1, 4, 0, 6, 2, 5,
          2, 8, 4, 2, 1, 7, 0, 9, 4, 3, 0, 4, 0, 4, 0, 0, 7, 4, 3, 4, 8, 4, 4,
          9, 7, 0, 7, 0, 3, 1, 2, 5, 1, 4, 2, 1, 0, 8, 5, 4, 7, 1, 5, 2, 0, 2,
          0, 0, 3, 7, 1, 7, 4, 2, 2, 4, 8, 5, 3, 5, 1, 5, 6, 2, 5, 7, 1, 0, 5,
          4, 2, 7, 3, 5, 7, 6, 0, 1, 0, 0, 1, 8, 5, 8, 7, 1, 1, 2, 4, 2, 6, 7,
          5, 7, 8, 1, 2, 5, 3, 5, 5, 2, 7, 1, 3, 6, 7, 8, 8, 0, 0, 5, 0, 0, 9,
          2, 9, 3, 5, 5, 6, 2, 1, 3, 3, 7, 8, 9, 0, 6, 2, 5, 1, 7, 7, 6, 3, 5,
          6, 8, 3, 9, 4, 0, 0, 2, 5, 0, 4, 6, 4, 6, 7, 7, 8, 1, 0, 6, 6, 8, 9,
          4, 5, 3, 1, 2, 5, 8, 8, 8, 1, 7, 8, 4, 1, 9, 7, 0, 0, 1, 2, 5, 2, 3,
          2, 3, 3, 8, 9, 0, 5, 3, 3, 4, 4, 7, 2, 6, 5, 6, 2, 5, 4, 4, 4, 0, 8,
          9, 2, 0, 9, 8, 5, 0, 0, 6, 2, 6, 1, 6, 1, 6, 9, 4, 5, 2, 6, 6, 7, 2,
          3, 6, 3, 2, 8, 1, 2, 5, 2, 2, 2, 0, 4, 4, 6, 0, 4, 9, 2, 5, 0, 3, 1,
          3, 0, 8, 0, 8, 4, 7, 2, 6, 3, 3, 3, 6, 1, 8, 1, 6, 4, 0, 6, 2, 5, 1,
          1, 1, 0, 2, 2, 3, 0, 2, 4, 6, 2, 5, 1, 5, 6, 5, 4, 0, 4, 2, 3, 6, 3,
          1, 6, 6, 8, 0, 9, 0, 8, 2, 0, 3, 1, 2, 5, 5, 5, 5, 1, 1, 1, 5, 1, 2,
          3, 1, 2, 5, 7, 8, 2, 7, 0, 2, 1, 1, 8, 1, 5, 8, 3, 4, 0, 4, 5, 4, 1,
          0, 1, 5, 6, 2, 5, 2, 7, 7, 5, 5, 5, 7, 5, 6, 1, 5, 6, 2, 8, 9, 1, 3,
          5, 1, 0, 5, 9, 0, 7, 9, 1, 7, 0, 2, 2, 7, 0, 5, 0, 7, 8, 1, 2, 5, 1,
          3, 8, 7, 7, 7, 8, 7, 8, 0, 7, 8, 1, 4, 4, 5, 6, 7, 5, 5, 2, 9, 5, 3,
          9, 5, 8, 5, 1, 1, 3, 5, 2, 5, 3, 9, 0, 6, 2, 5, 6, 9, 3, 8, 8, 9, 3,
          9, 0, 3, 9, 0, 7, 2, 2, 8, 3, 7, 7, 6, 4, 7, 6, 9, 7, 9, 2, 5, 5, 6,
          7, 6, 2, 6, 9, 5, 3, 1, 2, 5, 3, 4, 6, 9, 4, 4, 6, 9, 5, 1, 9, 5, 3,
          6, 1, 4, 1, 8, 8, 8, 2, 3, 8, 4, 8, 9, 6, 2, 7, 8, 3, 8, 1, 3, 4, 7,
          6, 5, 6, 2, 5, 1, 7, 3, 4, 7, 2, 3, 4, 7, 5, 9, 7, 6, 8, 0, 7, 0, 9,
          4, 4, 1, 1, 9, 2, 4, 4, 8, 1, 3, 9, 1, 9, 0, 6, 7, 3, 8, 2, 8, 1, 2,
          5, 8, 6, 7, 3, 6, 1, 7, 3, 7, 9, 8, 8, 4, 0, 3, 5, 4, 7, 2, 0, 5, 9,
          6, 2, 2, 4, 0, 6, 9, 5, 9, 5, 3, 3, 6, 9, 1, 4, 0, 6, 2, 5,
      };
  const uint8_t *pow5 =
      &number_of_digits_decimal_left_shift_table_powers_of_5[pow5_a];
  uint32_t i = 0;
  uint32_t n = pow5_b - pow5_a;
  for (; i < n; i++) {
    if (i >= h.num_digits) {
      return num_new_digits - 1;
    } else if (h.digits[i] == pow5[i]) {
      continue;
    } else if (h.digits[i] < pow5[i]) {
      return num_new_digits - 1;
    } else {
      return num_new_digits;
    }
  }
  return num_new_digits;
}

}  // end of anonymous namespace

static uint64_t round(decimal &h) {
  if ((h.num_digits == 0) || (h.decimal_point < 0)) {
    return 0;
  } else if (h.decimal_point > 18) {
    return UINT64_MAX;
  }
  // at this point, we know that h.decimal_point >= 0
  uint32_t dp = uint32_t(h.decimal_point);
  uint64_t n = 0;
  for (uint32_t i = 0; i < dp; i++) {
    n = (10 * n) + ((i < h.num_digits) ? h.digits[i] : 0);
  }
  bool round_up = false;
  if (dp < h.num_digits) {
    round_up = h.digits[dp] >= 5;  // normally, we round up
    // but we may need to round to even!
    if ((h.digits[dp] == 5) && (dp + 1 == h.num_digits)) {
      round_up = h.truncated || ((dp > 0) && (1 & h.digits[dp - 1]));
    }
  }
  if (round_up) {
    n++;
  }
  return n;
}

// computes h * 2^-shift
static void decimal_left_shift(decimal &h, uint32_t shift) {
  if (h.num_digits == 0) {
    return;
  }
  uint32_t num_new_digits = number_of_digits_decimal_left_shift(h, shift);
  int32_t read_index = int32_t(h.num_digits - 1);
  uint32_t write_index = h.num_digits - 1 + num_new_digits;
  uint64_t n = 0;

  while (read_index >= 0) {
    n += uint64_t(h.digits[read_index]) << shift;
    uint64_t quotient = n / 10;
    uint64_t remainder = n - (10 * quotient);
    if (write_index < max_digits) {
      h.digits[write_index] = uint8_t(remainder);
    } else if (remainder > 0) {
      h.truncated = true;
    }
    n = quotient;
    write_index--;
    read_index--;
  }
  while (n > 0) {
    uint64_t quotient = n / 10;
    uint64_t remainder = n - (10 * quotient);
    if (write_index < max_digits) {
      h.digits[write_index] = uint8_t(remainder);
    } else if (remainder > 0) {
      h.truncated = true;
    }
    n = quotient;
    write_index--;
  }
  h.num_digits += num_new_digits;
  if (h.num_digits > max_digits) {
    h.num_digits = max_digits;
  }
  h.decimal_point += int32_t(num_new_digits);
  trim(h);
}

// computes h * 2^shift
static void decimal_right_shift(decimal &h, uint32_t shift) {
  uint32_t read_index = 0;
  uint32_t write_index = 0;

  uint64_t n = 0;

  while ((n >> shift) == 0) {
    if (read_index < h.num_digits) {
      n = (10 * n) + h.digits[read_index++];
    } else if (n == 0) {
      return;
    } else {
      while ((n >> shift) == 0) {
        n = 10 * n;
        read_index++;
      }
      break;
    }
  }
  h.decimal_point -= int32_t(read_index - 1);
  if (h.decimal_point < -decimal_point_range) {  // it is zero
    h.num_digits = 0;
    h.decimal_point = 0;
    h.negative = false;
    h.truncated = false;
    return;
  }
  uint64_t mask = (uint64_t(1) << shift) - 1;
  while (read_index < h.num_digits) {
    uint8_t new_digit = uint8_t(n >> shift);
    n = (10 * (n & mask)) + h.digits[read_index++];
    h.digits[write_index++] = new_digit;
  }
  while (n > 0) {
    uint8_t new_digit = uint8_t(n >> shift);
    n = 10 * (n & mask);
    if (write_index < max_digits) {
      h.digits[write_index++] = new_digit;
    } else if (new_digit > 0) {
      h.truncated = true;
    }
  }
  h.num_digits = write_index;
  trim(h);
}

template <typename binary>
adjusted_mantissa compute_float(decimal &d) {
  adjusted_mantissa answer;
  if (d.num_digits == 0) {
    // should be zero
    answer.power2 = 0;
    answer.mantissa = 0;
    return answer;
  }
  // At this point, going further, we can assume that d.num_digits > 0.
  // We want to guard against excessive decimal point values because
  // they can result in long running times. Indeed, we do
  // shifts by at most 60 bits. We have that log(10**400)/log(2**60) ~= 22
  // which is fine, but log(10**299995)/log(2**60) ~= 16609 which is not
  // fine (runs for a long time).
  //
  if (d.decimal_point < -324) {
    // We have something smaller than 1e-324 which is always zero
    // in binary64 and binary32.
    // It should be zero.
    answer.power2 = 0;
    answer.mantissa = 0;
    return answer;
  } else if (d.decimal_point >= 310) {
    // We have something at least as large as 0.1e310 which is
    // always infinite.
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
    return answer;
  }

  static const uint32_t max_shift = 60;
  static const uint32_t num_powers = 19;
  static const uint8_t powers[19] = {
      0,  3,  6,  9,  13, 16, 19, 23, 26, 29,  //
      33, 36, 39, 43, 46, 49, 53, 56, 59,      //
  };
  int32_t exp2 = 0;
  while (d.decimal_point > 0) {
    uint32_t n = uint32_t(d.decimal_point);
    uint32_t shift = (n < num_powers) ? powers[n] : max_shift;
    decimal_right_shift(d, shift);
    if (d.decimal_point < -decimal_point_range) {
      // should be zero
      answer.power2 = 0;
      answer.mantissa = 0;
      return answer;
    }
    exp2 += int32_t(shift);
  }
  // We shift left toward [1/2 ... 1].
  while (d.decimal_point <= 0) {
    uint32_t shift;
    if (d.decimal_point == 0) {
      if (d.digits[0] >= 5) {
        break;
      }
      shift = (d.digits[0] < 2) ? 2 : 1;
    } else {
      uint32_t n = uint32_t(-d.decimal_point);
      shift = (n < num_powers) ? powers[n] : max_shift;
    }
    decimal_left_shift(d, shift);
    if (d.decimal_point > decimal_point_range) {
      // we want to get infinity:
      answer.power2 = 0xFF;
      answer.mantissa = 0;
      return answer;
    }
    exp2 -= int32_t(shift);
  }
  // We are now in the range [1/2 ... 1] but the binary format uses [1 ... 2].
  exp2--;
  constexpr int32_t minimum_exponent = binary::minimum_exponent();
  while ((minimum_exponent + 1) > exp2) {
    uint32_t n = uint32_t((minimum_exponent + 1) - exp2);
    if (n > max_shift) {
      n = max_shift;
    }
    decimal_right_shift(d, n);
    exp2 += int32_t(n);
  }
  if ((exp2 - minimum_exponent) >= binary::infinite_power()) {
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
    return answer;
  }

  const int mantissa_size_in_bits = binary::mantissa_explicit_bits() + 1;
  decimal_left_shift(d, mantissa_size_in_bits);

  uint64_t mantissa = round(d);
  // It is possible that we have an overflow, in which case we need
  // to shift back.
  if (mantissa >= (uint64_t(1) << mantissa_size_in_bits)) {
    decimal_right_shift(d, 1);
    exp2 += 1;
    mantissa = round(d);
    if ((exp2 - minimum_exponent) >= binary::infinite_power()) {
      answer.power2 = binary::infinite_power();
      answer.mantissa = 0;
      return answer;
    }
  }
  answer.power2 = exp2 - binary::minimum_exponent();
  if (mantissa < (uint64_t(1) << binary::mantissa_explicit_bits())) {
    answer.power2--;
  }
  answer.mantissa =
      mantissa & ((uint64_t(1) << binary::mantissa_explicit_bits()) - 1);
  return answer;
}

template <typename binary>
adjusted_mantissa parse_long_mantissa(const char *first) {
  decimal d = parse_decimal(first);
  return compute_float<binary>(d);
}

template <typename binary>
adjusted_mantissa parse_long_mantissa(const char *first, const char *end) {
  decimal d = parse_decimal(first, end);
  return compute_float<binary>(d);
}

double from_chars(const char *first) noexcept {
  bool negative = first[0] == '-';
  if (negative) {
    first++;
  }
  adjusted_mantissa am = parse_long_mantissa<binary_format<double>>(first);
  uint64_t word = am.mantissa;
  word |= uint64_t(am.power2)
          << binary_format<double>::mantissa_explicit_bits();
  word = negative ? word | (uint64_t(1) << binary_format<double>::sign_index())
                  : word;
  double value;
  std::memcpy(&value, &word, sizeof(double));
  return value;
}

double from_chars(const char *first, const char *end) noexcept {
  bool negative = first[0] == '-';
  if (negative) {
    first++;
  }
  adjusted_mantissa am = parse_long_mantissa<binary_format<double>>(first, end);
  uint64_t word = am.mantissa;
  word |= uint64_t(am.power2)
          << binary_format<double>::mantissa_explicit_bits();
  word = negative ? word | (uint64_t(1) << binary_format<double>::sign_index())
                  : word;
  double value;
  std::memcpy(&value, &word, sizeof(double));
  return value;
}

}  // namespace internal
}  // namespace simdjson
}  // namespace minijson

namespace minijson {
namespace simdjson {
namespace internal {
/*!
implements the Grisu2 algorithm for binary to decimal floating-point
conversion.
Adapted from JSON for Modern C++

This implementation is a slightly modified version of the reference
implementation which may be obtained from
http://florian.loitsch.com/publications (bench.tar.gz).
The code is distributed under the MIT license, Copyright (c) 2009 Florian
Loitsch. For a detailed description of the algorithm see: [1] Loitsch, "Printing
Floating-Point Numbers Quickly and Accurately with Integers", Proceedings of the
ACM SIGPLAN 2010 Conference on Programming Language Design and Implementation,
PLDI 2010 [2] Burger, Dybvig, "Printing Floating-Point Numbers Quickly and
Accurately", Proceedings of the ACM SIGPLAN 1996 Conference on Programming
Language Design and Implementation, PLDI 1996
*/
namespace dtoa_impl {

template <typename Target, typename Source>
Target reinterpret_bits(const Source source) {
  static_assert(sizeof(Target) == sizeof(Source), "size mismatch");

  Target target;
  std::memcpy(&target, &source, sizeof(Source));
  return target;
}

struct diyfp  // f * 2^e
{
  static constexpr int kPrecision = 64;  // = q

  std::uint64_t f = 0;
  int e = 0;

  constexpr diyfp(std::uint64_t f_, int e_) noexcept : f(f_), e(e_) {}

  /*!
  @brief returns x - y
  @pre x.e == y.e and x.f >= y.f
  */
  static diyfp sub(const diyfp &x, const diyfp &y) noexcept {
    return {x.f - y.f, x.e};
  }

  /*!
  @brief returns x * y
  @note The result is rounded. (Only the upper q bits are returned.)
  */
  static diyfp mul(const diyfp &x, const diyfp &y) noexcept {
    static_assert(kPrecision == 64, "internal error");

    // Computes:
    //  f = round((x.f * y.f) / 2^q)
    //  e = x.e + y.e + q

    // Emulate the 64-bit * 64-bit multiplication:
    //
    // p = u * v
    //   = (u_lo + 2^32 u_hi) (v_lo + 2^32 v_hi)
    //   = (u_lo v_lo         ) + 2^32 ((u_lo v_hi         ) + (u_hi v_lo )) +
    //   2^64 (u_hi v_hi         ) = (p0                ) + 2^32 ((p1 ) + (p2 ))
    //   + 2^64 (p3                ) = (p0_lo + 2^32 p0_hi) + 2^32 ((p1_lo +
    //   2^32 p1_hi) + (p2_lo + 2^32 p2_hi)) + 2^64 (p3                ) =
    //   (p0_lo             ) + 2^32 (p0_hi + p1_lo + p2_lo ) + 2^64 (p1_hi +
    //   p2_hi + p3) = (p0_lo             ) + 2^32 (Q ) + 2^64 (H ) = (p0_lo ) +
    //   2^32 (Q_lo + 2^32 Q_hi                           ) + 2^64 (H )
    //
    // (Since Q might be larger than 2^32 - 1)
    //
    //   = (p0_lo + 2^32 Q_lo) + 2^64 (Q_hi + H)
    //
    // (Q_hi + H does not overflow a 64-bit int)
    //
    //   = p_lo + 2^64 p_hi

    const std::uint64_t u_lo = x.f & 0xFFFFFFFFu;
    const std::uint64_t u_hi = x.f >> 32u;
    const std::uint64_t v_lo = y.f & 0xFFFFFFFFu;
    const std::uint64_t v_hi = y.f >> 32u;

    const std::uint64_t p0 = u_lo * v_lo;
    const std::uint64_t p1 = u_lo * v_hi;
    const std::uint64_t p2 = u_hi * v_lo;
    const std::uint64_t p3 = u_hi * v_hi;

    const std::uint64_t p0_hi = p0 >> 32u;
    const std::uint64_t p1_lo = p1 & 0xFFFFFFFFu;
    const std::uint64_t p1_hi = p1 >> 32u;
    const std::uint64_t p2_lo = p2 & 0xFFFFFFFFu;
    const std::uint64_t p2_hi = p2 >> 32u;

    std::uint64_t Q = p0_hi + p1_lo + p2_lo;

    // The full product might now be computed as
    //
    // p_hi = p3 + p2_hi + p1_hi + (Q >> 32)
    // p_lo = p0_lo + (Q << 32)
    //
    // But in this particular case here, the full p_lo is not required.
    // Effectively we only need to add the highest bit in p_lo to p_hi (and
    // Q_hi + 1 does not overflow).

    Q += std::uint64_t{1} << (64u - 32u - 1u);  // round, ties up

    const std::uint64_t h = p3 + p2_hi + p1_hi + (Q >> 32u);

    return {h, x.e + y.e + 64};
  }

  /*!
  @brief normalize x such that the significand is >= 2^(q-1)
  @pre x.f != 0
  */
  static diyfp normalize(diyfp x) noexcept {
    while ((x.f >> 63u) == 0) {
      x.f <<= 1u;
      x.e--;
    }

    return x;
  }

  /*!
  @brief normalize x such that the result has the exponent E
  @pre e >= x.e and the upper e - x.e bits of x.f must be zero.
  */
  static diyfp normalize_to(const diyfp &x,
                            const int target_exponent) noexcept {
    const int delta = x.e - target_exponent;

    return {x.f << delta, target_exponent};
  }
};

struct boundaries {
  diyfp w;
  diyfp minus;
  diyfp plus;
};

/*!
Compute the (normalized) diyfp representing the input number 'value' and its
boundaries.
@pre value must be finite and positive
*/
template <typename FloatType>
boundaries compute_boundaries(FloatType value) {
  // Convert the IEEE representation into a diyfp.
  //
  // If v is denormal:
  //      value = 0.F * 2^(1 - bias) = (          F) * 2^(1 - bias - (p-1))
  // If v is normalized:
  //      value = 1.F * 2^(E - bias) = (2^(p-1) + F) * 2^(E - bias - (p-1))

  static_assert(std::numeric_limits<FloatType>::is_iec559,
                "internal error: dtoa_short requires an IEEE-754 "
                "floating-point implementation");

  constexpr int kPrecision =
      std::numeric_limits<FloatType>::digits;  // = p (includes the hidden bit)
  constexpr int kBias =
      std::numeric_limits<FloatType>::max_exponent - 1 + (kPrecision - 1);
  constexpr int kMinExp = 1 - kBias;
  constexpr std::uint64_t kHiddenBit = std::uint64_t{1}
                                       << (kPrecision - 1);  // = 2^(p-1)

  using bits_type = typename std::conditional<kPrecision == 24, std::uint32_t,
                                              std::uint64_t>::type;

  const std::uint64_t bits = reinterpret_bits<bits_type>(value);
  const std::uint64_t E = bits >> (kPrecision - 1);
  const std::uint64_t F = bits & (kHiddenBit - 1);

  const bool is_denormal = E == 0;
  const diyfp v = is_denormal
                      ? diyfp(F, kMinExp)
                      : diyfp(F + kHiddenBit, static_cast<int>(E) - kBias);

  // Compute the boundaries m- and m+ of the floating-point value
  // v = f * 2^e.
  //
  // Determine v- and v+, the floating-point predecessor and successor if v,
  // respectively.
  //
  //      v- = v - 2^e        if f != 2^(p-1) or e == e_min                (A)
  //         = v - 2^(e-1)    if f == 2^(p-1) and e > e_min                (B)
  //
  //      v+ = v + 2^e
  //
  // Let m- = (v- + v) / 2 and m+ = (v + v+) / 2. All real numbers _strictly_
  // between m- and m+ round to v, regardless of how the input rounding
  // algorithm breaks ties.
  //
  //      ---+-------------+-------------+-------------+-------------+---  (A)
  //         v-            m-            v             m+            v+
  //
  //      -----------------+------+------+-------------+-------------+---  (B)
  //                       v-     m-     v             m+            v+

  const bool lower_boundary_is_closer = F == 0 && E > 1;
  const diyfp m_plus = diyfp(2 * v.f + 1, v.e - 1);
  const diyfp m_minus = lower_boundary_is_closer
                            ? diyfp(4 * v.f - 1, v.e - 2)   // (B)
                            : diyfp(2 * v.f - 1, v.e - 1);  // (A)

  // Determine the normalized w+ = m+.
  const diyfp w_plus = diyfp::normalize(m_plus);

  // Determine w- = m- such that e_(w-) = e_(w+).
  const diyfp w_minus = diyfp::normalize_to(m_minus, w_plus.e);

  return {diyfp::normalize(v), w_minus, w_plus};
}

// Given normalized diyfp w, Grisu needs to find a (normalized) cached
// power-of-ten c, such that the exponent of the product c * w = f * 2^e lies
// within a certain range [alpha, gamma] (Definition 3.2 from [1])
//
//      alpha <= e = e_c + e_w + q <= gamma
//
// or
//
//      f_c * f_w * 2^alpha <= f_c 2^(e_c) * f_w 2^(e_w) * 2^q
//                          <= f_c * f_w * 2^gamma
//
// Since c and w are normalized, i.e. 2^(q-1) <= f < 2^q, this implies
//
//      2^(q-1) * 2^(q-1) * 2^alpha <= c * w * 2^q < 2^q * 2^q * 2^gamma
//
// or
//
//      2^(q - 2 + alpha) <= c * w < 2^(q + gamma)
//
// The choice of (alpha,gamma) determines the size of the table and the form of
// the digit generation procedure. Using (alpha,gamma)=(-60,-32) works out well
// in practice:
//
// The idea is to cut the number c * w = f * 2^e into two parts, which can be
// processed independently: An integral part p1, and a fractional part p2:
//
//      f * 2^e = ( (f div 2^-e) * 2^-e + (f mod 2^-e) ) * 2^e
//              = (f div 2^-e) + (f mod 2^-e) * 2^e
//              = p1 + p2 * 2^e
//
// The conversion of p1 into decimal form requires a series of divisions and
// modulos by (a power of) 10. These operations are faster for 32-bit than for
// 64-bit integers, so p1 should ideally fit into a 32-bit integer. This can be
// achieved by choosing
//
//      -e >= 32   or   e <= -32 := gamma
//
// In order to convert the fractional part
//
//      p2 * 2^e = p2 / 2^-e = d[-1] / 10^1 + d[-2] / 10^2 + ...
//
// into decimal form, the fraction is repeatedly multiplied by 10 and the digits
// d[-i] are extracted in order:
//
//      (10 * p2) div 2^-e = d[-1]
//      (10 * p2) mod 2^-e = d[-2] / 10^1 + ...
//
// The multiplication by 10 must not overflow. It is sufficient to choose
//
//      10 * p2 < 16 * p2 = 2^4 * p2 <= 2^64.
//
// Since p2 = f mod 2^-e < 2^-e,
//
//      -e <= 60   or   e >= -60 := alpha

constexpr int kAlpha = -60;
constexpr int kGamma = -32;

struct cached_power  // c = f * 2^e ~= 10^k
{
  std::uint64_t f;
  int e;
  int k;
};

/*!
For a normalized diyfp w = f * 2^e, this function returns a (normalized) cached
power-of-ten c = f_c * 2^e_c, such that the exponent of the product w * c
satisfies (Definition 3.2 from [1])
     alpha <= e_c + e + q <= gamma.
*/
inline cached_power get_cached_power_for_binary_exponent(int e) {
  // Now
  //
  //      alpha <= e_c + e + q <= gamma                                    (1)
  //      ==> f_c * 2^alpha <= c * 2^e * 2^q
  //
  // and since the c's are normalized, 2^(q-1) <= f_c,
  //
  //      ==> 2^(q - 1 + alpha) <= c * 2^(e + q)
  //      ==> 2^(alpha - e - 1) <= c
  //
  // If c were an exact power of ten, i.e. c = 10^k, one may determine k as
  //
  //      k = ceil( log_10( 2^(alpha - e - 1) ) )
  //        = ceil( (alpha - e - 1) * log_10(2) )
  //
  // From the paper:
  // "In theory the result of the procedure could be wrong since c is rounded,
  //  and the computation itself is approximated [...]. In practice, however,
  //  this simple function is sufficient."
  //
  // For IEEE double precision floating-point numbers converted into
  // normalized diyfp's w = f * 2^e, with q = 64,
  //
  //      e >= -1022      (min IEEE exponent)
  //           -52        (p - 1)
  //           -52        (p - 1, possibly normalize denormal IEEE numbers)
  //           -11        (normalize the diyfp)
  //         = -1137
  //
  // and
  //
  //      e <= +1023      (max IEEE exponent)
  //           -52        (p - 1)
  //           -11        (normalize the diyfp)
  //         = 960
  //
  // This binary exponent range [-1137,960] results in a decimal exponent
  // range [-307,324]. One does not need to store a cached power for each
  // k in this range. For each such k it suffices to find a cached power
  // such that the exponent of the product lies in [alpha,gamma].
  // This implies that the difference of the decimal exponents of adjacent
  // table entries must be less than or equal to
  //
  //      floor( (gamma - alpha) * log_10(2) ) = 8.
  //
  // (A smaller distance gamma-alpha would require a larger table.)

  // NB:
  // Actually this function returns c, such that -60 <= e_c + e + 64 <= -34.

  constexpr int kCachedPowersMinDecExp = -300;
  constexpr int kCachedPowersDecStep = 8;

  static constexpr std::array<cached_power, 79> kCachedPowers = {{
      {0xAB70FE17C79AC6CA, -1060, -300}, {0xFF77B1FCBEBCDC4F, -1034, -292},
      {0xBE5691EF416BD60C, -1007, -284}, {0x8DD01FAD907FFC3C, -980, -276},
      {0xD3515C2831559A83, -954, -268},  {0x9D71AC8FADA6C9B5, -927, -260},
      {0xEA9C227723EE8BCB, -901, -252},  {0xAECC49914078536D, -874, -244},
      {0x823C12795DB6CE57, -847, -236},  {0xC21094364DFB5637, -821, -228},
      {0x9096EA6F3848984F, -794, -220},  {0xD77485CB25823AC7, -768, -212},
      {0xA086CFCD97BF97F4, -741, -204},  {0xEF340A98172AACE5, -715, -196},
      {0xB23867FB2A35B28E, -688, -188},  {0x84C8D4DFD2C63F3B, -661, -180},
      {0xC5DD44271AD3CDBA, -635, -172},  {0x936B9FCEBB25C996, -608, -164},
      {0xDBAC6C247D62A584, -582, -156},  {0xA3AB66580D5FDAF6, -555, -148},
      {0xF3E2F893DEC3F126, -529, -140},  {0xB5B5ADA8AAFF80B8, -502, -132},
      {0x87625F056C7C4A8B, -475, -124},  {0xC9BCFF6034C13053, -449, -116},
      {0x964E858C91BA2655, -422, -108},  {0xDFF9772470297EBD, -396, -100},
      {0xA6DFBD9FB8E5B88F, -369, -92},   {0xF8A95FCF88747D94, -343, -84},
      {0xB94470938FA89BCF, -316, -76},   {0x8A08F0F8BF0F156B, -289, -68},
      {0xCDB02555653131B6, -263, -60},   {0x993FE2C6D07B7FAC, -236, -52},
      {0xE45C10C42A2B3B06, -210, -44},   {0xAA242499697392D3, -183, -36},
      {0xFD87B5F28300CA0E, -157, -28},   {0xBCE5086492111AEB, -130, -20},
      {0x8CBCCC096F5088CC, -103, -12},   {0xD1B71758E219652C, -77, -4},
      {0x9C40000000000000, -50, 4},      {0xE8D4A51000000000, -24, 12},
      {0xAD78EBC5AC620000, 3, 20},       {0x813F3978F8940984, 30, 28},
      {0xC097CE7BC90715B3, 56, 36},      {0x8F7E32CE7BEA5C70, 83, 44},
      {0xD5D238A4ABE98068, 109, 52},     {0x9F4F2726179A2245, 136, 60},
      {0xED63A231D4C4FB27, 162, 68},     {0xB0DE65388CC8ADA8, 189, 76},
      {0x83C7088E1AAB65DB, 216, 84},     {0xC45D1DF942711D9A, 242, 92},
      {0x924D692CA61BE758, 269, 100},    {0xDA01EE641A708DEA, 295, 108},
      {0xA26DA3999AEF774A, 322, 116},    {0xF209787BB47D6B85, 348, 124},
      {0xB454E4A179DD1877, 375, 132},    {0x865B86925B9BC5C2, 402, 140},
      {0xC83553C5C8965D3D, 428, 148},    {0x952AB45CFA97A0B3, 455, 156},
      {0xDE469FBD99A05FE3, 481, 164},    {0xA59BC234DB398C25, 508, 172},
      {0xF6C69A72A3989F5C, 534, 180},    {0xB7DCBF5354E9BECE, 561, 188},
      {0x88FCF317F22241E2, 588, 196},    {0xCC20CE9BD35C78A5, 614, 204},
      {0x98165AF37B2153DF, 641, 212},    {0xE2A0B5DC971F303A, 667, 220},
      {0xA8D9D1535CE3B396, 694, 228},    {0xFB9B7CD9A4A7443C, 720, 236},
      {0xBB764C4CA7A44410, 747, 244},    {0x8BAB8EEFB6409C1A, 774, 252},
      {0xD01FEF10A657842C, 800, 260},    {0x9B10A4E5E9913129, 827, 268},
      {0xE7109BFBA19C0C9D, 853, 276},    {0xAC2820D9623BF429, 880, 284},
      {0x80444B5E7AA7CF85, 907, 292},    {0xBF21E44003ACDD2D, 933, 300},
      {0x8E679C2F5E44FF8F, 960, 308},    {0xD433179D9C8CB841, 986, 316},
      {0x9E19DB92B4E31BA9, 1013, 324},
  }};

  // This computation gives exactly the same results for k as
  //      k = ceil((kAlpha - e - 1) * 0.30102999566398114)
  // for |e| <= 1500, but doesn't require floating-point operations.
  // NB: log_10(2) ~= 78913 / 2^18
  const int f = kAlpha - e - 1;
  const int k = (f * 78913) / (1 << 18) + static_cast<int>(f > 0);

  const int index = (-kCachedPowersMinDecExp + k + (kCachedPowersDecStep - 1)) /
                    kCachedPowersDecStep;

  const cached_power cached = kCachedPowers[static_cast<std::size_t>(index)];

  return cached;
}

/*!
For n != 0, returns k, such that pow10 := 10^(k-1) <= n < 10^k.
For n == 0, returns 1 and sets pow10 := 1.
*/
inline int find_largest_pow10(const std::uint32_t n, std::uint32_t &pow10) {
  // LCOV_EXCL_START
  if (n >= 1000000000) {
    pow10 = 1000000000;
    return 10;
  }
  // LCOV_EXCL_STOP
  else if (n >= 100000000) {
    pow10 = 100000000;
    return 9;
  } else if (n >= 10000000) {
    pow10 = 10000000;
    return 8;
  } else if (n >= 1000000) {
    pow10 = 1000000;
    return 7;
  } else if (n >= 100000) {
    pow10 = 100000;
    return 6;
  } else if (n >= 10000) {
    pow10 = 10000;
    return 5;
  } else if (n >= 1000) {
    pow10 = 1000;
    return 4;
  } else if (n >= 100) {
    pow10 = 100;
    return 3;
  } else if (n >= 10) {
    pow10 = 10;
    return 2;
  } else {
    pow10 = 1;
    return 1;
  }
}

inline void grisu2_round(char *buf, int len, std::uint64_t dist,
                         std::uint64_t delta, std::uint64_t rest,
                         std::uint64_t ten_k) {
  //               <--------------------------- delta ---->
  //                                  <---- dist --------->
  // --------------[------------------+-------------------]--------------
  //               M-                 w                   M+
  //
  //                                  ten_k
  //                                <------>
  //                                       <---- rest ---->
  // --------------[------------------+----+--------------]--------------
  //                                  w    V
  //                                       = buf * 10^k
  //
  // ten_k represents a unit-in-the-last-place in the decimal representation
  // stored in buf.
  // Decrement buf by ten_k while this takes buf closer to w.

  // The tests are written in this order to avoid overflow in unsigned
  // integer arithmetic.

  while (rest < dist && delta - rest >= ten_k &&
         (rest + ten_k < dist || dist - rest > rest + ten_k - dist)) {
    buf[len - 1]--;
    rest += ten_k;
  }
}

/*!
Generates V = buffer * 10^decimal_exponent, such that M- <= V <= M+.
M- and M+ must be normalized and share the same exponent -60 <= e <= -32.
*/
inline void grisu2_digit_gen(char *buffer, int &length, int &decimal_exponent,
                             diyfp M_minus, diyfp w, diyfp M_plus) {
  static_assert(kAlpha >= -60, "internal error");
  static_assert(kGamma <= -32, "internal error");

  // Generates the digits (and the exponent) of a decimal floating-point
  // number V = buffer * 10^decimal_exponent in the range [M-, M+]. The diyfp's
  // w, M- and M+ share the same exponent e, which satisfies alpha <= e <=
  // gamma.
  //
  //               <--------------------------- delta ---->
  //                                  <---- dist --------->
  // --------------[------------------+-------------------]--------------
  //               M-                 w                   M+
  //
  // Grisu2 generates the digits of M+ from left to right and stops as soon as
  // V is in [M-,M+].

  std::uint64_t delta =
      diyfp::sub(M_plus, M_minus)
          .f;  // (significand of (M+ - M-), implicit exponent is e)
  std::uint64_t dist =
      diyfp::sub(M_plus, w)
          .f;  // (significand of (M+ - w ), implicit exponent is e)

  // Split M+ = f * 2^e into two parts p1 and p2 (note: e < 0):
  //
  //      M+ = f * 2^e
  //         = ((f div 2^-e) * 2^-e + (f mod 2^-e)) * 2^e
  //         = ((p1        ) * 2^-e + (p2        )) * 2^e
  //         = p1 + p2 * 2^e

  const diyfp one(std::uint64_t{1} << -M_plus.e, M_plus.e);

  auto p1 = static_cast<std::uint32_t>(
      M_plus.f >>
      -one.e);  // p1 = f div 2^-e (Since -e >= 32, p1 fits into a 32-bit int.)
  std::uint64_t p2 = M_plus.f & (one.f - 1);  // p2 = f mod 2^-e

  // 1)
  //
  // Generate the digits of the integral part p1 = d[n-1]...d[1]d[0]

  std::uint32_t pow10;
  const int k = find_largest_pow10(p1, pow10);

  //      10^(k-1) <= p1 < 10^k, pow10 = 10^(k-1)
  //
  //      p1 = (p1 div 10^(k-1)) * 10^(k-1) + (p1 mod 10^(k-1))
  //         = (d[k-1]         ) * 10^(k-1) + (p1 mod 10^(k-1))
  //
  //      M+ = p1                                             + p2 * 2^e
  //         = d[k-1] * 10^(k-1) + (p1 mod 10^(k-1))          + p2 * 2^e
  //         = d[k-1] * 10^(k-1) + ((p1 mod 10^(k-1)) * 2^-e + p2) * 2^e
  //         = d[k-1] * 10^(k-1) + (                         rest) * 2^e
  //
  // Now generate the digits d[n] of p1 from left to right (n = k-1,...,0)
  //
  //      p1 = d[k-1]...d[n] * 10^n + d[n-1]...d[0]
  //
  // but stop as soon as
  //
  //      rest * 2^e = (d[n-1]...d[0] * 2^-e + p2) * 2^e <= delta * 2^e

  int n = k;
  while (n > 0) {
    // Invariants:
    //      M+ = buffer * 10^n + (p1 + p2 * 2^e)    (buffer = 0 for n = k)
    //      pow10 = 10^(n-1) <= p1 < 10^n
    //
    const std::uint32_t d = p1 / pow10;  // d = p1 div 10^(n-1)
    const std::uint32_t r = p1 % pow10;  // r = p1 mod 10^(n-1)
    //
    //      M+ = buffer * 10^n + (d * 10^(n-1) + r) + p2 * 2^e
    //         = (buffer * 10 + d) * 10^(n-1) + (r + p2 * 2^e)
    //
    buffer[length++] = static_cast<char>('0' + d);  // buffer := buffer * 10 + d
    //
    //      M+ = buffer * 10^(n-1) + (r + p2 * 2^e)
    //
    p1 = r;
    n--;
    //
    //      M+ = buffer * 10^n + (p1 + p2 * 2^e)
    //      pow10 = 10^n
    //

    // Now check if enough digits have been generated.
    // Compute
    //
    //      p1 + p2 * 2^e = (p1 * 2^-e + p2) * 2^e = rest * 2^e
    //
    // Note:
    // Since rest and delta share the same exponent e, it suffices to
    // compare the significands.
    const std::uint64_t rest = (std::uint64_t{p1} << -one.e) + p2;
    if (rest <= delta) {
      // V = buffer * 10^n, with M- <= V <= M+.

      decimal_exponent += n;

      // We may now just stop. But instead look if the buffer could be
      // decremented to bring V closer to w.
      //
      // pow10 = 10^n is now 1 ulp in the decimal representation V.
      // The rounding procedure works with diyfp's with an implicit
      // exponent of e.
      //
      //      10^n = (10^n * 2^-e) * 2^e = ulp * 2^e
      //
      const std::uint64_t ten_n = std::uint64_t{pow10} << -one.e;
      grisu2_round(buffer, length, dist, delta, rest, ten_n);

      return;
    }

    pow10 /= 10;
    //
    //      pow10 = 10^(n-1) <= p1 < 10^n
    // Invariants restored.
  }

  // 2)
  //
  // The digits of the integral part have been generated:
  //
  //      M+ = d[k-1]...d[1]d[0] + p2 * 2^e
  //         = buffer            + p2 * 2^e
  //
  // Now generate the digits of the fractional part p2 * 2^e.
  //
  // Note:
  // No decimal point is generated: the exponent is adjusted instead.
  //
  // p2 actually represents the fraction
  //
  //      p2 * 2^e
  //          = p2 / 2^-e
  //          = d[-1] / 10^1 + d[-2] / 10^2 + ...
  //
  // Now generate the digits d[-m] of p1 from left to right (m = 1,2,...)
  //
  //      p2 * 2^e = d[-1]d[-2]...d[-m] * 10^-m
  //                      + 10^-m * (d[-m-1] / 10^1 + d[-m-2] / 10^2 + ...)
  //
  // using
  //
  //      10^m * p2 = ((10^m * p2) div 2^-e) * 2^-e + ((10^m * p2) mod 2^-e)
  //                = (                   d) * 2^-e + (                   r)
  //
  // or
  //      10^m * p2 * 2^e = d + r * 2^e
  //
  // i.e.
  //
  //      M+ = buffer + p2 * 2^e
  //         = buffer + 10^-m * (d + r * 2^e)
  //         = (buffer * 10^m + d) * 10^-m + 10^-m * r * 2^e
  //
  // and stop as soon as 10^-m * r * 2^e <= delta * 2^e

  int m = 0;
  for (;;) {
    // Invariant:
    //      M+ = buffer * 10^-m + 10^-m * (d[-m-1] / 10 + d[-m-2] / 10^2 + ...)
    //      * 2^e
    //         = buffer * 10^-m + 10^-m * (p2                                 )
    //         * 2^e = buffer * 10^-m + 10^-m * (1/10 * (10 * p2) ) * 2^e =
    //         buffer * 10^-m + 10^-m * (1/10 * ((10*p2 div 2^-e) * 2^-e +
    //         (10*p2 mod 2^-e)) * 2^e
    //
    p2 *= 10;
    const std::uint64_t d = p2 >> -one.e;      // d = (10 * p2) div 2^-e
    const std::uint64_t r = p2 & (one.f - 1);  // r = (10 * p2) mod 2^-e
    //
    //      M+ = buffer * 10^-m + 10^-m * (1/10 * (d * 2^-e + r) * 2^e
    //         = buffer * 10^-m + 10^-m * (1/10 * (d + r * 2^e))
    //         = (buffer * 10 + d) * 10^(-m-1) + 10^(-m-1) * r * 2^e
    //
    buffer[length++] = static_cast<char>('0' + d);  // buffer := buffer * 10 + d
    //
    //      M+ = buffer * 10^(-m-1) + 10^(-m-1) * r * 2^e
    //
    p2 = r;
    m++;
    //
    //      M+ = buffer * 10^-m + 10^-m * p2 * 2^e
    // Invariant restored.

    // Check if enough digits have been generated.
    //
    //      10^-m * p2 * 2^e <= delta * 2^e
    //              p2 * 2^e <= 10^m * delta * 2^e
    //                    p2 <= 10^m * delta
    delta *= 10;
    dist *= 10;
    if (p2 <= delta) {
      break;
    }
  }

  // V = buffer * 10^-m, with M- <= V <= M+.

  decimal_exponent -= m;

  // 1 ulp in the decimal representation is now 10^-m.
  // Since delta and dist are now scaled by 10^m, we need to do the
  // same with ulp in order to keep the units in sync.
  //
  //      10^m * 10^-m = 1 = 2^-e * 2^e = ten_m * 2^e
  //
  const std::uint64_t ten_m = one.f;
  grisu2_round(buffer, length, dist, delta, p2, ten_m);

  // By construction this algorithm generates the shortest possible decimal
  // number (Loitsch, Theorem 6.2) which rounds back to w.
  // For an input number of precision p, at least
  //
  //      N = 1 + ceil(p * log_10(2))
  //
  // decimal digits are sufficient to identify all binary floating-point
  // numbers (Matula, "In-and-Out conversions").
  // This implies that the algorithm does not produce more than N decimal
  // digits.
  //
  //      N = 17 for p = 53 (IEEE double precision)
  //      N = 9  for p = 24 (IEEE single precision)
}

/*!
v = buf * 10^decimal_exponent
len is the length of the buffer (number of decimal digits)
The buffer must be large enough, i.e. >= max_digits10.
*/
inline void grisu2(char *buf, int &len, int &decimal_exponent, diyfp m_minus,
                   diyfp v, diyfp m_plus) {
  //  --------(-----------------------+-----------------------)--------    (A)
  //          m-                      v                       m+
  //
  //  --------------------(-----------+-----------------------)--------    (B)
  //                      m-          v                       m+
  //
  // First scale v (and m- and m+) such that the exponent is in the range
  // [alpha, gamma].

  const cached_power cached = get_cached_power_for_binary_exponent(m_plus.e);

  const diyfp c_minus_k(cached.f, cached.e);  // = c ~= 10^-k

  // The exponent of the products is = v.e + c_minus_k.e + q and is in the range
  // [alpha,gamma]
  const diyfp w = diyfp::mul(v, c_minus_k);
  const diyfp w_minus = diyfp::mul(m_minus, c_minus_k);
  const diyfp w_plus = diyfp::mul(m_plus, c_minus_k);

  //  ----(---+---)---------------(---+---)---------------(---+---)----
  //          w-                      w                       w+
  //          = c*m-                  = c*v                   = c*m+
  //
  // diyfp::mul rounds its result and c_minus_k is approximated too. w, w- and
  // w+ are now off by a small amount.
  // In fact:
  //
  //      w - v * 10^k < 1 ulp
  //
  // To account for this inaccuracy, add resp. subtract 1 ulp.
  //
  //  --------+---[---------------(---+---)---------------]---+--------
  //          w-  M-                  w                   M+  w+
  //
  // Now any number in [M-, M+] (bounds included) will round to w when input,
  // regardless of how the input rounding algorithm breaks ties.
  //
  // And digit_gen generates the shortest possible such number in [M-, M+].
  // Note that this does not mean that Grisu2 always generates the shortest
  // possible number in the interval (m-, m+).
  const diyfp M_minus(w_minus.f + 1, w_minus.e);
  const diyfp M_plus(w_plus.f - 1, w_plus.e);

  decimal_exponent = -cached.k;  // = -(-k) = k

  grisu2_digit_gen(buf, len, decimal_exponent, M_minus, w, M_plus);
}

/*!
v = buf * 10^decimal_exponent
len is the length of the buffer (number of decimal digits)
The buffer must be large enough, i.e. >= max_digits10.
*/
template <typename FloatType>
void grisu2(char *buf, int &len, int &decimal_exponent, FloatType value) {
  static_assert(diyfp::kPrecision >= std::numeric_limits<FloatType>::digits + 3,
                "internal error: not enough precision");

  // If the neighbors (and boundaries) of 'value' are always computed for
  // double-precision numbers, all float's can be recovered using strtod (and
  // strtof). However, the resulting decimal representations are not exactly
  // "short".
  //
  // The documentation for 'std::to_chars'
  // (https://en.cppreference.com/w/cpp/utility/to_chars) says "value is
  // converted to a string as if by std::sprintf in the default ("C") locale"
  // and since sprintf promotes float's to double's, I think this is exactly
  // what 'std::to_chars' does. On the other hand, the documentation for
  // 'std::to_chars' requires that "parsing the representation using the
  // corresponding std::from_chars function recovers value exactly". That
  // indicates that single precision floating-point numbers should be recovered
  // using 'std::strtof'.
  //
  // NB: If the neighbors are computed for single-precision numbers, there is a
  // single float
  //     (7.0385307e-26f) which can't be recovered using strtod. The resulting
  //     double precision value is off by 1 ulp.
#if 0
    const boundaries w = compute_boundaries(static_cast<double>(value));
#else
  const boundaries w = compute_boundaries(value);
#endif

  grisu2(buf, len, decimal_exponent, w.minus, w.w, w.plus);
}

/*!
@brief appends a decimal representation of e to buf
@return a pointer to the element following the exponent.
@pre -1000 < e < 1000
*/
inline char *append_exponent(char *buf, int e) {
  if (e < 0) {
    e = -e;
    *buf++ = '-';
  } else {
    *buf++ = '+';
  }

  auto k = static_cast<std::uint32_t>(e);
  if (k < 10) {
    // Always print at least two digits in the exponent.
    // This is for compatibility with printf("%g").
    *buf++ = '0';
    *buf++ = static_cast<char>('0' + k);
  } else if (k < 100) {
    *buf++ = static_cast<char>('0' + k / 10);
    k %= 10;
    *buf++ = static_cast<char>('0' + k);
  } else {
    *buf++ = static_cast<char>('0' + k / 100);
    k %= 100;
    *buf++ = static_cast<char>('0' + k / 10);
    k %= 10;
    *buf++ = static_cast<char>('0' + k);
  }

  return buf;
}

/*!
@brief prettify v = buf * 10^decimal_exponent
If v is in the range [10^min_exp, 10^max_exp) it will be printed in fixed-point
notation. Otherwise it will be printed in exponential notation.
@pre min_exp < 0
@pre max_exp > 0
*/
inline char *format_buffer(char *buf, int len, int decimal_exponent,
                           int min_exp, int max_exp) {
  const int k = len;
  const int n = len + decimal_exponent;

  // v = buf * 10^(n-k)
  // k is the length of the buffer (number of decimal digits)
  // n is the position of the decimal point relative to the start of the buffer.

  if (k <= n && n <= max_exp) {
    // digits[000]
    // len <= max_exp + 2

    std::memset(buf + k, '0', static_cast<size_t>(n) - static_cast<size_t>(k));
    // Make it look like a floating-point number (#362, #378)
    buf[n + 0] = '.';
    buf[n + 1] = '0';
    return buf + (static_cast<size_t>(n)) + 2;
  }

  if (0 < n && n <= max_exp) {
    // dig.its
    // len <= max_digits10 + 1
    std::memmove(buf + (static_cast<size_t>(n) + 1), buf + n,
                 static_cast<size_t>(k) - static_cast<size_t>(n));
    buf[n] = '.';
    return buf + (static_cast<size_t>(k) + 1U);
  }

  if (min_exp < n && n <= 0) {
    // 0.[000]digits
    // len <= 2 + (-min_exp - 1) + max_digits10

    std::memmove(buf + (2 + static_cast<size_t>(-n)), buf,
                 static_cast<size_t>(k));
    buf[0] = '0';
    buf[1] = '.';
    std::memset(buf + 2, '0', static_cast<size_t>(-n));
    return buf + (2U + static_cast<size_t>(-n) + static_cast<size_t>(k));
  }

  if (k == 1) {
    // dE+123
    // len <= 1 + 5

    buf += 1;
  } else {
    // d.igitsE+123
    // len <= max_digits10 + 1 + 5

    std::memmove(buf + 2, buf + 1, static_cast<size_t>(k) - 1);
    buf[1] = '.';
    buf += 1 + static_cast<size_t>(k);
  }

  *buf++ = 'e';
  return append_exponent(buf, n - 1);
}

}  // namespace dtoa_impl

/*!
The format of the resulting decimal representation is similar to printf's %g
format. Returns an iterator pointing past-the-end of the decimal representation.
@note The input number must be finite, i.e. NaN's and Inf's are not supported.
@note The buffer must be large enough.
@note The result is NOT null-terminated.
*/
char *to_chars(char *first, const char *last, double value) {
  static_cast<void>(last);  // maybe unused - fix warning

  // bool negative = std::signbit(value);
  bool negative = (*reinterpret_cast<uint64_t *>(&value)) & (1 << 31ull);
  if (negative) {
    value = -value;
    *first++ = '-';
  }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

  if (value == 0)  // +-0
  {
    *first++ = '0';
    // Make it look like a floating-point number (#362, #378)
    *first++ = '.';
    *first++ = '0';
    return first;
  }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

  // Compute v = buffer * 10^decimal_exponent.
  // The decimal digits are stored in the buffer, which needs to be interpreted
  // as an unsigned decimal integer.
  // len is the length of the buffer, i.e. the number of decimal digits.
  int len = 0;
  int decimal_exponent = 0;
  dtoa_impl::grisu2(first, len, decimal_exponent, value);
  // Format the buffer like printf("%.*g", prec, value)
  constexpr int kMinExp = -4;
  constexpr int kMaxExp = std::numeric_limits<double>::digits10;

  return dtoa_impl::format_buffer(first, len, decimal_exponent, kMinExp,
                                  kMaxExp);
}
}  // namespace internal
}  // namespace simdjson
}  // namespace minijson

#endif  // !MINIJSON_USE_STRTOD

#endif  // MINIJSON_IMPLEMENTATION


namespace safetensors {

// Max header(JSON) size. 100 MB as done in original safetensors implementation.
constexpr size_t kMaxJSONSize = 1024ull * 1024ull * 100ull;

namespace detail {

#ifdef _WIN32
std::wstring UTF8ToWchar(const std::string &str) {
  int wstr_size =
      MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), nullptr, 0);
  std::wstring wstr(size_t(wstr_size), 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), &wstr[0],
                      int(wstr.size()));
  return wstr;
}

std::string WcharToUTF8(const std::wstring &wstr) {
  int str_size = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()),
                                     nullptr, 0, nullptr, nullptr);
  std::string str(size_t(str_size), 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()), &str[0],
                      int(str.size()), nullptr, nullptr);
  return str;
}
#endif

bool ReadWholeFile(std::vector<unsigned char> *out, std::string *err,
                   const std::string &filepath, void *) {
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
  if (asset_manager) {
    AAsset *asset = AAssetManager_open(asset_manager, filepath.c_str(),
                                       AASSET_MODE_STREAMING);
    if (!asset) {
      if (err) {
        (*err) += "File open error : " + filepath + "\n";
      }
      return false;
    }
    size_t size = AAsset_getLength(asset);
    if (size == 0) {
      if (err) {
        (*err) += "Invalid file size : " + filepath +
                  " (does the path point to a directory?)";
      }
      return false;
    }
    out->resize(size);
    AAsset_read(asset, reinterpret_cast<char *>(&out->at(0)), size);
    AAsset_close(asset);
    return true;
  } else {
    if (err) {
      (*err) += "No asset manager specified : " + filepath + "\n";
    }
    return false;
  }
#else
#ifdef _WIN32
#if defined(__GLIBCXX__)  // mingw
  int file_descriptor =
      _wopen(UTF8ToWchar(filepath).c_str(), _O_RDONLY | _O_BINARY);
  __gnu_cxx::stdio_filebuf<char> wfile_buf(file_descriptor, std::ios_base::in);
  std::istream f(&wfile_buf);
#elif defined(_MSC_VER) || defined(_LIBCPP_VERSION)
  // For libcxx, assume _LIBCPP_HAS_OPEN_WITH_WCHAR is defined to accept
  // `wchar_t *`
  std::ifstream f(UTF8ToWchar(filepath).c_str(), std::ifstream::binary);
#else
  // Unknown compiler/runtime
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
#else
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
  if (!f) {
    if (err) {
      (*err) += "File open error : " + filepath + "\n";
    }
    return false;
  }

  // For directory(and pipe?), peek() will fail(Posix gnustl/libc++ only)
  f.peek();
  if (!f) {
    if (err) {
      (*err) +=
          "File read error. Maybe empty file or invalid file : " + filepath +
          "\n";
    }
    return false;
  }

  f.seekg(0, f.end);
  size_t sz = static_cast<size_t>(f.tellg());

  // std::cout << "sz = " << sz << "\n";
  f.seekg(0, f.beg);

  if (int64_t(sz) < 0) {
    if (err) {
      (*err) += "Invalid file size : " + filepath +
                " (does the path point to a directory?)";
    }
    return false;
  } else if (sz == 0) {
    if (err) {
      (*err) += "File is empty : " + filepath + "\n";
    }
    return false;
  } else if (sz >= (std::numeric_limits<std::streamoff>::max)()) {
    if (err) {
      (*err) += "Invalid file size : " + filepath + "\n";
    }
    return false;
  }

  out->resize(sz);
  f.read(reinterpret_cast<char *>(&out->at(0)),
         static_cast<std::streamsize>(sz));

  return true;
#endif
}

bool parse_metadata(const ::minijson::value &v,
                    ordered_dict<std::string> &dst, std::string *err) {
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      ::minijson::value ov;
      if (!po->at(i, &ov)) {
          if (err) {
            (*err) +=
                "[Internal error] Invalid object found in __metadata__, at index " + std::to_string(i) + ".\n";
          }
          return false;
      }

      if (auto so = ov.as<std::string>()) {
        if (dst.count(po->keys()[i])) {
          // This should not be happen though
          if (err) {
            (*err) +=
                "Duplicate key `" + po->keys()[i] + "` found in __metadata__.\n";
          }
          return false;
        }

        dst.insert(po->keys()[i], *so);
      } else {
        if (err) {
          (*err) += "`" + po->keys()[i] + "` must be string value.\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) += "`__metadata__` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

bool parse_dtype(const ::minijson::value &v, safetensors::dtype &dtype,
                 std::string *err) {
  if (auto so = v.as<std::string>()) {
    if ((*so) == "BOOL") {
      dtype = safetensors::dtype::kBOOL;
    } else if ((*so) == "U8") {
      dtype = safetensors::dtype::kUINT8;
    } else if ((*so) == "I8") {
      dtype = safetensors::dtype::kINT8;
    } else if ((*so) == "U16") {
      dtype = safetensors::dtype::kUINT16;
    } else if ((*so) == "I16") {
      dtype = safetensors::dtype::kINT16;
    } else if ((*so) == "U32") {
      dtype = safetensors::dtype::kUINT32;
    } else if ((*so) == "I32") {
      dtype = safetensors::dtype::kINT32;
    } else if ((*so) == "U64") {
      dtype = safetensors::dtype::kUINT64;
    } else if ((*so) == "I64") {
      dtype = safetensors::dtype::kINT64;
    } else if ((*so) == "F16") {
      dtype = safetensors::dtype::kFLOAT16;
    } else if ((*so) == "BF16") {
      dtype = safetensors::dtype::kBFLOAT16;
    } else if ((*so) == "F32") {
      dtype = safetensors::dtype::kFLOAT32;
    } else if ((*so) == "F64") {
      dtype = safetensors::dtype::kFLOAT64;
    } else {
      if (err) {
        (*err) += "Unknown `dtype` string: " + *so + ".\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) +=
          "`dtype` item should be string type but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_shape(const ::minijson::value &v, std::vector<size_t> &dst,
                 std::string *err) {
  // NOTE:
  // - Empty tensors (tensors with 1 dimension being 0) are allowed
  // - [] is allowed(0-Rank tensor = merely a scalar)
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (dst.size() >= kMaxDim) {
          if (err) {
            (*err) += "`shape` length must be less than " +
                      std::to_string(kMaxDim) + " but got " +
                      std::to_string(dst.size()) + ".\n";
          }
          return false;
        }

        dst.push_back(size_t(*pn));

      } else {
        if (err) {
          (*err) += "Array item in `shape` must be number type, but got " +
                    i->type_name() + ".\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) +=
          "`shape` value must be JSON array, but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_data_offsets(const ::minijson::value &v, std::array<size_t, 2> &dst,
                        std::string *err) {
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;
    size_t cnt = 0;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (cnt >= 2) {
          if (err) {
            (*err) += "`data_offsets` length must be 2.\n";
          }
          return false;
        }

        dst[cnt] = size_t(*pn);

        cnt++;

      } else {
        if (err) {
          (*err) +=
              "Array item in `data_offsets` must be number type, but got " +
              i->type_name() + ".\n";
        }
        return false;
      }
    }

    if (cnt != 2) {
      if (err) {
        (*err) += "`data_offsets` length must be 2.\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) += "`data_offsets` value must be JSON array, but got " +
                v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_tensor(const std::string &name, const ::minijson::value &v,
                  tensor_t &tensor, std::string *err) {
  if (auto po = v.as<::minijson::object>()) {

    bool dtype_found{false};
    bool shape_found{false};
    bool data_offsets_found{false};

    dtype dtype;
    std::vector<size_t> shape;
    std::array<size_t, 2> data_offsets{};

    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "dtype") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `dtype` has invalid object.\n";
          }
          return false;
        }

        if (!parse_dtype(value, dtype, err)) {
          return false;
        }

        dtype_found = true;
      } else if (key == "shape") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `shape` has invalid object.\n";
          }
          return false;
        }

        if (!parse_shape(value, shape, err)) {
          return false;
        }

        shape_found = true;
      } else if (key == "data_offsets") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `data_offsets` has invalid object.\n";
          }
          return false;
        }
        if (!parse_data_offsets(value, data_offsets, err)) {
          return false;
        }

        data_offsets_found = true;
      } else {
        // Unknown key. Report error?
      }
    }

    if (!dtype_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `dtype` item.\n";
      }
      return false;
    }

    if (!shape_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `shape` item.\n";
      }
      return false;
    }

    bool is_empty_tensor{false};
    if ((shape.size() > 0)) {
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0) {
          is_empty_tensor = true;
          break;
        }
      }
    }

    if (is_empty_tensor) {
      // They are not storing any data in the databuffer, yet retaining size in
      // the header. So ignore data_offsets
      if (data_offsets_found) {
        // TODO: make this warn instead of err?
        if (err) {
          (*err) +=
              "`" + name +
              "` is empty tensors(tensors with 1 dimension being 0), and no "
              "data in databuffer, but `data_offsets` item is provided.\n";
        }
        // DO NOT RETURN FALSE, JUST CONTINUE
      }
    } else {
      if (!data_offsets_found) {
        if (err) {
          (*err) += "`" + name + "` does not have `data_offsets` item.\n";
        }
        return false;
      }
    }

    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.data_offsets = data_offsets;

  } else {
    if (err) {
      (*err) += "`" + name + "` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

// From llama.cpp
#if defined(_WIN32)
static std::string safetensors_format_win_err(DWORD err) {
  LPSTR buf;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0,
      NULL);
  if (!size) {
    return "FormatMessageA failed";
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

struct safetensors_file {
  // use FILE * so we don't have to re-open the file to mmap
  FILE *fp{nullptr};
  size_t size{0};
  mutable bool _valid{false};
  std::string _err;

  safetensors_file(const char *fname, const char *mode) {
    fp = std::fopen(fname, mode);
    if (fp == nullptr) {
      _err = "failed to open " + std::string(fname) + ":" +
             std::string(strerror(errno)) + "\n";
      _valid = false;
    } else {
      seek(0, SEEK_END);
      size = tell();
      seek(0, SEEK_SET);
      _valid = true;
    }
  }

  ~safetensors_file() {
    if (fp) {
      std::fclose(fp);
      fp = nullptr;
    }
  }

  size_t tell() const {
#ifdef _WIN32
    __int64 ret = _ftelli64(fp);
#else
    long ret = std::ftell(fp);
#endif
    if (ret == -1) {
      // this really shouldn't fail
      _valid = false;
      return 0;
    }

    return (size_t)ret;
  }

  void seek(size_t offset, int whence) const {
#ifdef _WIN32
    int ret = _fseeki64(fp, (__int64)offset, whence);
#else
    int ret = std::fseek(fp, (long)offset, whence);
#endif
    if (ret == 0) {
      _valid = false;
    }
  }

  bool &is_valid() const { return _valid; }

  const std::string &get_error() const { return _err; }
};

struct safetensors_mmap {
  uint8_t *addr{nullptr};
  size_t size{0};

  bool _valid{false};
  std::string _warn;
  std::string _err;

  const bool is_valid() const { return _valid; }

  const std::string &get_error() const { return _err; }

  const std::string &get_warning() const { return _warn; }

  safetensors_mmap(const safetensors_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
  static constexpr bool SUPPORTED = true;

  safetensors_mmap(struct safetensors_file *file,
                   size_t prefetch = (size_t)-1 /* -1 = max value */,
                   bool numa = false) {
    size = file->size;
    int fd = fileno(file->fp);
    int flags = MAP_SHARED;
    // prefetch/readahead impairs performance on NUMA systems
    if (numa) {
      prefetch = 0;
    }
#ifdef __linux__
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif
    addr = reinterpret_cast<uint8_t *>(
        mmap(NULL, file->size, PROT_READ, flags, fd, 0));
    if (addr == MAP_FAILED) {
      _valid = false;
      _err = "mmap failed: " + std::string(strerror(errno)) + "\n";

      size = 0;
      addr = nullptr;

      return;
    }

    if (prefetch > 0) {
      // Advise the kernel to preload the mapped memory
      if (posix_madvise(addr, std::min(file->size, prefetch),
                        POSIX_MADV_WILLNEED)) {
        _warn += "posix_madvise(.., POSIX_MADV_WILLNEED) failed: " +
                 std::string(strerror(errno)) + "\n";
      }
    }
    if (numa) {
      // advise the kernel not to use readahead
      // (because the next page might not belong on the same node)
      if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
        _warn += "posix_madvise(.., POSIX_MADV_RANDOM) failed: " +
                 std::string(strerror(errno)) + "\n";
      }
    }

    _valid = true;
  }

  ~safetensors_mmap() {
    if (_valid) {
      munmap(addr, size);
    }
    size = 0;
    addr = nullptr;
    _valid = false;
  }

#elif defined(_WIN32)
  static constexpr bool SUPPORTED = true;

  safetensors_mmap(struct safetensors_file *file, bool prefetch = true,
                   bool numa = false) {
    (void)numa;

    size = file->size;

    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    DWORD error = GetLastError();

    if (hMapping == NULL) {
      // TODO: get error message
      _err = "CreateFileMappingA failed: " + safetensors_format_win_err(error) +
             "\n";
      _valid = false;
      size = 0;
      addr = nullptr;
      return;
    }

    addr = reinterpret_cast<uint8_t *>(
        MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
    error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
      _err =
          "MapViewOfFile failed: " + safetensors_format_win_err(error) + "\n";
    }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
    if (prefetch) {
      // PrefetchVirtualMemory is only present on Windows 8 and above, so we
      // dynamically load it
      BOOL(WINAPI * pPrefetchVirtualMemory)
      (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      // may fail on pre-Windows 8 systems
      pPrefetchVirtualMemory =
          reinterpret_cast<decltype(pPrefetchVirtualMemory)>(
              GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

      if (pPrefetchVirtualMemory) {
        // advise the kernel to preload the mapped memory
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)size;
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          _warn += "PrefetchVirtualMemory failed: " +
                   safetensors_format_win_err(GetLastError()) + "\n";
        }
      }
    }
#endif
  }
  ~safetensors_mmap() {
    if (!UnmapViewOfFile(addr)) {
      _warn += "UnmapViewOfFile failed: " +
               safetensors_format_win_err(GetLastError()) + "\n";
    }
  }
#else
  static constexpr bool SUPPORTED = false;

  safetensors_mmap(struct safetensors_file *file, bool prefetch = true,
                   bool numa = false) {
    (void)file;
    (void)prefetch;
    (void)numa;

    _valid = false;
    _err = "mmap not supported\n";
    addr = nullptr;
    size = 0;
  }
#endif
};

// Based on MIOPen bfloat16
// https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/src/kernels/bfloat16_dev.hpp

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

typedef union cvt_bf16_fp32 {
  uint32_t u32;
  uint16_t ushortvec[2];

  float f32;
} cvt_bf16_fp32_t;

float bfloat16_to_float(uint16_t src_val) {
  cvt_bf16_fp32_t target_val;

  target_val.ushortvec[0] = 0;
  target_val.ushortvec[1] = src_val;

  return target_val.f32;
}

uint16_t float_to_bfloat16(float src_val) {
  cvt_bf16_fp32_t target_val;
  target_val.f32 = src_val;
  // BF16 round and NaN preservation code matches
  // https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
  if ((~target_val.u32 & 0x7f800000) == 0)  // Inf or NaN
  {
    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bloat16's mantissa bits are all 0.
    if ((target_val.u32 & 0xffff) != 0) {
      target_val.u32 |= 0x10000;  // Preserve signaling NaN
    }
  } else {
#if 1  // MIOPEN_USE_RNE_BFLOAT16
       // When the exponent bits are not all 1s, then the value is zero, normal,
       // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
       // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
       // This causes the bfloat16's mantissa to be incremented by 1 if the 16
       // least significant bits of the float mantissa are greater than 0x8000,
       // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif  // MIOPEN_USE_RNE_BFLOAT16
  }

  return target_val.ushortvec[1];
}

// half <-> float conversion based on: https://gist.github.com/rygorous/2156668
// (CC0 license)
//

// Little endian
union FP32le {
  unsigned int u;
  float f;
  struct {
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
  } s;
};

// Little endian
union float16le {
  unsigned short u;
  struct {
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
  } s;
};

float half_to_float_le(float16le h) {
  static const FP32le magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13;  // exponent mask after shift
  FP32le o;

  o.u = (h.u & 0x7fffU) << 13U;           // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)    // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  else if (exp_ == 0)         // Zero/Denormal?
  {
    o.u += 1 << 23;  // extra exp adjust
    o.f -= magic.f;  // renormalize
  }

  o.u |= (h.u & 0x8000U) << 16U;  // sign bit
  return o.f;
}

uint16_t float_to_half_full_le(float _f) {
  FP32le f;
  f.f = _f;
  float16le o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
  } else                                      // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31)  // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0)  // Underflow
    {
      if ((14 - newexp) <= 24)  // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1)  // Check for rounding
          o.u++;  // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.s.Exponent = static_cast<unsigned int>(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000)  // Check for rounding
        o.u++;                    // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;

  return o.u;
}

bool parse_safetensors_header(const uint8_t *addr, const size_t nbytes,
                              const std::string &filename, safetensors_t *st,
                              std::string *warn, std::string *err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  uint64_t header_size{0};
  memcpy(reinterpret_cast<unsigned char *>(&header_size), addr,
         sizeof(uint64_t));

  if (header_size < 4) {
    if (err) {
      (*err) += "Header size is too short.\n";
    }
    return false;
  }

  if ((8 + header_size) > nbytes) {
    if (err) {
      (*err) += "Header size " + std::to_string(header_size) +
                " + 8 exceeds input size " + std::to_string(nbytes) + " .\n";
    }
    return false;
  }

  if (header_size > kMaxJSONSize) {
    if (err) {
      (*err) += "Header JSON size exceeds the limit(" +
                std::to_string(kMaxJSONSize) + ").\n";
    }
    return false;
  }

  // assume JSON data is small enough.
  std::string json_str(reinterpret_cast<const char *>(&addr[8]), header_size);
  const char *p = json_str.c_str();

  ::minijson::value v;
  ::minijson::error e = ::minijson::parse(p, v);

  if (e != ::minijson::no_error) {
    if (err) {
      std::string json_err(::minijson::errstr(e));
      (*err) += "JSON parse error: " + json_err + "\n";
    }

    return false;
  }

  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;

  // root element must be dict.
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "__metadata__") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in __metadata__.\n";
          }
          return false;
        }

        if (!detail::parse_metadata(value, metadata, err)) {
          return false;
        }
      } else {
        // tensor

        if (tensors.count(key)) {
          if (err) {
            (*err) += "Duplicate key `" + key + "` found.\n";
          }
          return false;
        }

        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in `" + key + "`.\n";
          }
          return false;
        }

        tensor_t tensor;
        if (!detail::parse_tensor(key, value, tensor, err)) {
          return false;
        }

        tensors.insert(key, std::move(tensor));
      }
    }
  } else {
    if (err) {
      (*err) += "JSON root elements must be object(dict)\n";
    }
  }

  st->tensors = std::move(tensors);
  st->metadata = std::move(metadata);
  st->header_size = header_size;

#if 0
  size_t databuffer_size = nbytes - header_size - 8;

  st->storage.resize(nbytes);
  memcpy(st->storage.data(), addr + 8 + header_size, nbytes);

  st->mmaped = false;
  st->mmap_addr = addr + 8 + header_size;
  st->mmap_size = 0;
#endif

  return true;
}

}  // namespace detail

safetensors_t::~safetensors_t() {
  if (st_mmap) {
    detail::safetensors_mmap *p =
        reinterpret_cast<detail::safetensors_mmap *>(st_mmap);
    delete p;
    st_mmap = nullptr;
  }

  if (st_file) {
    detail::safetensors_file *p =
        reinterpret_cast<detail::safetensors_file *>(st_file);
    delete p;
    st_file = nullptr;
  }
}

//
// - 8byte: header_size
// - json data(header_size bytes)
// - tensor data(filesize - header_size)
//

bool load_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err) {
  std::vector<unsigned char> data;
  if (!detail::ReadWholeFile(&data, err, filename, nullptr)) {
    return false;
  }

  return load_from_memory(reinterpret_cast<const uint8_t *>(data.data()),
                          data.size(), filename, st, warn, err);
}

bool load_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  size_t databuffer_size = nbytes - st->header_size - 8;

  st->storage.resize(databuffer_size);
  memcpy(st->storage.data(), addr + 8 + st->header_size, databuffer_size);

  st->mmaped = false;
  st->mmap_addr = nullptr;
  st->mmap_size = 0;
  st->databuffer_addr = nullptr;
  st->databuffer_size = 0;

  return true;
}

bool mmap_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err) {
  if (!st) {
    return false;
  }

  detail::safetensors_file *pf =
      new detail::safetensors_file(filename.c_str(), "rb");
  if (!pf->is_valid()) {
    if (err) {
      (*err) += pf->get_error();
    }
    delete pf;
    return false;
  }

  // TODO: prefetch, numa
  detail::safetensors_mmap *pm = new detail::safetensors_mmap(pf);

  bool ret = mmap_from_memory(pm->addr, pm->size, filename, st, warn, err);

  if (!ret) {
    delete pm;
    delete pf;

    return false;
  }

  st->mmap_addr = pm->addr;
  st->mmap_size = pm->size;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  // retain pointer
  st->st_file = pf;
  st->st_mmap = pm;

  st->mmaped = true;

  return true;
}

bool mmap_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err) {
  if (!addr) {
    return false;
  }

  if (nbytes < 16) {
    return false;
  }

  if (!st) {
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  size_t databuffer_size = nbytes - st->header_size - 8;

  st->mmaped = true;

  st->mmap_addr = addr;
  st->mmap_size = nbytes;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  return true;
}

float bfloat16_to_float(uint16_t x) { return detail::bfloat16_to_float(x); }

uint16_t float_to_bfloat16(float x) { return detail::float_to_bfloat16(x); }

float fp16_to_float(uint16_t x) {
  detail::float16le src;
  src.u = x;
  return detail::half_to_float_le(src);
}

uint16_t float_to_fp16(float x) { return detail::float_to_half_full_le(x); }

size_t get_dtype_bytes(const safetensors::dtype dtype) {
  size_t sz = 0;

  switch (dtype) {
    case safetensors::dtype::kBOOL:
      // Original Rust implementaion uses 1.
      sz = 1;
      break;
    case safetensors::dtype::kUINT8:
      sz = 1;
      break;
    case safetensors::dtype::kINT8:
      sz = 1;
      break;
    case safetensors::dtype::kUINT16:
      sz = 2;
      break;
    case safetensors::dtype::kINT16:
      sz = 2;
      break;
    case safetensors::dtype::kINT32:
      sz = 4;
      break;
    case safetensors::dtype::kUINT32:
      sz = 4;
      break;
    case safetensors::dtype::kFLOAT16:
      sz = 2;
      break;
    case safetensors::dtype::kBFLOAT16:
      sz = 2;
      break;
    case safetensors::dtype::kFLOAT32:
      sz = 4;
      break;
    case safetensors::dtype::kFLOAT64:
      sz = 8;
      break;
    case safetensors::dtype::kINT64:
      sz = 8;
      break;
    case safetensors::dtype::kUINT64:
      sz = 8;
      break;
  }

  return sz;
}

std::string get_dtype_str(const safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBOOL:
      return "BOOL";
    case safetensors::dtype::kUINT8:
      return "U8";
    case safetensors::dtype::kINT8:
      return "I8";
    case safetensors::dtype::kUINT16:
      return "U16";
    case safetensors::dtype::kINT16:
      return "I16";
    case safetensors::dtype::kINT32:
      return "I32";
    case safetensors::dtype::kUINT32:
      return "U32";
    case safetensors::dtype::kFLOAT16:
      return "F16";
    case safetensors::dtype::kBFLOAT16:
      return "BF16";
    case safetensors::dtype::kFLOAT32:
      return "F32";
    case safetensors::dtype::kFLOAT64:
      return "F64";
    case safetensors::dtype::kINT64:
      return "I64";
    case safetensors::dtype::kUINT64:
      return "U64";
  }
  return "???";
}

// Empty Tensor returns 0.
// Zero-rank Tensor reuturns 1(scalar)
size_t get_shape_size(const tensor_t &t) {
  if (t.shape.empty()) {
    return 1;
  }

  if (t.shape.size() >= kMaxDim) {  // invalid ndim
    return 0;
  }

  size_t sz = 1;

  for (size_t i = 0; i < t.shape.size(); i++) {
    sz *= t.shape[i];
  }

  return sz;
}

bool validate_data_offsets(const safetensors_t &st, std::string &err) {
  bool valid{true};

  std::stringstream ss;

  size_t databuffersize;
  if (st.mmaped) {
    databuffersize = st.databuffer_size;
  } else {
    databuffersize = st.storage.size();
  }

  size_t ntensors{0};
  // Iterate with key insertion order.
  for (size_t i =0 ;i < st.tensors.size(); i++) {

    std::string key = st.tensors.keys()[i];

    tensor_t tensor;
    if (!st.tensors.at(i, &tensor)) {
      ss << "Internal error: Failed to get tensor at [" << i << "]\n";
      valid = false;
      continue;
    }

    if (tensor.data_offsets[0] > tensor.data_offsets[1]) {
      ss << key << ".data_offsets.BEGIN " << tensor.data_offsets[0]
         << " must be less than or equal to data_offsets.END "
         << tensor.data_offsets[1] << "\n";
      valid = false;
    }

    size_t tensor_size = get_dtype_bytes(tensor.dtype) * get_shape_size(tensor);

    if (tensor_size == 0) {
      // OK
      continue;
    }

    // data_offsets are absolute offset from the databuffer(file)
    if (tensor.data_offsets[0] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.BEGIN "
         << tensor.data_offsets[0] << " exceeds databuffer size "
         << databuffersize << ".\n";
      valid = false;
    }

    if (tensor.data_offsets[1] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.END "
         << tensor.data_offsets[1] << " exceeds databuffer size "
         << databuffersize << ".\n";
      valid = false;
    }

    size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];

    if (tensor_size != data_size) {
      ss << "Data size mismatch. The size in Tensor `" << key << "` is "
         << tensor_size << ", but the size from data_offsets is " << data_size
         << "\n";
      valid = false;
    }

    ntensors++;
    if (ntensors == st.tensors.size()) {
      // Last element's data_offsets[1] must be equal to databuffer size.
      if (tensor.data_offsets[1] != databuffersize) {
        ss << "The last tensor's data_offset.END(" << tensor.data_offsets[1]
           << ") must be equal to databufer size " << databuffersize << ".\n";
        valid = false;
      }
    }
  }

  if (!valid) {
    err = ss.str();
  }

  return valid;
}

bool save_to_memory(const safetensors_t &st, std::vector<uint8_t> *dst,
                    std::string *warn, std::string *err) {
  // directly serialize JSON string.
  std::stringstream ss;

  // NOTE: The last offset **must** be the end of the file,
  // so write __metadata__ first(if metadata part exists)

  std::string _err;
  if (!validate_data_offsets(st, _err)) {
    if (err) {
      (*err) += "Invalid safensors is provided.\n";
      (*err) += _err;
    }
    return false;
  }

  ss << "{";
  if (st.metadata.size()) {
    ss << "\"__metadata__\": {";
    size_t nmeta = 0;
    for (size_t i = 0; i < st.metadata.size(); i++) {
      std::string key = st.metadata.keys()[i];
      std::string value;
      st.metadata.at(i, &value);

      if (nmeta > 0) {
        ss << ", ";
      }
      ss << "\"" + key + "\": \"" << value << "\"";
      nmeta++;
    }
    ss << "}";

    if (st.tensors.size()) {
      ss << ", ";
    }
  }

  size_t ntensors = 0;
  {
    for (size_t i = 0; i < st.tensors.size(); i++) {

      std::string key = st.tensors.keys()[i];
      safetensors::tensor_t tensor;
      st.tensors.at(i, &tensor);

      if (tensor.shape.size() > safetensors::kMaxDim) {
        if (err) {
          (*err) += key + ".shape is too large.\n";
          (*err) += _err;
        }
        return false;
      }

      if (ntensors > 0) {
        ss << ", ";
      }
      ss << "\"" << key << "\": {";
      ss << "\"dtype\": \"" << safetensors::get_dtype_str(tensor.dtype)
         << "\", ";
      ss << "\"shape\": [";
      for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i > 0) {
          ss << ", ";
        }
        ss << tensor.shape[i];
      }
      ss << "]";
      ss << ", \"data_offsets\": [" << tensor.data_offsets[0] << ", "
         << tensor.data_offsets[1] << "]";
      ss << "}";
      ntensors++;
    }
  }
  ss << "}";

  std::string header_str = ss.str();

  uint64_t header_size = header_str.size();  // do not include '\n'

  const void *databuffer_addr{nullptr};
  size_t databuffer_size{0};
  if (st.mmaped) {
    databuffer_size = st.databuffer_size;
    databuffer_addr = st.databuffer_addr;
  } else {
    databuffer_size = st.storage.size();
    databuffer_addr = reinterpret_cast<const void *>(st.storage.data());
  }

  // make databuffer addr start from the multiple of 8.
  size_t pad_bytes = 0;
  if ((header_size % 8) != 0) {
    pad_bytes = 8 - (header_size % 8);
  }
  //printf("header_size = %d\n", int(header_size));
  //printf("pad_bytes = %d\n", int(pad_bytes));
  size_t padded_header_size = header_size + pad_bytes;
  dst->resize(8 + padded_header_size + databuffer_size);

  // write padded header_size
  memcpy(dst->data(), &padded_header_size, 8);

  // write header
  memcpy(dst->data() + 8, header_str.data(), header_size);

  // Use whitespace for trailing padding.
  memset(dst->data() + 8 + header_size, 0x20, pad_bytes);

  memcpy(dst->data() + 8 + padded_header_size, databuffer_addr,
         databuffer_size);

  return true;
}

bool save_to_file(const safetensors_t &st, const std::string &filename,
                  std::string *warn, std::string *err) {
  // TODO: Use more reliable io.
  std::ofstream ofs(filename, std::ios::binary);

  if (!ofs) {
    if (err) {
      (*err) += "Failed to open `" + filename +
                "` to write. File is either existing directory or "
                "write-protected, or disk is full?\n";
    }
    return false;
  }

  std::vector<uint8_t> buf;
  if (!save_to_memory(st, &buf, warn, err)) {
    return false;
  }

  ofs.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  if (!ofs) {
    if (err) {
      (*err) += "Failed to write safetensor data to `" + filename +
                "`. Maybe no disk space available?(Required bytes : " +
                std::to_string(buf.size()) + "\n";
    }
    return false;
  }

  return true;
}

}  // namespace safetensors

#endif
