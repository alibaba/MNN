#pragma once
#include <string>
#include <memory>
#include <vector>
#include <MNN/HalideRuntime.h>
#if defined(_MSC_VER) && PY_MAJOR_VERSION >= 3
#include <Windows.h>
#include <stringapiset.h>
#endif
#include "common.h"

using namespace std;
typedef vector<int> INTS;

// In python3, default str is unicode, then be transformed to UTF-8 bytes by pybind.
// In Windows, MNN library assume input bytes be encoded by CP_ACP.
// So we need: UTF-8 bytes -> unicodes -> CP_ACP bytes
inline std::string convertBytesEncodeIfNeed(const char* srcBytes) {
#if defined(_MSC_VER) && PY_MAJOR_VERSION >= 3
    int wideCharSize = MultiByteToWideChar(CP_UTF8, 0, srcBytes, -1, nullptr, 0);
    if (wideCharSize == 0) {
        return {};
    }
    std::unique_ptr<wchar_t[]> unicodes(new wchar_t[wideCharSize]);
    if (MultiByteToWideChar(CP_UTF8, 0, srcBytes, -1, unicodes.get(), wideCharSize) == 0) {
        return {};
    }
    int byteSize = WideCharToMultiByte(CP_ACP, 0, unicodes.get(), wideCharSize, nullptr, 0, nullptr, nullptr);
    if (byteSize == 0) {
        return {};
    }
    std::unique_ptr<char[]> dstBytes(new char[byteSize]);
    if (WideCharToMultiByte(CP_ACP, 0, unicodes.get(), wideCharSize, dstBytes.get(), byteSize, nullptr, nullptr) == 0) {
        return {};
    }
    return {dstBytes.get(), (size_t)byteSize};
#else
    return {srcBytes};
#endif
}

// Returns true if obj is a bytes/str or unicode object
inline bool checkString(PyObject* obj) {
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
}
// Convert PyBytes (PyString) or PyUnicode as std::string
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: this method requires the GIL
inline std::string object2String(PyObject* obj) {
  if (PyBytes_Check(obj)) {
      return std::string(PyBytes_AS_STRING(obj));
  }
  if (PyUnicode_Check(obj)) {
      PyObject *bytes = PyUnicode_AsUTF8String(obj);
      std::string s = std::string(PyBytes_AS_STRING(bytes));
      Py_XDECREF(bytes);
      return s;
  }
  //just to pass compile.It should never comes to here.
  return std::string("");
}

inline PyObject* char2Object(const char* str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(str);
#else
  return PyUnicode_FromString(str);
#endif
}
inline PyObject* string2Object(const std::string& str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(str.c_str());
#else
  return PyUnicode_FromString(str.c_str());
#endif
}
inline double unpackDouble(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  throw std::runtime_error("Overflow when unpacking double");
}
inline int64_t unpackLong(PyObject* obj) {
  int overflow;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  return (int64_t)value;
}
inline void store_scalar(void* data, int dtype, PyObject* obj) {
  switch (dtype) {
    case 4: *(uint8_t*)data = (uint8_t)unpackLong(obj); break;
    case 3: *(int32_t*)data = (int32_t)unpackLong(obj); break;
    case 9: *(int64_t*)data = unpackLong(obj); break;
    case 1: *(float*)data = (float)unpackDouble(obj); break;
    case 2: *(double*)data = (double)unpackDouble(obj); break;
    default: throw std::runtime_error("invalid type");
  }
}
template<class T>
class MNNPointer {
public:
  MNNPointer(): ptr(nullptr) {};
  explicit MNNPointer(T *ptr) noexcept : ptr(ptr) {};
  MNNPointer(MNNPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~MNNPointer() { free(); };
  T * get() { return ptr; }
  const T * get() const { return ptr; }
  T * release() { T *tmp = ptr; ptr = nullptr; return tmp; }
  operator T*() { return ptr; }
  MNNPointer& operator =(T *new_ptr) noexcept { free(); ptr = new_ptr; return *this; }
  MNNPointer& operator =(MNNPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
  T * operator ->() { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }

private:
  void free();
  T *ptr = nullptr;
};
template<>
void MNNPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}
using MNNObjectPtr = MNNPointer<PyObject>;
INTS getshape(PyObject* seq) {
  INTS shape;
  while (PySequence_Check(seq)) {
    auto length = PySequence_Length(seq);
    if (length < 0) throw std::exception();
    shape.push_back(length);
    if (shape.size() > 20) {
      throw std::runtime_error("max dimension greater than 20");
    }
    if (length == 0) break;
    auto seq_obj = MNNObjectPtr(PySequence_GetItem(seq, 0));
    seq = seq_obj.get();
  }
  return shape;
}
void recursive_store(char* data, INTS shape, INTS stride, int dim, PyObject* obj, int dtype, int elementSize) {
  auto ndim = shape.size();
  if(dim == ndim) {
     store_scalar(data, dtype, obj);
     return;
  }
  auto n = shape[dim];
  auto seq = PySequence_Fast(obj, "not a sequence");
  if (!seq) throw std::exception();
  auto seq_size = PySequence_Fast_GET_SIZE(seq);
  if (seq_size != n) {
     throw std::exception();
  }
  PyObject** items = PySequence_Fast_ITEMS(seq);
  for (int i = 0; i < n; i++) {
    recursive_store(data, shape, stride, dim + 1, items[i], dtype, elementSize);
    data +=  stride[dim] * elementSize;
  }
  Py_XDECREF(seq);
}
enum DType {
      DType_FLOAT = 1,
      DType_DOUBLE = 2,
      DType_INT32 = 3,
      DType_UINT8 = 4,
      DType_INT8 = 6,
      DType_STRING = 7,
      DType_INT64 = 9,
}; //ruhuan match DType to DataType in flatbuffer
DType htype2dtype(halide_type_t type) {
    if (type.code == halide_type_float) {
        return DType_FLOAT;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return DType_UINT8;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return DType_INT32;
    }
    if (type.code == halide_type_int && type.bits == 64) {
        return DType_INT64;
    } 
    if (type.code == halide_type_handle) {
	return DType_STRING; 
    } 
    return DType_FLOAT;
}
#define CONVERT(src, dst, f)\
  if (f == src) return dst;
halide_type_t dtype2htype(DType dtype) {
    CONVERT(DType_FLOAT, halide_type_of<float>(), dtype);
    CONVERT(DType_INT32, halide_type_of<int32_t>(), dtype);
    CONVERT(DType_INT64, halide_type_of<int32_t>(), dtype);
    CONVERT(DType_UINT8, halide_type_of<uint8_t>(), dtype);
    CONVERT(DType_INT8, halide_type_of<int8_t>(), dtype);
    return halide_type_of<float>();
}
#ifdef PYMNN_NUMPY_USABLE
inline int getitemsize(int dtype, int npy_type) {
    switch(dtype) {
      case DType_FLOAT:
        if(npy_type != NPY_FLOAT) {
          throw std::runtime_error("numpy type does not match");
        }
        return 4;
      case DType_DOUBLE:
        if(npy_type != NPY_DOUBLE) {
          throw std::runtime_error("numpy type does not match");
        }
        return 8;
      case DType_INT32:
        if(npy_type != NPY_INT) {
          throw std::runtime_error("numpy type does not match");
        }
        return 4;
      case DType_INT64:
        if(npy_type != NPY_INT64) {
          throw std::runtime_error("numpy type does not match");
        }
        return 8;
      case DType_UINT8:
        if(npy_type != NPY_UINT8) {
          throw std::runtime_error("numpy type does not match");
        }
        return 1;
      default:
        throw std::runtime_error("does not support this dtype");
    }
}
#endif
inline int getitemsize(int dtype) {
    switch(dtype) {
      case DType_FLOAT:
        return 4;
      case DType_DOUBLE:
        return 8;
      case DType_INT32:
        return 4;
      case DType_INT64:
        return 8;
      case DType_UINT8:
        return 1;
      case DType_STRING:
        return 4;
      default:
        throw std::runtime_error("does not support this dtype");
    }
}

