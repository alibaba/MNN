#pragma once
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <MNN/HalideRuntime.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#if defined(_MSC_VER) && PY_MAJOR_VERSION >= 3
#include <Windows.h>
#include <stringapiset.h>
#endif
#include "common.h"

#define PARSE(obj, default, func) ((obj) == nullptr ? (default) : func(obj))
#define MAX_CONFIG_SIZE 5

using namespace std;
typedef vector<int> INTS;
#define PyMNN_ERROR_LOG(x) PyErr_SetString(PyExc_TypeError, x);MNN_PRINT(x);
#define PyMNN_ERROR(x) PyMNN_ERROR_LOG(x) \
    Py_RETURN_NONE
#if PY_MAJOR_VERSION < 3
#define PySlice_Unpack _PySlice_Unpack
#define Py_hash_t long
#endif

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
  PyMNN_ERROR_LOG("Overflow when unpacking double");
  return 0;
}
inline int64_t unpackLong(PyObject* obj) {
  int overflow;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    PyMNN_ERROR_LOG("unpackLong: Error!");
  }
  if (overflow != 0) {
    PyMNN_ERROR_LOG("Overflow when unpacking long");
  }
  return (int64_t)value;
}
inline double unpackDoubleOrLong(PyObject* obj) {
    if (PyLong_Check(obj)
#if PY_MAJOR_VERSION < 3
    || PyInt_Check(obj)
#endif
    ) {
        return static_cast<float>(unpackLong(obj));
    }
    return unpackDouble(obj);
}
inline void store_scalar(void* data, int dtype, PyObject* obj) {
  switch (dtype) {
    case 4: *(uint8_t*)data = (uint8_t)unpackLong(obj); break;
    case 3: *(int32_t*)data = (int32_t)unpackLong(obj); break;
    case 9: *(int64_t*)data = unpackLong(obj); break;
    case 1: *(float*)data = (float)unpackDoubleOrLong(obj); break;
    case 2: *(double*)data = (double)unpackDoubleOrLong(obj); break;
    case 6: *(int8_t*)data = (int8_t)unpackLong(obj); break;
    default: PyMNN_ERROR_LOG("store_scalar: invalid type");
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
    if (length < 0) {
        PyMNN_ERROR_LOG("Error: getshape sequence length < 0!")
        return shape;
    }
    shape.push_back(length);
    if (shape.size() > 20) {
        PyMNN_ERROR_LOG("max dimension greater than 20");
        return shape;
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
  if (!seq) {
      PyMNN_ERROR_LOG("Error: recursive_store not a sequence")
      return;
  }
  auto seq_size = PySequence_Fast_GET_SIZE(seq);
  if (seq_size != n) {
    PyMNN_ERROR_LOG("Error: seq_size != n")
    return;
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
    if (type.code == halide_type_int && type.bits == 8) {
        return DType_INT8;
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
inline int getnpysize(int npy_type) {
    switch(npy_type) {
      case NPY_FLOAT:
        return 4;
      case NPY_DOUBLE:
        return 8;
      case NPY_INT64:
        return 8;
      case NPY_UINT8:
        return 1;
      default:
        // NPY_INT(np.int) and NPY_INT32(np.int32) may be different enum on some platform
        // use `if` instead of `switch case`(when NPY_INT is same as NPY_INT32, two same case value is not support)
        if (npy_type == NPY_INT || npy_type == NPY_INT32) {
            return 4;
        }
        PyMNN_ERROR_LOG("does not support this npy_type");
        return 0;
    }
}
inline int getitemsize(int dtype, int npy_type) {
    switch(dtype) {
      case DType_FLOAT:
        if(npy_type != NPY_FLOAT) {
          PyMNN_ERROR_LOG("numpy type does not match");
        }
        return 4;
      case DType_DOUBLE:
        if(npy_type != NPY_DOUBLE) {
          PyMNN_ERROR_LOG("numpy type does not match");
        }
        return 8;
      case DType_INT32:
        if(npy_type != NPY_INT && npy_type != NPY_INT32) {
          PyMNN_ERROR_LOG("numpy type does not match");
        }
        return 4;
      case DType_INT64:
        if(npy_type != NPY_INT64) {
          PyMNN_ERROR_LOG("numpy type does not match");
        }
        return 8;
      case DType_UINT8:
        if(npy_type != NPY_UINT8) {
          PyMNN_ERROR_LOG("numpy type does not match");
        }
        return 1;
      default:
        PyMNN_ERROR_LOG("does not support this dtype");
        return 0;
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
        PyMNN_ERROR_LOG("does not support this dtype");
        return 0;
    }
}

// define a submodule of module m
static PyObject* def_submodule(PyObject* m, const char* name) {
    std::string full_name = std::string(PyModule_GetName(m)) + "." + name;
    PyObject* submodule = PyImport_AddModule(full_name.c_str());
    PyObject_SetAttrString(m, name, submodule);
    return submodule;
}

// define a method of module m
static void def_method(PyObject* m, PyMethodDef* method) {
    PyModule_AddObject(m, method->ml_name, PyCFunction_New(method, 0));
}
// Basic type of cpp to python Object Wrapper-Func
static inline PyObject* toPyObj(bool val) {
    if (val) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}
static inline PyObject* toPyObj(uint8_t val) {
    return PyLong_FromLong((long)val);
}
static inline PyObject* toPyObj(int val) {
    return PyLong_FromLong(val);
}
static inline PyObject* toPyObj(size_t val) {
    return PyLong_FromLong(val);
}
static inline PyObject* toPyObj(float val) {
    return PyFloat_FromDouble(val);
}
static inline PyObject* toPyObj(const char* val) {
    return char2Object(val);
}
static inline PyObject* toPyObj(string val) {
    return string2Object(val);
}
template <typename T, PyObject*(*Func)(T)=toPyObj>
static PyObject* toPyObj(vector<T> values) {
    PyObject* obj = PyList_New(values.size());
    for (int i = 0; i < values.size(); i++) {
        PyList_SetItem(obj, i, Func(values[i]));
    }
    return obj;
}
template <typename K, PyObject*(*FuncK)(K)=toPyObj,
          typename V, PyObject*(*FuncV)(V)=toPyObj>
static PyObject* toPyObj(pair<K, V> value) {
    PyObject* obj = PyTuple_New(2);
    PyTuple_SetItem(obj, 0, FuncK(value.first));
    PyTuple_SetItem(obj, 1, FuncV(value.second));
    return obj;
}
template <typename K, PyObject*(*FuncK)(K)=toPyObj,
          typename V, PyObject*(*FuncV)(V)=toPyObj>
static PyObject* toPyObj(map<K, V> values) {
    PyObject* obj = PyDict_New();
    for (auto iter = values.begin(); iter != values.end(); iter++) {
        auto key = FuncK(iter->first), val = FuncV(iter->second);
        PyDict_SetItem(obj, key, val);
        Py_XDECREF(key);
        Py_XDECREF(val);
    }
    return obj;
}
// Python Object to basic type of cpp Wrapper-Func
static inline bool isString(PyObject* obj) {
    return PyBytes_Check(obj) || PyUnicode_Check(obj);
}
static inline bool isInt(PyObject* obj) {
    return PyLong_Check(obj)
#if PY_MAJOR_VERSION < 3
    || PyInt_Check(obj)
#endif
    ;
}
static inline bool isFloat(PyObject* obj) {
    return PyFloat_Check(obj);
}
static inline bool isNone(PyObject* obj) {
    return (obj == Py_None);
}
static inline bool isPySequence(PyObject* obj) {
    // ndarray in PySequence_Check is true;
    // when PYMNN_NUMPY_USABLE is close will get some wrong judge
    // use isPySequence replace PySequence_Check
    return PyTuple_Check(obj) || PyList_Check(obj) || PyBytes_Check(obj);
}
static inline int PySequenceSize(PyObject* obj) {
    if (PyTuple_Check(obj)) return PyTuple_Size(obj);
    if (PyList_Check(obj)) return PyList_Size(obj);
    if (PyBytes_Check(obj)) return PyBytes_Size(obj);
    return 0;
}
static inline bool isVals(PyObject* obj) {
    return
#ifdef PYMNN_NUMPY_USABLE
    PyArray_Check(obj) ||
#endif
    PyCapsule_CheckExact(obj) ||
    isPySequence(obj);
}
template <bool (*Func)(PyObject*)>
static bool isVec(PyObject* obj) {
#ifdef PYMNN_NUMPY_USABLE
    if(PyArray_Check(obj)) {
        return true;
    }
#endif
    if (PyTuple_Check(obj)) {
        if (PyTuple_Size(obj) > 0) {
            return Func(PyTuple_GetItem(obj, 0));
        } else return true;
    } else if (PyList_Check(obj)) {
        if (PyList_Size(obj) > 0) {
            return Func(PyList_GetItem(obj, 0));
        } else return true;
    }
    return false;
}
static inline bool isInts(PyObject* obj) {
    return isInt(obj) || isVec<isInt>(obj);
}
static inline bool isFloats(PyObject* obj) {
    return isFloat(obj) || isVec<isFloat>(obj);
}
static inline string toString(PyObject* obj) {
    return object2String(obj);
}
static inline int toInt(PyObject* obj) {
    return static_cast<int>(unpackLong(obj));
}
static inline float toFloat(PyObject* obj) {
    return static_cast<float>(unpackDouble(obj));
}
template <typename T, T (*Func)(PyObject*)>
static vector<T> toVec(PyObject* obj) {
    vector<T> values;
#ifdef PYMNN_NUMPY_USABLE
    if(PyArray_Check(obj)) {
        int total_length = PyArray_Size(obj);
        if (total_length == 0) {
            return values;
        }
        int item_size = getnpysize(PyArray_TYPE((const PyArrayObject*)obj));
        PyArrayObject *obj_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)obj);
        auto tmpBuffer = PyArray_DATA(obj_cont);
        if(NULL == tmpBuffer) {
            PyMNN_ERROR_LOG("numpy failed to get buffer");
            return values;
        }
        values.resize(total_length);
        memcpy(values.data(), tmpBuffer, total_length * item_size);
        Py_XDECREF(obj_cont);
        return values;
    }
#endif
    if (PyTuple_Check(obj)) {
        size_t size = PyTuple_Size(obj);
        values.resize(size);
        for (int i = 0; i < size; i++) {
            values[i] = Func(PyTuple_GetItem(obj, i));
        }
        return values;
    }
    if (PyList_Check(obj)) {
        size_t size = PyList_Size(obj);
        values.resize(size);
        for (int i = 0; i < size; i++) {
            values[i] = Func(PyList_GetItem(obj, i));
        }
        return values;
    }
    values.push_back(Func(obj));
    return values;
}
static inline std::vector<int> toInts(PyObject* obj) {
    if (isInt(obj)) { return { toInt(obj) }; }
    return toVec<int, toInt>(obj);
}
static inline std::vector<float> toFloats(PyObject* obj) {
    if (isFloat(obj)) { return { toFloat(obj) }; }
    return toVec<float, toFloat>(obj);
}
static inline std::vector<string> toStrings(PyObject* obj) {
    return toVec<string, toString>(obj);
}
template <typename K, K(*FuncK)(PyObject*),
          typename V, V(*FuncV)(PyObject*)>
static map<K, V> toMap(PyObject* obj) {
    map<K, V> values;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        values.insert(make_pair(FuncK(key), FuncV(value)));
    }
    return values;
}
static void* toPtr(PyObject *obj, DType dtype, int64_t& total_length, void* data = nullptr) {
#ifdef PYMNN_NUMPY_USABLE
    if(PyArray_Check(obj)) {
        //numpy support
        if (total_length < 0) {
            total_length = PyArray_Size(obj);
        } else if (total_length != PyArray_Size(obj)) {
            PyMNN_ERROR_LOG("data size does not match each other");
            return data;
        }
        int npy_type = PyArray_TYPE((const PyArrayObject*)obj);
        int itemsize = getitemsize(dtype, npy_type);
        PyArrayObject *obj_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)obj);
        auto tmpBuffer = PyArray_DATA(obj_cont);
        if(NULL == tmpBuffer) {
            PyMNN_ERROR_LOG("numpy failed to get buffer");
            return data;
        }
        if (!data) data = malloc(total_length * itemsize);
        if (nullptr == data) {
            PyMNN_ERROR_LOG("call to writeMap meet a error");
            return data;
        }
        memcpy(data, tmpBuffer, total_length * itemsize);
        Py_XDECREF(obj_cont);
        return data;
    }
#endif
    INTS shapeData = getshape(obj);
    int64_t totalLengthData = 1;
    INTS stride;
    for(size_t i = 0; i < shapeData.size(); i++) {
        totalLengthData *= shapeData[i];
    }
    if (totalLengthData == 0) {
        PyMNN_ERROR_LOG("input data is empty!");
        return data;
    }
    int totalStride = 1;
    for (int i = shapeData.size() - 1; i >= 0; i--) {
        if (i + 1 < shapeData.size()) {
            totalStride *= shapeData[i+1];
        }
        stride.push_back(totalStride);
    }
    std::reverse(stride.begin(), stride.end());
    if (total_length < 0) {
        total_length = totalLengthData;
    } else if (totalLengthData != total_length) {
        PyMNN_ERROR_LOG("data size does not match each other");
        return data;
    }
    if(DType_FLOAT == dtype) {
        if (!data) data = malloc(total_length * sizeof(float));
        if (nullptr == data) {
            PyMNN_ERROR_LOG("not enough memory");
            return data;
        }
        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(float));
    }
    else if(DType_INT32 == dtype) {
        if (!data) data = malloc(total_length * sizeof(int));
        if (nullptr == data) {
            PyMNN_ERROR_LOG("not enough memory");
            return data;
        }
        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int));
    }
    else if(DType_UINT8 == dtype) {
        if (!data) data = malloc(total_length * sizeof(uint8_t));
        if (nullptr == data) {
            PyMNN_ERROR_LOG("not enough memory");
            return data;
        }
        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(uint8_t));
    }
    else if(DType_INT8 == dtype) {
        if (!data) data = malloc(total_length * sizeof(int8_t));
        if (nullptr == data) {
            PyMNN_ERROR_LOG("not enough memory");
            return data;
        }
        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int8_t));
    }
    return data;
}

namespace ec {
    int getVectorByKey(PyObject* dict, const char *key, std::vector<std::string>& result){
        PyObject *saveTensors = PyDict_GetItemString(dict, key);
        int count = 0;
        if (saveTensors) {
            if (!PyTuple_Check(saveTensors)) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: saveTensors must be a tuple");
                return -1;
            }

            size_t saveTensorsCount = PyTuple_Size(saveTensors);
            for (size_t i = 0; i < saveTensorsCount; i++) {
                PyObject *tensorNameItem = PyTuple_GetItem(saveTensors, i);
                if (!checkString(tensorNameItem)) {
                    PyErr_SetString(PyExc_Exception,
                                    "PyMNNInterpreter_createSession: saveTensors's member must be string");
                    return -1;
                }


                result.push_back(object2String(tensorNameItem));
                count++;
            }
        }
        return count;
    }
}

inline bool getScheduleConfig(PyObject* dict, MNN::ScheduleConfig &config) {
    auto backendConfig = config.backendConfig;
    if (dict) {
        PyObject *backend = PyDict_GetItemString(dict, "backend");
        config.type = MNN_FORWARD_CPU;
        if (backend && checkString(backend)) {
            auto backend_name = object2String(backend);
            // Avoid misusing backend not supported by the bridge and corresponding MNN library on python level,
            // then user will ask for right version bridge library to us, same like MNN.expr.Backend.* python enum
            std::unordered_map<std::string, MNNForwardType> backend_map = {
                // Don't care whether MNN library support corresponding backend, all backend type are usable by user,
                // which make MNN.whl setup.py easy
                {"CPU", MNN_FORWARD_CPU},
                {"OPENCL", MNN_FORWARD_OPENCL},
                {"OPENGL", MNN_FORWARD_OPENGL},
                {"VULKAN", MNN_FORWARD_VULKAN},
                {"METAL", MNN_FORWARD_METAL},
                {"TRT", MNN_FORWARD_USER_1},
                {"CUDA", MNN_FORWARD_CUDA},
                {"HIAI", MNN_FORWARD_USER_0},
                {"NN", MNN_FORWARD_NN},
                {"AUTO", MNN_FORWARD_AUTO}
            };
            auto iter = backend_map.find(backend_name);
            if (iter == backend_map.end()) {
                // backend not support, issue on python level when development
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: backend not support");
                return false;
            }
            config.type = iter->second;
        } else if (backend && isInt(backend)) {
            config.type = (MNNForwardType)toInt(backend); // {'backend': 1L} for example
        }
        PyObject *numThread = PyDict_GetItemString(dict, "numThread");
        if (numThread) {
            if (!isInt(numThread)) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: numThread must be a integer");
                return false;
            }
            config.numThread = (int)toInt(numThread);
        }
        {
            //power
            PyObject *obj = PyDict_GetItemString(dict, "power");
            if (obj) {
                if (isInt(obj)) {
                    backendConfig->power = (MNN::BackendConfig::PowerMode)toInt(obj);
                }
            }
        }
        {
            //memory
            PyObject *obj = PyDict_GetItemString(dict, "memory");
            if (obj) {
                if (isInt(obj)) {
                    backendConfig->memory = (MNN::BackendConfig::MemoryMode)toInt(obj);
                }
            }
        }
        {
            //precision
            PyObject *obj = PyDict_GetItemString(dict, "precision");
            if (obj) {
                if (isInt(obj)) {
                    backendConfig->precision = (MNN::BackendConfig::PrecisionMode)toInt(obj);
                } else {
                    // For compability
                    auto obj_name = object2String(obj);
                    if (!obj_name.compare("low")) {
                        MNN_PRINT("MNN use low precision\n");
                        backendConfig->precision = MNN::BackendConfig::Precision_Low;
                    }
                    if (!obj_name.compare("Low_BF16")) {
                        MNN_PRINT("MNN use lowBF precision\n");
                        backendConfig->precision = MNN::BackendConfig::Precision_Low_BF16;
                    }
                    if (!obj_name.compare("high")) {
                        MNN_PRINT("MNN use high precision\n");
                        backendConfig->precision = MNN::BackendConfig::Precision_High;
                    }
                }
            }
        }

        if (-1 == ec::getVectorByKey(dict, "saveTensors", config.saveTensors)
            || -1 == ec::getVectorByKey(dict, "inputPaths", config.path.inputs)
            || -1 == ec::getVectorByKey(dict, "outputPaths", config.path.outputs)){
            return false;
        }
    }
    return true;
}


//------------------------ macro_utils start -------------------------
#define arg_half_size(...) \
         arg_half_size_(__VA_ARGS__, arg_half_rseq_n())
#define arg_half_size_(...) \
         arg_n(__VA_ARGS__)
#define arg_n( \
         _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,  \
         _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
         _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
         _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
         _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
         _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
         _61,_62,_63,_64,_65,_66,_67,_68,_69,_70, \
         _71,_72,_73,_74,_75,_76,_77,_78,_79,_80, \
         _81,_82,_83,_84,_85,_86,_87,_88,_89,_90, \
         _91,_92,_93,_94,_95,_96,_97,_98,_99,_100, \
         _101,_102,_103,_104,_105,_106,_107,_108,_109,_110, \
         _111,_112,_113,_114,_115,_116,_117,_118,_119,_120, \
         N,...) N
#define arg_half_rseq_n() \
        60,60,59,59,58,58,57,57, \
        56,56,55,55,54,54,53,53,52,52,51,51, \
        50,50,49,49,48,48,47,47,46,46,45,45, \
        44,44,43,43,42,42,41,41,40,40,39,39, \
        38,38,37,37,36,36,35,35,34,34,33,33, \
        32,32,31,31,30,30,29,29,28,28,27,27, \
        26,26,25,25,24,24,23,23,22,22,21,21, \
        20,20,19,19,18,18,17,17,16,16,15,15, \
        14,14,13,13,12,12,11,11,10,10, 9, 9, \
        8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0

#define arg_concat_impl(x, y) x ## y
#define arg_concat(x, y) arg_concat_impl(x, y)
#define arg_if_1(THEN, ELSE) THEN
#define arg_if_0(THEN, ELSE) ELSE
// just support COND = 0 or 1
#define arg_if(COND, THEN, ELSE) arg_concat(arg_if_, COND)(THEN, ELSE)
#define expand_item_0(...)
#define expand_item_1(macro, context, key, value, ...) \
    macro(context, key, value)
#define expand_item_2(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_1(macro, context, __VA_ARGS__)
#define expand_item_3(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_2(macro, context, __VA_ARGS__)
#define expand_item_4(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_3(macro, context, __VA_ARGS__)
#define expand_item_5(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_4(macro, context, __VA_ARGS__)
#define expand_item_6(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_5(macro, context, __VA_ARGS__)
#define expand_item_7(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_6(macro, context, __VA_ARGS__)
#define expand_item_8(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_7(macro, context, __VA_ARGS__)
#define expand_item_9(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_8(macro, context, __VA_ARGS__)
#define expand_item_10(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_9(macro, context, __VA_ARGS__)
#define expand_item_11(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_10(macro, context, __VA_ARGS__)
#define expand_item_12(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_11(macro, context, __VA_ARGS__)
#define expand_item_13(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_12(macro, context, __VA_ARGS__)
#define expand_item_14(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_13(macro, context, __VA_ARGS__)
#define expand_item_15(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_14(macro, context, __VA_ARGS__)
#define expand_item_16(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_15(macro, context, __VA_ARGS__)
#define expand_item_17(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_16(macro, context, __VA_ARGS__)
#define expand_item_18(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_17(macro, context, __VA_ARGS__)
#define expand_item_19(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_18(macro, context, __VA_ARGS__)
#define expand_item_20(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_19(macro, context, __VA_ARGS__)
#define expand_item_21(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_20(macro, context, __VA_ARGS__)
#define expand_item_22(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_21(macro, context, __VA_ARGS__)
#define expand_item_23(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_22(macro, context, __VA_ARGS__)
#define expand_item_24(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_23(macro, context, __VA_ARGS__)
#define expand_item_25(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_24(macro, context, __VA_ARGS__)
#define expand_item_26(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_25(macro, context, __VA_ARGS__)
#define expand_item_27(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_26(macro, context, __VA_ARGS__)
#define expand_item_28(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_27(macro, context, __VA_ARGS__)
#define expand_item_29(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_28(macro, context, __VA_ARGS__)
#define expand_item_30(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_29(macro, context, __VA_ARGS__)
#define expand_item_31(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_30(macro, context, __VA_ARGS__)
#define expand_item_32(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_31(macro, context, __VA_ARGS__)
#define expand_item_33(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_32(macro, context, __VA_ARGS__)
#define expand_item_34(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_33(macro, context, __VA_ARGS__)
#define expand_item_35(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_34(macro, context, __VA_ARGS__)
#define expand_item_36(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_35(macro, context, __VA_ARGS__)
#define expand_item_37(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_36(macro, context, __VA_ARGS__)
#define expand_item_38(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_37(macro, context, __VA_ARGS__)
#define expand_item_39(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_38(macro, context, __VA_ARGS__)
#define expand_item_40(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_39(macro, context, __VA_ARGS__)
#define expand_item_41(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_40(macro, context, __VA_ARGS__)
#define expand_item_42(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_41(macro, context, __VA_ARGS__)
#define expand_item_43(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_42(macro, context, __VA_ARGS__)
#define expand_item_44(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_43(macro, context, __VA_ARGS__)
#define expand_item_45(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_44(macro, context, __VA_ARGS__)
#define expand_item_46(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_45(macro, context, __VA_ARGS__)
#define expand_item_47(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_46(macro, context, __VA_ARGS__)
#define expand_item_48(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_47(macro, context, __VA_ARGS__)
#define expand_item_49(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_48(macro, context, __VA_ARGS__)
#define expand_item_50(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_49(macro, context, __VA_ARGS__)
#define expand_item_51(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_50(macro, context, __VA_ARGS__)
#define expand_item_52(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_51(macro, context, __VA_ARGS__)
#define expand_item_53(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_52(macro, context, __VA_ARGS__)
#define expand_item_54(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_53(macro, context, __VA_ARGS__)
#define expand_item_55(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_54(macro, context, __VA_ARGS__)
#define expand_item_56(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_55(macro, context, __VA_ARGS__)
#define expand_item_57(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_56(macro, context, __VA_ARGS__)
#define expand_item_58(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_57(macro, context, __VA_ARGS__)
#define expand_item_59(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_58(macro, context, __VA_ARGS__)
#define expand_item_60(macro, context, key, value, ...) \
    macro(context, key, value) \
    expand_item_59(macro, context, __VA_ARGS__)
#define expand_items(macro, context, ...) \
    arg_concat(expand_item_, arg_half_size(__VA_ARGS__))(macro, context, __VA_ARGS__)
//------------------------ macro_utils end -------------------------
// ------------------------ enum start -----------------------------
typedef struct {
    PyObject_HEAD
    int value;
} PyMNNEnum;
static PyObject* PyEnum_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNEnum* self = (PyMNNEnum *)type->tp_alloc(type, 0);
    long val = 0;
    if (PyTuple_Size(args)) {
        if (!PyArg_ParseTuple(args, "l", &val)) {
            return NULL;
        }
    }
    self->value = (int)val;
    return (PyObject*)self;
}
Py_hash_t PyEnum_hash(PyObject* x) {
    return static_cast<Py_hash_t>(((PyMNNEnum*)x)->value);
}
static PyObject* toPyEnum(PyObject* type, int val) {
    auto args = PyTuple_New(1);
    PyTuple_SetItem((PyObject*)args, 0, PyLong_FromLong((long)val));
    auto e = PyObject_Call(type, args, NULL);
    Py_XDECREF(args);
    if (!e) {
        PyErr_SetString(PyExc_Exception,
                        "toEnum: PyMNNEnum instance create failed");
        return NULL;
    }
    return e;
}
template <typename T>
static T toEnum(PyObject* e) {
    if (!e) {
        return static_cast<T>(0);
    }
    return static_cast<T>(((PyMNNEnum*)e)->value);
}
#define declare_map_item(_, key, value)  { static_cast<int>(key), value },
#define register_item(context, key, value) { \
    auto pykey = toPyObj(key); \
    PyObject_SetAttrString(scope, value, pykey); \
    PyDict_SetItemString(dict, value, pykey); \
    Py_XDECREF(pykey); }

#define def_enum_repr(NAME, ...) \
static PyObject* PyEnum_##NAME##_repr(PyObject *self) { \
    std::string str = #NAME "."; \
    std::map<int, const char*> items = { \
        expand_items(declare_map_item, _, __VA_ARGS__) \
    }; \
    int key = ((PyMNNEnum*)self)->value; \
    auto iter = items.find(key); \
    str += (iter != items.end() ? iter->second : "???"); \
    return Py_BuildValue("s", str.c_str()); \
}

#define def_enum_to(NAME, TYPE) \
static PyObject* toPyObj(TYPE value) { \
    return toPyEnum((PyObject*)PyType_FindTLSType(&PyEnum_##NAME), static_cast<int>(value)); \
}

#define def_enum_register(NAME, ...) \
static void def_##NAME(PyObject *scope) { \
    if (PyType_Ready(&PyEnum_##NAME) < 0) { \
        PyErr_SetString(PyExc_Exception, "init " #NAME ": PyType_Ready failed"); \
    } \
    PyObject* self = (PyObject *)PyType_FindTLSType(&PyEnum_##NAME); \
    PyObject* dict = ((PyTypeObject *)self)->tp_dict; \
    PyModule_AddObject(scope, #NAME, self); \
    expand_items(register_item, NAME, __VA_ARGS__) \
}

#define def_enum(NAME, TYPE, ...) \
def_enum_repr(NAME, __VA_ARGS__) \
PyObject *PyEnum_##NAME##richcompare(PyObject *self, PyObject *other, int op); \
static PyTypeObject PyEnum_##NAME = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    #NAME,                                    /*tp_name*/\
    sizeof(PyMNNEnum),                        /*tp_basicsize*/\
    0,                                        /*tp_itemsize*/\
    0,                                        /*tp_dealloc*/\
    0,                                        /*tp_print*/\
    0,                                        /*tp_getattr*/\
    0,                                        /*tp_setattr*/\
    0,                                        /*tp_compare*/\
    PyEnum_##NAME##_repr,                     /*tp_repr*/\
    0,                                        /*tp_as_number*/\
    0,                                        /*tp_as_sequence*/\
    0,                                        /*tp_as_mapping*/\
    PyEnum_hash,                              /*tp_hash*/\
    0,                                        /*tp_call*/\
    PyEnum_##NAME##_repr,                     /*tp_str*/\
    0,                                        /*tp_getattro*/\
    0,                                        /*tp_setattro*/\
    0,                                        /*tp_as_buffer*/\
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/\
    "PyMNNEnum",                              /*tp_doc*/\
    0,                                        /*tp_traverse*/\
    0,                                        /*tp_clear*/\
    &PyEnum_##NAME##richcompare,              /*tp_richcompare*/\
    0,                                        /*tp_weaklistoffset*/\
    0,                                        /*tp_iter*/\
    0,                                        /*tp_iternext*/\
    0,                                        /*tp_methods*/\
    0,                                        /*tp_members*/\
    0,                                        /*tp_getset*/\
    0,                                        /*tp_base*/\
    0,                                        /*tp_dict*/\
    0,                                        /*tp_descr_get*/\
    0,                                        /*tp_descr_set*/\
    0,                                        /*tp_dictoffset*/\
    0,                                        /*tp_init*/\
    0,                                        /*tp_alloc*/\
    PyEnum_new                                /*tp_new*/\
};\
static inline bool is##NAME(PyObject* obj) { return Py_TYPE(obj) == PyType_FindTLSType(&PyEnum_##NAME); } \
PyObject *PyEnum_##NAME##richcompare(PyObject *self, PyObject *other, int op) { \
    if (!is##NAME(other)) Py_RETURN_FALSE; \
    int l = ((PyMNNEnum*)self)->value, r = ((PyMNNEnum*)other)->value; \
    switch (op) { \
        case Py_LT: return toPyObj(l < r); \
        case Py_LE: return toPyObj(l <= r); \
        case Py_EQ: return toPyObj(l == r); \
        case Py_NE: return toPyObj(l != r); \
        case Py_GT: return toPyObj(l > r); \
        case Py_GE: return toPyObj(l >= r); \
    } \
    Py_RETURN_FALSE; \
} \
def_enum_to(NAME, TYPE) \
def_enum_register(NAME, __VA_ARGS__)
// ------------------------ enum end --------------------------
// ------------------------ func start ------------------------
#define def_methods(MODULE, NAME) \
for (int i = 0; i < (sizeof(PyMNN##NAME##_methods) / sizeof(PyMethodDef)); i++) { \
    def_method(MODULE, &PyMNN##NAME##_methods[i]); \
}

#define register_func(SCOPE, NAME, DOC) {#NAME, (PyCFunction)PyMNN##SCOPE##_##NAME, METH_VARARGS, DOC},
#define register_func_kw(SCOPE, NAME, DOC) {#NAME, (PyCFunction)PyMNN##SCOPE##_##NAME, METH_VARARGS|METH_KEYWORDS, DOC},
#define register_methods(SCOPE, ...) expand_items(register_func, SCOPE, __VA_ARGS__)
#define register_methods_kw(SCOPE, ...) expand_items(register_func_kw, SCOPE, __VA_ARGS__)
#define declare_unary(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    PyObject *x; \
    if (PyArg_ParseTuple(args, "O", &x) && isVar(x)) { \
        return toPyObj(FUNC(toVar(x))); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var)"); \
}
#define declare_binary(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    PyObject *l, *r; \
    if (PyArg_ParseTuple(args, "OO", &l, &r) && isVar(l) && isVar(r)) { \
        return toPyObj(FUNC(toVar(l), toVar(r))); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, Var)"); \
}
#define declare_reduce(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    INTS default_shape = {}; \
    PyObject *x, *axis = nullptr; \
    int keep_dims = 0; \
    if (PyArg_ParseTuple(args, "O|Oi", &x, &axis, &keep_dims) \
        && isVar(x) && (axis == nullptr || isInts(axis))) { \
        return toPyObj(FUNC(toVar(x), PARSE(axis, default_shape, toInts), keep_dims)); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, |[int], bool)"); \
}
#define declare_eltwise(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    PyObject *l, *r, *coeff; \
    if (PyArg_ParseTuple(args, "OOO", &l, &r, &coeff) \
        && isVar(l) && isVar(r) && isFloats(coeff)) { \
        return toPyObj(FUNC(toVar(l), toVar(r), toFloats(coeff))); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, Var, [floats])"); \
}
#define declare_axis_op(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    PyObject *x; \
    int axis; \
    if (PyArg_ParseTuple(args, "Oi", &x, &axis) && isVar(x)) { \
        return toPyObj(FUNC(toVar(x), axis)); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, int)"); \
}
#define declare_axiss_op(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    INTS default_axis = {}; \
    PyObject *x, *axis = nullptr; \
    if (PyArg_ParseTuple(args, "O|O", &x, &axis) \
        && isVar(x) && (axis == nullptr || isInts(axis))) { \
        return toPyObj(FUNC(toVar(x), PARSE(axis, default_axis, toInts))); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, |[int])"); \
}
#define declare_triple(SCOPE, NAME, FUNC) \
static PyObject* PyMNN##SCOPE##_##NAME(PyObject *self, PyObject *args) { \
    PyObject *x, *y, *z; \
    if (PyArg_ParseTuple(args, "OOO", &x, &y, &z) \
        && isVar(x) && isVar(y) && isVar(z)) { \
        return toPyObj(FUNC(toVar(x), toVar(y), toVar(z))); \
    } \
    PyMNN_ERROR(#NAME " require args: (Var, Var, Var)"); \
}
#define def_unary(SCOPE, ...) expand_items(declare_unary, SCOPE, __VA_ARGS__)
#define def_binary(SCOPE, ...) expand_items(declare_binary, SCOPE, __VA_ARGS__)
#define def_reduce(SCOPE, ...) expand_items(declare_reduce, SCOPE, __VA_ARGS__)
#define def_eltwise(SCOPE, ...) expand_items(declare_eltwise, SCOPE, __VA_ARGS__)
#define def_axis_op(SCOPE, ...) expand_items(declare_axis_op, SCOPE, __VA_ARGS__)
#define def_axiss_op(SCOPE, ...) expand_items(declare_axiss_op, SCOPE, __VA_ARGS__)
#define def_triple(SCOPE, ...) expand_items(declare_triple, SCOPE, __VA_ARGS__)
// ------------------------ func end ---------------------------
// ------------------------ class start ------------------------
#define declare_getter(SCOPE, NAME, _) \
    static PyObject* PyMNN##SCOPE##_get##NAME(PyMNN##SCOPE *self, void *closure);
#define declare_setter_impl(SCOPE, NAME) \
    static int PyMNN##SCOPE##_set##NAME(PyMNN##SCOPE *self, PyObject *value, void *closure);
#define declare_setter(SCOPE, NAME, HASSET) \
    arg_if(HASSET, declare_setter_impl(SCOPE, NAME), )

#define declare_method(SCOPE, NAME, X) \
    static PyObject* PyMNN##SCOPE##_##NAME(PyMNN##SCOPE *self, PyObject *args);

#define register_set(SCOPE, NAME) (setter)PyMNN##SCOPE##_set##NAME
#define register_getset(SCOPE, NAME, HASSET) \
    {#NAME, (getter)PyMNN##SCOPE##_get##NAME, arg_if(HASSET, register_set(SCOPE, NAME), NULL), #NAME, NULL},

#define def_class_register(NAME) \
static void def_##NAME(PyObject *scope) { \
    if (PyType_Ready(&PyMNN##NAME##Type) < 0) { \
        PyErr_SetString(PyExc_Exception, "init" #NAME ": PyType_Ready PyMNN" #NAME "Type failed"); \
    } \
    PyObject* self = (PyObject *)PyType_FindTLSType(&PyMNN##NAME##Type); \
    PyModule_AddObject(scope, #NAME, self); \
}

#define def_class_start(NAME, TYPE) \
typedef struct { \
    PyObject_HEAD \
    TYPE* ptr; \
} PyMNN##NAME;
#define def_class_getset(NAME, ...) \
expand_items(declare_getter, NAME, __VA_ARGS__) \
expand_items(declare_setter, NAME, __VA_ARGS__) \
static PyGetSetDef PyMNN##NAME##_getsetters[] = { \
    expand_items(register_getset, NAME, __VA_ARGS__) \
    {NULL}  /* Sentinel */ \
};
#define def_class_methods(NAME, ...) \
expand_items(declare_method, NAME, __VA_ARGS__) \
static PyMethodDef PyMNN##NAME##_methods[] = { \
    expand_items(register_func, NAME, __VA_ARGS__) \
    {NULL}  /* Sentinel */ \
};
#define def_class_end(NAME, TYPE) \
static PyObject* PyMNN##NAME##_new(PyTypeObject *type, PyObject *args, PyObject *kwds); \
static int PyMNN##NAME##_init(PyTypeObject *self, PyObject *args, PyObject *kwds); \
static PyObject* PyMNN##NAME##_call(PyObject *self, PyObject *args, PyObject *kwds); \
static void PyMNN##NAME##_dealloc(PyMNN##NAME *self) { \
    if (self->ptr) { \
        delete self->ptr; \
    } \
    Py_TYPE(self)->tp_free((PyObject *) self); \
} \
static PyTypeObject PyMNN##NAME##Type = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    #NAME,                                    /*tp_name*/\
    sizeof(PyMNN##NAME),                      /*tp_basicsize*/\
    0,                                        /*tp_itemsize*/\
    (destructor)PyMNN##NAME##_dealloc,        /*tp_dealloc*/\
    0,                                        /*tp_print*/\
    0,                                        /*tp_getattr*/\
    0,                                        /*tp_setattr*/\
    0,                                        /*tp_compare*/\
    0,                                        /*tp_repr*/\
    0,                                        /*tp_as_number*/\
    0,                                        /*tp_as_sequence*/\
    0,                                        /*tp_as_mapping*/\
    0,                                        /*tp_hash*/\
    PyMNN##NAME##_call,                       /*tp_call*/\
    0,                                        /*tp_str*/\
    0,                                        /*tp_getattro*/\
    0,                                        /*tp_setattro*/\
    0,                                        /*tp_as_buffer*/\
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/\
    "MNN " #NAME " objects",                  /*tp_doc*/\
    0,                                        /*tp_traverse*/\
    0,                                        /*tp_clear*/\
    0,                                        /*tp_richcompare*/\
    0,                                        /*tp_weaklistoffset*/\
    0,                                        /*tp_iter*/\
    0,                                        /*tp_iternext*/\
    PyMNN##NAME##_methods,                    /*tp_methods*/\
    0,                                        /*tp_members*/\
    PyMNN##NAME##_getsetters,                 /*tp_getset*/\
    0,                                        /*tp_base*/\
    0,                                        /*tp_dict*/\
    0,                                        /*tp_descr_get*/\
    0,                                        /*tp_descr_set*/\
    0,                                        /*tp_dictoffset*/\
    (initproc)PyMNN##NAME##_init,             /*tp_init*/\
    0,                                        /*tp_alloc*/\
    PyMNN##NAME##_new                         /*tp_new*/\
};\
def_class_register(NAME) \
static PyMNN##NAME* get##NAME() { \
    return (PyMNN##NAME *)PyObject_CallObject((PyObject*)PyType_FindTLSType(&PyMNN##NAME##Type), NULL); \
} \
static PyObject* toPyObj(TYPE* x) { \
    auto ret = get##NAME(); \
    ret->ptr = x; \
    return (PyObject*)ret; \
} \
static TYPE* to##NAME(PyObject* m) { \
    return ((PyMNN##NAME*)m)->ptr; \
}

// define an empty list for class without getter/setter
#define def_class_without_getset(NAME) \
static PyGetSetDef PyMNN##NAME##_getsetters[] = { \
    {NULL}  /* Sentinel */ \
};
// define a basic new impl for class
#define class_basic_new_impl(NAME) \
static PyObject* PyMNN##NAME##_new(PyTypeObject *type, PyObject *args, PyObject *kwds) { \
    PyMNN##NAME *self = (PyMNN##NAME *)type->tp_alloc(type, 0); \
    return (PyObject*)self; \
}
#define class_basic_init_impl(NAME) \
static int PyMNN##NAME##_init(PyTypeObject *self, PyObject *args, PyObject *kwds) { \
    return 0; \
}
#define class_basic_call_impl(NAME) \
static PyObject* PyMNN##NAME##_call(PyObject *self, PyObject *args, PyObject *kwds) { \
    return (PyObject*)self; \
}
// ------------------------ class start ------------------------
// ------------------------ capsule start ------------------------


#define def_class_smart_start(NAME, TYPE) \
typedef struct { \
    PyObject_HEAD \
    std::shared_ptr<TYPE>* ptr; \
} PyMNN##NAME;
#define def_class_smart_end(NAME, TYPE) \
static PyObject* PyMNN##NAME##_new(PyTypeObject *type, PyObject *args, PyObject *kwds); \
static PyObject* PyMNN##NAME##_call(PyObject *self, PyObject *args, PyObject *kwds); \
static void PyMNN##NAME##_dealloc(PyMNN##NAME *self) { \
    if (self->ptr) { \
        delete self->ptr; \
    } \
    Py_TYPE(self)->tp_free((PyObject *) self); \
} \
static PyTypeObject PyMNN##NAME##Type = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    #NAME,                                    /*tp_name*/\
    sizeof(PyMNN##NAME),                      /*tp_basicsize*/\
    0,                                        /*tp_itemsize*/\
    (destructor)PyMNN##NAME##_dealloc,        /*tp_dealloc*/\
    0,                                        /*tp_print*/\
    0,                                        /*tp_getattr*/\
    0,                                        /*tp_setattr*/\
    0,                                        /*tp_compare*/\
    0,                                        /*tp_repr*/\
    0,                                        /*tp_as_number*/\
    0,                                        /*tp_as_sequence*/\
    0,                                        /*tp_as_mapping*/\
    0,                                        /*tp_hash*/\
    PyMNN##NAME##_call,                       /*tp_call*/\
    0,                                        /*tp_str*/\
    0,                                        /*tp_getattro*/\
    0,                                        /*tp_setattro*/\
    0,                                        /*tp_as_buffer*/\
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/\
    "MNN " #NAME " objects",                  /*tp_doc*/\
    0,                                        /*tp_traverse*/\
    0,                                        /*tp_clear*/\
    0,                                        /*tp_richcompare*/\
    0,                                        /*tp_weaklistoffset*/\
    0,                                        /*tp_iter*/\
    0,                                        /*tp_iternext*/\
    PyMNN##NAME##_methods,                    /*tp_methods*/\
    0,                                        /*tp_members*/\
    PyMNN##NAME##_getsetters,                 /*tp_getset*/\
    0,                                        /*tp_base*/\
    0,                                        /*tp_dict*/\
    0,                                        /*tp_descr_get*/\
    0,                                        /*tp_descr_set*/\
    0,                                        /*tp_dictoffset*/\
    0,                                        /*tp_init*/\
    0,                                        /*tp_alloc*/\
    PyMNN##NAME##_new                         /*tp_new*/\
};\
def_class_register(NAME) \
static PyMNN##NAME* get##NAME() { \
    return (PyMNN##NAME *)PyObject_CallObject((PyObject*)PyType_FindTLSType(&PyMNN##NAME##Type), NULL); \
} \
static PyObject* toPyObj(TYPE* x) { \
    auto ret = get##NAME(); \
    (*(ret->ptr)).reset(x); \
    return (PyObject*)ret; \
} \
static std::shared_ptr<TYPE>* to##NAME(PyObject* m) { \
    return ((PyMNN##NAME*)m)->ptr; \
}


#define def_capsule(TYPE) \
static void del_##TYPE(PyObject *obj) { \
    free(PyCapsule_GetPointer(obj, #TYPE)); \
} \
static TYPE* to##TYPE(PyObject *obj) { \
    return (TYPE *)PyCapsule_GetPointer(obj, #TYPE); \
} \
static PyObject* from##TYPE(TYPE *p) { \
    return PyCapsule_New(p, #TYPE, del_##TYPE); \
}
// ------------------------ capsule start ------------------------
