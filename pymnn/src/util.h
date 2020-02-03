#pragma once
#include <string>
using namespace MNN;
using namespace MNN::Express;
using namespace std;
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
inline void store_scalar(void* data, DataType dtype, PyObject* obj) {
  switch (dtype) {
    case DataType_DT_UINT8: *(uint8_t*)data = (uint8_t)unpackLong(obj); break;
    case DataType_DT_INT32: *(int32_t*)data = (int32_t)unpackLong(obj); break;
    case DataType_DT_INT64: *(int64_t*)data = unpackLong(obj); break;
    case DataType_DT_FLOAT: *(float*)data = (float)unpackDouble(obj); break;
    case DataType_DT_DOUBLE: *(double*)data = (double)unpackDouble(obj); break;
    default: throw std::runtime_error("invalid type");
  }
}
INTS getshape(PyObject* seq) {
  INTS shape;
  while (PySequence_Check(seq)) {
    auto length = PySequence_Length(seq);
    if (length < 0) throw std::exception();
    shape.push_back(length);
    if (shape.size() > 20) {
      throw std::exception();
    }
    if (length == 0) break;
    seq = PySequence_GetItem(seq,0);
  }
  return shape;
}
void recursive_store(char* data, INTS shape, INTS stride, int dim, PyObject* obj, DataType dtype, int elementSize) {
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
}
