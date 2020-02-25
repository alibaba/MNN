#pragma once
#include <string>
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

