#pragma once

#ifndef PYMNN_USE_ALINNPYTHON
#ifndef PYMNN_EXPR_API
#error PYMNN_EXPR_API macro should be define on official python (PYMNN_USE_ALINNPYTHON=OFF)
#endif // PYMNN_EXPR_API
#ifndef PYMNN_NUMPY_USABLE
#error PYMNN_NUMPY_USABLE macro should be define on official python (PYMNN_USE_ALINNPYTHON=OFF)
#endif // PYMNN_NUMPY_USABLE
#endif // PYMNN_USE_ALINNPYTHON

#if defined(ANDROID) || defined(__ANDROID__)
#undef _FILE_OFFSET_BITS
#endif
#include <fstream>

// NOTE: global_new_python_flag be declared here to avoid affect by AliNNPython
#ifdef PYMNN_USE_ALINNPYTHON
#ifdef PYMNN_RUNTIME_CHECK_VM
#ifdef WIN32
#define EXTERN_IMPORT(type) extern "C" __declspec(dllimport) type
#else
#define EXTERN_IMPORT(type) extern "C" type
#endif // WIN32
EXTERN_IMPORT(int) global_new_python_flag;
#else
#ifdef PYMNN_NEW_PYTHON
static int global_new_python_flag = 1;
#else
static int global_new_python_flag = 0;
#endif // PYMNN_NEW_PYTHON
#endif // PYMNN_RUNTIME_CHECK_VM
#else
static int global_new_python_flag = 0;
#endif // PYMNN_USE_ALINNPYTHON

#ifdef PYMNN_USE_ALINNPYTHON
#include <AliNNPython/Python.h>
#include <AliNNPython/frameobject.h>
#include <AliNNPython/pythread.h>
#include "renameForAliNNPython.h"

#ifdef PYMNN_NUMPY_USABLE
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#endif // PYMNN_NUMPY_USABLE

#else
#define PyType_FindTLSType
#include <Python.h>
#include <pythread.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#endif // PYMNN_USE_ALINNPYTHON
