#pragma once

#ifndef PYMNN_USE_ALINNPYTHON
#ifndef PYMNN_EXPR_API
#error PYMNN_EXPR_API macro should be define on official python (PYMNN_USE_ALINNPYTHON=OFF)
#endif
#ifndef PYMNN_NUMPY_USABLE
#error PYMNN_NUMPY_USABLE macro should be define on official python (PYMNN_USE_ALINNPYTHON=OFF)
#endif
#endif

#if defined(ANDROID) || defined(__ANDROID__)
#undef _FILE_OFFSET_BITS
#endif
#include <fstream>

#ifdef PYMNN_USE_ALINNPYTHON
#include <AliNNPython/Python.h>
#include <AliNNPython/frameobject.h>
#include <AliNNPython/pythread.h>
#include "renameForAliNNPython.h"

#ifdef PYMNN_NUMPY_USABLE
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#endif

#else
#define PyType_FindTLSType
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#endif
