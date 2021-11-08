/*
    MNN python module
    PYMNN_EXPR_API: MNN.expr, MNN.nn
    PYMNN_TRAIN_API: MNN.nn.compress, MNN.nn.loss, MNN.data, MNN.optim
*/
#include "MNNPyBridge.h"
#include "common.h"
#include "util.h"

static int tls_key = 0;
static int tls_key_2 = 0;

#ifdef PYMNN_EXPR_API
#ifdef PYMNN_USE_ALINNPYTHON
#include "pybind_private/pybind11.h"
#include "pybind_private/stl.h"
#include "pybind_private/operators.h"
#else
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"
#endif // PYMNN_USE_ALINNPYTHON
#endif // PYMNN_EXPR_API

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#ifdef PYMNN_EXPR_API
namespace py = pybind11;
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
//#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
using namespace MNN::Express;
#endif // PYMNN_EXPR_API

#ifdef BUILD_OPTYPE
#include "MNN_generated.h"
#endif // BUILD_OPTYPE

#ifdef PYMNN_TRAIN_API
#include "NN.hpp"
#include "OpGrad.hpp"
#include "ParameterOptimizer.hpp"
#include "SGD.hpp"
#include "ADAM.hpp"
#include "Dataset.hpp"
#include "DataLoader.hpp"
#include "Loss.hpp"
#include "Transformer.hpp"
#include "PipelineModule.hpp"
#include "cpp/ConvertToFullQuant.hpp"
#include "cpp/HelperFuncs.hpp"
using namespace MNN::Train;
#endif // PYMNN_TRAIN_API

#include <mutex>
#include <unordered_map>

using namespace MNN;

using namespace std;

struct MNN_TLSData {
    PyObject *PyMNNHalideTypeInt = NULL;
    PyObject *PyMNNHalideTypeInt64 = NULL;
    PyObject *PyMNNHalideTypeFloat = NULL;
    PyObject *PyMNNHalideTypeDouble = NULL;
    PyObject *PyMNNHalideTypeUint8 = NULL;
    PyObject *PyMNNHalideTypeString = NULL;
    std::unordered_map<std::string, Interpreter *> *interpreterMap = NULL;
    std::unordered_map<std::string, Session *> *sessionCacheMap = NULL;
};
static MNN_TLSData* old_python_data = NULL;
static MNN_TLSData * getTLSData() {
    if(global_new_python_flag > 0) {
        return static_cast<MNN_TLSData*>(PyThread_get_key_value(tls_key));
    }else{
        return old_python_data;
    }
}
static void setTLSData(MNN_TLSData* tlsData) {
    if(global_new_python_flag > 0) {
        PyThread_set_key_value(tls_key, tlsData);
    } else {
        old_python_data = tlsData;
    }
}

#if defined(PYMNN_EXPR_API) && defined(PYMNN_USE_ALINNPYTHON)
static void set_rh_tls_data(py::detail::rh_tls* rh_tls) {
    if(global_new_python_flag > 0) {
        PyThread_set_key_value(tls_key_2, rh_tls);
    } else {
        py::detail::old_rh_tls_data = rh_tls;
    }
}
#endif

#ifdef PYMNN_EXPR_API
namespace py = pybind11;
#endif
static PyObject *importName(const char *name, const char *symbol)
{
    PyObject *u_name, *module;
    u_name = PyUnicode_FromString(name);
    module = PyImport_Import(u_name);
    if (!module) {
        return NULL;
    }
    Py_DECREF(u_name);
    return PyObject_GetAttrString(module, symbol);
}

typedef struct {
    PyObject_HEAD
    std::string *modelPath;
    Interpreter *interpreter;
} PyMNNInterpreter;

typedef struct {
    PyObject_HEAD
    std::string *modelPath;
    Session *session;
} PyMNNSession;

typedef struct {
    PyObject_HEAD
    Tensor *tensor;
    int owner;
} PyMNNTensor;

typedef struct {
    PyObject_HEAD
    CV::ImageProcess *imageProcess;
} PyMNNCVImageProcess;

typedef struct {
    PyObject_HEAD
    CV::Matrix *matrix;
} PyMNNCVMatrix;

typedef struct {
    PyObject_HEAD
    const OperatorInfo *opInfo;
}PyMNNOpInfo;
halide_type_t* httInt() {
    static halide_type_t httInt = halide_type_of<int>();
    return &httInt;
}

halide_type_t* httInt64() {
    static halide_type_t httInt64 = halide_type_of<int64_t>();
    return &httInt64;
}

halide_type_t* httFloat() {
    static halide_type_t httFloat = halide_type_of<float>();
    return &httFloat;
}

halide_type_t* httDouble() {
    static halide_type_t httDouble = halide_type_of<double>();
    return &httDouble;
}

halide_type_t* httUint8() {
    static halide_type_t httUint8 = halide_type_of<uint8_t>();
    return &httUint8;
}

halide_type_t* httString() {
    static halide_type_t httString = halide_type_t(halide_type_handle, sizeof(void*)*8);
    return &httString;
}

/// MNN NetInstance Type
static PyObject* PyMNNInterpreter_createRuntime(PyObject *self, PyObject *args);
static PyObject* PyMNNInterpreter_createSession(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_resizeSession(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_resizeTensor(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_runSession(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_runSessionWithCallBack(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_runSessionWithCallBackInfo(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getSessionInput(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getSessionOutput(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getSessionInputAll(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getSessionOutputAll(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getSessionInfo(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_setCacheFile(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_updateCacheFile(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_cache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_removeCache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_updateSessionToModel(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static int PyMNNInterpreter_init(PyMNNInterpreter *self, PyObject *args, PyObject *kwds);
static void PyMNNInterpreter_dealloc(PyMNNInterpreter *);

static PyMethodDef PyMNNInterpreter_methods[] = {
    {"createRuntime", (PyCFunction)PyMNNInterpreter_createRuntime, METH_VARARGS | METH_STATIC, "create runtime"},
    {"createSession", (PyCFunction)PyMNNInterpreter_createSession, METH_VARARGS, "create session"},
    {"setCacheFile", (PyCFunction)PyMNNInterpreter_setCacheFile, METH_VARARGS, "set cache file for create session"},
    {"updateCacheFile", (PyCFunction)PyMNNInterpreter_updateCacheFile, METH_VARARGS, "update cache file after resize session"},
    {"resizeSession", (PyCFunction)PyMNNInterpreter_resizeSession, METH_VARARGS, "resize session"},
    {"runSession", (PyCFunction)PyMNNInterpreter_runSession, METH_VARARGS, "run session"},
    {"runSessionWithCallBack", (PyCFunction)PyMNNInterpreter_runSessionWithCallBack, METH_VARARGS, "run session with callback"},
    {"runSessionWithCallBackInfo", (PyCFunction)PyMNNInterpreter_runSessionWithCallBackInfo, METH_VARARGS, "run session with callback info"},
    {"getSessionOutput", (PyCFunction)PyMNNInterpreter_getSessionOutput, METH_VARARGS, "get session output"},
    {"getSessionInput", (PyCFunction)PyMNNInterpreter_getSessionInput, METH_VARARGS, "get session input"},
    {"getSessionOutputAll", (PyCFunction)PyMNNInterpreter_getSessionOutputAll, METH_VARARGS, "get session output all"},
    {"getSessionInputAll", (PyCFunction)PyMNNInterpreter_getSessionInputAll, METH_VARARGS, "get session input all"},
    {"getSessionInfo", (PyCFunction)PyMNNInterpreter_getSessionInfo, METH_VARARGS, "getSessionInfo"},
    {"resizeTensor", (PyCFunction)PyMNNInterpreter_resizeTensor, METH_VARARGS, "resize tensor"},
    {"cache", (PyCFunction)PyMNNInterpreter_cache, METH_VARARGS, "cache current net instance"},
    {"removeCache", (PyCFunction)PyMNNInterpreter_removeCache, METH_VARARGS, "remove cache with given path"},
    {"updateSessionToModel", (PyCFunction)PyMNNInterpreter_updateSessionToModel, METH_VARARGS, "updateSessionToModel"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNInterpreterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.Interpreter",                   /*tp_name*/
    sizeof(PyMNNInterpreter),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNInterpreter_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN Interpreter objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNInterpreter_methods,                      /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNInterpreter_init,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNInterpreter_new,                          /* tp_new */
};

/// MNN Session Type
static PyObject* PyMNNSession_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static void PyMNNSession_dealloc(PyMNNSession *);
static PyObject* PyMNNSession_cache(PyMNNSession *self, PyObject *args);
static PyObject* PyMNNSession_removeCache(PyMNNSession *self, PyObject *args);

static PyMethodDef PyMNNSession_methods[] = {
    {"cache", (PyCFunction)PyMNNSession_cache, METH_VARARGS, "cache current session instance"},
    {"removeCache", (PyCFunction)PyMNNSession_removeCache, METH_VARARGS, "remove session cache with given path"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNSessionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.Session",                   /*tp_name*/
    sizeof(PyMNNSession),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNSession_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN Session objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNSession_methods,                   /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNSession_new,                          /* tp_new */
};

/// MNN Tensor Type
static PyObject* PyMNNTensor_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static void PyMNNTensor_dealloc(PyMNNTensor *);
static int PyMNNTensor_init(PyMNNTensor *self, PyObject *args, PyObject *kwds);
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNTensor_fromNumpy(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getNumpyData(PyMNNTensor *self, PyObject *args);
#endif
static PyObject* PyMNNTensor_printTensorData(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getShape(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getDataType(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getDimensionType(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getData(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getHost(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_copyFrom(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_copyToHostTensor(PyMNNTensor *self, PyObject *args);

static PyMethodDef PyMNNTensor_methods[] = {
#ifdef PYMNN_NUMPY_USABLE
    {"fromNumpy", (PyCFunction)PyMNNTensor_fromNumpy, METH_VARARGS, "copy data from numpy"},
    {"getNumpyData", (PyCFunction)PyMNNTensor_getNumpyData, METH_NOARGS, "get tensor data (numpy)"},
#endif
    {"printTensorData", (PyCFunction)PyMNNTensor_printTensorData, METH_NOARGS, "print tensor data"},
    {"getShape", (PyCFunction)PyMNNTensor_getShape, METH_NOARGS, "get tensor shape"},
    {"getDataType", (PyCFunction)PyMNNTensor_getDataType, METH_NOARGS, "get tensor data type"},
    {"getData", (PyCFunction)PyMNNTensor_getData, METH_NOARGS, "get tensor data (tuple)"},
    {"getHost", (PyCFunction)PyMNNTensor_getHost, METH_NOARGS, "get tensor host"},
    {"getDimensionType", (PyCFunction)PyMNNTensor_getDimensionType, METH_NOARGS, "get dimension data"},
    {"copyFrom", (PyCFunction)PyMNNTensor_copyFrom, METH_VARARGS, "copy data from host tensor"},
    {"copyFromHostTensor", (PyCFunction)PyMNNTensor_copyFrom, METH_VARARGS, "copy data from host tensor"},
    {"copyToHostTensor", (PyCFunction)PyMNNTensor_copyToHostTensor, METH_VARARGS, "copy data to host tensor"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.Tensor",                   /*tp_name*/
    sizeof(PyMNNTensor),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNTensor_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN Tensor objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNTensor_methods,                                   /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNTensor_init,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNTensor_new,                          /* tp_new */
};

/// MNN ImageProcess Type
static PyObject* PyMNNCVImageProcess_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static void PyMNNCVImageProcess_dealloc(PyMNNCVImageProcess *);
static int PyMNNCVImageProcess_init(PyMNNCVImageProcess *self, PyObject *args, PyObject *kwds);
static PyObject* PyMNNCVImageProcess_setMatrix(PyMNNCVImageProcess *self, PyObject *args);
static PyObject* PyMNNCVImageProcess_convert(PyMNNCVImageProcess *self, PyObject *args);
static PyObject* PyMNNCVImageProcess_createImageTensor(PyMNNCVImageProcess *self, PyObject *args);

static PyMethodDef PyMNNCVImageProcess_methods[] = {
    {"setMatrix", (PyCFunction)PyMNNCVImageProcess_setMatrix, METH_VARARGS, "ImageProcess setMatrix"},
    {"convert", (PyCFunction)PyMNNCVImageProcess_convert, METH_VARARGS, "ImageProcess convert"},
    {"createImageTensor", (PyCFunction)PyMNNCVImageProcess_createImageTensor, METH_VARARGS, "ImageProcess create Image Tensor"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNCVImageProcessType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.CVImageProcess",                   /*tp_name*/
    sizeof(PyMNNCVImageProcess),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNCVImageProcess_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN CVImageProcess objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNCVImageProcess_methods,                                   /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNCVImageProcess_init,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNCVImageProcess_new,                          /* tp_new */
};

/// MNN CVMatrix Type
static PyObject* PyMNNCVMatrix_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static void PyMNNCVMatrix_dealloc(PyMNNCVMatrix *);
/// scale
static PyObject* PyMNNCVMatrix_setScale(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_preScale(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_postScale(PyMNNCVMatrix *, PyObject *args);
/// rotate
static PyObject* PyMNNCVMatrix_setRotate(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_preRotate(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_postRotate(PyMNNCVMatrix *, PyObject *args);
/// translate
static PyObject* PyMNNCVMatrix_setTranslate(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_preTranslate(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_postTranslate(PyMNNCVMatrix *, PyObject *args);

static PyObject* PyMNNCVMatrix_invert(PyMNNCVMatrix *);

static PyMethodDef PyMNNCVMatrix_methods[] = {
    {"setScale", (PyCFunction)PyMNNCVMatrix_setScale, METH_VARARGS, "MNNCVMatrix setScale"},
    {"preScale", (PyCFunction)PyMNNCVMatrix_preScale, METH_VARARGS, "MNNCVMatrix preScale"},
    {"postScale", (PyCFunction)PyMNNCVMatrix_postScale, METH_VARARGS, "MNNCVMatrix postScale"},

    {"setRotate", (PyCFunction)PyMNNCVMatrix_setRotate, METH_VARARGS, "MNNCVMatrix setRotate"},
    {"preRotate", (PyCFunction)PyMNNCVMatrix_preRotate, METH_VARARGS, "MNNCVMatrix preRotate"},
    {"postRotate", (PyCFunction)PyMNNCVMatrix_postRotate, METH_VARARGS, "MNNCVMatrix postRotate"},

    {"setTranslate", (PyCFunction)PyMNNCVMatrix_setTranslate, METH_VARARGS, "MNNCVMatrix setTranslate"},
    {"preTranslate", (PyCFunction)PyMNNCVMatrix_preTranslate, METH_VARARGS, "MNNCVMatrix preTranslate"},
    {"postTranslate", (PyCFunction)PyMNNCVMatrix_postTranslate, METH_VARARGS, "MNNCVMatrix postTranslate"},

    {"invert", (PyCFunction)PyMNNCVMatrix_invert, METH_VARARGS, "MNNCVMatrix invert"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNCVMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.CVImageProcess",                   /*tp_name*/
    sizeof(PyMNNCVMatrix),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNCVMatrix_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN CVMatrix objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNCVMatrix_methods,                                   /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNCVMatrix_new,                          /* tp_new */
};

/// MNN NetInstance implementation
// 用来缓存net的实例

std::unordered_map<std::string, Interpreter *> *interpreterMap() {
//    static std::unordered_map<std::string, Interpreter *> *interpreterMap = nullptr; // <path, instance>
//    static std::once_flag flag;
//    std::call_once(flag, [](){interpreterMap = new std::unordered_map<std::string, Interpreter *>();});
    struct MNN_TLSData *tlsData = getTLSData();
    if (tlsData == nullptr) {
        return nullptr;
    }
    return tlsData->interpreterMap;
}

std::unordered_map<std::string, Session *> *sessionCacheMap() {
    struct MNN_TLSData *tlsData = getTLSData();
    if (tlsData == nullptr) {
        return nullptr;
    }
    return tlsData->sessionCacheMap;
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

static std::pair<bool, std::pair<ScheduleConfig, std::shared_ptr<BackendConfig>>> createScheduleConfig(PyObject* dict) {
    std::pair<bool, std::pair<ScheduleConfig, std::shared_ptr<BackendConfig>>> result;
    result.first = false;
    auto& config = result.second.first;
    auto& backendConfig = result.second.second;
    backendConfig.reset(new BackendConfig);
    config.backendConfig = backendConfig.get();
    
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
                {"AUTO", MNN_FORWARD_AUTO}
            };
            auto iter = backend_map.find(backend_name);
            if (iter == backend_map.end()) {
                // backend not support, issue on python level when development
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: backend not support");
                return result;
            }
            config.type = iter->second;
        } else if (backend && PyLong_Check(backend)) {
            config.type = (MNNForwardType)PyLong_AsLong(backend); // {'backend': 1L} for example
        }
        PyObject *numThread = PyDict_GetItemString(dict, "numThread");
        if (numThread) {
            if (!PyLong_Check(numThread)) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: numThread must be a integer");
                return result;
            }
            config.numThread = (int)PyLong_AsLong(numThread);
        }

        {
            //precision
            PyObject *obj = PyDict_GetItemString(dict, "precision");
            if (obj) {
                auto obj_name = object2String(obj);
                if (!obj_name.compare("low")) {
                    MNN_PRINT("MNN use low precision\n");
                    backendConfig->precision = MNN::BackendConfig::Precision_Low;
                }
            }
        }

        if (-1 == ec::getVectorByKey(dict, "saveTensors", config.saveTensors)
            || -1 == ec::getVectorByKey(dict, "inputPaths", config.path.inputs)
            || -1 == ec::getVectorByKey(dict, "outputPaths", config.path.outputs)){
            return result;
        }
    }
    result.first = true;
    return result;
}

static void _runtime_capsule_deleter(PyObject *obj) {
    auto info = (RuntimeInfo*)PyCapsule_GetPointer(obj, NULL);
    if (info != nullptr) {
        delete info;
    }
}

static PyObject* PyMNNInterpreter_createRuntime(PyObject* self, PyObject* args) {
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dicts = NULL;
    if (!PyArg_ParseTuple(args, "|O", &dicts)) {
        return NULL;
    }
    if (!PySequence_Check(dicts)) {
        return Py_None;
    }
    // BackendConfig lifetime management
    std::vector<ScheduleConfig> configs;
    std::vector<std::shared_ptr<BackendConfig>> backend_configs_backup;
    for (auto i = 0; i < PySequence_Size(dicts); ++i) {
        auto config = createScheduleConfig(PySequence_GetItem(dicts, i));
        if (!config.first) {
            return Py_None;
        }
        configs.push_back(config.second.first);
        backend_configs_backup.push_back(config.second.second);
    }
    
    auto info = new RuntimeInfo;
    *info = instance->interpreter->createRuntime(configs);
    auto res = PyTuple_New(2), runtime_exists = PyTuple_New(configs.size());
    for (size_t i = 0; i < configs.size(); ++i) {
        bool runtime_exist = (info->first.find(configs[i].type) != info->first.end());
        PyTuple_SetItem(runtime_exists, i, (runtime_exist ? Py_True : Py_False));
    }
    PyTuple_SetItem(res, 0, PyCapsule_New(info, NULL, _runtime_capsule_deleter));
    PyTuple_SetItem(res, 1, runtime_exists);
    return res;
}

static PyObject* PyMNNInterpreter_createSession(PyMNNInterpreter *self, PyObject *args) {
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dict = NULL, *rtinfo_py = NULL;
    if (!PyArg_ParseTuple(args, "|OO", &dict, &rtinfo_py)) {
        return NULL;
    }

    PyObject *f = importName("MNN", "Session");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session not found");
        return NULL;
    }

    // create a new session
    PyMNNSession *session = (PyMNNSession *)PyObject_Call(f, PyTuple_New(0), NULL);
    if (!session) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session instance create failed");
        return NULL;
    }

    if (self->modelPath && (*sessionCacheMap())[*self->modelPath]) {
        session->modelPath = self->modelPath;
        session->session = (*sessionCacheMap())[*self->modelPath];
        return (PyObject *)session;
    }

    auto config = createScheduleConfig(dict);
    if (!config.first) {
        return NULL;
    }
    Session* s;
    if (rtinfo_py == NULL) {
        s = instance->interpreter->createSession(config.second.first);
    } else {
        auto runtimeinfo = *(RuntimeInfo*)PyCapsule_GetPointer(rtinfo_py, NULL);
        s = instance->interpreter->createSession(config.second.first, runtimeinfo);
    }
    if (!s) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: NetInstance createSession failed");
        return NULL;
    }

    session->session = s;
    session->modelPath = instance->modelPath;

    return (PyObject *)session;
}

static PyObject* PyMNNInterpreter_resizeSession(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    if (!PyArg_ParseTuple(args, "O", &session)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_resizeSession: First argument is not a MNN.Session instance");
        return NULL;
    }

    self->interpreter->resizeSession(session->session);
    Py_RETURN_TRUE;
}

static PyObject* PyMNNInterpreter_resizeTensor(PyMNNInterpreter *self, PyObject *args) {
    PyMNNTensor* tensor = NULL;
    PyObject* shape = NULL;
    if (!PyArg_ParseTuple(args, "OO", &tensor, &shape)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(tensor, PyType_FindTLSType(&PyMNNTensorType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_resizeTensor: First argument is not a MNN.Tensor instance");
        return NULL;
    }

    if (!PyTuple_Check(shape)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_resizeTensor: Second argument is not a tuple");
        return NULL;
    }

    size_t shapeSize = PyTuple_Size(shape);

    std::vector<int> vShape;
    for (size_t i = 0; i < shapeSize; i++) {
        int shapeItem = (int)PyLong_AsLong(PyTuple_GetItem(shape, i));
        vShape.push_back(shapeItem);
    }

    self->interpreter->resizeTensor(tensor->tensor, vShape);
    Py_RETURN_NONE;
}
static PyObject* PyMNNInterpreter_setCacheFile(PyMNNInterpreter *self, PyObject *args) {
    char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_setCacheFile: Not string input");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    self->interpreter->setCacheFile(path);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static PyObject* PyMNNInterpreter_updateCacheFile(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    int flag = 0;
    if (!PyArg_ParseTuple(args, "Oi", &session, &flag)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_updateCacheFile: First argument is not a MNN.Session instance");
        return NULL;
    }

    ErrorCode r = NO_ERROR;
    r = self->interpreter->updateCacheFile(session->session, flag);
    return PyLong_FromLong(r);
}
static PyObject* PyMNNInterpreter_runSession(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    if (!args) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSession: No argument passed, expect 1");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O", &session)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSession: First argument is not a MNN.Session instance");
        return NULL;
    }
    ErrorCode r = NO_ERROR;
    Py_BEGIN_ALLOW_THREADS
    r = self->interpreter->runSession(session->session);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(r);
}
static PyObject* PyMNNInterpreter_runSessionWithCallBack(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    PyObject *beginCallback = NULL;
    PyObject *endCallback = NULL;
    if (!args) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBack: No argument passed, expect 1 or 3");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O|OO", &session, &beginCallback, &endCallback)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBack: First argument is not a AliNN.Session instance");
        return NULL;
    }

    TensorCallBack begin = [beginCallback](const std::vector<Tensor*>& tensors, const std::string& name){

        if (!beginCallback || !PyCallable_Check(beginCallback)) {

            return true;
        }

        PyObject *f = importName("MNN", "Tensor");
            if (!f || !PyCallable_Check(f)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBack: MNN.Tensor not found");
             return true;
        }

        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBack: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        //printf("begincallback name=%s\n",name.c_str());
        PyObject *weStringData = char2Object(name.c_str());
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, weStringData);
        bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(beginCallback, args, NULL)));
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };
    TensorCallBack end = [endCallback](const std::vector<Tensor*>& tensors, const std::string& name){
        if (!endCallback || !PyCallable_Check(endCallback)) {
            return true;
        }
        PyObject *f = importName("MNN", "Tensor");
            if (!f || !PyCallable_Check(f)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBack: MNN.Tensor not found");
             return true;
        }
        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBack: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        PyObject *weStringData = char2Object(name.c_str());
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, weStringData);
        bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(endCallback, args, NULL)));
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };

    ErrorCode r = NO_ERROR;
    //Py_BEGIN_ALLOW_THREADS
    r = self->interpreter->runSessionWithCallBack(session->session, begin, end);
    //Py_END_ALLOW_THREADS
    return PyLong_FromLong(r);
}

static PyObject* PyMNNInterpreter_runSessionWithCallBackInfo(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    PyObject *beginCallback = NULL;
    PyObject *endCallback = NULL;
    if (!args) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: No argument passed, expect 1 or 3");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O|OO", &session, &beginCallback, &endCallback)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: First argument is not a AliNN.Session instance");
        return NULL;
    }

    TensorCallBackWithInfo begin = [beginCallback](const std::vector<Tensor*>& tensors, const OperatorInfo* info){

        if (!beginCallback || !PyCallable_Check(beginCallback)) {

            return true;
        }

        PyObject *ftensor = importName("MNN", "Tensor");
        PyObject *finfo = importName("MNN", "OpInfo");
        if (!ftensor || !PyCallable_Check(ftensor)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBackINfo: MNN.Tensor not found");
             return true;
        }
        if (!finfo || !PyCallable_Check(finfo)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBackInfo: MNN.OpInfo not found");
             return true;
        }

        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(ftensor, PyTuple_New(0), NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        //printf("begincallback name=%s\n",name.c_str());
        PyMNNOpInfo *pyinfo = (PyMNNOpInfo *)PyObject_Call(finfo,PyTuple_New(0), NULL);
        if(!pyinfo){
            PyErr_SetString(PyExc_Exception,
                    "PyMNNInterpreter_runSessionWithCallBackInfo: create OpInfo failed");
            return true;
        }
        pyinfo->opInfo = info;
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, (PyObject *)pyinfo);
        bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(beginCallback, args, NULL)));
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };
    TensorCallBackWithInfo end = [endCallback](const std::vector<Tensor*>& tensors, const OperatorInfo* info){
        if (!endCallback || !PyCallable_Check(endCallback)) {
            return true;
        }
        PyObject *ftensor = importName("MNN", "Tensor");
        PyObject *finfo = importName("MNN", "OpInfo");
        if (!ftensor || !PyCallable_Check(ftensor)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBackInfo: MNN.Tensor not found");
             return true;
        }
        if (!finfo || !PyCallable_Check(finfo)) {
                    PyErr_SetString(PyExc_Exception,
                             "PyMNNInterpreter_runSessionWithCallBackInfo: MNN.OpInfo not found");
             return true;
        }
        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(ftensor, PyTuple_New(0), NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        PyMNNOpInfo *pyinfo = (PyMNNOpInfo *)PyObject_Call(finfo,PyTuple_New(0), NULL);
        if(!pyinfo){
            PyErr_SetString(PyExc_Exception,
                    "PyMNNInterpreter_runSessionWithCallBackInfo: create OpInfo failed");
            return true;
        }
        pyinfo->opInfo = info;
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, (PyObject *)pyinfo);
        bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(endCallback, args, NULL)));
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };

    ErrorCode r = NO_ERROR;
    //Py_BEGIN_ALLOW_THREADS
    r = self->interpreter->runSessionWithCallBackInfo(session->session, begin, end);
    //Py_END_ALLOW_THREADS
    return PyLong_FromLong(r);
}


static PyObject* PyMNNInterpreter_getSessionOutput(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    char* name = NULL;
    if (!PyArg_ParseTuple(args, "O|s", &session, &name)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionOutput: First argument is not a MNN.Session instance");
        return NULL;
    }

    Tensor *t = self->interpreter->getSessionOutput(session->session, name);
    if (!t) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionOutput: Get output failed");
        return NULL;
    }

    PyObject *f = importName("MNN", "Tensor");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionOutput: MNN.Tensor not found");
        return NULL;
    }

    // create a new tensor
    PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session instance create failed");
        return NULL;
    }

    tensor->tensor = t;
    return (PyObject *)tensor;
}

static PyObject* PyMNNInterpreter_getSessionInput(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    char* name = NULL;
    if (!PyArg_ParseTuple(args, "O|s", &session, &name)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionInput: First argument is not a MNN.Session instance");
        return NULL;
    }

    Tensor *t = self->interpreter->getSessionInput(session->session, name);
    if (!t) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionInput: Get input failed");
        return NULL;
    }

    PyObject *f = importName("MNN", "Tensor");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionInput: MNN.Tensor not found");
        return NULL;
    }

    // create a new tensor
    PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session instance create failed");
        return NULL;
    }

    tensor->tensor = t;
    return (PyObject *)tensor;
}

static PyObject* PyMNNInterpreter_getSessionOutputAll(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    if (!PyArg_ParseTuple(args, "O", &session)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionOutputAll: First argument is not a MNN.Session instance");
        return NULL;
    }
    PyObject *f = importName("MNN", "Tensor");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionOutputAll: MNN.Tensor not found");
        return NULL;
    }
    auto map = self->interpreter->getSessionOutputAll(session->session);
    PyObject* output = PyDict_New();
    for (auto it=map.begin(); it!=map.end(); ++it) {
        PyObject *tensor = PyObject_Call(f, PyTuple_New(0), NULL);
        if (!tensor) {
            PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionOutputAll: MNN.Tensor instance create failed");
            return NULL;
        }
        ((PyMNNTensor*)tensor)->tensor = it->second;
        PyDict_SetItem(output, char2Object(it->first.c_str()), tensor);
    }
    return output;
}

static PyObject* PyMNNInterpreter_getSessionInfo(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    int info_type_val;
    if (!PyArg_ParseTuple(args, "Oi", &session, &info_type_val)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionInfo: First argument is not a MNN.Session instance");
        return NULL;
    }

    auto info_type = (MNN::Interpreter::SessionInfoCode)info_type_val;
    if (info_type == MNN::Interpreter::BACKENDS) {
        int backendType[2];
        self->interpreter->getSessionInfo(session->session, info_type, backendType);
        return PyLong_FromLong(backendType[0]);
    }
    float result;
    self->interpreter->getSessionInfo(session->session, info_type, &result);
    return PyFloat_FromDouble(result);
}

static PyObject* PyMNNInterpreter_getSessionInputAll(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    if (!PyArg_ParseTuple(args, "O", &session)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionInputAll: First argument is not a MNN.Session instance");
        return NULL;
    }
    PyObject *f = importName("MNN", "Tensor");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionInputAll: MNN.Tensor not found");
        return NULL;
    }
    auto map = self->interpreter->getSessionInputAll(session->session);
    PyObject* output = PyDict_New();
    for (auto it=map.begin(); it!=map.end(); ++it) {
        PyObject *tensor = PyObject_Call(f, PyTuple_New(0), NULL);
        if (!tensor) {
            PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionInputAll: MNN.Tensor instance create failed");
            return NULL;
        }
        ((PyMNNTensor*)tensor)->tensor = it->second;
        PyDict_SetItem(output, char2Object(it->first.c_str()), tensor);
    }
    return output;
}

PyObject* PyMNNInterpreter_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNInterpreter* self = (PyMNNInterpreter *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyMNNInterpreter_init(PyMNNInterpreter *self, PyObject *args, PyObject *kwds) {
    char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_new: PyArg_ParseTuple failed");
        return -1;
    }
    auto converted_path = convertBytesEncodeIfNeed(path);
    self->modelPath = new std::string(converted_path.data());
    if (!self->modelPath) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_new: create modelPath string failed");
        return -1;
    }

    if ((*interpreterMap())[*self->modelPath]) {
        self->interpreter = (*interpreterMap())[*self->modelPath];
    } else {
        self->interpreter = Interpreter::createFromFile(path);
    }
    if (!self->interpreter) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_new: NetInstance::createFromFile failed");
        return -1;
    }

    return 0;
}

static PyObject* PyMNNInterpreter_cache(PyMNNInterpreter *self, PyObject *args) {
    if (self->modelPath && !(*interpreterMap())[*self->modelPath]) {
        (*interpreterMap())[*self->modelPath] = self->interpreter;
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNInterpreter_removeCache(PyMNNInterpreter *self, PyObject *args) {
    if (!self->modelPath) {
        Py_RETURN_NONE;
    }
    Interpreter* net = (*interpreterMap())[*self->modelPath];
    if (net) {
        interpreterMap()->erase(*self->modelPath);
        //delete net;
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNInterpreter_updateSessionToModel(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    char* name = NULL;
    if (!PyArg_ParseTuple(args, "O|s", &session, &name)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(session, PyType_FindTLSType(&PyMNNSessionType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_updateSessionToModel: First argument is not a MNN.Session instance");
        return NULL;
    }

    self->interpreter->updateSessionToModel(session->session);
    if(name){
        auto modelBuffer = self->interpreter->getModelBuffer();
        ofstream output(name);
        output.write((const char*)modelBuffer.first, modelBuffer.second);
    }
    Py_RETURN_NONE;
}

static void PyMNNInterpreter_dealloc(PyMNNInterpreter *self) {
    if (!self->modelPath) {
        return;
    }
    Interpreter* net = (*interpreterMap())[*self->modelPath];
    // 如果对象不存在缓存中， 则释放实例
    if (!net && self->interpreter) {
        delete self->interpreter;
        self->interpreter = NULL;
    }
    delete self->modelPath;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/// MNN Session implementation
static PyObject* PyMNNSession_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNSession* self = (PyMNNSession *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void PyMNNSession_dealloc(PyMNNSession *self) {
    self->session = NULL;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// cache session
static PyObject* PyMNNSession_cache(PyMNNSession *self, PyObject *args) {
    if (!self->modelPath) {
        Py_RETURN_NONE;
    }
    if (!(*sessionCacheMap())[*self->modelPath]) {
        (*sessionCacheMap())[*self->modelPath] = self->session;
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNSession_removeCache(PyMNNSession *self, PyObject *args) {
    if (!self->modelPath) {
        Py_RETURN_NONE;
    }
    Session* s = (*sessionCacheMap())[*self->modelPath];
    if (s) {
        sessionCacheMap()->erase(*self->modelPath);
    }
    Py_RETURN_NONE;
}

/// MNN Tensor implementation
static PyObject* PyMNNTensor_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNTensor* self = (PyMNNTensor *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void PyMNNTensor_dealloc(PyMNNTensor *self) {
    if (self->owner) {
        if (self->tensor->host<void *>()) {
            free(self->tensor->host<void *>());
        }
        delete self->tensor;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyMNNTensor_init(PyMNNTensor *self, PyObject *args, PyObject *kwds) {
    if (!PyTuple_Size(args)) {
        return 0;
    }

    PyObject *shape, *dataType, *data;
    long dimensionType;
    if (!PyArg_ParseTuple(args, "OOOl", &shape, &dataType, &data, &dimensionType)) {
        return -1;
    }

    size_t shapeSize = PyTuple_Size(shape);

    std::vector<int> vShape;
    size_t dataSize = 1;
    for (size_t i = 0; i<shapeSize; i++) {
        int shapeItem = (int)PyLong_AsLong(PyTuple_GetItem(shape, i));
        vShape.push_back(shapeItem);
        dataSize *= shapeItem;
    }
    bool isNumpy = false;
    void *pData = NULL;
    if(PyTuple_Check(data)) {
        if(dataSize != PyTuple_Size(data)) {
            PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_init: Tensor Dim not match");
            return -1;
        }
    }
#ifdef PYMNN_NUMPY_USABLE
    else {
        if(PyArray_Check(data)) {
            isNumpy = true;
            if(dataSize != PyArray_Size(data)) {
                PyErr_SetString(PyExc_Exception, "PyMNNTensor_init: numpy array size does not match shape requirement");
                return -1;
            }
        }
        else {
            PyErr_SetString(PyExc_Exception, "PyMNNTensor_init: data is not tuple/numpy");
            return -1;
        }
    }
#endif
    halide_type_t htt;
    struct MNN_TLSData *tlsData = getTLSData();
    if (dataType == tlsData->PyMNNHalideTypeInt) {
        htt = halide_type_of<int32_t>();
    }
    else if(dataType == tlsData->PyMNNHalideTypeFloat) {
        htt = halide_type_of<float>();
    }
    else if(dataType == tlsData->PyMNNHalideTypeDouble) {
        htt = halide_type_of<float>();
    }
    else if(dataType == tlsData->PyMNNHalideTypeUint8) {
        htt = halide_type_of<uint8_t>();
    }
    else if(dataType == tlsData->PyMNNHalideTypeInt64) {
        htt = halide_type_of<int64_t>();
    }
    else if(dataType == tlsData->PyMNNHalideTypeString) {
        htt = *httString();
    }
    else {
        PyErr_SetString(PyExc_Exception,"PyMNNTensor_create: unsupported data type");
        return -1;
    }
    DType dtype = htype2dtype(htt);
    if(!isNumpy) {
        int itemsize = getitemsize(dtype);
        pData = malloc(dataSize * itemsize);
        if(NULL == pData) {
            PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
            return -1;
        }
        if (dataType == tlsData->PyMNNHalideTypeInt) {
            for (size_t i = 0; i < dataSize; i++) {
                ((int *)pData)[i] = (int)PyLong_AsLong(PyTuple_GetItem(data, i));
            }
        } else if (dataType == tlsData->PyMNNHalideTypeFloat) {
            for (size_t i = 0; i < dataSize; i++) {
                ((float *)pData)[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(data, i));
            }
        } else if (dataType == tlsData->PyMNNHalideTypeDouble) {
            for (size_t i = 0; i < dataSize; i++) {
               ((double *)pData)[i] = PyFloat_AsDouble(PyTuple_GetItem(data, i));
            }
        } else if (dataType == tlsData->PyMNNHalideTypeUint8) {
            for (size_t i = 0; i < dataSize; i++) {
               ((uint8_t *)pData)[i] = (uint8_t)PyLong_AsLong(PyTuple_GetItem(data, i));
            }
        } else if (dataType == tlsData->PyMNNHalideTypeInt64) {
            for (size_t i = 0; i < dataSize; i++) {
               ((int64_t *)pData)[i] = (int64_t)PyLong_AsLong(PyTuple_GetItem(data, i));
            }
         } else if (dataType == tlsData->PyMNNHalideTypeString) {
            for (size_t i = 0; i < dataSize; i++) {
               char *item = (char *)object2String(PyTuple_GetItem(data, i)).c_str();
               ((char **)pData)[i] = item;
            }
        }
    }
#ifdef PYMNN_NUMPY_USABLE
    else {
        int npy_type = PyArray_TYPE(data);
        int itemsize = getitemsize(dtype, npy_type);
        pData = malloc(dataSize * itemsize);
        if(NULL == pData) {
            PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
            return -1;
        }
        PyArrayObject *data_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)data);
        auto tmpBuffer = PyArray_DATA(data_cont);
        if(NULL == tmpBuffer) {
             PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: ndarry failed to get buffer data");
             return -1;
        }
        memcpy(pData, tmpBuffer, dataSize * itemsize);
        Py_XDECREF(data_cont);
     }
 #endif
    Tensor *tensor = Tensor::create(vShape
                               , htt
                               , pData
                               , (Tensor::DimensionType)dimensionType
                               );
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_create: Tensor create failed");
        return -1;
    }
    self->tensor = tensor;
    self->owner = 1;
    return 0;
}
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNTensor_fromNumpy(PyMNNTensor *self, PyObject *args) {
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        return NULL;
    }
    if (!PyArray_Check(data)) {
        PyErr_SetString(PyExc_Exception,"PyMNNTensor_fromNumpy: input is not a numpy");
    }
    if (self->owner){
        if(self->tensor->size() != PyArray_Size(data)) {
            PyErr_SetString(PyExc_Exception,"PyMNNTensor_fromNumpy: tensor/numpy size does not match each other");
            return NULL;
        }
        DType dtype = htype2dtype(self->tensor->getType());
        int npy_type = PyArray_TYPE(data);
        int itemsize = getitemsize(dtype, npy_type);
        PyArrayObject *data_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)data);
        auto tmpBuffer = PyArray_DATA(data_cont);
        if(NULL == tmpBuffer) {
             PyErr_SetString(PyExc_Exception,"PyMNNTensor_fromNumpy: ndarry failed to get buffer data");
             return NULL;
        }
        memcpy(self->tensor->host<void *>(), tmpBuffer, self->tensor->size() * itemsize);
        Py_XDECREF(data_cont);
    }
    Py_RETURN_NONE;
}
#endif
static PyObject* PyMNNTensor_printTensorData(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        // Do nothing
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNTensor_getHost(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        return PyCapsule_New(self->tensor->host<void *>(), NULL, NULL);
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNTensor_getDataType(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        halide_type_t t = self->tensor->getType();
        PyObject *type;
        struct MNN_TLSData *tlsData =getTLSData();
        if (t == *httInt()) {
            type = tlsData->PyMNNHalideTypeInt;
        } else if (t == *httUint8()) {
            type = tlsData->PyMNNHalideTypeUint8;
        } else if (t == *httInt64()) {
            type = tlsData->PyMNNHalideTypeInt64;
        } else if (t == *httFloat()) {
            type = tlsData->PyMNNHalideTypeFloat;
        } else if (t == *httDouble()) {
            type = tlsData->PyMNNHalideTypeDouble;
        } else if (t == *httString()) {
            type = tlsData->PyMNNHalideTypeString;
        } else {
            Py_RETURN_NONE;
        }
        Py_XINCREF(type);
        return type;
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNTensor_getData(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        halide_type_t t = self->tensor->getType();
        size_t size = self->tensor->elementSize();
        PyObject *outputData = PyTuple_New(size);
        if (t == *httInt()) {
            auto data = self->tensor->host<int32_t>();
            for (size_t i = 0; i < size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
         } else if (t == *httUint8()) {
            auto data = self->tensor->host<uint8_t>();
            for (size_t i = 0; i < size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
         } else if (t == *httInt64()) {
            auto data = self->tensor->host<int64_t>();
            for (size_t i = 0; i < size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
         } else if (t == *httFloat()) {
            auto data = self->tensor->host<float>();
            for (size_t i = 0; i < size; i++) {
                PyTuple_SetItem(outputData, i, PyFloat_FromDouble(data[i]));
            }
         } else if (t == *httDouble()) {
            auto data = self->tensor->host<double>();
            for (size_t i = 0; i < size; i++) {
                PyTuple_SetItem(outputData, i, PyFloat_FromDouble(data[i]));
            }
         } else if (t == *httString()) {
            auto data = self->tensor->host<char *>();
            for (size_t i = 0; i < size; i++) {
                char *dataItem = data[i];
                PyTuple_SetItem(outputData, i, char2Object(dataItem?dataItem:""));
            }
         } else {
            Py_RETURN_NONE;
         }
         return outputData;
    }
    Py_RETURN_NONE;
}

#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNTensor_getNumpyData(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        halide_type_t t = self->tensor->getType();
        std::vector<npy_intp> npy_dims;
        for(const auto dim : self->tensor->shape()) {
            npy_dims.push_back(dim);
        }
        PyObject* obj;
        if (t == *httInt()) {
            auto data = self->tensor->host<int32_t>();
            obj = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT32, data);
        } else if (t == *httUint8()) {
            auto data = self->tensor->host<uint8_t>();
            obj = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_UINT8, data);
        } else if (t == *httInt64()) {
            auto data = self->tensor->host<int64_t>();
            obj = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT64, data);
        } else if (t == *httFloat()) {
            auto data = self->tensor->host<float>();
            obj = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_FLOAT, data);
        } else if (t == *httDouble()) {
            auto data = self->tensor->host<double>();
            obj = PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_DOUBLE, data);
        } else {
            PyErr_SetString(PyExc_Exception, "tensor can not be read as numpy");
            Py_RETURN_NONE;
        }
        return obj;
    }
    Py_RETURN_NONE;
}
#endif

static PyObject* PyMNNTensor_getDimensionType(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        return PyLong_FromLong(self->tensor->getDimensionType());
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNTensor_copyFrom(PyMNNTensor *self, PyObject *args) {
    PyMNNTensor *fromTensor = NULL;
    if (!PyArg_ParseTuple(args, "O", &fromTensor)) {
        return NULL;
    }

    if (!fromTensor->tensor || !self->tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_copyFrom: source or destination tensor is null");
    }

    bool r = self->tensor->copyFromHostTensor(fromTensor->tensor);
    if (!r) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject* PyMNNTensor_copyToHostTensor(PyMNNTensor *self, PyObject *args) {
    PyMNNTensor *toTensor = NULL;
    if (!PyArg_ParseTuple(args, "O", &toTensor)) {
        return NULL;
    }

    if (!toTensor->tensor || !self->tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_copyTo: source or destination tensor is null");
    }

    bool r = self->tensor->copyToHostTensor(toTensor->tensor);
    if (!r) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject* PyMNNTensor_getShape(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        PyObject *shape = PyTuple_New(self->tensor->shape().size());
        for (size_t i = 0; i < self->tensor->shape().size(); i++) {
            PyTuple_SetItem(shape, i, PyLong_FromLong(self->tensor->shape()[i]));
        }
        return shape;
    }
    Py_RETURN_NONE;
}

/// MNN ImageProcess implementation
static PyObject* PyMNNCVImageProcess_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNCVImageProcess* self = (PyMNNCVImageProcess *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void PyMNNCVImageProcess_dealloc(PyMNNCVImageProcess *self) {
    delete self->imageProcess;
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static int PyMNNCVImageProcess_init(PyMNNCVImageProcess *self, PyObject *args, PyObject *kwds) {
    PyObject *config = NULL, *destinationTensor = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &config, &destinationTensor)) {
        return -1;
    }

    Tensor *t = NULL;
    if (destinationTensor
        && PyObject_TypeCheck(destinationTensor, PyType_FindTLSType(&PyMNNTensorType))) {
        t = ((PyMNNTensor *)destinationTensor)->tensor;
    }

    CV::ImageProcess::Config c;
    if (PyDict_Check(config)) {
        PyObject *filterType = PyDict_GetItemString(config, "filterType");
        if (filterType && PyLong_Check(filterType)) {
            c.filterType = (CV::Filter)PyLong_AsLong(filterType);
        }

        PyObject *sourceFormat = PyDict_GetItemString(config, "sourceFormat");
        if (sourceFormat && PyLong_Check(sourceFormat)) {
            c.sourceFormat = (CV::ImageFormat)PyLong_AsLong(sourceFormat);
        }

        PyObject *destFormat = PyDict_GetItemString(config, "destFormat");
        if (destFormat && PyLong_Check(destFormat)) {
            c.destFormat = (CV::ImageFormat)PyLong_AsLong(destFormat);
        }

        PyObject *wrap = PyDict_GetItemString(config, "wrap");
        if (wrap && PyLong_Check(wrap)) {
            c.wrap = (CV::Wrap)PyLong_AsLong(wrap);
        }

        PyObject *mean = PyDict_GetItemString(config, "mean");
        if (mean) {
            if (!PyTuple_Check(mean) || PyTuple_Size(mean) != 4) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNCVImageProcess_init: mean must be a tuple with 4 elements");
                return -1;
            }
            for (int i = 0; i < 4; i++) {
                c.mean[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(mean, i));
            }
        }

        PyObject *normal = PyDict_GetItemString(config, "normal");
        if (normal) {
            if (!PyTuple_Check(normal) || PyTuple_Size(normal) != 4) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNCVImageProcess_init: normal must be a tuple with 4 elements");
                return -1;
            }
            for (int i = 0; i < 4; i++) {
                c.normal[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(normal, i));
            }
        }
    }

    CV::ImageProcess *imageProcess = CV::ImageProcess::create(c, t);
    if (!imageProcess) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_init: ImageProcess create failed");
        return -1;
    }

    self->imageProcess = imageProcess;
    return 0;
}

static PyObject* PyMNNCVImageProcess_setMatrix(PyMNNCVImageProcess *self, PyObject *args) {
    PyObject *matrix;
    if (!PyArg_ParseTuple(args, "O", &matrix)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(matrix, PyType_FindTLSType(&PyMNNCVMatrixType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_setMatrix: argument is not a matrix");
        return NULL;
    }

    self->imageProcess->setMatrix(*((PyMNNCVMatrix *)matrix)->matrix);
    Py_RETURN_NONE;
}

static PyObject* PyMNNCVImageProcess_convert(PyMNNCVImageProcess *self, PyObject *args) {
    PyObject *source, *dest;
    int iw, ih, stride;
    if (!PyArg_ParseTuple(args, "OiiiO", &source, &iw, &ih, &stride, &dest)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(dest, PyType_FindTLSType(&PyMNNTensorType))) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_convert: argument 4 is not a MNNTensor");
        return NULL;
    }

    if (PyCapsule_CheckExact(source)) {
        // Capsule Pointer
        ErrorCode ret = self->imageProcess->convert((const uint8_t *)PyCapsule_GetPointer(source, NULL),
                                                    iw, ih, stride,
                                                    ((PyMNNTensor *)dest)->tensor);
        return PyLong_FromLong(ret);
    } else if (PyTuple_Check(source)) {
        // Tuple Data
        size_t size = PyTuple_Size(source);

        void *pData = malloc(size * sizeof(uint8_t));
        for (size_t i = 0; i < size; i++) {
            ((uint8_t *)pData)[i] = (uint8_t)PyLong_AsLong(PyTuple_GetItem(source, i));
        }

        ErrorCode ret = self->imageProcess->convert((const uint8_t *)pData,
                                                    iw, ih, stride,
                                                    ((PyMNNTensor *)dest)->tensor);

        free(pData);

        return PyLong_FromLong(ret);
    }
#ifdef PYMNN_NUMPY_USABLE
    else if(PyArray_Check(source)) {
        // Array Data
        int npy_type = PyArray_TYPE(source);
        if(npy_type != NPY_UINT8) {
            PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_convert: only numpy.uint8 is supported for numpy");
            return NULL;
        }
        int64_t total_length = 1;
        for (size_t i = 0; i < ((PyMNNTensor *)dest)->tensor->shape().size(); i++) {
            total_length *= ((PyMNNTensor *)dest)->tensor->shape()[i];
        }
        if(PyArray_Size(source) < total_length) //as input may contain stride, so we can only do basic check
        {
            PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_convert: data length does not match tensor size");
            return NULL;
        }
        PyArrayObject *data_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)source);
        auto tmpBuffer = PyArray_DATA(data_cont);
        if(NULL == tmpBuffer) {
             PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: ndarry failed to get buffer data");
             return NULL;
        }
        ErrorCode ret = self->imageProcess->convert((const uint8_t *)tmpBuffer,
                                                    iw, ih, stride,
                                                    ((PyMNNTensor *)dest)->tensor);
        Py_XDECREF(data_cont);
        return PyLong_FromLong(ret);
    }
#endif

    PyErr_SetString(PyExc_Exception, "PyMNNCVImageProcess_convert: argument 0 is not a capsule or tuple or numpy");

    return NULL;
}


static PyObject* PyMNNCVImageProcess_createImageTensor(PyMNNCVImageProcess *self, PyObject *args) {

    PyObject *dataType;
    int width, height, bpp;
    PyObject *data;

    if (!PyArg_ParseTuple(args, "OiiiO", &dataType, &width, &height, &bpp, &data)) {
        return NULL;
    }


//    if (nullptr != data && !PyCapsule_CheckExact(data)) {
//        PyErr_SetString(PyExc_Exception,
//                        "PyMNNCVImageProcess_createImageTensor: argument 4 is not a capsule");
//        return NULL;
//    }

    std::vector<int> vShape = {1, height, width, bpp};

    halide_type_t htt;
    struct MNN_TLSData *tlsData = getTLSData();
    if (dataType == tlsData->PyMNNHalideTypeInt) {
        htt = halide_type_of<int32_t>();
    } else if (dataType == tlsData->PyMNNHalideTypeFloat) {
        htt = halide_type_of<float>();
    } else if (dataType == tlsData->PyMNNHalideTypeDouble) {
        htt = halide_type_of<double>();
    } else if (dataType == tlsData->PyMNNHalideTypeUint8) {
        htt = halide_type_of<uint8_t>();
    } else if (dataType == tlsData->PyMNNHalideTypeInt64) {
        htt = halide_type_of<int64_t>();
    } else if (dataType == tlsData->PyMNNHalideTypeString) {
        htt = *httString();
    }

    Tensor *tensor = Tensor::create(vShape, htt);
//    Tensor *tensor = Tensor::create(vShape, htt, PyCapsule_GetPointer(data, NULL));TODO
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_createImageTensor: Tensor create failed");
        return NULL;
    }

    PyObject *f = importName("MNN", "Tensor");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_createImageTensor: MNN.Tensor not found");
        return NULL;
    }

    PyMNNTensor *t = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
    if (!t) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_createImageTensor: create image tensor failed");
        return NULL;
    }

    t->tensor = tensor;
    t->owner = 1;
    return (PyObject *)t;
}

/// MNN CVMatrix implementation
static PyObject* PyMNNCVMatrix_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNCVMatrix* self;
    self = (PyMNNCVMatrix *)type->tp_alloc(type, 0);
    self->matrix = new CV::Matrix();
    return (PyObject*)self;
}

static void PyMNNCVMatrix_dealloc(PyMNNCVMatrix *self) {
    delete self->matrix;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// type: 0 set; 1 pre; 2 post
static PyObject* _PyMNNCVMatrix_Rotate(PyMNNCVMatrix *self, PyObject *args, int type) {
    float degrees, px = 0.0, py = 0.0;
    size_t argsCount = PyTuple_Size(args);
    if (argsCount == 1) {
        if (!PyArg_ParseTuple(args, "f", &degrees)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_Rotate: PyArg_ParseTuple failed");
            return NULL;
        }
    } else if (argsCount == 3) {
        if (!PyArg_ParseTuple(args, "fff", &degrees, &px, &py)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_Rotate: PyArg_ParseTuple failed");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVMatrix_Rotate: argument count error (should be 1 or 3)");
        return NULL;
    }

    if (argsCount == 1) {
        switch (type) {
            case 0:
                self->matrix->setRotate(degrees);
                break;
            case 1:
                self->matrix->preRotate(degrees);
                break;
            case 2:
                self->matrix->postRotate(degrees);
                break;
            default:
                break;
        }

    } else if (argsCount == 3) {
        switch (type) {
            case 0:
                self->matrix->setRotate(degrees, px, py);
                break;
            case 1:
                self->matrix->preRotate(degrees, px, py);
                break;
            case 2:
                self->matrix->postRotate(degrees, px, py);
                break;
            default:
                break;
        }
    }
    Py_RETURN_NONE;
}
// set
static PyObject* PyMNNCVMatrix_setRotate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Rotate(self, args, 0);
}
// pre
static PyObject* PyMNNCVMatrix_preRotate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Rotate(self, args, 1);
}
// post
static PyObject* PyMNNCVMatrix_postRotate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Rotate(self, args, 2);
}

static PyObject* _PyMNNCVMatrix_Scale(PyMNNCVMatrix *self, PyObject *args, int type) {
    float sx, sy, px = 0.0, py = 0.0;
    size_t argsCount = PyTuple_Size(args);
    if (argsCount == 2) {
        if (!PyArg_ParseTuple(args, "ff", &sx, &sy)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_Scale: PyArg_ParseTuple failed");
            return NULL;
        }
    } else if (argsCount == 4) {
        if (!PyArg_ParseTuple(args, "ffff", &sx, &sy, &px, &py)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_Scale: PyArg_ParseTuple failed");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVMatrix_Scale: argument count error (should be 2 or 4)");
        return NULL;
    }

    if (argsCount == 2) {
        switch (type) {
            case 0:
                self->matrix->setScale(sx, sy);
                break;
            case 1:
                self->matrix->preScale(sx, sy);
                break;
            case 2:
                self->matrix->postScale(sx, sy);
                break;
            default:
                break;
        }
    } else if (argsCount == 4) {
        switch (type) {
            case 0:
                self->matrix->setScale(sx, sy, px, py);
                break;
            case 1:
                self->matrix->preScale(sx, sy, px, py);
                break;
            case 2:
                self->matrix->postScale(sx, sy, px, py);
                break;
            default:
                break;
        }
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNCVMatrix_setScale(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Scale(self, args, 0);
}
static PyObject* PyMNNCVMatrix_preScale(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Scale(self, args, 1);
}
static PyObject* PyMNNCVMatrix_postScale(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Scale(self, args, 2);
}

static PyObject* _PyMNNCVMatrix_Translate(PyMNNCVMatrix *self, PyObject *args, int type) {
    float dx = 0.0, dy = 0.0;
    size_t argsCount = PyTuple_Size(args);
    if (argsCount == 2) {
        if (!PyArg_ParseTuple(args, "ff", &dx, &dy)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_postScale: PyArg_ParseTuple failed");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVMatrix_postScale: argument count error (should be 2 or 4)");
        return NULL;
    }

    switch (type) {
        case 0:
            self->matrix->setTranslate(dy, dy);
            break;
        case 1:
            self->matrix->preTranslate(dy, dy);
            break;
        case 2:
            self->matrix->postTranslate(dy, dy);
            break;
        default:
            break;
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNCVMatrix_setTranslate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Translate(self, args, 0);
}
static PyObject* PyMNNCVMatrix_preTranslate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Translate(self, args, 1);
}
static PyObject* PyMNNCVMatrix_postTranslate(PyMNNCVMatrix *self, PyObject *args) {
    return _PyMNNCVMatrix_Translate(self, args, 2);
}

static PyObject* PyMNNCVMatrix_invert(PyMNNCVMatrix *self) {

    self->matrix->invert(self->matrix);
    Py_RETURN_NONE;
}
static PyObject* PyMNNOpInfo_getName(PyMNNOpInfo *self, PyObject *args);
static PyObject* PyMNNOpInfo_getType(PyMNNOpInfo *self, PyObject *args);

static void PyMNNOpInfo_dealloc(PyMNNOpInfo *self);
static PyObject* PyMNNOpInfo_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static int PyMNNOpInfo_init(PyMNNOpInfo *info, PyObject *args, PyObject *kwds);

static PyMethodDef PyMNNOpInfo_methods[] = {
    {"getName", (PyCFunction)PyMNNOpInfo_getName, METH_VARARGS, "get op name"},
    {"getType", (PyCFunction)PyMNNOpInfo_getType, METH_VARARGS, "get op type"},
    {NULL}  /* Sentinel */
};
static PyObject* PyMNNOpInfo_getName(PyMNNOpInfo *self, PyObject *args) {
    PyObject *name = NULL;
    if (self->opInfo) {
        name = char2Object(self->opInfo->name().c_str());
    }
    return name;
}
static PyObject* PyMNNOpInfo_getType(PyMNNOpInfo *self, PyObject *args) {
    PyObject *type = NULL;
    if (self->opInfo) {
        type = char2Object(self->opInfo->type().c_str());
    }
    return type;
}
static PyTypeObject PyMNNOpInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.OpInfo",                   /*tp_name*/
    sizeof(PyMNNOpInfo),                      /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNOpInfo_dealloc,          /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN OpInfo objects",                    /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNOpInfo_methods,                                   /* tp_methods */
    0,                      /* tp_members */
    0,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNOpInfo_init,               /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNOpInfo_new,                          /* tp_new */
};
static PyObject* PyMNNOpInfo_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNOpInfo* self = (PyMNNOpInfo *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static int PyMNNOpInfo_init(PyMNNOpInfo *info, PyObject *args, PyObject *kwds) {
    return 0;
}

static void PyMNNOpInfo_dealloc(PyMNNOpInfo *self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}
/// module init
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

int add (int a , int b) {
    return a + b;
}

// _MOD_NAME [_mnncengine or MNN]
// MOD_NAME ["_mnncengine" or "MNN"]
#if PYMNN_USE_ALINNPYTHON
#if PYMNN_EXPR_API
#define _MOD_NAME _mnncengine
#else
#define _MOD_NAME MNN
#endif
#else
#define _MOD_NAME _mnncengine
#endif
#define _STRINGIFY(str) #str
#define STRINGIFY(macro) _STRINGIFY(macro)
#define MOD_NAME STRINGIFY(_MOD_NAME)

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    MOD_NAME,     /* m_name */
    "MNNEngine",  /* m_doc */
    -1,                  /* m_size */
    module_methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};
#define MOD_INIT_FUNC_NAME(name) PyInit_##name
#else
#define MOD_INIT_FUNC_NAME(name) init##name
#endif
// MOD_INIT_FUNC [PyInit_{MOD_NAME} or init{MOD_NAME}]
#define _MOD_INIT_FUNC(macro) MOD_INIT_FUNC_NAME(macro)
#define MOD_INIT_FUNC _MOD_INIT_FUNC(_MOD_NAME)

static std::once_flag mLoadFlag1;

PyMODINIT_FUNC MOD_INIT_FUNC(void) {
#if PY_MAJOR_VERSION >= 3
#define ERROR_RETURN return NULL;
#else
#define ERROR_RETURN return;
#endif

#ifdef PYMNN_USE_ALINNPYTHON
    std::call_once(mLoadFlag1, [&](){
        if (global_new_python_flag > 0) {
            tls_key = PyThread_create_key();
            tls_key_2 = PyThread_create_key();
        }
    });
#endif

    if (PyType_Ready(&PyMNNInterpreterType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNInterpreterType failed");
        ERROR_RETURN
    }
    if (PyType_Ready(&PyMNNSessionType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNSessionType failed");
        ERROR_RETURN
    }
    if (PyType_Ready(&PyMNNTensorType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNTensorType failed");
        ERROR_RETURN
    }
    if (PyType_Ready(&PyMNNCVImageProcessType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNCVImageProcessType failed");
        ERROR_RETURN
    }
    if (PyType_Ready(&PyMNNCVMatrixType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNCVMatrixType failed");
        ERROR_RETURN
    }
    if (PyType_Ready(&PyMNNOpInfoType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: PyType_Ready PyMNNOpInfoType failed");
        ERROR_RETURN
    }
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    PyObject *m = Py_InitModule3(MOD_NAME, module_methods, "MNN Module");
#endif
    // module import failed!
    if (!m) {
        PyErr_SetString(PyExc_Exception, "initMNN: import MNN failed");
        ERROR_RETURN
    }
#ifdef PYMNN_NUMPY_USABLE
    if(_import_array() < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN: init numpy failed");
        ERROR_RETURN
    }
#endif

    PyModule_AddObject(m, "Interpreter", (PyObject*)PyType_FindTLSType(&PyMNNInterpreterType));
    PyModule_AddObject(m, "Session", (PyObject*)PyType_FindTLSType(&PyMNNSessionType));
    PyModule_AddObject(m, "Tensor", (PyObject*)PyType_FindTLSType(&PyMNNTensorType));
    PyModule_AddObject(m, "CVImageProcess", (PyObject*)PyType_FindTLSType(&PyMNNCVImageProcessType));
    PyModule_AddObject(m, "CVMatrix", (PyObject*)PyType_FindTLSType(&PyMNNCVMatrixType));
    PyModule_AddObject(m, "OpInfo", (PyObject*)PyType_FindTLSType(&PyMNNOpInfoType));

    // Tensor::DimensionType
    PyObject *DimensionType_Tensorflow = PyLong_FromLong(Tensor::TENSORFLOW);
    PyObject *DimensionType_Caffe = PyLong_FromLong(Tensor::CAFFE);
    PyObject *DimensionType_Caffe_C4 = PyLong_FromLong(Tensor::CAFFE_C4);
    PyModule_AddObject(m, "Tensor_DimensionType_Tensorflow", DimensionType_Tensorflow);
    PyModule_AddObject(m, "Tensor_DimensionType_Caffe", DimensionType_Caffe);
    PyModule_AddObject(m, "Tensor_DimensionType_Caffe_C4", DimensionType_Caffe_C4);

    struct MNN_TLSData *tlsData = static_cast<MNN_TLSData *>(malloc(sizeof(MNN_TLSData)));
    setTLSData(tlsData);
    tlsData->interpreterMap = new std::unordered_map<std::string, Interpreter *>();
    tlsData->sessionCacheMap = new std::unordered_map<std::string, Session *>();

    // halide_type
    tlsData->PyMNNHalideTypeInt = PyCapsule_New(httInt(), NULL, NULL);
    tlsData->PyMNNHalideTypeInt64 = PyCapsule_New(httInt64(), NULL, NULL);
    tlsData->PyMNNHalideTypeFloat = PyCapsule_New(httFloat(), NULL, NULL);
    tlsData->PyMNNHalideTypeDouble = PyCapsule_New(httDouble(), NULL, NULL);
    tlsData->PyMNNHalideTypeUint8 = PyCapsule_New(httUint8(), NULL, NULL);
    tlsData->PyMNNHalideTypeString = PyCapsule_New(httString(), NULL, NULL);

#if defined(PYMNN_USE_ALINNPYTHON) && defined(PYMNN_EXPR_API)
    struct py::detail::rh_tls *rh_tls = static_cast<py::detail::rh_tls *>(malloc(sizeof(py::detail::rh_tls)));
    if(nullptr == rh_tls) {
        throw runtime_error("rh_tls malloc fail");
    }
    set_rh_tls_data(rh_tls);
    rh_tls->internals_pp = nullptr;
    rh_tls->locals = new py::detail::type_map<py::detail::type_info *>;
#endif

    PyModule_AddObject(m, "Halide_Type_Int", tlsData->PyMNNHalideTypeInt);
    PyModule_AddObject(m, "Halide_Type_Int64", tlsData->PyMNNHalideTypeInt64);
    PyModule_AddObject(m, "Halide_Type_Float", tlsData->PyMNNHalideTypeFloat);
    PyModule_AddObject(m, "Halide_Type_Double", tlsData->PyMNNHalideTypeDouble);
    PyModule_AddObject(m, "Halide_Type_Uint8", tlsData->PyMNNHalideTypeUint8);
    PyModule_AddObject(m, "Halide_Type_String", tlsData->PyMNNHalideTypeString);

    // CV
    // ImageFormat
    PyObject *CV_ImageFormat_RGBA = PyLong_FromLong(CV::RGBA);
    PyObject *CV_ImageFormat_RGB = PyLong_FromLong(CV::RGB);
    PyObject *CV_ImageFormat_BGR = PyLong_FromLong(CV::BGR);
    PyObject *CV_ImageFormat_GRAY = PyLong_FromLong(CV::GRAY);
    PyObject *CV_ImageFormat_BGRA = PyLong_FromLong(CV::BGRA);
    PyObject *CV_ImageFormat_YUV_NV21 = PyLong_FromLong(CV::YUV_NV21);
    PyModule_AddObject(m, "CV_ImageFormat_RGBA", CV_ImageFormat_RGBA);
    PyModule_AddObject(m, "CV_ImageFormat_RGB", CV_ImageFormat_RGB);
    PyModule_AddObject(m, "CV_ImageFormat_BGR", CV_ImageFormat_BGR);
    PyModule_AddObject(m, "CV_ImageFormat_GRAY", CV_ImageFormat_GRAY);
    PyModule_AddObject(m, "CV_ImageFormat_BGRA", CV_ImageFormat_BGRA);
    PyModule_AddObject(m, "CV_ImageFormat_YUV_NV21", CV_ImageFormat_YUV_NV21);
    // Filter
    PyObject *CV_Filter_NEAREST = PyLong_FromLong(CV::NEAREST);
    PyObject *CV_Filter_BILINEAL = PyLong_FromLong(CV::BILINEAR);
    PyObject *CV_Filter_BICUBIC = PyLong_FromLong(CV::BICUBIC);
    PyModule_AddObject(m, "CV_Filter_NEAREST", CV_Filter_NEAREST);
    PyModule_AddObject(m, "CV_Filter_BILINEAL", CV_Filter_BILINEAL);
    PyModule_AddObject(m, "CV_Filter_BICUBIC", CV_Filter_BICUBIC);
    // wrap
    PyObject *CV_Wrap_CLAMP_TO_EDGE = PyLong_FromLong(CV::CLAMP_TO_EDGE);
    PyObject *CV_Wrap_ZERO = PyLong_FromLong(CV::ZERO);
    PyObject *CV_Wrap_REPEAT = PyLong_FromLong(CV::REPEAT);
    PyModule_AddObject(m, "CV_Wrap_CLAMP_TO_EDGE", CV_Wrap_CLAMP_TO_EDGE);
    PyModule_AddObject(m, "CV_Wrap_ZERO", CV_Wrap_ZERO);
    PyModule_AddObject(m, "CV_Wrap_REPEAT", CV_Wrap_REPEAT);

    // static variable initialize
    interpreterMap();
    sessionCacheMap();

#ifdef PYMNN_EXPR_API
    auto py_module = py::reinterpret_borrow<py::module>(m);
    INTS default_shape = {};
    auto expr_module = py_module.def_submodule("_expr");
    py::enum_<Dimensionformat> (expr_module, "data_format")
        .value("NHWC", NHWC)
        .value("NC4HW4", NC4HW4)
        .value("NCHW", NCHW)
        .export_values();
    py::enum_<DType> (expr_module, "dtype")
        .value("float", DType_FLOAT)
        .value("double", DType_DOUBLE)
        .value("int", DType_INT32)
        .value("int64", DType_INT64)
        .value("uint8", DType_UINT8)
        .export_values();

    py::enum_<PaddingMode> (expr_module, "Padding_Mode")
        .value("CAFFE", CAFFE)
        .value("VALID", VALID)
        .value("SAME", SAME)
        .export_values();
    py::enum_<MNN::Express::PadValueMode> (expr_module, "PadValue_Mode")
        .value("CONSTANT", CONSTANT)
        .value("REFLECT", REFLECT)
        .value("SYMMETRIC", SYMMETRIC)
        .export_values();
    py::enum_<PoolingMode> (expr_module, "Pooling_Mode")
        .value("MAXPOOL", MAXPOOL)
        .value("AVEPOOL", AVEPOOL)
        .export_values();
    py::enum_<InterpolationMethod> (expr_module, "Interp_Method")
        .value("BILINEAR", BILINEAR)
        .value("NEAREST", NEAREST)
        .export_values();
    py::class_<VARP>(expr_module, "Var")
        .def_property_readonly("shape",
	    [](VARP *self){
            auto info = (*self)->getInfo();
            if(nullptr == info) {
                throw std::runtime_error("unable to get variable info");
            }
            return info->dim;
	    })
        .def_property_readonly("valid",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info) {
                    return false;
                }
                return true;
            })
        .def_property_readonly("data_format",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                    throw std::runtime_error("unable to get variable info");
                return info->order;
            })
        .def_property_readonly("dtype",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                   throw std::runtime_error("unable to get variable info");
                return htype2dtype(info->type);
            })
         .def_property_readonly("size",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info) {
                   throw std::runtime_error("unable to get variable info");
                }
                return info->size;
            })
        .def_property("name",
            [](VARP *self){
                auto name = (*self)->name();
                return name;
            },
            [] (VARP* self, std::string name) {
                (*self)->setName(name);
            })
#ifdef BUILD_OPTYPE
        .def_property_readonly("op_type",
            [](VARP *self){
                auto op = (*self)->expr().first->get();
                if (nullptr == op) {
                    switch ((*self)->expr().first->inputType()) {
                        case VARP::INPUT:
                            return std::string("Input");
                        case VARP::CONSTANT:
                            return std::string("Const");
                        case VARP::TRAINABLE:
                            return std::string("Trainable");
                    }
                }

                auto type = op->type();
                if (type == OpType_BinaryOp) {
                    return std::string(MNN::EnumNameBinaryOpOperation((BinaryOpOperation)op->main_as_BinaryOp()->opType()));
                }
                if (type == OpType_UnaryOp) {
                    return std::string(MNN::EnumNameUnaryOpOperation((UnaryOpOperation)op->main_as_UnaryOp()->opType()));
                }
                return std::string(MNN::EnumNameOpType(type));
            })
#endif
        .def_property_readonly("inputs",
            [] (VARP* self) {
                return (*self)->expr().first->inputs();
            })
        .def("fix_as_placeholder",
            [] (VARP* self) {
                (*self).fix(VARP::INPUT);
            })

        .def("fix_as_const",
            [] (VARP* self) {
                (*self).fix(VARP::CONSTANT);
            })
        .def("fix_as_trainable",
            [] (VARP* self) {
                (*self).fix(VARP::TRAINABLE);
            })
        .def("close",
            [] (VARP* self) {
                (*self)->input(VARP(nullptr));
            })
        .def("copy_from",
            [] (VARP* self, VARP source) {
                bool res = (*self)->input(source);
                if (!res) {
                    throw std::runtime_error("Copy from souce Error");
                }
            })
        .def("set_inputs",
            [] (VARP* self, std::vector<VARP> source) {
                if (source.empty()) {
                    throw std::runtime_error("Empty source");
                }
                auto expr = (*self)->expr();
                auto newExpr = Expr::create(expr.first->extra(), std::move(source), expr.first->outputSize());
                Expr::replace(expr.first, newExpr);
            })
        .def("replace",
            [] (VARP* self, VARP source) {
                Variable::replace(*self, source);
            })
        .def("reorder",
            [] (VARP* self, Dimensionformat order) {
                auto newInput = _ChangeInputFormat(*self, order);
                (*self) = newInput;
            })
        .def("resize",
            [] (VARP* self, const std::vector<int>& shape) {
                (*self)->resize(shape);
            })
#ifdef PYMNN_NUMPY_USABLE
        .def("read",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                   throw std::runtime_error("unable to get variable info");
                auto dtype = htype2dtype(info->type);
                auto shape = info->dim;
                int64_t total_length = info->size;
                auto readptr = [self](DType dtype, INTS shape, int64_t total_length) {
                    void *dataPtr = (void *) (*self)->readMap<void>();
                    std::vector<npy_intp> npy_dims;
                    for(const auto dim: shape) {
                        npy_dims.push_back(dim);
                    }

                    switch(dtype) {
                       case DType_FLOAT:
                           return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_FLOAT, dataPtr);
                       case DType_DOUBLE:
                           return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_DOUBLE, dataPtr);
                       case DType_INT32:
                           return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT32, dataPtr);
                       case DType_INT64:
                           return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT64, dataPtr);
                       case DType_UINT8:
                           return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_UINT8, dataPtr);
                       default:
                          throw std::runtime_error("does not support this dtype");
                    }
                    if (nullptr == dataPtr) {
                        throw std::runtime_error("call to readMap meet a error");
                    }
                };
                auto data = readptr(dtype, shape, total_length);
                (*self)->unMap();
                return py::reinterpret_steal<py::object>(data);
            })
#endif
        .def("read_as_tuple",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                   throw std::runtime_error("unable to get variable info");
                auto dtype = htype2dtype(info->type);
                auto shape = info->dim;
                size_t total_length = info->size;
                auto readptr = [self](DType dtype, INTS shape, size_t total_length) {
                    void *dataPtr = (void *) (*self)->readMap<void>();
                    auto obj = PyTuple_New(total_length);
                    if(DType_FLOAT == dtype) {
                        auto data = (float*)dataPtr;
                        for(size_t i = 0; i < total_length; i++) {
                            PyTuple_SetItem(obj, i, PyFloat_FromDouble(data[i]));
                        }
                    } else if(DType_INT32 == dtype) {
                        auto data = (int32_t*)dataPtr;
                        for(size_t i = 0; i < total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                    } else if(DType_UINT8 == dtype) {
                        auto data = (uint8_t*)dataPtr;
                        for(size_t i = 0; i < total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                    } else if(DType_INT8 == dtype) {
                        auto data = (int8_t*)dataPtr;
                        for(size_t i = 0; i < total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                    } else {
                        throw std::runtime_error("Don't support data type");
                    }
                    return obj;
                };
                auto data = readptr(dtype, shape, total_length);
                (*self)->unMap();
                return py::reinterpret_steal<py::object>(data);
            })
        .def("write",
            [](VARP *self, py::object data) {
                auto info = (*self)->getInfo();
                if(nullptr == info) {
                    throw std::runtime_error("unable to get variable info");
                }
                auto dtype = htype2dtype(info->type);
                auto shape = info->dim;
                int64_t total_length = info->size;
                PyObject *obj = data.ptr();
                auto write = [self](PyObject *obj, DType dtype, int64_t total_length) {
 #ifdef PYMNN_NUMPY_USABLE
                    if(PyArray_Check(obj)) {
                        //numpy support
                        if(total_length != PyArray_Size(obj)) {
                            throw std::runtime_error("data size does not match each other");
                        }
                        int npy_type = PyArray_TYPE(obj);
                        int itemsize = getitemsize(dtype, npy_type);
                        PyArrayObject *obj_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)obj);
                        auto tmpBuffer = PyArray_DATA(obj_cont);
                        if(NULL == tmpBuffer) {
                            throw std::runtime_error("numpy failed to get buffer");
                        }
                        auto data = (*self)->writeMap<void>();
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
                        }
                        memcpy(data, tmpBuffer, total_length * itemsize);
                        Py_XDECREF(obj_cont);
                        return;
                    }
#endif
                    INTS shapeData = getshape(obj);
                    int64_t totalLengthData = 1;
                    INTS stride;
                    for (size_t i = 0; i < shapeData.size(); i++) {
                        totalLengthData *= shapeData[i];
                    }
                    int totalStride = 1;
                    for (int i = shapeData.size() - 1; i >= 0; i--) {
                       if(i + 1 < shapeData.size()) {
                           totalStride *= shapeData[i+1];
                       }
                       stride.push_back(totalStride);
                    }
                    std::reverse(stride.begin(), stride.end());
                    if(totalLengthData != total_length) {
                        throw std::runtime_error("data size does not match each other");
                    }
                    if(DType_FLOAT == dtype) {
                        auto data = (*self)->writeMap<float>();
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(float));
                    }
                    else if(DType_INT32 == dtype) {
                        auto data = (*self)->writeMap<int>();
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int));
                    }
                    else if(DType_UINT8 == dtype) {
                        auto data = (*self)->writeMap<uint8_t>();
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(uint8_t));
                    }
                    else if(DType_INT8 == dtype) {
                        auto data = (*self)->writeMap<uint8_t>();
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int8_t));
                    }
                };
                write(obj, dtype, total_length);
                (*self)->unMap();
            });
    // Load And Save
    expr_module.def("load_as_list",
    		[](std::string fileName) {
                auto variable = Variable::load(fileName.c_str());
			    return variable;
    });
    expr_module.def("save",
    		[](const std::vector<VARP>& vars, std::string fileName, bool forInference = true) {
                std::vector<VARP> newVars;
                for (auto v : vars) {
                    if (v.get() != nullptr) {
                        newVars.emplace_back(v);
                    }
                }
#ifdef PYMNN_TRAIN_API
                if (forInference) {
                    Transformer::turnModelToInfer()->onExecute(newVars);
                }
#endif
                Variable::save(newVars, fileName.c_str());
#ifdef PYMNN_TRAIN_API
                ConvertToFullQuant::convert(fileName);
#endif
    }, py::arg("variables"), py::arg("file_name"), py::arg("for_inference") = true);
    expr_module.def("load_as_dict",
    		[](std::string fileName) {
                auto variable = Variable::loadMap(fileName.c_str());
			    return variable;
    });
    expr_module.def("get_inputs_and_outputs", &Variable::getInputAndOutput);
    // Executor
    expr_module.def("gc", [](bool full) {
        auto exe = Executor::getGlobalExecutor();
        if (full) {
            exe->gc(Executor::FULL);
        } else {
            exe->gc(Executor::PART);
        }
    });

    // Don't care whether MNN library support corresponding backend, all backend type are usable by user,
    // which make MNN.whl setup.py easy
    py::enum_<MNNForwardType>(expr_module, "Backend")
        .value("CPU", MNN_FORWARD_CPU)
        .value("OPENCL", MNN_FORWARD_OPENCL)
        .value("OPENGL", MNN_FORWARD_OPENGL)
        .value("VULKAN", MNN_FORWARD_VULKAN)
        .value("METAL", MNN_FORWARD_METAL)
        .value("TRT", MNN_FORWARD_USER_1)
        .value("CUDA", MNN_FORWARD_CUDA)
        .value("HIAI", MNN_FORWARD_USER_0)
        .export_values();

    using MemoryMode = BackendConfig::MemoryMode;
    using PowerMode = BackendConfig::PowerMode;
    using PrecisionMode = BackendConfig::PrecisionMode;
    py::enum_<MemoryMode>(expr_module, "MemoryMode")
        .value("Normal", MemoryMode::Memory_Normal)
        .value("High", MemoryMode::Memory_High)
        .value("Low", MemoryMode::Memory_Low)
        .export_values();
    py::enum_<PowerMode>(expr_module, "PowerMode")
        .value("Normal", PowerMode::Power_Normal)
        .value("High", PowerMode::Power_High)
        .value("Low", PowerMode::Power_Low)
        .export_values();
    py::enum_<PrecisionMode>(expr_module, "PrecisionMode")
        .value("Normal", PrecisionMode::Precision_Normal)
        .value("High", PrecisionMode::Precision_High)
        .value("Low", PrecisionMode::Precision_Low)
        .export_values();
    expr_module.def("set_config",
    		[](MNNForwardType backend, MemoryMode memory_mode, PowerMode power_mode, PrecisionMode precision_mode, int thread_num) {
                if (thread_num < 1 || thread_num > 8) {
                    PyErr_SetString(PyExc_Exception, "thread_num should bigger than 0 and less than 9");
                }
                thread_num = std::max(std::min(thread_num, 8), 1);
                //auto exe = ExecutorScope::Current();
                auto exe = Executor::getGlobalExecutor();
                BackendConfig config;
                config.memory = memory_mode;
                config.power = power_mode;
                config.precision = precision_mode;
                exe->setGlobalExecutorConfig(backend, config, thread_num);
            },
            py::arg("backend")=MNN_FORWARD_CPU, py::arg("memory_mode")=MemoryMode::Memory_Normal,
            py::arg("power_mode")=PowerMode::Power_Normal, py::arg("precision_mode")=PrecisionMode::Precision_Normal,
            py::arg("thread_num")=1);

    //Begin of Math OPS
    //Unary OPS
    expr_module.def("sign", &Express::_Sign);
    expr_module.def("abs", &Express::_Abs);
    expr_module.def("negative", &Express::_Negative);
    expr_module.def("floor", &Express::_Floor);
    expr_module.def("ceil", &Express::_Ceil);
    expr_module.def("square", &Express::_Square);
    expr_module.def("sqrt", &Express::_Sqrt);
    expr_module.def("rsqrt", &Express::_Rsqrt);
    expr_module.def("exp", &Express::_Exp);
    expr_module.def("log", &Express::_Log);
    expr_module.def("sin", &Express::_Sin);
    expr_module.def("cos", &Express::_Cos);
    expr_module.def("tan", &Express::_Tan);
    expr_module.def("asin", &Express::_Asin);
    expr_module.def("acos", &Express::_Acos);
    expr_module.def("atan", &Express::_Atan);
    expr_module.def("reciprocal", &Express::_Reciprocal);
    expr_module.def("log1p", &Express::_Log1p);
    expr_module.def("tanh", &Express::_Tanh);
    expr_module.def("sigmoid", &Express::_Sigmoid);
    //Binary OPS
    expr_module.def("add", &Express::_Add);
    expr_module.def("subtract", &Express::_Subtract);
    expr_module.def("multiply", &Express::_Multiply);
    expr_module.def("divide", &Express::_Divide);
    expr_module.def("pow", &Express::_Pow);
    expr_module.def("minimum", &Express::_Minimum);
    expr_module.def("maximum", &Express::_Maximum);
    expr_module.def("bias_add", &Express::_BiasAdd);
    expr_module.def("greater", &Express::_Greater);
    expr_module.def("greater_equal", &Express::_GreaterEqual);
    expr_module.def("less", &Express::_Less);
    expr_module.def("floordiv", &Express::_FloorDiv);
    expr_module.def("squared_difference", &Express::_SquaredDifference);
    expr_module.def("equal", &Express::_Equal);
    expr_module.def("not_equal", &Express::_NotEqual);
    expr_module.def("less_equal", &Express::_LessEqual);
    expr_module.def("floormod", &Express::_FloorMod);
    //Reduce OPS
    expr_module.def("reduce_sum",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceSum(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_mean",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceMean(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_max",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceMax(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_min",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceMin(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_prod",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceProd(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_any",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceAny(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    expr_module.def("reduce_all",
                    [](VARP input, INTS axis, bool keep_dims) {
                        return _ReduceAll(input, axis, keep_dims);
                    }, py::arg("input"), py::arg("axis")=default_shape, py::arg("keep_dims")=false);
    //Eltwise OPS
    expr_module.def("eltwise_prod", &Express::_Prod);
    expr_module.def("eltwise_sum", &Express::_Sum);
    expr_module.def("eltwise_max", &Express::_Max);
    expr_module.def("eltwise_sub", &Express::_Sub);
    //Other OPS
    expr_module.def("cast",
		    [](VARP x, DType dtype) {
			return _Cast(x, dtype2htype(dtype));
                    });
    expr_module.def("matmul", &Express::_MatMul, py::arg("a"), py::arg("b"), py::arg("tranposeA")=false, py::arg("tranposeB")=false);
    expr_module.def("normalize", &Express::_Normalize);
    expr_module.def("argmax",
		   [](VARP input, int axis) {
			return _ArgMax(input, axis);
                   }, py::arg("input"), py::arg("axis")=0);
    expr_module.def("unravel_index", &Express::_UnravelIndex, py::arg("indices"), py::arg("dims"));
    expr_module.def("scatter_nd", &Express::_ScatterNd, py::arg("indices"), py::arg("updates"), py::arg("shape"));
    expr_module.def("one_hot",
		   [](VARP indices, int depth, float onValue, float offValue, int axis) {
			return _OneHot(indices, _Scalar<int>(depth), _Scalar<float>(onValue), _Scalar<float>(offValue), axis);
                   },py::arg("indices"), py::arg("depth"), py::arg("on_value")=1, py::arg("off_value")=0, py::arg("axis")=-1);
    expr_module.def("broadcast_to", &Express::_BroadcastTo, py::arg("input"), py::arg("shape"));
    //End of Math OPS

    //Begin of NN OPS
    expr_module.def("placeholder",
                  [](INTS shape,Dimensionformat data_format, DType dtype)->VARP{
    			return _Input(shape, data_format, dtype2htype(dtype));
                  },
                  py::arg("shape")=default_shape,
                  py::arg("data_format")=NCHW,
                  py::arg("dtype")=DType_FLOAT);
    expr_module.def("clone",
                   [](VARP source, bool deepCopy) {
			return _Clone(source, deepCopy);
                   }, py::arg("source"), py::arg("deep_copy")=false);
    INTS default_pads = {0, 0};
    INTS default_axis = {};
    expr_module.def("const",
            [](py::object value, INTS shape, Dimensionformat data_format, DType dtype) {
                int64_t total_length = 1;
                for(size_t i = 0; i < shape.size(); i++) {
                    if (data_format == NC4HW4 && 1 == i)
                    {
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#endif
                        total_length *= ROUND_UP(shape[i], 4);
                    }
                    else
                    {
                        total_length *= shape[i];
                    }
                }
                PyObject *obj = value.ptr();
                auto write = [](PyObject *obj, DType dtype, int64_t total_length) {
 #ifdef PYMNN_NUMPY_USABLE
                    if(PyArray_Check(obj)) {
                        //numpy support
                        if(total_length != PyArray_Size(obj)) {
                            throw std::runtime_error("data size does not match each other");
                        }
                        int npy_type = PyArray_TYPE(obj);
                        int itemsize = getitemsize(dtype, npy_type);
                        PyArrayObject *obj_cont= PyArray_GETCONTIGUOUS((PyArrayObject*)obj);
                        auto tmpBuffer = PyArray_DATA(obj_cont);
                        if(NULL == tmpBuffer) {
                            throw std::runtime_error("numpy failed to get buffer");
                        }
                        auto data = malloc(total_length * itemsize);
                        if (nullptr == data) {
                            throw std::runtime_error("call to writeMap meet a error");
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
                    int totalStride = 1;
                    for (int i = shapeData.size() - 1; i >= 0; i--) {
                       if (i + 1 < shapeData.size()) {
                           totalStride *= shapeData[i+1];
                       }
                       stride.push_back(totalStride);
                    }
                    std::reverse(stride.begin(), stride.end());
                    if(totalLengthData != total_length) {
                        throw std::runtime_error("data size does not match each other");
                    }
                    void *data = nullptr;
                    if(DType_FLOAT == dtype) {
                        data = malloc(total_length * sizeof(float));
                        if (nullptr == data) {
                            throw std::runtime_error("not enough memory");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(float));
                    }
                    else if(DType_INT32 == dtype) {
                        data = malloc(total_length * sizeof(int));
                        if (nullptr == data) {
                            throw std::runtime_error("not enough memory");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int));
                    }
                    else if(DType_UINT8 == dtype) {
                        data = malloc(total_length * sizeof(uint8_t));
                        if (nullptr == data) {
                            throw std::runtime_error("not enough memory");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(uint8_t));
                    }
                    else if(DType_INT8 == dtype) {
                        data = malloc(total_length * sizeof(int8_t));
                        if (nullptr == data) {
                            throw std::runtime_error("not enough memory");
                        }
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int8_t));
                    }
                    return data;
                };
                auto data = write(obj, dtype, total_length);
                VARP ret = nullptr;
                if(data) {
                    ret = _Const((const void*)data, shape, data_format, dtype2htype(dtype));
                    free(data);
                }
                return ret;
            },py::arg("value_list"), py::arg("shape"), py::arg("data_format")=NCHW, py::arg("dtype")=DType::DType_FLOAT);
    INTS default_stride = {1, 1};
    INTS default_dialate = {1, 1};
    expr_module.def("conv2d",
            [](VARP input, VARP weight, VARP bias, INTS stride, INTS padding, INTS dilate, int group, PaddingMode padding_mode) {
                return _Conv(weight, bias, input, padding_mode, stride, dilate, group, padding);
            },py::arg("input"), py::arg("weight"), py::arg("bias"),
            py::arg("stride")=default_stride,
            py::arg("padding")=default_pads,
            py::arg("dilate")=default_dialate,
            py::arg("group")=1,
            py::arg("padding_mode")=VALID);
    expr_module.def("conv2d_transpose",
            [](VARP input, VARP weight, VARP bias, INTS stride, INTS padding, INTS dilate, int group, PaddingMode padding_mode) {
                return _Deconv(weight, bias, input, padding_mode, stride, dilate, group, padding);
            },py::arg("input"), py::arg("weight"), py::arg("bias"),
            py::arg("stride")=default_stride,
            py::arg("padding")=default_pads,
            py::arg("dilate")=default_dialate,
            py::arg("group")=1,
            py::arg("padding_mode")=VALID);
    expr_module.def("max_pool",
                   [](VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
                        return _MaxPool(x, kernel, stride, pad, pads);
                   }, py::arg("input"), py::arg("kernel"), py::arg("stride"),
		   py::arg("padding_mode")=VALID,
		   py::arg("pads")=default_pads);
    expr_module.def("avg_pool",
                   [](VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
                        return _AvePool(x, kernel, stride, pad, pads);
                   }, py::arg("input"), py::arg("kernel"), py::arg("stride"),
                   py::arg("padding_mode")=VALID,
                   py::arg("pads")=default_pads);
    expr_module.def("reshape",
                   [](VARP x, INTS shape, Dimensionformat original_format) {
                        return _Reshape(x, shape, original_format);
                   }, py::arg("x"), py::arg("shape"), py::arg("original_format")=NCHW);
    expr_module.def("reshape",
                   [](VARP x, VARP shape) {
                        return _Reshape(x, shape);
                   });
    expr_module.def("scale", &Express::_Scale, py::arg("x"), py::arg("channels"), py::arg("scales"), py::arg("bias"));
    expr_module.def("relu",
                   [](VARP x, float slope) {
                        return _Relu(x, slope);
                   }, py::arg("x"), py::arg("slope")=0.0f);
    expr_module.def("relu6", &Express::_Relu6, py::arg("x"), py::arg("min") = 0.0f, py::arg("max") = 6.0f);
    expr_module.def("prelu", &Express::_PRelu, py::arg("x"), py::arg("slopes"));
    expr_module.def("softmax",
                   [](VARP logits, int axis) {
                        return _Softmax(logits, axis);
                   }, py::arg("logits"), py::arg("axis")=-1);
    expr_module.def("softplus", &Express::_Softplus, py::arg("features"));
    expr_module.def("softsign", &Express::_Softsign, py::arg("features"));
    expr_module.def("slice", &Express::_Slice, py::arg("input"), py::arg("starts"), py::arg("sizes"));
    expr_module.def("split", &Express::_Split, py::arg("input"), py::arg("size_splits"), py::arg("axis"));
    expr_module.def("strided_slice", &Express::_StridedSlice, py::arg("input"), py::arg("begin"), py::arg("end"),
        py::arg("strides"), py::arg("begin_mask"), py::arg("end_mask"), py::arg("ellipsis_mask"), py::arg("new_axis_mask"), py::arg("shrink_axis_mask"));
    expr_module.def("concat", &Express::_Concat, py::arg("values"), py::arg("axis"));
    expr_module.def("convert", &Express::_Convert, py::arg("input"), py::arg("format"));
    expr_module.def("transpose",
                   [](VARP x, INTS perm) {
                        return _Transpose(x, perm);
                   }, py::arg("x"), py::arg("perm"));
    expr_module.def("transpose",
                   [](VARP x, VARP perm) {
                        return _Transpose(x, perm);
                   });
    expr_module.def("channel_shuffle", &Express::_ChannelShuffle);
    // change_inputformat not exposed because it's for static graphs.
    //expr_module.def("change_inputformat", &Express::_ChangeInputFormat);
    //
    expr_module.def("reverse_sequence", &Express::_ReverseSequence, py::arg("x"), py::arg("y"), py::arg("batch_dim"), py::arg("seq_dim"));
    expr_module.def("crop", &Express::_Crop, py::arg("images"), py::arg("size"), py::arg("axis"), py::arg("offset"));
    expr_module.def("resize", &Express::_Resize, py::arg("images"), py::arg("x_scale"), py::arg("y_scale"));
    expr_module.def("pad",
                   [](VARP x, VARP paddings, MNN::Express::PadValueMode mode) {
                        return Express::_Pad(x, paddings, mode);
                   }, py::arg("x"), py::arg("paddings"), py::arg("mode")=CONSTANT);
    expr_module.def("expand_dims",
                   [](VARP input, int axis) {
                        return _ExpandDims(input, axis);
                   });
    expr_module.def("expand_dims",
                   [](VARP input, VARP axis) {
                        return _ExpandDims(input, axis);
                   });
    expr_module.def("shape",
                    [](VARP input) {
                        return Express::_Shape(input, false);
                    }, py::arg("input"));
    expr_module.def("stack",
                   [](VARPS values, int axis) {
                        return _Stack(values, axis);
 		   }, py::arg("values"), py::arg("axis")=0);
    expr_module.def("crop_and_resize",
                   [](VARP image, VARP boxes, VARP box_ind, VARP crop_size, InterpolationMethod method, float extrapolation_value) {
                        return _CropAndResize(image, boxes, box_ind, crop_size, method, extrapolation_value);
                   }, py::arg("image"), py::arg("boxes"), py::arg("box_ind"), py::arg("crop_size"),
		   py::arg("method")=BILINEAR, py::arg("extrapolation_value")=0.0f);
    expr_module.def("fill", &Express::_Fill, py::arg("dims"), py::arg("value"));
    expr_module.def("tile", &Express::_Tile, py::arg("input"), py::arg("multiples"));
    expr_module.def("gather", &Express::_Gather, py::arg("params"), py::arg("indices"));
    expr_module.def("select", &Express::_Select);

    // Currently only axis == 0 is supported, which is the same as gather.
    /*
    expr_module.def("gather_v2",
                   [](VARP params, VARP indices, VARP axis = nullptr) {
                        return _GatherV2(params, indices, axis);
                   }, py::arg("params"), py::arg("indices"), py::arg("axis")=nullptr);
                   */

    expr_module.def("squeeze",
                   [](VARP input, INTS axis) {
                        return _Squeeze(input, axis);
                   }, py::arg("input"), py::arg("axis")=default_axis);
    expr_module.def("unsqueeze",
                   [](VARP input, INTS axis) {
                        return _Unsqueeze(input, axis);
                   }, py::arg("input"), py::arg("axis")=default_axis);
    expr_module.def("batch_to_space_nd", &Express::_BatchToSpaceND, py::arg("input"), py::arg("block_shape"), py::arg("crops"));
    expr_module.def("gather_nd", &Express::_GatherND, py::arg("params"), py::arg("indices"));
    expr_module.def("selu", &Express::_Selu, py::arg("features"), py::arg("scale"), py::arg("alpha"));
    expr_module.def("size", &Express::_Size, py::arg("input"));
    expr_module.def("elu",
                   [](VARP features, float alpha) {
                        return _Elu(features, alpha);
                   }, py::arg("features"), py::arg("alpha")=1.0);
    expr_module.def("matrix_band_part", &Express::_MatrixBandPart, py::arg("input"), py::arg("num_lower"), py::arg("num_upper"));
    expr_module.def("moments", &Express::_Moments, py::arg("x"), py::arg("axes"), py::arg("shift"), py::arg("keep_dims"));
    expr_module.def("setdiff1d", &Express::_SetDiff1D, py::arg("x"), py::arg("y"));
    expr_module.def("space_to_depth", &Express::_SpaceToDepth, py::arg("input"), py::arg("block_size"));
    expr_module.def("space_to_batch_nd", &Express::_SpaceToBatchND, py::arg("input"), py::arg("block_shape"), py::arg("paddings"));
    expr_module.def("zeros_like", &Express::_ZerosLike, py::arg("input"));
    expr_module.def("unstack",
                   [](VARP value, int axis) {
                        return _Unstack(value, axis);
                   }, py::arg("value"), py::arg("axis")=0);
    expr_module.def("rank", &Express::_Rank, py::arg("input"));
    expr_module.def("range", &Express::_Range, py::arg("start"), py::arg("limit"), py::arg("delta"));
    expr_module.def("depth_to_space", &Express::_DepthToSpace, py::arg("input"), py::arg("block_size"));
    expr_module.def("detection_post_process", &Express::_DetectionPostProcess,
                   py::arg("encode_boxes"), py::arg("class_predictions"), py::arg("anchors"),
                   py::arg("num_classes"), py::arg("max_detections"), py::arg("max_class_per_detection"),
                   py::arg("detections_per_class"), py::arg("nms_threshold"), py::arg("iou_threshold"),
                   py::arg("use_regular_nms")=false, py::arg("centersize_encoding"));
    //End of NN OPS
    auto cv_module = py_module.def_submodule("cv");
    py::enum_<CV::ImageFormat>(cv_module, "Format")
        .value("RGBA", CV::RGBA)
        .value("RGB", CV::RGB)
        .value("GRAY", CV::GRAY)
        .value("BGR", CV::BGR)
        .value("YUV_NV21", CV::YUV_NV21)
        .value("YUV_NV12", CV::YUV_NV12)
        .export_values();

    auto nn_module = py_module.def_submodule("_nn");

    class PyModule : public Module {
    public:
        using Module::Module;
        using Module::registerModel;

        virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
            PYBIND11_OVERLOAD_PURE(std::vector<Express::VARP>, Module, forward, inputs);
        }
    };
    py::class_<Module, PyModule, std::shared_ptr<Module>>(nn_module, "_Module")
        .def(py::init())
        .def("__call__", &Module::forward)
        .def("__call__", &Module::onForward)
        .def("forward", &Module::forward)
        .def("forward", &Module::onForward)
        .def_property_readonly("name", &Module::name) // TODO: too ugly, find way to fix it
        .def("set_name", &Module::setName)
        .def_property_readonly("is_training", &Module::getIsTraining)
        .def("train", &Module::setIsTraining, py::arg("is_training") = true)
        .def_property_readonly("parameters", &Module::parameters)
        .def("load_parameters", &Module::loadParameters)
        .def("clear_cache", &Module::clearCache)
        .def("_register_submodules", &PyModule::registerModel)
        .def("_add_parameter", &Module::addParameter);

    nn_module.def("load_module", [](vector<VARP> inputs, vector<VARP> outputs, bool fortrain){
#ifdef PYMNN_TRAIN_API
        return NN::extract(inputs, outputs, fortrain);
#else
        return Module::extract(inputs, outputs, fortrain);
#endif
    });
    nn_module.def("load_module_from_file", [](const vector<string>& inputs, const vector<string>& outputs,
                                              const char* file_name, bool dynamic, bool shape_mutable, bool rearrange,
                                              MNNForwardType backend, MemoryMode memory_mode, PowerMode power_mode,
                                              PrecisionMode precision_mode, int thread_num) -> Module* {
        BackendConfig backend_config;
        backend_config.memory = memory_mode;
        backend_config.power = power_mode;
        backend_config.precision = precision_mode;

        Module::BackendInfo backend_info;
        backend_info.type = backend;
        backend_info.config = &backend_config;

        Module::Config config;
        config.dynamic = dynamic;
        config.shapeMutable = shape_mutable;
        config.rearrange = rearrange;
        config.backend = &backend_info;

        auto converted_file_name = convertBytesEncodeIfNeed(file_name);
        auto m_ptr = Module::load(inputs, outputs, converted_file_name.data(), &config);
        if (m_ptr == nullptr) {
            std::string mnn_errno = "load_module_from_file failed ";
            mnn_errno = mnn_errno + std::string(file_name);
            PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
        }
        return m_ptr;
    });

#ifdef PYMNN_TRAIN_API
    py_module.def("get_model_uuid", &HelperFuncs::getModelUUID, "helper to get model uuid");

    // CNN
    nn_module.def("conv", [](int in_channel, int out_channel, INTS kernel_size, INTS stride, INTS padding,
                             INTS dilation, bool depthwise, bool bias, PaddingMode padding_mode) {
            NN::ConvOption option;
            option.channel = {in_channel, out_channel};
            option.kernelSize = kernel_size;
            if (!stride.empty()) {
                option.stride = stride;
            }
            option.padMode = padding_mode;
            if (!padding.empty()) {
                option.pads = padding;
            }
            if (!dilation.empty()) {
                option.dilate = dilation;
            }
            option.depthwise = depthwise;
            return NN::Conv(std::move(option), bias);
        },
        py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
        py::arg("stride") = std::vector<int>({1, 1}),
        py::arg("padding") = std::vector<int>({0, 0}),
        py::arg("dilation") = std::vector<int>({1, 1}),
        py::arg("depthwise") = false,
        py::arg("bias") = true,
        py::arg("padding_mode") = PaddingMode::VALID
    );
    nn_module.def("linear", [](int in_channel, int out_channel, bool bias) {
            return NN::Linear(in_channel, out_channel, bias);
        },
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("bias") = true
    );
    nn_module.def("batch_norm", &NN::BatchNorm, py::arg("channels"), py::arg("dims") = 4, py::arg("momentum") = 0.99, py::arg("epsilon") = 1e-5);
    nn_module.def("dropout", &NN::Dropout, py::arg("dropout_ratio"));

    auto optim_module = py_module.def_submodule("_optim");

    {
        py::enum_<ParameterOptimizer::RegularizationMethod>(optim_module, "Regularization_Method")
            .value("L1", ParameterOptimizer::RegularizationMethod::L1)
            .value("L2", ParameterOptimizer::RegularizationMethod::L2)
            .value("L1L2", ParameterOptimizer::RegularizationMethod::L1L2)
            .export_values();

        py::class_<ParameterOptimizer>(optim_module, "_Optimizer")
            .def_property("learning_rate", [](ParameterOptimizer* self) {
                    return ((SGD*)self)->currentLearningRate();
                },
                [](ParameterOptimizer* self, float lr) {
                    ((SGD*)self)->setLearningRate(lr);
                }
            )
            .def_property("momentum", [](ParameterOptimizer* self) {
                    return ((SGD*)self)->getMomentum();
                },
                [](ParameterOptimizer* self, float m) {
                    ((SGD*)self)->setMomentum(m);
                }
            )
            .def_property("momentum2", [](ParameterOptimizer* self) {
                    return ((ADAM*)self)->getMomentum2();
                },
                [](ParameterOptimizer* self, float m) {
                    ((ADAM*)self)->setMomentum2(m);
                }
            )
            .def_property("weight_decay", [](ParameterOptimizer* self) {
                    return ((SGD*)self)->getWeightDecay();
                },
                [](ParameterOptimizer* self, float decay) {
                    ((SGD*)self)->setWeightDecay(decay);
                }
            )
            .def_property("eps", [](ParameterOptimizer* self) {
                    return ((ADAM*)self)->getEps();
                },
                [](ParameterOptimizer* self, float eps) {
                    ((ADAM*)self)->setEps(eps);
                }
            )
            .def_property("regularization_method", [](ParameterOptimizer* self) {
                    return ((SGD*)self)->getRegularizationMethod();
                },
                [](ParameterOptimizer* self, ParameterOptimizer::RegularizationMethod method) {
                    ((SGD*)self)->setRegularizationMethod(method);
                }
            )
            .def("step", [](ParameterOptimizer* self, Express::VARP loss) {
                return self->step(loss);
            })
        ;

        optim_module.def("SGD", &ParameterOptimizer::createSGD,
                        py::arg("module"),
                        py::arg("learning_rate"), py::arg("momentum") = 0.9, py::arg("weight_decay") = 0,
                        py::arg("regularization_method") = ParameterOptimizer::RegularizationMethod::L2);
        optim_module.def("ADAM", &ParameterOptimizer::createADAM,
                        py::arg("module"),
                        py::arg("learning_rate") = 1e-3, py::arg("momentum") = 0.9, py::arg("momentum2") = 0.999,
                        py::arg("weight_decay") = 0.0, py::arg("eps") = 1e-8,
                        py::arg("regularization_method") = ParameterOptimizer::RegularizationMethod::L2);
    }


    {
        class PyDataset : public Dataset {
        public:
            using Dataset::Dataset;

            virtual Example get(size_t index) override {
                PYBIND11_OVERLOAD_PURE(Example, Dataset, __getitem__, index);
            }
            virtual size_t size() override {
                PYBIND11_OVERLOAD_PURE(size_t, Dataset, __len__);
            }
        };

        auto data_module = py_module.def_submodule("_data");
        py::class_<Dataset, PyDataset, std::shared_ptr<Dataset>>(data_module, "Dataset")
            .def(py::init())
            .def("__getitem__", &Dataset::get, py::arg("index"))
            .def("__len__", &Dataset::size)
        ;

        py::class_<DataLoader>(data_module, "DataLoader")
            .def(py::init([](std::shared_ptr<Dataset> dataset, const int batchsize, const bool shuffle, const int numWorkers) {
                bool stack = true;
                //TODO:hardcode numworkers as 0, as to enable workers, we need gil, in private pybind, gil is removed.
                return DataLoader::makeDataLoader(dataset, batchsize, stack, shuffle, 0);
            }), py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = true, py::arg("num_workers") = 0)
            .def_property_readonly("iter_number", &DataLoader::iterNumber)
            .def_property_readonly("size", &DataLoader::size)
            .def("reset", &DataLoader::reset)
            .def("next", [](DataLoader* self) {
                return self->next()[0]; // since we always stack
            })
        ;
    }

    {
        // Loss
        auto loss_module = nn_module.def_submodule("loss");
        loss_module.def("cross_entropy", _CrossEntropy, py::arg("predicts"), py::arg("one_hot_targets"));
        loss_module.def("kl", _KLDivergence, py::arg("predicts"), py::arg("one_hot_targets"));
        loss_module.def("mse", _MSE, py::arg("predicts"), py::arg("one_hot_targets"));
        loss_module.def("mae", _MAE, py::arg("predicts"), py::arg("one_hot_targets"));
        loss_module.def("hinge", _Hinge, py::arg("predicts"), py::arg("one_hot_targets"));
    }

    {
        auto compress_module = nn_module.def_submodule("compress");
        py::enum_<NN::FeatureScaleStatMethod>(compress_module, "Feature_Scale_Method")
            .value("PER_TENSOR", NN::PerTensor)
            .value("PER_CHANNEL", NN::PerChannel)
            .export_values();
        py::enum_<NN::ScaleUpdateMethod>(compress_module, "Scale_Update_Method")
            .value("MAXIMUM", NN::Maximum)
            .value("MOVING_AVERAGE", NN::MovingAverage)
            .export_values();
        compress_module.def("train_quant", &NN::turnQuantize,
            py::arg("module"),
            py::arg("quant_bits") = 8,
            py::arg("feature_scale_method") = NN::FeatureScaleStatMethod::PerTensor,
            py::arg("scale_update_method") = NN::ScaleUpdateMethod::MovingAverage);
    }
    // End of Train
#endif
#endif
#if PY_MAJOR_VERSION >= 3
    return m;
#else
    return;
#endif
}

// MNNPyBridge invoke loadMNN by static block on Windows / Linux / Mac / Android
#if defined(PYMNN_USE_ALINNPYTHON) && !defined(TARGET_OS_IOS)
static std::once_flag mLoadFlag2;
// Declared (extern "C" PYMNN_PUBLIC) in MNNPyBridge
void loadMNN() {
    std::call_once(mLoadFlag2, [](){
        WeImport_AppendInittab(MOD_NAME, MOD_INIT_FUNC);
    });
}
static auto registerMNN = []() {
    loadMNN();
    return true;
}();
#endif
