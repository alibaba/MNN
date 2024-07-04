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

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#ifdef PYMNN_EXPR_API
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
using namespace MNN::Express;
#ifdef PYMNN_OPENCV_API
#include "cv/cv.hpp"
#endif
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

// import submodules define
#ifdef PYMNN_EXPR_API
#include "expr.h"
#include "nn.h"
#ifdef PYMNN_TRAIN_API
using RegularizationMethod = ParameterOptimizer::RegularizationMethod;
#include "optim.h"
#include "data.h"
#include "loss.h"
#include "compress.h"
#endif
#ifdef PYMNN_OPENCV_API
#include "cv.h"
#endif
#endif

#ifdef PYMNN_LLM_API
#include "llm.h"
#endif

#ifdef PYMNN_INTERNAL_SERVING
#include <MNN/AutoTime.hpp>
#include "internal/monitor_service.h"
#include "internal/verify_service.h"
#endif

struct MNN_TLSData {
    PyObject *PyMNNHalideTypeInt = NULL;
    PyObject *PyMNNHalideTypeInt64 = NULL;
    PyObject *PyMNNHalideTypeFloat = NULL;
    PyObject *PyMNNHalideTypeDouble = NULL;
    PyObject *PyMNNHalideTypeUint8 = NULL;
    PyObject *PyMNNHalideTypeString = NULL;
    std::unordered_map<std::string, Interpreter *> *interpreterMap = NULL;
    std::unordered_map<std::string, Session *> *sessionCacheMap = NULL;
#ifdef PYMNN_EXPR_API
#if TARGET_OS_IPHONE
    ExecutorScope* scope = NULL;
#endif
#endif
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

static PyObject *importName(const char *name, const char *symbol)
{
    PyObject *u_name, *module;
    u_name = PyUnicode_FromString(name);
    module = PyImport_Import(u_name);
    if (!module) {
        return NULL;
    }
    auto f = PyObject_GetAttrString(module, symbol);
    Py_XDECREF(module);
    Py_XDECREF(u_name);
    return f;
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
    // owner: 1: own tensor and data; 2. own tensor and tensor own data.
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
static PyObject* PyMNNInterpreter_setExternalFile(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_updateCacheFile(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_setSessionMode(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_setSessionHint(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_cache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_removeCache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_updateSessionToModel(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_getModelVersion(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static int PyMNNInterpreter_init(PyMNNInterpreter *self, PyObject *args, PyObject *kwds);
static void PyMNNInterpreter_dealloc(PyMNNInterpreter *);

#ifdef PYMNN_INTERNAL_SERVING
static PyObject* PyMNNInterpreter_createSessionWithToken(PyMNNInterpreter *self, PyObject *args);
#endif

static PyMethodDef PyMNNInterpreter_methods[] = {
    {"createRuntime", (PyCFunction)PyMNNInterpreter_createRuntime, METH_VARARGS | METH_STATIC, "create runtime"},
    {"createSession", (PyCFunction)PyMNNInterpreter_createSession, METH_VARARGS, "create session"},
    {"setCacheFile", (PyCFunction)PyMNNInterpreter_setCacheFile, METH_VARARGS, "set cache file for create session"},
    {"setExternalFile", (PyCFunction)PyMNNInterpreter_setExternalFile, METH_VARARGS, "set external data file for create session"},
    {"updateCacheFile", (PyCFunction)PyMNNInterpreter_updateCacheFile, METH_VARARGS, "update cache file after resize session"},
    {"setSessionMode", (PyCFunction)PyMNNInterpreter_setSessionMode, METH_VARARGS, "set session mode before create session"},
    {"setSessionHint", (PyCFunction)PyMNNInterpreter_setSessionHint, METH_VARARGS, "set session hint before create session"},
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
    {"getModelVersion", (PyCFunction)PyMNNInterpreter_getModelVersion, METH_VARARGS, "getModelVersion"},
#ifdef PYMNN_INTERNAL_SERVING
    {"createSessionWithToken", (PyCFunction)PyMNNInterpreter_createSessionWithToken, METH_VARARGS, "create session with token"},
#endif
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNInterpreterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.Interpreter",                        /*tp_name*/
    sizeof(PyMNNInterpreter),                 /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNInterpreter_dealloc,     /*tp_dealloc*/
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
    "MNN Interpreter objects",                /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNInterpreter_methods,                 /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNInterpreter_init,          /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNInterpreter_new,                     /* tp_new */
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
static PyObject* PyMNNTensor_repr(PyObject *self);
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNTensor_fromNumpy(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getNumpyData(PyMNNTensor *self, PyObject *args);
static bool gNumpyValid = false;
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
    PyMNNTensor_repr,                         /*tp_repr*/
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
static PyObject* PyMNNCVImageProcess_setPadding(PyMNNCVImageProcess *self, PyObject *args);

static PyMethodDef PyMNNCVImageProcess_methods[] = {
    {"setMatrix", (PyCFunction)PyMNNCVImageProcess_setMatrix, METH_VARARGS, "ImageProcess setMatrix"},
    {"convert", (PyCFunction)PyMNNCVImageProcess_convert, METH_VARARGS, "ImageProcess convert"},
    {"createImageTensor", (PyCFunction)PyMNNCVImageProcess_createImageTensor, METH_VARARGS, "ImageProcess create Image Tensor"},
    {"setPadding", (PyCFunction)PyMNNCVImageProcess_setPadding, METH_VARARGS, "ImageProcess setPadding"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNCVImageProcessType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.CVImageProcess",                     /*tp_name*/
    sizeof(PyMNNCVImageProcess),              /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNCVImageProcess_dealloc,  /*tp_dealloc*/
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
    "MNN CVImageProcess objects",             /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNCVImageProcess_methods,              /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNCVImageProcess_init,       /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNCVImageProcess_new,                  /* tp_new */
};

/// MNN CVMatrix Type
static PyObject* PyMNNCVMatrix_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static void PyMNNCVMatrix_dealloc(PyMNNCVMatrix *);
static PyObject* PyMNNCVMatrix_repr(PyObject *self);
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
/// poly2poly
static PyObject* PyMNNCVMatrix_setPolyToPoly(PyMNNCVMatrix *, PyObject *args);

static PyObject* PyMNNCVMatrix_invert(PyMNNCVMatrix *);
static PyObject* PyMNNCVMatrix_write(PyMNNCVMatrix *, PyObject *args);
static PyObject* PyMNNCVMatrix_read(PyMNNCVMatrix *);

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

    {"setPolyToPoly", (PyCFunction)PyMNNCVMatrix_setPolyToPoly, METH_VARARGS, "MNNCVMatrix setPolyToPoly"},
    {"invert", (PyCFunction)PyMNNCVMatrix_invert, METH_VARARGS, "MNNCVMatrix invert"},
    {"write", (PyCFunction)PyMNNCVMatrix_write, METH_VARARGS, "MNNCVMatrix write from list"},
    {"read", (PyCFunction)PyMNNCVMatrix_read, METH_VARARGS, "MNNCVMatrix read as list"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNCVMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.CVMatrix",                           /*tp_name*/
    sizeof(PyMNNCVMatrix),                    /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNCVMatrix_dealloc,        /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    PyMNNCVMatrix_repr,                       /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    PyMNNCVMatrix_repr,                       /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MNN CVMatrix objects",                   /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNCVMatrix_methods,                    /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNCVMatrix_new,                        /* tp_new */
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

static void _runtime_capsule_deleter(PyObject *obj) {
    auto info = (RuntimeInfo*)PyCapsule_GetPointer(obj, NULL);
    if (info != nullptr) {
        delete info;
    }
}

static PyObject* PyMNNInterpreter_createRuntime(PyObject* self, PyObject* args) {
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dicts = NULL;
    if (!PyArg_ParseTuple(args, "O", &dicts)) {
        return NULL;
    }
    if (!PySequence_Check(dicts)) {
        return Py_None;
    }
    // BackendConfig lifetime management
    if(PySequence_Size(dicts) > MAX_CONFIG_SIZE) {
        MNN_PRINT("Error: MNN support max ScheduleConfig size is %d\n", MAX_CONFIG_SIZE);
        return Py_None;
    }
    std::vector<ScheduleConfig> configs;
    ScheduleConfig config[MAX_CONFIG_SIZE];
    BackendConfig backendConfig[MAX_CONFIG_SIZE];
    for (auto i = 0; i < PySequence_Size(dicts); ++i) {
        config[i].backendConfig = &backendConfig[i];
        bool ret = getScheduleConfig(PySequence_GetItem(dicts, i), config[i]);
        if (!ret) {
            return Py_None;
        }
        configs.push_back(config[i]);
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

static PyObject* createSession(PyMNNInterpreter *self, PyObject* dict, PyObject *rtinfo_py) {
    PyObject *f = importName("MNN", "Session");
    if (!f || !PyCallable_Check(f)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session not found");
        return NULL;
    }

    // create a new session
    PyMNNSession *session = (PyMNNSession *)PyObject_CallObject(f, NULL);
    Py_XDECREF(f);
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

    ScheduleConfig config;
    BackendConfig backendConfig;
    config.backendConfig = &backendConfig;
    bool ret = getScheduleConfig(dict, config);
    if (!ret) {
        return NULL;
    }
    Session* s;
    if (rtinfo_py == NULL) {
        s = self->interpreter->createSession(config);
    } else {
        auto runtimeinfo = *(RuntimeInfo*)PyCapsule_GetPointer(rtinfo_py, NULL);
        s = self->interpreter->createSession(config, runtimeinfo);
    }
    if (!s) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: NetInstance createSession failed");
        return NULL;
    }

    session->session = s;
    session->modelPath = self->modelPath;

    return (PyObject *)session;
}

static PyObject* PyMNNInterpreter_createSession(PyMNNInterpreter *self, PyObject *args) {
#ifdef PYMNN_INTERNAL_SERVING
    PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: unsupported interface, should use createSessionWithToken.");
    return NULL;
#endif
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dict = NULL, *rtinfo_py = NULL;
    if (!PyArg_ParseTuple(args, "|OO", &dict, &rtinfo_py)) {
        return NULL;
    }

    return createSession(instance, dict, rtinfo_py);
}

#ifdef PYMNN_INTERNAL_SERVING
static PyObject* PyMNNInterpreter_createSessionWithToken(PyMNNInterpreter *self, PyObject *args) {
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dict = NULL, *rtinfo_py = NULL;
    char *token = NULL;
    char *scene = NULL;
    char *app_key = NULL;
    if (!PyArg_ParseTuple(args, "sss|OO", &token, &scene, &app_key, &dict, &rtinfo_py)) {
        return NULL;
    }

    if (!token || !scene || !app_key) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSessionWithToken: input invalid, token, scene or app_key is null.");
        return NULL;
    }

    bool ret = VerifyService::GetInstance().VerifyToken(std::string(token), std::string(scene), std::string(app_key));
    if (!ret) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNNN_createSessionWithToken: check token failed, return null session.");
        return NULL;
    }

    return createSession(instance, dict, rtinfo_py);
}
#endif

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
static PyObject* PyMNNInterpreter_setExternalFile(PyMNNInterpreter *self, PyObject *args) {
    char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_setExternalFile: Not string input");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    self->interpreter->setExternalFile(path);
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

    ErrorCode r;
    r = self->interpreter->updateCacheFile(session->session, flag);
    return PyLong_FromLong(r);
}
static PyObject* PyMNNInterpreter_setSessionMode(PyMNNInterpreter *self, PyObject *args) {
    int session_val;
    if (!PyArg_ParseTuple(args, "i", &session_val)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_setSessionMode: Not interger input");
        return NULL;
    }
    auto mode = (MNN::Interpreter::SessionMode)session_val;
    self->interpreter->setSessionMode(mode);
    Py_RETURN_NONE;
}
static PyObject* PyMNNInterpreter_setSessionHint(PyMNNInterpreter *self, PyObject *args) {
    int type_val = 0;
    int num_val = 0;
    if (!PyArg_ParseTuple(args, "ii", &type_val, &num_val)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_setSessionHint: Not interger input and interger input");
        return NULL;
    }

    auto type = (MNN::Interpreter::HintMode)type_val;
    self->interpreter->setSessionHint(type, num_val);
    Py_RETURN_NONE;
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
    ErrorCode r;
    Py_BEGIN_ALLOW_THREADS

#ifdef PYMNN_INTERNAL_SERVING
    Timer timer;
    r = self->interpreter->runSession(session->session);
    float cost_time = (float)timer.durationInUs() / (float)1000;
    MNN::Interpreter::SessionInfoCode info_type = MNN::Interpreter::BACKENDS;
    int backendType[MNN_FORWARD_ALL];
    self->interpreter->getSessionInfo(session->session, info_type, backendType);
    std::string mBizCode = self->interpreter->bizCode() ? self->interpreter->bizCode() : "";
    std::string mUuid = self->interpreter->uuid() ? self->interpreter->uuid() : "";
    MonitorService::GetInstance().Track(cost_time, std::to_string(*backendType), "RUN_SESSION",
                                             "PyMNNInterpreter_runSession", std::to_string(r), mBizCode, mUuid);
#else
    r = self->interpreter->runSession(session->session);
#endif

    Py_END_ALLOW_THREADS
    return PyLong_FromLong(r);
}
static PyMNNTensor* getTensor() {
    PyMNNTensor *tensor = (PyMNNTensor *)PyObject_CallObject((PyObject*)PyType_FindTLSType(&PyMNNTensorType), NULL);
    if (tensor) {
        tensor->tensor = nullptr;
    }
    return tensor;
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
        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor* tensor = getTensor();
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
        auto pyret = PyObject_Call(beginCallback, args, NULL);
        bool ret = static_cast<bool>(PyLong_AsLong(pyret));
        Py_XDECREF(pyret);
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };
    TensorCallBack end = [endCallback](const std::vector<Tensor*>& tensors, const std::string& name){
        if (!endCallback || !PyCallable_Check(endCallback)) {
            return true;
        }
        PyObject *args = PyTuple_New(2);
        size_t size_tensors = tensors.size();
        PyObject *weTensorData = PyTuple_New(size_tensors);
        for (size_t i = 0; i < size_tensors; i++) {
            // create a new tensor
            PyMNNTensor* tensor = getTensor();
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
        auto pyret = PyObject_Call(endCallback, args, NULL);
        bool ret = static_cast<bool>(PyLong_AsLong(pyret));
        Py_XDECREF(pyret);
        Py_XDECREF(args);//del all the C++ created python api parameters
        return ret;
    };

    ErrorCode r;
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
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_CallObject(ftensor, NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        //printf("begincallback name=%s\n",name.c_str());
        PyMNNOpInfo *pyinfo = (PyMNNOpInfo *)PyObject_CallObject(finfo, NULL);
        if(!pyinfo){
            PyErr_SetString(PyExc_Exception,
                    "PyMNNInterpreter_runSessionWithCallBackInfo: create OpInfo failed");
            return true;
        }
        pyinfo->opInfo = info;
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, (PyObject *)pyinfo);
        auto pyret = PyObject_Call(beginCallback, args, NULL);
        bool ret = static_cast<bool>(PyLong_AsLong(pyret));
        Py_XDECREF(pyret);
        Py_XDECREF(args);//del all the C++ created python api parameters
        Py_XDECREF(ftensor);
        Py_XDECREF(finfo);
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
            PyMNNTensor *tensor = (PyMNNTensor *)PyObject_CallObject(ftensor, NULL);
            if (!tensor) {
                PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_runSessionWithCallBackInfo: create Tensor failed");
                return true;
            }
            tensor->tensor = tensors[i];
            PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        }
        PyMNNOpInfo *pyinfo = (PyMNNOpInfo *)PyObject_CallObject(finfo, NULL);
        if(!pyinfo){
            PyErr_SetString(PyExc_Exception,
                    "PyMNNInterpreter_runSessionWithCallBackInfo: create OpInfo failed");
            return true;
        }
        pyinfo->opInfo = info;
        PyTuple_SetItem(args, 0, weTensorData);
        PyTuple_SetItem(args, 1, (PyObject *)pyinfo);
        auto pyret = PyObject_Call(endCallback, args, NULL);
        bool ret = static_cast<bool>(PyLong_AsLong(pyret));
        Py_XDECREF(pyret);
        Py_XDECREF(args);//del all the C++ created python api parameters
        Py_XDECREF(ftensor);
        Py_XDECREF(finfo);
        return ret;
    };

    ErrorCode r;
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
    PyMNNTensor *tensor = (PyMNNTensor *)PyObject_CallObject(f, NULL);
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session instance create failed");
        return NULL;
    }

    tensor->tensor = t;
    Py_XDECREF(f);
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
    PyMNNTensor *tensor = (PyMNNTensor *)PyObject_CallObject(f, NULL);
    if (!tensor) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_createSession: MNN.Session instance create failed");
        return NULL;
    }

    tensor->tensor = t;
    Py_XDECREF(f);
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
        PyObject *tensor = PyObject_CallObject(f, NULL);
        if (!tensor) {
            PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionOutputAll: MNN.Tensor instance create failed");
            return NULL;
        }
        ((PyMNNTensor*)tensor)->tensor = it->second;
        PyDict_SetItemString(output, it->first.c_str(), tensor);
        Py_XDECREF(tensor);
    }
    Py_XDECREF(f);
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
        PyObject *tensor = PyObject_CallObject(f, NULL);
        if (!tensor) {
            PyErr_SetString(PyExc_Exception,"PyMNNInterpreter_getSessionInputAll: MNN.Tensor instance create failed");
            return NULL;
        }
        ((PyMNNTensor*)tensor)->tensor = it->second;
        PyDict_SetItemString(output, it->first.c_str(), tensor);
        Py_XDECREF(tensor);
    }
    Py_XDECREF(f);
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
                        "PyMNNInterpreter_new: NetInstance::createFromFile failed. Invalid model file, Check console log messages!");
        return -1;
    }

#ifdef PYMNN_INTERNAL_SERVING
    // initialize MonitorService
    MonitorService::GetInstance().Start();
    VerifyService::GetInstance().Start();
#endif

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

static PyObject* PyMNNInterpreter_getModelVersion(PyMNNInterpreter *self, PyObject *args) {
    return toPyObj(self->interpreter->getModelVersion());
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
bool isTensor(PyObject* t) {
    return PyObject_IsInstance(t, (PyObject*)PyType_FindTLSType(&PyMNNTensorType));
}
Tensor* toTensor(PyObject* t) {
    return ((PyMNNTensor*)t)->tensor;
}
static PyObject* PyMNNTensor_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNTensor* self = (PyMNNTensor *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void PyMNNTensor_dealloc(PyMNNTensor *self) {
    if (self->owner) {
        if (self->tensor->host<void *>() && self->owner != 2) {
            free(self->tensor->host<void *>());
        }
        delete self->tensor;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyMNNTensor_init(PyMNNTensor *self, PyObject *args, PyObject *kwds) {
    int argc = PyTuple_Size(args);
    PyObject *shape, *dataType, *data = nullptr, *input_tensor = nullptr, *input_var = nullptr;
    long dimensionType = -1;
    bool parse_res = false;
    switch (argc) {
        case 0:
            // just return, using in `PyMNNInterpreter_getSessionInputAll`;
            return 0;
#ifdef PYMNN_EXPR_API
        case 1:
            parse_res = PyArg_ParseTuple(args, "O", &input_var)
                        && isVar(input_var);
            break;
        case 2:
            parse_res = PyArg_ParseTuple(args, "Ol", &input_tensor, &dimensionType)
                        && (isTensor(input_tensor) || isVar(input_tensor));
            if (isVar(input_tensor)) {
                input_var = input_tensor;
                input_tensor = nullptr;
            }
            break;
#else
        case 2:
            parse_res = PyArg_ParseTuple(args, "Ol", &input_tensor, &dimensionType)
                        && isTensor(input_tensor);
            break;
#endif
        case 3:
            parse_res = PyArg_ParseTuple(args, "OOl", &shape, &dataType, &dimensionType)
                        && isInts(shape);
            break;
        case 4:
            parse_res = PyArg_ParseTuple(args, "OOOl", &shape, &dataType, &data, &dimensionType)
                        && isInts(shape) && (isVals(data) || isInt(data));
            break;
        default:
            parse_res = false;
    }
    if (!parse_res) {
        PyMNN_ERROR_LOG("Tensor init require args as belows:\n"
                        "\t0. (Var)\n"
                        "\t1. (Tensor/Var, DimensionType)\n"
                        "\t2. ([int], DataType, DimensionType)\n"
                        "\t3. ([int], DataType, ndarray/list/tuple/bytes/PyCapsule/int_addr, DimensionType)\n");
        return -1;
    }
#ifdef PYMNN_EXPR_API
    // 0. create Tensor by Var
    if (input_var) {
        auto var = toVar(input_var);
        auto info = var->getInfo();
        void* ptr = const_cast<void*>(var->readMap<void>());
        Tensor::DimensionType type = Tensor::TENSORFLOW;
        if (dimensionType < 0) {
            if (info->order == NCHW) type = Tensor::CAFFE;
            else if (info->order == NC4HW4) type = Tensor::CAFFE_C4;
        } else {
            type = static_cast<Tensor::DimensionType>(dimensionType);
        }
        Tensor *tensor = Tensor::create(info->dim, info->type, ptr, type);
        if (!tensor) {
            PyMNN_ERROR_LOG("PyMNNTensor_create: Tensor create failed");
            return -1;
        }
        self->tensor = tensor;
        self->owner = 2;
        return 0;
    }
#endif
    // 1. create Tensor by Tensor
    if (input_tensor) {
        Tensor *tensor = new Tensor(toTensor(input_tensor), (Tensor::DimensionType)dimensionType, true);
        if (!tensor) {
            PyMNN_ERROR_LOG("PyMNNTensor_create: Tensor create failed");
            return -1;
        }
        self->tensor = tensor;
        self->owner = 2;
        return 0;
    }
    // 2,3. create Tensor by shape and data
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
        PyMNN_ERROR_LOG("PyMNNTensor_create: unsupported data type");
        return -1;
    }
    DType dtype = htype2dtype(htt);
    int itemsize = getitemsize(dtype);
    size_t shapeSize = PySequenceSize(shape);
    std::vector<int> vShape = toInts(shape);
    size_t dataSize = 1;
    for (auto i : vShape) {
        dataSize *= i;
    }
    void *pData = NULL;
    if (data && !PyCapsule_CheckExact(data) && !isInt(data)) {
        if (PyBytes_Check(data)) {
            int64_t total_len = PyBytes_Size(data);
            if (dataSize * itemsize != total_len) {
                PyMNN_ERROR_LOG("PyMNNTensor_init: Tensor Dim not match");
                return -1;
            }
            pData = toPtr(data, DType_UINT8, total_len);
        } else {
            if (isPySequence(data)) {
                int inputSize = PySequenceSize(data);
                if (dataSize != inputSize) {
                    PyMNN_ERROR_LOG("PyMNNTensor_init: Tensor Dim not match");
                    return -1;
                }
            }
#ifdef PYMNN_NUMPY_USABLE
            else if (gNumpyValid && PyArray_Check(data)) {
                if(dataSize != PyArray_Size(data)) {
                    PyMNN_ERROR_LOG("PyMNNTensor_init: numpy array size does not match shape requirement");
                    return -1;
                }
            }
#endif
            else {
                PyMNN_ERROR_LOG("PyMNNTensor_init: data is not tuple/list/bytes/ndarray");
                return -1;
            }
            int64_t total_len = dataSize;
            pData = toPtr(data, dtype, total_len);
        }
        if(NULL == pData) {
            PyMNN_ERROR_LOG("PyMNNTensor_init: data load failed");
            return -1;
        }
    } else {
        // no data input, set all zeros
        // pycapsule/int_addr input, copy data
        pData = malloc(dataSize * itemsize);
        if (data) {
            void* srcPtr = nullptr;
            if (PyCapsule_CheckExact(data)) {
                srcPtr = PyCapsule_GetPointer(data, NULL);
            } else {
                srcPtr = PyLong_AsVoidPtr(data);
            }
            if (srcPtr == nullptr) {
                PyMNN_ERROR_LOG("PyMNNTensor_init: PyCapsule/int_addr pointer is null.");
                return -1;
            }
            memcpy(pData, srcPtr, dataSize * itemsize);
        } else {
            memset(pData, 0, dataSize * itemsize);
        }
    }
    Tensor *tensor = Tensor::create(vShape
                               , htt
                               , pData
                               , (Tensor::DimensionType)dimensionType
                               );
    if (!tensor) {
        PyMNN_ERROR_LOG("PyMNNTensor_create: Tensor create failed");
        return -1;
    }
    self->tensor = tensor;
    self->owner = 1;
    return 0;
}
static PyObject* PyMNNTensor_repr(PyObject *self) {
    auto tensor = ((PyMNNTensor*)self)->tensor;
    if (tensor == nullptr || tensor->host<void>() == nullptr) {
        return toPyObj("array([])");
    }
#ifdef PYMNN_NUMPY_USABLE
    auto content = PyMNNTensor_getNumpyData(((PyMNNTensor*)self), NULL);
#else
    auto content = PyMNNVar_read_as_tuple((PyMNNVar*)self, NULL);
#endif
    auto reprfunc = PyObject_GetAttrString(content, "__repr__");
    auto str = PyEval_CallObject(reprfunc, NULL);
    Py_DECREF(content);
    Py_DECREF(reprfunc);
    return str;
}
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNTensor_fromNumpy(PyMNNTensor *self, PyObject *args) {
    if (!gNumpyValid) {
        PyErr_SetString(PyExc_Exception,"PyMNNTensor_fromNumpy: numpy not valid");
        return NULL;
    }
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
        int npy_type = PyArray_TYPE((const PyArrayObject*)data);
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
    if (!gNumpyValid) {
        PyErr_SetString(PyExc_Exception,"PyMNNTensor_getNumpyData: numpy not valid");
        return NULL;
    }
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

    if (isInt(source)) {
        auto ptr = PyLong_AsVoidPtr(source);
        if (ptr == NULL) {
            Py_RETURN_NONE;
        }
        ErrorCode ret = self->imageProcess->convert(reinterpret_cast<const uint8_t *>(ptr),
                                                    iw, ih, stride,
                                                    ((PyMNNTensor *)dest)->tensor);
        return PyLong_FromLong(ret);
    } else if (PyCapsule_CheckExact(source)) {
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
    else if(gNumpyValid && PyArray_Check(source)) {
        // Array Data
        int npy_type = PyArray_TYPE((const PyArrayObject*)source);
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

    PyErr_SetString(PyExc_Exception, "PyMNNCVImageProcess_convert: argument 0 is not a long or capsule or tuple or numpy");

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

    PyMNNTensor *t = (PyMNNTensor *)PyObject_CallObject(f, NULL);
    if (!t) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_createImageTensor: create image tensor failed");
        return NULL;
    }

    t->tensor = tensor;
    t->owner = 1;
    Py_XDECREF(f);
    return (PyObject *)t;
}

static PyObject* PyMNNCVImageProcess_setPadding(PyMNNCVImageProcess *self, PyObject *args) {
    int value;
    if (!PyArg_ParseTuple(args, "i", &value)) {
        PyMNN_ERROR("setPadding require args: (int)");
    }
    self->imageProcess->setPadding(static_cast<uint8_t>(value));
    Py_RETURN_NONE;
}

/// MNN CVMatrix implementation
bool isMatrix(PyObject* obj) {
    return PyObject_IsInstance(obj, (PyObject*)PyType_FindTLSType(&PyMNNCVMatrixType));
}
CV::Matrix toMatrix(PyObject* obj) {
    return *(((PyMNNCVMatrix*)obj)->matrix);
}
PyObject* toPyObj(CV::Matrix m) {
    PyMNNCVMatrix *ret = (PyMNNCVMatrix *)PyObject_CallObject((PyObject*)PyType_FindTLSType(&PyMNNCVMatrixType), NULL);
    ret->matrix = new CV::Matrix();
    *(ret->matrix) = m;
    return (PyObject*)ret;
}

bool isPoint(PyObject* obj) {
    return (isFloats(obj) && toFloats(obj).size() == 2) ||
           (isInts(obj) && toInts(obj).size() == 2);
}
CV::Point toPoint(PyObject* obj) {
    CV::Point point;
    if (isFloats(obj)) {
        auto vals = toFloats(obj);
        MNN_ASSERT(vals.size() == 2);
        point.set(vals[0], vals[1]);
    } else if (isInts(obj)) {
        auto vals = toInts(obj);
        MNN_ASSERT(vals.size() == 2);
        point.set(vals[0], vals[1]);
    }
    return point;
}
bool isPoints(PyObject* obj) {
    return (isFloats(obj) && toFloats(obj).size() % 2 == 0) ||
           (isInts(obj) && toInts(obj).size() % 2 == 0) || isVar(obj);
}
std::vector<CV::Point> toPoints(PyObject* obj) {
    if (isFloats(obj)) {
        auto vals = toFloats(obj);
        MNN_ASSERT(vals.size() % 2 == 0);
        std::vector<CV::Point> points(vals.size() / 2);
        for (int i = 0; i < points.size(); i++) {
            points[i].set(vals[i*2], vals[i*2+1]);
        }
        return points;
    }
    if (isInts(obj)) {
        auto vals = toInts(obj);
        MNN_ASSERT(vals.size() % 2 == 0);
        std::vector<CV::Point> points(vals.size() / 2);
        for (int i = 0; i < points.size(); i++) {
            points[i].set(vals[i*2], vals[i*2+1]);
        }
        return points;
    }
    if (isVar(obj)) {
        auto vals = toVar(obj);
        auto info = vals->getInfo();
        auto size = info->size;
        MNN_ASSERT(size % 2 == 0);
        std::vector<CV::Point> points(size / 2);
        if (info->type == halide_type_of<float>()) {
            auto ptr = vals->readMap<float>();
            for (int i = 0; i < points.size(); i++) {
                points[i].set(ptr[i*2], ptr[i*2+1]);
            }
        } else if (info->type == halide_type_of<int>()) {
            auto ptr = vals->readMap<int>();
            for (int i = 0; i < points.size(); i++) {
                points[i].set(ptr[i*2], ptr[i*2+1]);
            }
        } else {
            PyMNN_ERROR_LOG("Point data type must be int32 or float32.");
        }
        return points;
    }
    return {};
}
PyObject* toPyObj(std::vector<CV::Point> _points) {
    std::vector<float> points(_points.size() * 2);
    for (int i = 0; i < _points.size(); i++) {
        points[2 * i + 0] = _points[i].fX;
        points[2 * i + 1] = _points[i].fY;
    }
    return toPyObj(points);
}

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

static PyObject* PyMNNCVMatrix_repr(PyObject *self) {
    float mat[9];
    ((PyMNNCVMatrix *)self)->matrix->get9(mat);
    char buffer [100];
    sprintf(buffer, "[[%f\t%f\t%f]\n [%f\t%f\t%f]\n [%f\t%f\t%f]]",
            mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);
    return toPyObj(buffer);
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

static PyObject* PyMNNCVMatrix_setPolyToPoly(PyMNNCVMatrix *self, PyObject *args) {
    PyObject *src, *dst;
    if (PyArg_ParseTuple(args, "OO", &src, &dst) && isPoints(src) && isPoints(dst)) {
        auto s = toPoints(src);
        auto d = toPoints(dst);
        self->matrix->setPolyToPoly(s.data(), d.data(), s.size());
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("setPolyToPoly require args: ([float], [float])");
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
            self->matrix->setTranslate(dx, dy);
            break;
        case 1:
            self->matrix->preTranslate(dx, dy);
            break;
        case 2:
            self->matrix->postTranslate(dx, dy);
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
static PyObject* PyMNNCVMatrix_write(PyMNNCVMatrix *self, PyObject *args) {
    PyObject* data = NULL;
    if (PyArg_ParseTuple(args, "O", &data) && isFloats(data)) {
        auto vec = toFloats(data);
        int write_size = vec.size() > 9 ? 9 : vec.size();
        for (int i = 0; i < write_size; i++) {
            self->matrix->set(i, vec[i]);
        }
        Py_RETURN_NONE;
    }
    PyMNN_ERROR("write require args: ([float])");
}
static PyObject* PyMNNCVMatrix_read(PyMNNCVMatrix *self) {
    std::vector<float> mat(9);
    self->matrix->get9(mat.data());
    return toPyObj(mat);
}
static PyObject* PyMNNOpInfo_getName(PyMNNOpInfo *self, PyObject *args);
static PyObject* PyMNNOpInfo_getType(PyMNNOpInfo *self, PyObject *args);
static PyObject* PyMNNOpInfo_getFlops(PyMNNOpInfo *self, PyObject *args);

static void PyMNNOpInfo_dealloc(PyMNNOpInfo *self);
static PyObject* PyMNNOpInfo_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static int PyMNNOpInfo_init(PyMNNOpInfo *info, PyObject *args, PyObject *kwds);

static PyMethodDef PyMNNOpInfo_methods[] = {
    {"getName", (PyCFunction)PyMNNOpInfo_getName, METH_VARARGS, "get op name"},
    {"getType", (PyCFunction)PyMNNOpInfo_getType, METH_VARARGS, "get op type"},
    {"getFlops", (PyCFunction)PyMNNOpInfo_getFlops, METH_VARARGS, "get op flops"},
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
static PyObject* PyMNNOpInfo_getFlops(PyMNNOpInfo *self, PyObject *args) {
    PyObject *flops = NULL;
    if (self->opInfo) {
        flops = toPyObj(self->opInfo->flops());
    }
    return flops;
}
static PyTypeObject PyMNNOpInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MNN.OpInfo",                             /*tp_name*/
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

#ifdef PYMNN_TRAIN_API
static PyObject* PyMNN_get_model_uuid(PyObject *self, PyObject *args) {
    const char* modelFile;
    if (!PyArg_ParseTuple(args, "s", &modelFile)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
    return toPyObj(HelperFuncs::getModelUUID(modelFile));
}
#endif
static PyObject* PyMNN_version(PyObject *self, PyObject *args) {
    return toPyObj(MNN::getVersion());
}
/// module init
static PyMethodDef module_methods[] = {
#ifdef PYMNN_TRAIN_API
    {"get_model_uuid", (PyCFunction)PyMNN_get_model_uuid, METH_VARARGS, "get model's uuid"},
#endif
    {"version", (PyCFunction)PyMNN_version, METH_VARARGS, "get MNN version number"},
    {NULL, NULL, 0, NULL}
};

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
    gNumpyValid = true;
    if(_import_array() < 0) {
        MNN_PRINT("[Warnning] import numpy failed, please reinstall numpy!\n");
        PyErr_Clear();
        gNumpyValid = false;
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
    // for expr multi-thread
#ifdef PYMNN_USE_ALINNPYTHON
    // create different excutor for each thread when alinn python
    BackendConfig bnConfig;
    auto threadExecutor = Executor::newExecutor(MNN_FORWARD_CPU, bnConfig, 1);
    // close lazy evaluation in python for speed and memory
    threadExecutor->lazyEval = false;
    #if TARGET_OS_IPHONE
    tlsData->scope = new ExecutorScope(threadExecutor);
    #else
    static thread_local ExecutorScope scope(threadExecutor);
    #endif
#else
    // use the same excutor for each thread
    auto exe = ExecutorScope::Current();
    // close lazy evaluation in python for speed and memory
    exe->lazyEval = false;
#endif
    // _expr submodule
    auto expr_module = def_submodule(m, "_expr");
    if (PyType_Ready(&PyMNNVarType) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN.expr: PyType_Ready PyMNNVarType failed");
        ERROR_RETURN
    }
    PyModule_AddObject(expr_module, "Var", (PyObject *)PyType_FindTLSType(&PyMNNVarType));
    // def enum
    def_data_format(expr_module);
    def_dtype(expr_module);
    def_Padding_Mode(expr_module);
    def_PadValue_Mode(expr_module);
    def_Pooling_Mode(expr_module);
    def_Interp_Method(expr_module);
    def_Backend(expr_module);
    def_MemoryMode(expr_module);
    def_PowerMode(expr_module);
    def_PrecisionMode(expr_module);
    // add methods of expr
    constexpr int expr_method_num = sizeof(PyMNNExpr_methods) / sizeof(PyMethodDef);
    for (int i = 0; i < expr_method_num; i++) {
        def_method(expr_module, &PyMNNExpr_methods[i]);
    }
    // _nn module
    auto nn_module = def_submodule(m, "_nn");
    def__Module(nn_module);
    def_RuntimeManager(nn_module);
    def_methods(nn_module, NN)
#ifdef PYMNN_TRAIN_API
    // _optim module
    auto optim_module = def_submodule(m, "_optim");
    def_Regularization_Method(optim_module);
    def_Optimizer(optim_module);
    def_methods(optim_module, Optim);
    // _data module
    auto data_module = def_submodule(m, "_data");
    def_Dataset(data_module);
    def_DataLoader(data_module);
    // loss module
    auto loss_module = def_submodule(nn_module, "loss");
    def_methods(loss_module, Loss);
    // define compress module
    auto compress_module = def_submodule(nn_module, "compress");
    def_Feature_Scale_Method(compress_module);
    def_Scale_Update_Method(compress_module);
    def_methods(compress_module, Compress);
#endif
#ifdef PYMNN_OPENCV_API
    // cv submodule
    auto cv_module = def_submodule(m, "cv");
    // add methods of cv
    constexpr int cv_method_num = sizeof(PyMNNCV_methods) / sizeof(PyMethodDef);
    for (int i = 0; i < cv_method_num; i++) {
        def_method(cv_module, &PyMNNCV_methods[i]);
    }
#endif
#endif
#ifdef PYMNN_LLM_API
    // llm submodule
    auto llm_module = def_submodule(m, "llm");
    if (PyType_Ready(&PyMNNLLM) < 0) {
        PyErr_SetString(PyExc_Exception, "initMNN.llm: PyType_Ready PyMNNLLM failed");
        ERROR_RETURN
    }
    PyModule_AddObject(llm_module, "LLM", (PyObject *)PyType_FindTLSType(&PyMNNLLM));
    // add methods of llm
    constexpr int llm_method_num = sizeof(PyMNNLLM_static_methods) / sizeof(PyMethodDef);
    for (int i = 0; i < llm_method_num; i++) {
        def_method(llm_module, &PyMNNLLM_static_methods[i]);
    }
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

#if defined(PYMNN_USE_ALINNPYTHON)
extern "C" MNN_PUBLIC void* memoryToVar(const void* ptr, int h, int w, int c, int type) {
    auto var = Express::_Const(ptr, {h, w, c}, NHWC, dtype2htype(static_cast<DType>(type)));
    return reinterpret_cast<void*>(toPyObj(var));
}
#endif
