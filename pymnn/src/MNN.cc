/*
    MNN python module 
*/
#include <fstream>

#ifdef USE_PRIVATE
#include "private_define.h"
#else
#include <Python.h>
#include "structmember.h"
#endif
#include <mutex>
#include <unordered_map>
#if __has_include(<MNN/Interpreter.hpp>)
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#else
#include "Interpreter.hpp"
#include "ImageProcess.hpp"
#endif
#include "util.h"
using namespace MNN;
using namespace std;

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

static PyObject *PyMNNHalideTypeInt = NULL;
static PyObject *PyMNNHalideTypeInt64 = NULL;
static PyObject *PyMNNHalideTypeFloat = NULL;
static PyObject *PyMNNHalideTypeDouble = NULL;
static PyObject *PyMNNHalideTypeUint8 = NULL;
static PyObject *PyMNNHalideTypeString = NULL;

/// MNN NetInstance Type
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
static PyObject* PyMNNInterpreter_cache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_removeCache(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_updateSessionToModel(PyMNNInterpreter *self, PyObject *args);
static PyObject* PyMNNInterpreter_new(struct _typeobject *type, PyObject *args, PyObject *kwds);
static int PyMNNInterpreter_init(PyMNNInterpreter *self, PyObject *args, PyObject *kwds);
static void PyMNNInterpreter_dealloc(PyMNNInterpreter *);

static PyMethodDef PyMNNInterpreter_methods[] = {
    {"createSession", (PyCFunction)PyMNNInterpreter_createSession, METH_VARARGS, "create session"},
    {"resizeSession", (PyCFunction)PyMNNInterpreter_resizeSession, METH_VARARGS, "resize session"},
    {"runSession", (PyCFunction)PyMNNInterpreter_runSession, METH_VARARGS, "run session"},
    {"runSessionWithCallBack", (PyCFunction)PyMNNInterpreter_runSessionWithCallBack, METH_VARARGS, "run session with callback"},
    {"runSessionWithCallBackInfo", (PyCFunction)PyMNNInterpreter_runSessionWithCallBackInfo, METH_VARARGS, "run session with callback info"},
    {"getSessionOutput", (PyCFunction)PyMNNInterpreter_getSessionOutput, METH_VARARGS, "get session output"},
    {"getSessionInput", (PyCFunction)PyMNNInterpreter_getSessionInput, METH_VARARGS, "get session input"},
    {"getSessionOutputAll", (PyCFunction)PyMNNInterpreter_getSessionOutputAll, METH_VARARGS, "get session output all"},
    {"getSessionInputAll", (PyCFunction)PyMNNInterpreter_getSessionInputAll, METH_VARARGS, "get session input all"},
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
static PyObject* PyMNNTensor_printTensorData(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getShape(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getDataType(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getDimensionType(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getData(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_getHost(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_copyFrom(PyMNNTensor *self, PyObject *args);
static PyObject* PyMNNTensor_copyToHostTensor(PyMNNTensor *self, PyObject *args);

static PyMethodDef PyMNNTensor_methods[] = {
    {"printTensorData", (PyCFunction)PyMNNTensor_printTensorData, METH_NOARGS, "print tensor data"},
    {"getShape", (PyCFunction)PyMNNTensor_getShape, METH_NOARGS, "get tensor shape"},
    {"getDataType", (PyCFunction)PyMNNTensor_getDataType, METH_NOARGS, "get tensor data type"},
    {"getData", (PyCFunction)PyMNNTensor_getData, METH_NOARGS, "get tensor data"},
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

static PyMethodDef PyMNNCVImageProcess_methods[] = {
    {"setMatrix", (PyCFunction)PyMNNCVImageProcess_setMatrix, METH_VARARGS, "ImageProcess setMatrix"},
    {"convert", (PyCFunction)PyMNNCVImageProcess_convert, METH_VARARGS, "ImageProcess convert"},
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
static PyObject* PyMNNCVMatrix_postScale(PyMNNCVMatrix *, PyObject *args);

static PyMethodDef PyMNNCVMatrix_methods[] = {
    {"postScale", (PyCFunction)PyMNNCVMatrix_postScale, METH_VARARGS, "MNNCVMatrix postScale"},
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
    static std::unordered_map<std::string, Interpreter *> *interpreterMap = nullptr; // <path, instance>
    static std::once_flag flag;
    std::call_once(flag, [](){interpreterMap = new std::unordered_map<std::string, Interpreter *>();});
    return interpreterMap;
}

std::unordered_map<std::string, Session *> *sessionCacheMap() {
    static std::unordered_map<std::string, Session *> *sessionCacheMap = nullptr; // <path, instance>
    static std::once_flag flag;
    std::call_once(flag, [](){sessionCacheMap = new std::unordered_map<std::string, Session *>();});
    return sessionCacheMap;
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
            for (int i=0; i<saveTensorsCount; i++) {
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

static PyObject* PyMNNInterpreter_createSession(PyMNNInterpreter *self, PyObject *args) {
    PyMNNInterpreter* instance = (PyMNNInterpreter *)self;
    PyObject* dict = NULL;
    if (!PyArg_ParseTuple(args, "|O", &dict)) {
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
    
    ScheduleConfig config;
    if (dict) {
        PyObject *numThread = PyDict_GetItemString(dict, "numThread");
        if (numThread) {
            if (!PyLong_Check(numThread)) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: numThread must be a integer");
                return NULL;
            }
            
            config.numThread = (int)PyLong_AsLong(numThread);
        }
        
        if (-1 == ec::getVectorByKey(dict, "saveTensors", config.saveTensors)
            || -1 == ec::getVectorByKey(dict, "inputPaths", config.path.inputs)
            || -1 == ec::getVectorByKey(dict, "outputPaths", config.path.outputs)){
            return NULL;
        }

    }
    
    Session *s = instance->interpreter->createSession(config);
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
    
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
    
    if (!PyObject_TypeCheck(tensor, &PyMNNTensorType)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_resizeTensor: First argument is not a MNN.Tensor instance");
        return NULL;
    }
    
    size_t shapeSize = PyTuple_Size(shape);
    
    std::vector<int> vShape;
    for (size_t i=0; i<shapeSize; i++) {
        int shapeItem = (int)PyLong_AsLong(PyTuple_GetItem(shape, i));
        vShape.push_back(shapeItem);
    }
    
    self->interpreter->resizeTensor(tensor->tensor, vShape);
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
    
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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

    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
        for (int i=0; i<size_tensors; i++) {
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
        for (int i=0; i<size_tensors; i++) {
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

    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
        for (int i=0; i<size_tensors; i++) {
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
        for (int i=0; i<size_tensors; i++) {
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
    
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
    
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionInput: First argument is not a MNN.Session instance");
        return NULL;
    }
    
    Tensor *t = self->interpreter->getSessionInput(session->session, name);
    if (!t) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNInterpreter_getSessionInput: Get output failed");
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
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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

static PyObject* PyMNNInterpreter_getSessionInputAll(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    if (!PyArg_ParseTuple(args, "O", &session)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
    
    self->modelPath = new std::string(path);
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
        delete net;
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNInterpreter_updateSessionToModel(PyMNNInterpreter *self, PyObject *args) {
    PyMNNSession* session = NULL;
    char* name = NULL;
    if (!PyArg_ParseTuple(args, "O|s", &session, &name)) {
        return NULL;
    }
    
    if (!PyObject_TypeCheck(session, &PyMNNSessionType)) {
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
    for (size_t i=0; i<shapeSize; i++) {
        int shapeItem = (int)PyLong_AsLong(PyTuple_GetItem(shape, i));
        vShape.push_back(shapeItem);
        dataSize *= shapeItem;
    }
    bool isNumpy = false;
    void *pData = NULL;
    if(PyTuple_Check(data)){
        if(dataSize != PyTuple_Size(data)){
            PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_init: Tensor Dim not match");
            return -1;
        }
    }
    else
    {
        PyObject* ndarray = importName("numpy", "ndarray");
        if(!ndarray || !PyObject_IsInstance(data, ndarray)){
            PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_init: data is not tuple/np.ndarray");
            return -1;
        }
        isNumpy = true;
        PyObject* sizeSrc = PyObject_GetAttrString(data, "size");
        if(dataSize != PyLong_AsLong(sizeSrc)){
            PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_init: Tensor Dim not match");
            return -1;
        }
        PyObject* reshape_func = PyObject_GetAttrString(data, "reshape");
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyLong_FromLong(dataSize));
        PyObject* reshaped_array = PyObject_Call(reshape_func, args, NULL);
        PyObject* reshaped_tuple = PySequence_Tuple(reshaped_array);
        data = reshaped_tuple;
        Py_XDECREF(reshaped_array);
        Py_XDECREF(args);
        Py_XDECREF(reshape_func);
        Py_XDECREF(sizeSrc);
    }
    halide_type_t htt;
    if (dataType == PyMNNHalideTypeInt) {
        htt = halide_type_of<int32_t>();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(int));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                ((int *)pData)[i] = (int)PyLong_AsLong(PyTuple_GetItem(data, i));
            }
        }
    } else if (dataType == PyMNNHalideTypeFloat) {
        htt = halide_type_of<float>();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(float));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                ((float *)pData)[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(data, i));
            }}
    } else if (dataType == PyMNNHalideTypeDouble) {
        htt = halide_type_of<double>();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(double));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                ((double *)pData)[i] = PyFloat_AsDouble(PyTuple_GetItem(data, i));
            }}
    } else if (dataType == PyMNNHalideTypeUint8) {
        htt = halide_type_of<uint8_t>();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(uint8_t));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                ((uint8_t *)pData)[i] = (uint8_t)PyLong_AsLong(PyTuple_GetItem(data, i));
            }}
    } else if (dataType == PyMNNHalideTypeInt64) {
        htt = halide_type_of<int64_t>();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(int64_t));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                ((int64_t *)pData)[i] = (int64_t)PyLong_AsLong(PyTuple_GetItem(data, i));
            }}
    } else if (dataType == PyMNNHalideTypeString) {
        htt = *httString();
        if (dataSize > 0) {
            pData = malloc(dataSize * sizeof(void *));
            if(NULL == pData){
                PyErr_SetString(PyExc_Exception,"PyMNNTensor_init: malloc failed");
                return -1;
            }
            for (int i=0; i<dataSize; i++) {
                char *item = (char *)object2String(PyTuple_GetItem(data, i)).c_str();
                ((char **)pData)[i] = item;
            }}
    } else {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNTensor_create: unsupported data type");
        return -1;
    }

    
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
    //decrease the ref count of data only when data is a numpy in fact  
    if(isNumpy){
        Py_XDECREF(data);
    }
    return 0;
}

static PyObject* PyMNNTensor_printTensorData(PyMNNTensor *self, PyObject *args) {
    if (self->tensor) {
        self->tensor->print();
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
        if (t == *httInt()) {
            type = PyMNNHalideTypeInt;
        } else if (t == *httUint8()) {
            type =  PyMNNHalideTypeUint8;
        } else if (t == *httInt64()) {
            type = PyMNNHalideTypeInt64;
        } else if (t == *httFloat()) {
            type = PyMNNHalideTypeFloat;
        } else if (t == *httDouble()) {
            type = PyMNNHalideTypeDouble;
        } else if (t == *httString()) {
            type = PyMNNHalideTypeString;
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
            for (int i=0; i<size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
        } else if (t == *httUint8()) {
            auto data = self->tensor->host<uint8_t>();
            for (int i=0; i<size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
        } else if (t == *httInt64()) {
            auto data = self->tensor->host<int64_t>();
            for (int i=0; i<size; i++) {
                PyTuple_SetItem(outputData, i, PyLong_FromLong(data[i]));
            }
        } else if (t == *httFloat()) {
            auto data = self->tensor->host<float>();
            for (int i=0; i<size; i++) {
                PyTuple_SetItem(outputData, i, PyFloat_FromDouble(data[i]));
            }
        } else if (t == *httDouble()) {
            auto data = self->tensor->host<double>();
            for (int i=0; i<size; i++) {
                PyTuple_SetItem(outputData, i, PyFloat_FromDouble(data[i]));
            }
        } else if (t == *httString()) {
            auto data = self->tensor->host<char *>();
            for (int i=0; i<size; i++) {
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
        for (int i=0; i<self->tensor->shape().size(); i++) {
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
        && PyObject_TypeCheck(destinationTensor, &PyMNNTensorType)) {
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
            for (int i=0; i<4; i++) {
                c.mean[0] = (float)PyFloat_AsDouble(PyTuple_GetItem(mean, i));
            }
        }

        PyObject *normal = PyDict_GetItemString(config, "normal");
        if (normal) {
            if (!PyTuple_Check(normal) || PyTuple_Size(normal) != 4) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNCVImageProcess_init: normal must be a tuple with 4 elements");
                return -1;
            }
            for (int i=0; i<4; i++) {
                c.normal[0] = (float)PyFloat_AsDouble(PyTuple_GetItem(normal, i));
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
    
    if (!PyObject_TypeCheck(matrix, &PyMNNCVMatrixType)) {
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
    
    if (!PyCapsule_CheckExact(source)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_convert: argument 0 is not a capsule");
        return NULL;
    }
    
    if (!PyObject_TypeCheck(dest, &PyMNNTensorType)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVImageProcess_convert: argument 4 is not a MNNTensor");
        return NULL;
    }
    
    ErrorCode ret = self->imageProcess->convert((const uint8_t *)PyCapsule_GetPointer(source, NULL)
                                                , iw, ih, stride
                                                , ((PyMNNTensor *)dest)->tensor);
    return PyLong_FromLong(ret);
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

static PyObject* PyMNNCVMatrix_postScale(PyMNNCVMatrix *self, PyObject *args) {
    float sx, sy, px, py;
    size_t argsCount = PyTuple_Size(args);
    if (argsCount == 2) {
        if (!PyArg_ParseTuple(args, "ff", &sx, &sy)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_postScale: PyArg_ParseTuple failed");
            return NULL;
        }
    } else if (argsCount == 4) {
        if (!PyArg_ParseTuple(args, "ffff", &sx, &sy, &px, &py)) {
            PyErr_SetString(PyExc_Exception,
                            "PyMNNCVMatrix_postScale: PyArg_ParseTuple failed");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNCVMatrix_postScale: argument count error (should be 2 or 4)");
        return NULL;
    }

    if (argsCount == 2) {
        self->matrix->postScale(sx, sy);
    } else if (argsCount == 4) {
        self->matrix->postScale(sx, sy, px, py);
    }
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

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "MNN",     /* m_name */
        "MNNEngine",  /* m_doc */
        -1,                  /* m_size */
        module_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#endif
MOD_INIT(MNN)
{
    #if PY_MAJOR_VERSION >= 3
        if (PyType_Ready(&PyMNNInterpreterType) < 0) {
            printf("initMNN: PyType_Ready PyMNNInterpreterType failed");
            return NULL;
        }
    
        if (PyType_Ready(&PyMNNSessionType) < 0) {
             printf("initMNN: PyType_Ready PyMNNSessionType failed");
             return NULL;
        }
    
        if (PyType_Ready(&PyMNNTensorType) < 0) {
            printf("initMNN: PyType_Ready PyMNNTensorType failed");
            return NULL;
        }
    
        if (PyType_Ready(&PyMNNCVImageProcessType) < 0) {
            printf("initMNN: PyType_Ready PyMNNCVImageProcessType failed");
            return NULL;
        }
    
        if (PyType_Ready(&PyMNNCVMatrixType) < 0) {
            printf("initMNN: PyType_Ready PyMNNCVMatrixType failed");
            return NULL;
        }

        if (PyType_Ready(&PyMNNOpInfoType) < 0) {
            printf("initMNN: PyType_Ready PyMNNOpInfoType failed");
            return NULL;
        }
    
        PyObject *m = PyModule_Create(&moduledef);

         // module import failed!
        if (!m) {
            printf("initMNN: import MNN failed");
            return NULL;
        }
    #else
        if (PyType_Ready(&PyMNNInterpreterType) < 0) {
            printf("initMNN: PyType_Ready PyMNNInterpreterType failed");
            return;
        }
    
        if (PyType_Ready(&PyMNNSessionType) < 0) {
             printf("initMNN: PyType_Ready PyMNNSessionType failed");
             return;
        }
    
        if (PyType_Ready(&PyMNNTensorType) < 0) {
            printf("initMNN: PyType_Ready PyMNNTensorType failed");
            return;
        }
    
        if (PyType_Ready(&PyMNNCVImageProcessType) < 0) {
            printf("initMNN: PyType_Ready PyMNNCVImageProcessType failed");
            return;
        }
    
        if (PyType_Ready(&PyMNNCVMatrixType) < 0) {
            printf("initMNN: PyType_Ready PyMNNCVMatrixType failed");
            return;
        }

        if (PyType_Ready(&PyMNNOpInfoType) < 0) {
            printf("initMNN: PyType_Ready PyMNNOpInfoType failed");
            return;
        }
    
        PyObject *m = Py_InitModule3("MNN", module_methods, "MNN Module");

         // module import failed!
        if (!m) {
            printf("initMNN: import MNN failed");
            return;
        }
    #endif
    
    
    PyModule_AddObject(m, "Interpreter", (PyObject*)&PyMNNInterpreterType);
    PyModule_AddObject(m, "Session", (PyObject*)&PyMNNSessionType);
    PyModule_AddObject(m, "Tensor", (PyObject*)&PyMNNTensorType);
    PyModule_AddObject(m, "CVImageProcess", (PyObject*)&PyMNNCVImageProcessType);
    PyModule_AddObject(m, "CVMatrix", (PyObject*)&PyMNNCVMatrixType);
    PyModule_AddObject(m, "OpInfo", (PyObject*)&PyMNNOpInfoType);
    
    // Tensor::DimensionType
    PyObject *DimensionType_Tensorflow = PyLong_FromLong(Tensor::TENSORFLOW);
    PyObject *DimensionType_Caffe = PyLong_FromLong(Tensor::CAFFE);
    PyObject *DimensionType_Caffe_C4 = PyLong_FromLong(Tensor::CAFFE_C4);
    PyModule_AddObject(m, "Tensor_DimensionType_Tensorflow", DimensionType_Tensorflow);
    PyModule_AddObject(m, "Tensor_DimensionType_Caffe", DimensionType_Caffe);
    PyModule_AddObject(m, "Tensor_DimensionType_Caffe_C4", DimensionType_Caffe_C4);

    // halide_type
    PyMNNHalideTypeInt = PyCapsule_New(httInt(), NULL, NULL);
    PyMNNHalideTypeInt64 = PyCapsule_New(httInt64(), NULL, NULL);
    PyMNNHalideTypeFloat = PyCapsule_New(httFloat(), NULL, NULL);
    PyMNNHalideTypeDouble = PyCapsule_New(httDouble(), NULL, NULL);
    PyMNNHalideTypeUint8 = PyCapsule_New(httUint8(), NULL, NULL);
    PyMNNHalideTypeString = PyCapsule_New(httString(), NULL, NULL);

    PyModule_AddObject(m, "Halide_Type_Int", PyMNNHalideTypeInt);
    PyModule_AddObject(m, "Halide_Type_Int64", PyMNNHalideTypeInt64);
    PyModule_AddObject(m, "Halide_Type_Float", PyMNNHalideTypeFloat);
    PyModule_AddObject(m, "Halide_Type_Double", PyMNNHalideTypeDouble);
    PyModule_AddObject(m, "Halide_Type_Uint8", PyMNNHalideTypeUint8);
    PyModule_AddObject(m, "Halide_Type_String", PyMNNHalideTypeString);
    
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
    #if PY_MAJOR_VERSION >= 3
        return m;
    #else
        return;
    #endif
}


