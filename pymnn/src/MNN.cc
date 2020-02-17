/*
    MNN python module 
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
#include "Session.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "Utils.hpp"
#include "MNN_generated.h"
#else
#include "Interpreter.hpp"
#include "ImageProcess.hpp"
#endif
#include "util.h"
#include "NN.hpp"
#include "OpGrad.hpp"
#include "SGD.hpp"
#include "ADAM.hpp"
#include "MnistDataset.hpp"
#include "DataLoader.hpp"
#include "Loss.hpp"

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Express;
using namespace std;
namespace py = pybind11;
int add(int i, int j) {
    return i + j;
}

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
    Tensor *tensor;
    int owner;
} PyMNNTensor;

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
        if (PyType_Ready(&PyMNNTensorType) < 0) {
            printf("initMNN: PyType_Ready PyMNNTensorType failed");
            return NULL;
        }
    
        PyObject *m = PyModule_Create(&moduledef);

         // module import failed!
        if (!m) {
            printf("initMNN: import MNN failed");
            return NULL;
        }
    #else
        if (PyType_Ready(&PyMNNTensorType) < 0) {
            printf("initMNN: PyType_Ready PyMNNTensorType failed");
            return;
        }
    
        PyObject *m = Py_InitModule3("MNN", module_methods, "MNN Module");

         // module import failed!
        if (!m) {
            printf("initMNN: import MNN failed");
            return;
        }
    #endif
    
    
    //PyModule_AddObject(m, "Interpreter", (PyObject*)&PyMNNInterpreterType);
    //PyModule_AddObject(m, "Session", (PyObject*)&PyMNNSessionType);
    PyModule_AddObject(m, "Tensor", (PyObject*)&PyMNNTensorType);
    //PyModule_AddObject(m, "CVImageProcess", (PyObject*)&PyMNNCVImageProcessType);
    //PyModule_AddObject(m, "CVMatrix", (PyObject*)&PyMNNCVMatrixType);
    //PyModule_AddObject(m, "OpInfo", (PyObject*)&PyMNNOpInfoType);
    
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
    //interpreterMap();
    //sessionCacheMap();
    ScheduleConfig config;
    auto py_module = py::reinterpret_borrow<py::module>(m);
    py::enum_<CV::Filter>(py_module, "Filter")
        .value("NEAREST", CV::NEAREST)
        .value("BILINEAR", CV::BILINEAR)
        .value("BICUBIC", CV::BICUBIC)
        .export_values();
    py::enum_<CV::Wrap>(py_module, "Wrap")
        .value("CLAMP_TO_EDGE", CV::CLAMP_TO_EDGE)
        .value("ZERO", CV::ZERO)
        .value("REPEAT", CV::REPEAT)
        .export_values();
    py::enum_<CV::ImageFormat>(py_module, "ImageFormat")
        .value("RGBA", CV::RGBA)
        .value("RGB", CV::RGB)
        .value("BGR", CV::BGR)
        .value("GRAY", CV::GRAY)
        .value("BGRA", CV::BGRA)
        .value("YUV_NV21", CV::YUV_NV21)
        .export_values();
    py::class_<CV::Matrix>(py_module, "CVMatrix")
        .def(py::init<>())
        .def("postScale", (void (CV::Matrix::*)(float, float)) &CV::Matrix::postScale)
        .def("postScale", (void (CV::Matrix::*)(float, float, float, float)) &CV::Matrix::postScale);
    py::class_<CV::ImageProcess>(py_module, "CVImageProcess")
        //.def(py::init((CV::ImageProcess* (CV::ImageProcess::*)(const config&, float)) &CV::ImageProcess::create))
        .def("setMatrix", &CV::ImageProcess::setMatrix)
        .def("convert",
	    [](CV::ImageProcess *self,const uint8_t* source, int iw, int ih, int stride, py::object destOrigin) {
                PyMNNTensor *tensor = (PyMNNTensor*)destOrigin.ptr();
	        self->convert(source, iw, ih, stride, tensor->tensor); 
            });
    py::class_<ScheduleConfig>(py_module, "ScheduleConfig")
        .def(py::init([](int numThread, std::vector<std::string> saveTensors, std::vector<std::string> inputPaths, std::vector<std::string> outputPaths){
		auto config = new ScheduleConfig();
		config->numThread = numThread;
            	config->saveTensors = saveTensors;
	        config->path.inputs = inputPaths;
            	config->path.outputs = outputPaths;
		return config;}
        	))
        .def_readwrite("numThread",&ScheduleConfig::numThread)
        .def_readwrite("saveTensors",&ScheduleConfig::saveTensors);
        //.def_readwrite("inputPaths",&ScheduleConfig::path.inputs);
        //.def_readwrite("outputPaths",&ScheduleConfig::Path::outputs);
    py::class_<Session>(py_module, "Session");
    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("NO_ERROR",NO_ERROR)
        .value("OUT_OF_MEMORY",OUT_OF_MEMORY)
        .value("NOT_SUPPORT",NOT_SUPPORT)
        .value("COMPUTE_SIZE_ERROR", COMPUTE_SIZE_ERROR)
	.value("NO_EXECUTION", NO_EXECUTION)
        .value("INPUT_DATA_ERROR", INPUT_DATA_ERROR)
        .value("CALL_BACK_STOP", CALL_BACK_STOP)
        .value("TENSOR_NOT_SUPPORT", TENSOR_NOT_SUPPORT)
        .value("TENSOR_NEED_DIVIDE", TENSOR_NEED_DIVIDE)
        .export_values();
    //py::class_<Variable> (m, "Variable")
    INTS default_shape = {};
    auto expr_module = py_module.def_submodule("expr");
    py::enum_<VARP::InputType> (expr_module, "tensor_type")
        .value("PlaceHolder", VARP::INPUT)
        .value("Trainable", VARP::TRAINABLE)
        .value("Const", VARP::CONST)
        .export_values();
    py::enum_<Dimensionformat> (expr_module, "data_format")
        .value("NHWC", NHWC)
        .value("NC4HW4", NC4HW4)
        .value("NCHW", NCHW)
        .export_values();
    py::enum_<DataType> (expr_module, "dtype")
        .value("float", DataType_DT_FLOAT)
        .value("double", DataType_DT_DOUBLE)
        .value("int", DataType_DT_INT32)
        .value("int64", DataType_DT_INT64)
        .value("uint8", DataType_DT_UINT8)
        .export_values();
    py::enum_<PaddingMode> (expr_module, "padding_mode")
        .value("caffe", CAFFE)
        .value("valid", VALID)
        .value("same", SAME)
        .export_values();
    py::enum_<MNN::Express::PadValueMode> (expr_module, "padvalue_mode")
        .value("constant", CONSTANT)
        .value("reflect", REFLECT)
        .value("symmetric", SYMMETRIC)
        .export_values();
    py::enum_<PoolingMode> (expr_module, "polling_mode")
        .value("maxpool", MAXPOOL)
        .value("avepool", AVEPOOL)
        .export_values();
    py::enum_<InterpolationMethod> (expr_module, "interp_method")
        .value("bilinear", BILINEAR)
        .value("nearest", NEAREST)
        .export_values();
    py::class_<VARP>(expr_module, "VARP")
        .def_property_readonly("shape",
	    [](VARP *self){
            auto info = (*self)->getInfo();
            if(nullptr == info) {
                throw std::exception();
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
                    throw std::exception();
                return info->order;
            })
        .def_property_readonly("dtype",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                   throw std::exception();
                return Utils::convertDataType(info->type);
            })
         .def_property_readonly("length",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info) {
                   throw std::exception();
                }
                return info->size;
            })
        .def_property_readonly("name",
            [](VARP *self){
                auto name = (*self)->name();
                return name;
            })
        .def_property_readonly("op_type",
            [](VARP *self){
                auto op = (*self)->expr().first->get();
                if (nullptr == op) {
                    switch ((*self)->expr().first->inputType()) {
                        case VARP::INPUT:
                            return std::string("Input");
                        case VARP::CONST:
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
        .def_property_readonly("inputs",
            [] (VARP* self) {
                return (*self)->expr().first->inputs();
            })
        .def("fix",
            [] (VARP* self, VARP::InputType type) {
                (*self).fix(type);
            })
        .def("close",
            [] (VARP* self) {
                (*self)->input(VARP(nullptr));
            })
        .def("input",
            [] (VARP* self, VARP source) {
                bool res = (*self)->input(source);
                if (!res) {
                    MNN_ERROR("Input Error\n");
                    throw std::exception();
                }
            })
        .def("setInputs",
            [] (VARP* self, std::vector<VARP> source) {
                if (source.empty()) {
                    MNN_ERROR("Empty source\n");
                    throw std::exception();
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
        .def("setName",
            [] (VARP* self, std::string name) {
                (*self)->setName(name);
            })
	    .def("read",
            [](VARP *self){
                auto info = (*self)->getInfo();
                if(nullptr == info)
                   throw std::exception();
                auto dtype = Utils::convertDataType(info->type);
                auto shape = info->dim;
                int64_t total_length = info->size;
                auto readptr = [self](DataType dtype, int64_t total_length) {
                    auto dataPtr = (*self)->readMap<void>();
                    if (nullptr == dataPtr) {
                        throw std::exception();
                    }
                    if(DataType_DT_FLOAT == dtype) {
                        auto data = (float*)dataPtr;
                        auto obj = PyTuple_New(total_length);
                        for(int64_t i=0; i< total_length; i++) {
			                PyTuple_SetItem(obj, i, PyFloat_FromDouble(data[i]));
                        }
                        return obj;
                    }
                    else if(DataType_DT_INT32 == dtype) {
                        auto data = (int32_t*)dataPtr;
                        auto obj = PyTuple_New(total_length);
                        for(int64_t i=0; i< total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                        return obj;
                    }
                    else if(DataType_DT_UINT8 == dtype) {
                        auto data = (uint8_t*)dataPtr;
                        auto obj = PyTuple_New(total_length);
                        for(int64_t i=0; i< total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                        return obj;
                    } else if(DataType_DT_INT8 == dtype) {
                        auto data = (int8_t*)dataPtr;
                        auto obj = PyTuple_New(total_length);
                        for(int64_t i=0; i< total_length; i++) {
                            PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
                        }
                        return obj;
                    } else {
                        MNN_ERROR("Don't support data type\n");
                        throw std::exception();
                    }
                };
                try{
                    auto data = readptr(dtype, total_length);
                    (*self)->unMap();
                    return py::reinterpret_borrow<py::object>(data);
                }
                catch(const std::exception& e)
                {
                    throw std::exception();
                }
            })
        .def("write",
            [](VARP *self, py::object data) {
                auto info = (*self)->getInfo();
                if(nullptr == info) {
                    throw std::exception();
                }
                auto dtype = Utils::convertDataType(info->type);
                auto shape = info->dim;
                int64_t total_length = info->size;
                PyObject *obj = data.ptr();
                auto write = [self](PyObject *obj, DataType dtype, int64_t total_length) {
                    INTS shapeData = getshape(obj);
                    int64_t totalLengthData = 1;
                    INTS stride;
                    for(int i=0; i< shapeData.size(); i++) {
                        totalLengthData *= shapeData[i];
                        if(i < shapeData.size()-1) {
                            stride.push_back(shapeData[i+1]);
                        }
                        else
                        {
                            stride.push_back(1);
                        }
                    }
                    if(totalLengthData != total_length) {
			            throw std::exception();
                    }
                    if(DataType_DT_FLOAT == dtype) {
                        auto data = (*self)->writeMap<float>();
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(float));
                    }
                    else if(DataType_DT_INT32 == dtype) {
                        auto data = (*self)->writeMap<int>();
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int));
                    }
                    else if(DataType_DT_UINT8 == dtype) {
                        auto data = (*self)->writeMap<uint8_t>();
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(uint8_t));
                    }
                    else if(DataType_DT_INT8 == dtype) {
                        auto data = (*self)->writeMap<uint8_t>();
                        recursive_store((char*)data, shapeData, stride, 0, obj, dtype, sizeof(int8_t));
                    }
                };
                try{
                    write(obj, dtype, total_length);
                    (*self)->unMap();
                }
                catch(const std::exception& e)
                {
                    throw std::exception();
                }
            });
    // Load And Save
    expr_module.def("load",
    		[](std::string fileName) {
                auto variable = Variable::load(fileName.c_str());
                if (variable.empty()) {
                    throw std::exception();
                }
			    return variable;
    });
    expr_module.def("save",
    		[](const std::vector<VARP>& vars, std::string fileName) {
                Variable::save(vars, fileName.c_str());
    });
    expr_module.def("loadMap",
    		[](std::string fileName) {
                auto variable = Variable::loadMap(fileName.c_str());
                if (variable.empty()) {
                    throw std::exception();
                }
			    return variable;
    });

    //Begin of Math OPS
    //Unary OPS
    expr_module.def("Sign", &_Sign);  
    expr_module.def("Abs", &_Abs);
    expr_module.def("Negative", &_Negative);
    expr_module.def("Floor", &_Floor);
    expr_module.def("Ceil", &_Ceil);
    expr_module.def("Square", &_Square);
    expr_module.def("Sqrt", &_Sqrt);
    expr_module.def("Rsqrt", &_Rsqrt);
    expr_module.def("Exp", &_Exp);
    expr_module.def("Log", &_Log);
    expr_module.def("Sin", &_Sin);
    expr_module.def("Cos", &_Cos);
    expr_module.def("Tan", &_Tan);
    expr_module.def("Asin", &_Asin);
    expr_module.def("Acos", &_Acos);
    expr_module.def("Atan", &_Atan);
    expr_module.def("Reciprocal", &_Reciprocal);
    expr_module.def("Log1p", &_Log1p);
    expr_module.def("Tanh", &_Tanh);
    expr_module.def("Sigmoid", &_Sigmoid);
    //Binary OPS
    expr_module.def("Add", &_Add);
    expr_module.def("Subtract", &_Subtract);
    expr_module.def("Multiply", &_Multiply);
    expr_module.def("Divide", &_Divide);
    expr_module.def("Pow", &_Pow);
    expr_module.def("Minimum", &_Minimum);
    expr_module.def("Maximum", &_Maximum);
    expr_module.def("BiasAdd", &_BiasAdd);
    expr_module.def("Greater", &_Greater);
    expr_module.def("GreaterEqual", &_GreaterEqual);
    expr_module.def("Less", &_Less);
    expr_module.def("FloorDiv", &_FloorDiv);
    expr_module.def("SquaredDifference", &_SquaredDifference);
    expr_module.def("Equal", &_Equal);
    expr_module.def("LessEqual", &_LessEqual);
    expr_module.def("FloorMod", &_FloorMod);
    //Reduce OPS
    expr_module.def("ReduceSum", &_ReduceSum);
    expr_module.def("ReduceMean", &_ReduceMean);
    expr_module.def("ReduceMax", &_ReduceMax);
    expr_module.def("ReduceMin", &_ReduceMin);
    expr_module.def("ReduceProd", &_ReduceProd);
    expr_module.def("ReduceAny", &_ReduceAny);
    expr_module.def("ReduceAll", &_ReduceAll);
    //Eltwise OPS
    expr_module.def("EltwiseProd", &_Prod);
    expr_module.def("EltwiseSum", &_Sum);
    expr_module.def("EltwiseMax", &_Max);
    expr_module.def("EltwiseSub", &_Sub);
    //Other OPS
    expr_module.def("Cast", 
		    [](VARP x, DataType dtype) {
			return _Cast(x, Utils::revertDataType(dtype));
                    });
    expr_module.def("MatMul", &_MatMul, py::arg("a"), py::arg("b"), py::arg("tranposeA")=false, py::arg("tranposeB")=false);
    expr_module.def("Normalize", &_Normalize);
    expr_module.def("ArgMax", 
		   [](VARP input, int axis) {
			return _ArgMax(input, axis);
                   }, py::arg("input"), py::arg("axis")=0);
    expr_module.def("BatchMatMul",
		   [](VARP x, VARP y, bool adj_x, bool adj_y) {
                   }, py::arg("x"), py::arg("y"), py::arg("adj_x")=false, py::arg("adj_y")=false);
    expr_module.def("UnravelIndex", &_UnravelIndex);
    expr_module.def("ScatterNd", &_ScatterNd);
    expr_module.def("OneHot",
		   [](VARP indices, VARP depth, VARP onValue, VARP offValue, int axis) {
			return _OneHot(indices, depth, onValue, offValue, axis);
                   },py::arg("indices"), py::arg("depth"), py::arg("onValue"), py::arg("offValue"), py::arg("axis")=-1);
    expr_module.def("BroadcastTo", &_BroadcastTo);
    //End of Math OPS
 
    //Begin of NN OPS
    expr_module.def("Input",
                  [](INTS shape,Dimensionformat data_format, DataType dtype)->VARP{
    			return _Input(shape, data_format, Utils::revertDataType(dtype));
                  },
                  py::arg("shape")=default_shape,
                  py::arg("data_format")=NC4HW4,
                  py::arg("dtype")=DataType_DT_FLOAT);
    expr_module.def("Clone",
                   [](VARP source, bool deepCopy) {
			return _Clone(source, deepCopy);
                   }, py::arg("source"), py::arg("deepCopy")=false);
    INTS default_pads = {0, 0};
    INTS default_axis = {};
    expr_module.def("MaxPool",
                   [](VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
                        return _MaxPool(x, kernel, stride, pad, pads);
                   }, py::arg("x"), py::arg("kernel"), py::arg("stride"),
		   py::arg("pad")=VALID,
		   py::arg("pads")=default_pads);
    expr_module.def("AvePool",
                   [](VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
                        return _AvePool(x, kernel, stride, pad, pads);
                   }, py::arg("x"), py::arg("kernel"), py::arg("stride"),
                   py::arg("pad")=VALID,
                   py::arg("pads")=default_pads);
    expr_module.def("Reshape",
                   [](VARP x, INTS shape, Dimensionformat original_format) {
                        return _Reshape(x, shape, original_format);
                   }, py::arg("x"), py::arg("shape"), py::arg("original_format")=NHWC);
    expr_module.def("Reshape",
                   [](VARP x, VARP shape) {
                        return _Reshape(x, shape);
                   });
    expr_module.def("Scale", &_Scale);
    expr_module.def("Relu",
                   [](VARP x, float slope) {
                        return _Relu(x, slope);
                   }, py::arg("x"), py::arg("slope")=0.0f);
    expr_module.def("Relu6", &_Relu6);
    expr_module.def("PRelu", &_PRelu);
    expr_module.def("Softmax",
                   [](VARP logits, int axis) {
                        return _Softmax(logits, axis);
                   }, py::arg("logits"), py::arg("axis")=-1);
    expr_module.def("Softplus", &_Softplus);
    expr_module.def("Softsign", &_Softsign);
    expr_module.def("Slice", &_Slice);
    expr_module.def("StridedSlice", &_StridedSlice);
    expr_module.def("Concat", &_Concat);
    expr_module.def("Convert", &_Convert);
    expr_module.def("Transpose",
                   [](VARP x, INTS perm) {
                        return _Transpose(x, perm);
                   });
    expr_module.def("Transpose",
                   [](VARP x, VARP perm) {
                        return _Transpose(x, perm);
                   });
    expr_module.def("ChannelShuffle", &_ChannelShuffle);
    expr_module.def("ChangeInputFormat", &_ChangeInputFormat);
    expr_module.def("ReverseSequence", &_ReverseSequence);
    expr_module.def("Crop", &_Crop);
    expr_module.def("Resize", &_Resize);
    expr_module.def("Pad",
                   [](VARP x, VARP paddings, MNN::Express::PadValueMode mode) {
                        return _Pad(x, paddings, mode);
                   }, py::arg("x"), py::arg("paddings"), py::arg("mode")=CONSTANT);
    expr_module.def("ExpandDims",
                   [](VARP input, int axis) {
                        return _ExpandDims(input, axis);
                   });
    expr_module.def("ExpandDims",
                   [](VARP input, VARP axis) {
                        return _ExpandDims(input, axis);
                   });
    expr_module.def("Shape", &_Shape);
    expr_module.def("Stack",
                   [](VARPS values, int axis) {
                        return _Stack(values, axis);
 		   }, py::arg("values"), py::arg("axis")=0);
    expr_module.def("CropAndResize",
                   [](VARP image, VARP boxes, VARP box_ind, VARP crop_size, InterpolationMethod method, float extrapolation_value) {
                        return _CropAndResize(image, boxes, box_ind, crop_size, method, extrapolation_value);
                   }, py::arg("image"), py::arg("boxes"), py::arg("box_ind"), py::arg("crop_size"),
		   py::arg("method")=BILINEAR, py::arg("extrapolation_value")=0.0f);
    expr_module.def("Fill", &_Fill);
    expr_module.def("Tile", &_Tile);
    expr_module.def("Gather", &_Gather);   
     
    expr_module.def("GatherV2",
                   [](VARP params, VARP indices, VARP axis = nullptr) {
                        return _GatherV2(params, indices, axis);
                   }, py::arg("params"), py::arg("indices"), py::arg("axis")=nullptr);
    expr_module.def("Squeeze",
                   [](VARP input, INTS axis) {
                        return _Squeeze(input, axis);
                   }, py::arg("input"), py::arg("axis")=default_axis);
    expr_module.def("Unsqueeze",
                   [](VARP input, INTS axis) {
                        return _Unsqueeze(input, axis);
                   }, py::arg("input"), py::arg("axis")=default_axis);
    expr_module.def("BatchToSpaceND", &_BatchToSpaceND);
    expr_module.def("GatherND", &_GatherND);
    expr_module.def("Selu", &_Selu);
    expr_module.def("Size", &_Size);
    expr_module.def("Elu",
                   [](VARP features, float alpha) {
                        return _Elu(features, alpha);
                   }, py::arg("features"), py::arg("alpha")=1.0);
    expr_module.def("MatrixBandPart", &_MatrixBandPart);
    expr_module.def("Moments", &_Moments);
    expr_module.def("SetDiff1D", &_SetDiff1D);
    expr_module.def("SpaceToDepth", &_SpaceToDepth);
    expr_module.def("SpaceToBatchND", &_SpaceToBatchND);
    expr_module.def("ZerosLike", &_ZerosLike);
    expr_module.def("Unstack",
                   [](VARP value, int axis) {
                        return _Unstack(value, axis);
                   }, py::arg("value"), py::arg("axis")=0);
    expr_module.def("Rank", &_Rank);
    expr_module.def("Range", &_Range);
    expr_module.def("DepthToSpace", &_DepthToSpace);
    //End of NN OPS

    //Begin of Train
    auto train_module = py_module.def_submodule("train");

    py::class_<ParameterOptimizer>(train_module, "Optimizer")
        .def("step", &ParameterOptimizer::step)
        .def("append", &ParameterOptimizer::append)
    ;
    train_module.def("SGD", &ParameterOptimizer::createSGD);
    train_module.def("ADAM", &ParameterOptimizer::createADAM);

    py::class_<Module>(train_module, "CppModule")
        .def("__call__", &Module::forward)
        .def("forward", &Module::forward)
        .def("forwardArray", &Module::onForward)
        .def("setName", &Module::setName)
        .def("name", &Module::name)
        .def("train", &Module::setIsTraining)
        .def("parameters", &Module::parameters)
        .def("loadParameters", &Module::loadParameters)
    ;
    {
        auto data_module = train_module.def_submodule("data");
        py::class_<DataLoader>(data_module, "DataLoader")
            .def("iterNumber", &DataLoader::iterNumber)
            .def("size", &DataLoader::size)
            .def("reset", &DataLoader::reset)
            .def("next", &DataLoader::next)
        ;
        py::class_<DatasetPtr>(data_module, "Dataset")
            .def("createLoader", &DatasetPtr::createLoader)
        ;
        auto mnist_module = data_module.def_submodule("mnist");
        py::enum_<MnistDataset::Mode>(mnist_module, "Mode")
            .value("Train", MnistDataset::TRAIN)
            .value("Test", MnistDataset::TEST)
            .export_values();
        mnist_module.def("create", &MnistDataset::create);
    }
    {
        // Loss
        auto loss_module = train_module.def_submodule("loss");
        loss_module.def("CrossEntropy", _CrossEntropy);
        loss_module.def("KLDivergence", _KLDivergence);
        loss_module.def("MSE", _MSE);
        loss_module.def("MAE", _MAE);
        loss_module.def("Hinge", _Hinge);
    }
    {
        // CNN
        auto cnn_module = train_module.def_submodule("cnn");
        cnn_module.def("Conv",
                    [](int in_channel, int out_channel,
                        INTS kernel_size,
                        INTS stride,
                        INTS padding,
                        INTS dilation,
                        bool depthwise,
                        bool bias
                        ) {
                        NN::ConvOption option;
                        option.channel = {in_channel, out_channel};
                        option.kernelSize = kernel_size;
                        if (!stride.empty()) {
                            option.stride = stride;
                        }
                        if (!padding.empty()) {
                            option.pads = padding;
                        }
                        if (!dilation.empty()) {
                            option.dilate = dilation;
                        }
                        option.depthwise = depthwise;
                        return NN::Conv(std::move(option), bias);
                    },
                    py::arg("in_channel"),
                    py::arg("out_channel"),
                    py::arg("kernel_size"),
                    py::arg("stride") = std::vector<int>(),
                    py::arg("padding") = std::vector<int>(),
                    py::arg("dilation") = std::vector<int>(),
                    py::arg("depthwise") = false,
                    py::arg("bias") = true
                    );
        cnn_module.def("Linear",
                    [](int in_channel, int out_channel) {
                        return NN::Linear(in_channel, out_channel);
                    }
                    );

        cnn_module.def("BatchNorm", &NN::BatchNorm);
        cnn_module.def("Dropout", &NN::Dropout);
    }

    // End of Train

    py::class_<Interpreter>(m, "Interpreter")
        .def(py::init(&Interpreter::createFromFile))
        .def("createSession", &Interpreter::createSession, py::arg("config")=config, py::return_value_policy::reference)
        .def("resizeSession", &Interpreter::resizeSession)
        .def("runSession", &Interpreter::runSession)
        .def("updateSessionToModel", 
            [](Interpreter *self, Session* session, const char* name) {
                self->updateSessionToModel(session);
                try
                {
                    if(name){
                        auto modelBuffer = self->getModelBuffer();
                        ofstream output(name);
                        output.write((const char*)modelBuffer.first, modelBuffer.second);
                    }
                }
                catch(const std::exception& e)
                {
                    throw std::exception();
                }
            },
            "name:full file name to save the model, if not set, does not save to file", 
            py::arg("session"), py::arg("name")=(const char*)nullptr)
        .def("getSessionInput", 
            [](Interpreter *self, const Session* session, const char* name) -> py::object {
		Tensor *t = self->getSessionInput(session,name);
                if(nullptr == t) {
		    throw std::exception();
                }
                PyObject *f = importName("MNN", "Tensor");
                if(nullptr == f) {
		    throw std::exception();
                }
   		PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
                if(nullptr == tensor) {
                    throw std::exception();
                }
	        tensor->tensor = t;
    		return py::reinterpret_borrow<py::object>((PyObject*)tensor);
    	    }, 
            "name:tensor's name, if not set, gets the default", 
            py::arg("session"), py::arg("name")=(const char*)nullptr,py::return_value_policy::reference)
        .def("getSessionOutput", 
            [](Interpreter *self, const Session* session, const char* name) {
                Tensor *t = self->getSessionOutput(session,name);
                if(nullptr == t) {
                    throw std::exception();
                }
                PyObject *f = importName("MNN", "Tensor");
                if(nullptr == f) {
                    throw std::exception();
                }
                PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
                if(nullptr == tensor) {
                    throw std::exception();
                }
                tensor->tensor = t;
                return py::reinterpret_borrow<py::object>((PyObject*)tensor);
            },
            "name:tensor's name, if not set, gets the default", 
            py::arg("session"), py::arg("name")=(const char*)nullptr,py::return_value_policy::reference)   
        .def("getSessionInputAll", 
            [](Interpreter *self, const Session* session) -> py::dict {
                PyObject *f = importName("MNN", "Tensor");
                if(nullptr == f) {
                    throw std::exception();
                }
                auto inputs = self->getSessionInputAll(session);
                py::dict dict = py::dict();
                for (auto it=inputs.begin(); it!=inputs.end(); ++it) {
                    PyObject *tensor = PyObject_Call(f, PyTuple_New(0), NULL);
                    ((PyMNNTensor*)tensor)->tensor = it->second;
                    dict[it->first.c_str()] = tensor;    
                }
                return dict;
            })
        .def("getSessionOutputAll", 
            [](Interpreter *self, const Session* session) -> py::dict {
                PyObject *f = importName("MNN", "Tensor");
                if(nullptr == f) {
                    throw std::exception();
                }
                auto outputs = self->getSessionOutputAll(session);
                py::dict dict = py::dict();
                for (auto it=outputs.begin(); it!=outputs.end(); ++it) {
                    PyObject *tensor = PyObject_Call(f, PyTuple_New(0), NULL);
                    ((PyMNNTensor*)tensor)->tensor = it->second;
                    dict[it->first.c_str()] = tensor;    
                }
                return dict;
            })
        .def("runSessionWithCallBack",
            [](Interpreter *self, const Session* session, py::function begin_func, py::function end_func) {
                PyObject *beginCallBack = begin_func.ptr();
                PyObject *endCallBack = end_func.ptr();
                TensorCallBack begin = [beginCallBack](const std::vector<Tensor*>& tensors, const std::string& name) {
			        PyObject *f = importName("MNN", "Tensor");
                    if(nullptr == f) {
                        throw std::exception();
                    }
			        PyObject *args = PyTuple_New(2);
        		    size_t size_tensors = tensors.size();
        		    PyObject *weTensorData = PyTuple_New(size_tensors);
        		    for (int i=0; i<size_tensors; i++) {
            			// create a new tensor
            			PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
            			tensor->tensor = tensors[i];
            			PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
        		    }
        		    PyObject *weStringData = char2Object(name.c_str());
        		    PyTuple_SetItem(args, 0, weTensorData);
        		    PyTuple_SetItem(args, 1, weStringData);
                    bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(beginCallBack, args, NULL)));
        		    Py_XDECREF(args);
        		    return ret;
    		        };
                TensorCallBack end = [endCallBack](const std::vector<Tensor*>& tensors, const std::string& name) {
			        PyObject *f = importName("MNN", "Tensor");
                    if(nullptr == f) {
                        throw std::exception();
                    }                        
			        PyObject *args = PyTuple_New(2);
                        size_t size_tensors = tensors.size();
                        PyObject *weTensorData = PyTuple_New(size_tensors);
                        for (int i=0; i<size_tensors; i++) {
                                // create a new tensor
                                PyMNNTensor *tensor = (PyMNNTensor *)PyObject_Call(f, PyTuple_New(0), NULL);
                                tensor->tensor = tensors[i];
                                PyTuple_SetItem(weTensorData, i, (PyObject *)tensor);
                        }
                        PyObject *weStringData = char2Object(name.c_str());
                        PyTuple_SetItem(args, 0, weTensorData);
                        PyTuple_SetItem(args, 1, weStringData);
                        bool ret = static_cast<bool>(PyLong_AsLong(PyObject_Call(endCallBack, args, NULL)));
                        Py_XDECREF(args);//del all the C++ created python api parameters
                        return ret;
                    };
                return self->runSessionWithCallBack(session, begin, end);
            });
    #if PY_MAJOR_VERSION >= 3
        return m;
    #else
        return;
    #endif
}


