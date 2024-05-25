/*
    MNN python module
*/
#include <Python.h>
#include "structmember.h"
#include "util.h"
#include "MNN_generated.h"
#include "config.hpp"
#include "cli.hpp"
#include "calibration.hpp"
#include "logkit.h"
#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <vector>
using namespace MNN;
using namespace std;
/// module init

static PyObject* PyTool_Converter(PyObject *self, PyObject *argsTuple) {
    int tupleSize = PyTuple_GET_SIZE(argsTuple);
    if (tupleSize < 1) {
        MNN_ERROR("Invalid input for Converter\n");
        return nullptr;
    }
    PyObject* args = PyTuple_GET_ITEM(argsTuple, 0);
    int argSize = PyList_Size(args);
    std::vector<char*> argsCpp(argSize);
    std::vector<PyObject*> argsContant(argSize);
    for (int i=0; i<argSize; ++i) {
        argsContant[i] = PyList_GetItem(args, i);
        PyArg_Parse(argsContant[i], "s", argsCpp.data() + i);
    }
    modelConfig modelPath;
    auto res = MNN::Cli::initializeMNNConvertArgs(modelPath, argSize, argsCpp.data());
    if (!res) {
        Py_RETURN_TRUE;
    }
    MNN::Cli::convertModel(modelPath);
    Py_RETURN_TRUE;
}

static PyObject* PyTool_Quantization(PyObject *self, PyObject *argsTuple) {
    int tupleSize = PyTuple_GET_SIZE(argsTuple);
    if (tupleSize < 1) {
        MNN_ERROR("Invalid input for Converter\n");
        return nullptr;
    }
    PyObject* args = PyTuple_GET_ITEM(argsTuple, 0);
    int argSize = PyList_Size(args);
    std::vector<const char*> argsCpp(argSize);
    std::vector<PyObject*> argsContant(argSize);
    for (int i=0; i<argSize; ++i) {
        argsContant[i] = PyList_GetItem(args, i);
        PyArg_Parse(argsContant[i], "s", argsCpp.data() + i);
    }
    quant_main(argSize, argsCpp.data());
    Py_RETURN_TRUE;
}

static PyMethodDef module_methods[] = {
    { "mnnconvert", (PyCFunction)PyTool_Converter, METH_VARARGS, NULL },
    { "mnnquant", (PyCFunction)PyTool_Quantization, METH_VARARGS, NULL },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_tools",     /* m_name */
        "MNNTools",  /* m_doc */
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
MOD_INIT(_tools) {
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
    // module import failed!
    if (!m) {
        printf("import Tools failed");
        return NULL;
    }
    return m;
#else
    PyObject *m = Py_InitModule3("_tools", module_methods, "MNNTools Module");
    // module import failed!
    if (!m) {
        printf("import Tools failed");
        return;
    }
    return;
#endif
}
