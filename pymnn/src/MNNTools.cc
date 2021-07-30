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

static PyObject* PyTool_Quantization(PyObject *self, PyObject *args) {
    const char* modelFile      = NULL;
    const char* preTreatConfig = NULL;
    const char* dstFile        = NULL;
    if (!PyArg_ParseTuple(args, "sss", &modelFile, &dstFile, &preTreatConfig)) {
        return NULL;
    }
    DLOG(INFO) << ">>> modelFile: " << modelFile;
    DLOG(INFO) << ">>> preTreatConfig: " << preTreatConfig;
    DLOG(INFO) << ">>> dstFile: " << dstFile;
    std::unique_ptr<MNN::NetT> netT;
    {
        std::ifstream input(modelFile);
        std::ostringstream outputOs;
        outputOs << input.rdbuf();
        netT = MNN::UnPackNet(outputOs.str().c_str());
    }

    // temp build net for inference
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto ocontent = builder.GetBufferPointer();

    // model buffer for creating mnn Interpreter
    std::unique_ptr<uint8_t> modelForInference(new uint8_t[size]);
    memcpy(modelForInference.get(), ocontent, size);

    std::unique_ptr<uint8_t> modelOriginal(new uint8_t[size]);
    memcpy(modelOriginal.get(), ocontent, size);

    netT.reset();
    netT = MNN::UnPackNet(modelOriginal.get());

    // quantize model's weight
    DLOG(INFO) << "Calibrate the feature and quantize model...";
    std::shared_ptr<Calibration> calibration(
        new Calibration(netT.get(), modelForInference.get(), size, preTreatConfig, std::string(modelFile), std::string(dstFile)));
    calibration->runQuantizeModel();
    calibration->dumpTensorScales(dstFile);
    DLOG(INFO) << "Quantize model done!";

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
