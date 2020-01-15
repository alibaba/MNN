/*
    MNN python module
*/
#include <Python.h>
#include "structmember.h"

#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include "writeFb.hpp"
#include "config.hpp"
#include "calibration.hpp"
#include "logkit.h"
using namespace MNN;
using namespace std;
/// module init

static PyObject* PyTool_Converter(PyObject *self, PyObject *args) {

    const char* mnnModel = NULL;
    const char* modelFile = NULL;
    PyObject* frameworkType = NULL;
    PyObject* fp16 = NULL;
    const char* prototxtFile = NULL;
    if (!PyArg_ParseTuple(args, "ssOO|s", &mnnModel, &modelFile, &frameworkType, &fp16, &prototxtFile)) {
        return NULL;
    }
    struct modelConfig modelPath;
    modelPath.MNNModel = std::string(mnnModel);
    modelPath.modelFile = std::string(modelFile);
    modelPath.model = static_cast<modelConfig::MODEL_SOURCE>(PyLong_AsLong(frameworkType));
    modelPath.bizCode = std::string("");
    modelPath.benchmarkModel = false;
    modelPath.saveHalfFloat = static_cast<bool>(PyLong_AsLong(fp16));
    modelPath.forTraining = false;
    if(prototxtFile){
	    modelPath.prototxtFile = std::string(prototxtFile);
    }

    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    if (modelPath.model == modelConfig::CAFFE) {
        caffe2MNNNet(modelPath.prototxtFile, modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TENSORFLOW) {
        tensorflow2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::MNN) {
        addBizCode(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::ONNX) {
        onnx2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else if (modelPath.model == modelConfig::TFLITE) {
        tflite2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
    } else {
        std::cout << "Not Support Model Type" << std::endl;
    }

    if (modelPath.model != modelConfig::MNN) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining);
        writeFb(newNet, modelPath.MNNModel, modelPath.benchmarkModel,modelPath.saveHalfFloat);
    } else {
        writeFb(netT, modelPath.MNNModel, modelPath.benchmarkModel,modelPath.saveHalfFloat);
    }
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
        new Calibration(netT.get(), modelForInference.get(), size, preTreatConfig));
    calibration->runQuantizeModel();
    DLOG(INFO) << "Quantize model done!";

    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);

    {
        std::ofstream output(dstFile);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }

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
        "Tools",     /* m_name */
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
MOD_INIT(Tools)
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *m = PyModule_Create(&moduledef);
        // module import failed!
        if (!m) {
            printf("import Tools failed");
            return NULL;
        }
        return m;
    #else
        PyObject *m = Py_InitModule3("Tools", module_methods, "MNNTools Module");
        // module import failed!
        if (!m) {
            printf("import Tools failed");
            return;
        }
        return;
    #endif
}
