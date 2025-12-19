#ifdef PYMNN_LLM_API
#include <sstream>
#include <iostream>
#ifdef BUILD_FOR_IOS
#include "MNN/llm/reranker.hpp"
#include "MNN/llm/llm.hpp"
#else
#include "llm/reranker.hpp"
#include "llm/llm.hpp"
#endif

typedef struct {
    PyObject_HEAD
    MNN::Transformer::RerankerBase* reranker;
} Reranker;

static PyObject* PyMNNReranker_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    Reranker* self = (Reranker *)type->tp_alloc(type, 0);
    if (self) {
        self->reranker = nullptr;
    }
    return (PyObject*)self;
}

static void PyMNNReranker_dealloc(Reranker *self) {
    if (nullptr != self->reranker) {
        delete self->reranker;
        self->reranker = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMNNReranker_str(PyObject *self) {
    Reranker* reranker = (Reranker*)self;
    if (!reranker) {
        Py_RETURN_NONE;
    }
    return toPyObj("reranker");
}

static PyObject* PyMNNReranker_setInstruct(Reranker *self, PyObject *args) {
    if (!self->reranker) {
        PyErr_SetString(PyExc_RuntimeError, "Reranker not initialized");
        Py_RETURN_NONE;
    }
    
    const char* instruct = NULL;
    if (!PyArg_ParseTuple(args, "s", &instruct)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Usage: set_instruct(instruct)");
        Py_RETURN_NONE;
    }
    
    self->reranker->setInstruct(std::string(instruct));
    Py_RETURN_NONE;
}

static PyObject* PyMNNReranker_load(Reranker *self, PyObject *args) {
    if (!self->reranker) {
        PyErr_SetString(PyExc_RuntimeError, "Reranker not initialized");
        Py_RETURN_NONE;
    }
    
    self->reranker->load();
    Py_RETURN_NONE;
}

static PyObject* PyMNNReranker_compute_scores(Reranker *self, PyObject *args) {
    if (!self->reranker) {
        PyErr_SetString(PyExc_RuntimeError, "Reranker not initialized");
        Py_RETURN_NONE;
    }
    
    const char* query = NULL;
    PyObject* documents_list = NULL;
    
    if (!PyArg_ParseTuple(args, "sO", &query, &documents_list)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Usage: compute_scores(query, documents)");
        Py_RETURN_NONE;
    }
    
    if (!PyList_Check(documents_list)) {
        PyErr_SetString(PyExc_ValueError, "documents must be a list of strings");
        Py_RETURN_NONE;
    }
    
    Py_ssize_t doc_count = PyList_Size(documents_list);
    std::vector<std::string> documents;
    documents.reserve(doc_count);
    
    for (Py_ssize_t i = 0; i < doc_count; i++) {
        PyObject* doc_obj = PyList_GetItem(documents_list, i);
        if (!isString(doc_obj)) {
            PyErr_SetString(PyExc_ValueError, "All documents must be strings");
            Py_RETURN_NONE;
        }
        documents.push_back(object2String(doc_obj));
    }
    auto scores = self->reranker->compute_scores(std::string(query), documents);
    return toPyObj<float, toPyObj>(scores);
}

static PyObject* PyMNNReranker_get_llm(Reranker *self, PyObject *args) {
    if (!self->reranker) {
        PyErr_SetString(PyExc_RuntimeError, "Reranker not initialized");
        Py_RETURN_NONE;
    }
    
    MNN::Transformer::Llm* llm_ptr = self->reranker->get_llm();
    if (!llm_ptr) {
        PyErr_SetString(PyExc_RuntimeError, "LLM not available");
        Py_RETURN_NONE;
    }
    
    // Create a borrowed LLM object - reuse existing LLM type but mark as borrowed
    LLM *llm_obj = (LLM *)PyObject_Call((PyObject*)PyType_FindTLSType(&PyMNNLLM), PyTuple_New(0), NULL);
    if (!llm_obj) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create LLM object");
        Py_RETURN_NONE;
    }
    
    // Borrow the LLM pointer - don't take ownership
    llm_obj->llm = llm_ptr;
    llm_obj->is_embedding = false;
    llm_obj->is_borrowed = true;  // Mark as borrowed to prevent double-free
    
    return (PyObject*)llm_obj;
}

static PyMethodDef PyMNNReranker_methods[] = {
    {"set_instruct", (PyCFunction)PyMNNReranker_setInstruct, METH_VARARGS, "Set instruction for the reranker."},
    {"load", (PyCFunction)PyMNNReranker_load, METH_VARARGS, "Load the reranker model."},
    {"compute_scores", (PyCFunction)PyMNNReranker_compute_scores, METH_VARARGS, "Compute scores for documents given a query."},
    {"get_llm", (PyCFunction)PyMNNReranker_get_llm, METH_VARARGS, "Get the underlying LLM instance for configuration."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNReranker = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Reranker",                                    /*tp_name*/
    sizeof(Reranker),                              /*tp_basicsize*/
    0,                                             /*tp_itemsize*/
    (destructor)PyMNNReranker_dealloc,             /*tp_dealloc*/
    0,                                             /*tp_print*/
    0,                                             /*tp_getattr*/
    0,                                             /*tp_setattr*/
    0,                                             /*tp_compare*/
    PyMNNReranker_str,                             /*tp_repr*/
    0,                                             /*tp_as_number*/
    0,                                             /*tp_as_sequence*/
    0,                                             /*tp_as_mapping*/
    0,                                             /*tp_hash */
    0,                                             /*tp_call*/
    PyMNNReranker_str,                             /*tp_str*/
    0,                                             /*tp_getattro*/
    0,                                             /*tp_setattro*/
    0,                                             /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,      /*tp_flags*/
    "Reranker is mnn-llm's `RerankerBase` python wrapper",  /* tp_doc */
    0,                                             /* tp_traverse */
    0,                                             /* tp_clear */
    0,                                             /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    0,                                             /* tp_iter */
    0,                                             /* tp_iternext */
    PyMNNReranker_methods,                         /* tp_methods */
    0,                                             /* tp_members */
    0,                                             /* tp_getset */
    0,                                             /* tp_base */
    0,                                             /* tp_dict */
    0,                                             /* tp_descr_get */
    0,                                             /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    0,                                             /* tp_init */
    0,                                             /* tp_alloc */
    PyMNNReranker_new,                             /* tp_new */
};

static PyObject* PyMNNReranker_create_reranker(PyObject *self, PyObject *args) {
    const char* model_type = NULL;
    const char* config_path = NULL;
    if (!PyArg_ParseTuple(args, "ss", &model_type, &config_path)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Usage: create_reranker(model_type, config_path)");
        return NULL;
    }
    
    Reranker *reranker = (Reranker *)PyObject_Call((PyObject*)PyType_FindTLSType(&PyMNNReranker), PyTuple_New(0), NULL);
    if (!reranker) {
        return NULL;
    }
    
    std::string model_type_str(model_type);
    if (model_type_str == "qwen3") {
        reranker->reranker = new MNN::Transformer::Qwen3Reranker(std::string(config_path));
    } else if (model_type_str == "gte") {
        reranker->reranker = new MNN::Transformer::GteReranker(std::string(config_path));
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported model type. Supported types: 'qwen3', 'gte'");
        Py_DECREF(reranker);
        return NULL;
    }
    
    return (PyObject*)reranker;
}

static PyMethodDef PyMNNReranker_static_methods[] = {
    {"create_reranker", PyMNNReranker_create_reranker, METH_VARARGS, "Create a reranker instance. Usage: create_reranker(model_type, config_path). Supported model types: 'qwen3', 'gte'."},
};

#endif
