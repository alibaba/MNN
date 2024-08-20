#include "llm/llm.hpp"

typedef struct {
    PyObject_HEAD
    MNN::Transformer::Llm* llm;
} LLM;

static PyObject* PyMNNLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static PyObject* Py_str(PyObject *self) {
    LLM* llm = (LLM*)self;
    if (!llm) {
        Py_RETURN_NONE;
    }
    return toPyObj("llm");
}

static PyObject* PyMNNLLM_load(LLM *self, PyObject *args) {
    self->llm->load();
    Py_RETURN_NONE;
}

static PyObject* PyMNNLLM_generate(LLM *self, PyObject *args) {
    PyObject *input_ids = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input_ids) && isInts(input_ids)) {
        Py_RETURN_NONE;
    }
    auto output_ids = self->llm->generate(toInts(input_ids));
    return toPyObj<int, toPyObj>(output_ids);
}

static PyObject* PyMNNLLM_response(LLM *self, PyObject *args) {
    const char* query = NULL;
    int stream = 0;
    if (!PyArg_ParseTuple(args, "s|p", &query, &stream)) {
        Py_RETURN_NONE;
    }
    MNN::Transformer::LlmStreamBuffer buffer(nullptr);
    std::ostream null_os(&buffer);
    auto res = self->llm->response(query, stream ? &std::cout : &null_os);
    return string2Object(res);
}

static PyMethodDef PyMNNLLM_methods[] = {
    {"load", (PyCFunction)PyMNNLLM_load, METH_VARARGS, "load model."},
    {"generate", (PyCFunction)PyMNNLLM_generate, METH_VARARGS, "generate `output_ids` by `input_ids`."},
    {"response", (PyCFunction)PyMNNLLM_response, METH_VARARGS, "response `query` without hsitory."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNLLM = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "LLM",                                    /*tp_name*/
    sizeof(LLM),                              /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    0,                                        /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    Py_str,                                   /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    Py_str,                                   /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "LLM is mnn-llm's `Llm` python wrapper",  /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    PyMNNLLM_methods,                            /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNLLM_new,                                /* tp_new */
};

static PyObject* PyMNNLLM_create(PyObject *self, PyObject *args) {
    if (!PyTuple_Size(args)) {
        return NULL;
    }
    const char* path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }
    LLM *llm = (LLM *)PyObject_Call((PyObject*)&PyMNNLLM, PyTuple_New(0), NULL);
    if (!llm) {
        return NULL;
    }
    llm->llm = MNN::Transformer::Llm::createLLM(path);
    return (PyObject*)llm;
}

static PyMethodDef PyMNNLLM_static_methods[] = {
    {"create", PyMNNLLM_create, METH_VARARGS}
};
