#include <sstream>
#include "llm/llm.hpp"

typedef struct {
    PyObject_HEAD
    MNN::Transformer::Llm* llm;
    bool is_embedding = false;
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

static PyObject* PyMNNLLM_forward(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    PyObject *input_ids = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input_ids) && isInts(input_ids)) {
        Py_RETURN_NONE;
    }
    auto logits = getVar();
    self->llm->generate_init();
    *(logits->var) = self->llm->forward(toInts(input_ids));
    self->llm->reset();
    return (PyObject *)logits;
}

static PyObject* PyMNNLLM_generate(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    PyObject *input_ids = nullptr;
    if (!PyArg_ParseTuple(args, "O", &input_ids) && isInts(input_ids)) {
        Py_RETURN_NONE;
    }

    auto output_ids = self->llm->generate(toInts(input_ids));
    return toPyObj<int, toPyObj>(output_ids);
}

static PyObject* PyMNNLLM_eraseHistory(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    size_t history = 0;
    size_t end = 0;
    if (!PyArg_ParseTuple(args, "LL", &history, &end)) {
        Py_RETURN_NONE;
    }
    self->llm->eraseHistory(history, end);
    Py_RETURN_NONE;
}
static PyObject* PyMNNLLM_getCurrentHistory(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    auto history = self->llm->getCurrentHistory();
    return PyLong_FromLong(history);
}
static PyObject* PyMNNLLM_response(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* query = NULL;
    int stream = 0;
    if (!PyArg_ParseTuple(args, "s|p", &query, &stream)) {
        Py_RETURN_NONE;
    }
    std::ostringstream null_os;
    self->llm->response(query, stream ? &std::cout : &null_os);
    return string2Object(null_os.str());
}

static PyObject* PyMNNLLM_tokenizer_encode(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* prompt = NULL;
    int use_template = 0;
    if (!PyArg_ParseTuple(args, "s|p", &prompt, &use_template)) {
        Py_RETURN_NONE;
    }
    auto ids = self->llm->tokenizer_encode(prompt, use_template);
    return toPyObj<int, toPyObj>(ids);
}

static PyObject* PyMNNLLM_tokenizer_decode(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    PyObject *id = nullptr;
    if (!PyArg_ParseTuple(args, "O", &id) && isInt(id)) {
        Py_RETURN_NONE;
    }
    auto query = self->llm->tokenizer_decode(toInt(id));
    return string2Object(query);
}

static PyObject* PyMNNLLM_txt_embedding(LLM *self, PyObject *args) {
    if (!self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* query = NULL;
    if (!PyArg_ParseTuple(args, "s", &query)) {
        Py_RETURN_NONE;
    }
    auto embeds = getVar();
    *(embeds->var) = ((MNN::Transformer::Embedding*)self->llm)->txt_embedding(query);
    return (PyObject *)embeds;
}

static PyObject* PyMNNLLM_apply_lora(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        Py_RETURN_NONE;
    }
    int model_index = self->llm->apply_lora(path);
    return toPyObj(model_index);
}

static PyObject* PyMNNLLM_select_module(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    PyObject *index = nullptr;
    if (!PyArg_ParseTuple(args, "O", &index) && isInt(index)) {
        Py_RETURN_NONE;
    }
    bool res = self->llm->select_module(toInt(index));
    return toPyObj(res);
}

static PyObject* PyMNNLLM_release_module(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    PyObject *index = nullptr;
    if (!PyArg_ParseTuple(args, "O", &index) && isInt(index)) {
        Py_RETURN_NONE;
    }
    bool res = self->llm->release_module(toInt(index));
    return toPyObj(res);
}

static PyMethodDef PyMNNLLM_methods[] = {
    {"load", (PyCFunction)PyMNNLLM_load, METH_VARARGS, "load model."},
    {"forward", (PyCFunction)PyMNNLLM_forward, METH_VARARGS, "forward `logits` by `input_ids`."},
    {"generate", (PyCFunction)PyMNNLLM_generate, METH_VARARGS, "generate `output_ids` by `input_ids`."},
    {"response", (PyCFunction)PyMNNLLM_response, METH_VARARGS, "response `query` without hsitory."},
    {"get_current_history", (PyCFunction)PyMNNLLM_getCurrentHistory, METH_VARARGS, "Get Current History."},
    {"erase_history", (PyCFunction)PyMNNLLM_eraseHistory, METH_VARARGS, "Erase History."},
    {"tokenizer_encode", (PyCFunction)PyMNNLLM_tokenizer_encode, METH_VARARGS, "tokenizer encode."},
    {"tokenizer_decode", (PyCFunction)PyMNNLLM_tokenizer_decode, METH_VARARGS, "tokenizer decode."},
    {"txt_embedding", (PyCFunction)PyMNNLLM_txt_embedding, METH_VARARGS, "txt embedding."},
    {"apply_lora", (PyCFunction)PyMNNLLM_apply_lora, METH_VARARGS, "apply_lora."},
    {"select_module", (PyCFunction)PyMNNLLM_select_module, METH_VARARGS, "select_module."},
    {"release_module", (PyCFunction)PyMNNLLM_release_module, METH_VARARGS, "release_module."},
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
    int embedding_model = 0;
    if (!PyArg_ParseTuple(args, "s|p", &path, &embedding_model)) {
        return NULL;
    }
    LLM *llm = (LLM *)PyObject_Call((PyObject*)&PyMNNLLM, PyTuple_New(0), NULL);
    if (!llm) {
        return NULL;
    }
    if (embedding_model) {
        llm->llm = MNN::Transformer::Embedding::createEmbedding(path);
        llm->is_embedding = true;
    } else {
        llm->llm = MNN::Transformer::Llm::createLLM(path);
    }

    return (PyObject*)llm;
}

static PyMethodDef PyMNNLLM_static_methods[] = {
    {"create", PyMNNLLM_create, METH_VARARGS}
};
