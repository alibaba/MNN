#include <sstream>
#include <iostream>
#ifdef BUILD_FOR_IOS
#include "MNN/llm/llm.hpp"
#else
#include "llm/llm.hpp"
#endif
#ifdef PYMNN_LLM_COLLECTION
#include "cpp/getLinearInput.hpp"
#endif
typedef struct {
    PyObject_HEAD
    MNN::Transformer::Llm* llm = nullptr;
    bool is_embedding = false;
} LLM;

static PyObject* PyMNNLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static void PyMNNLLM_dealloc(LLM *self) {
    if (nullptr != self->llm) {
        MNN::Transformer::Llm::destroy(self->llm);
        self->llm = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
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
    int max_new_tokens = 0;
    if (!PyArg_ParseTuple(args, "O|i", &input_ids, &max_new_tokens) && isInts(input_ids)) {
        Py_RETURN_NONE;
    }

    auto output_ids = self->llm->generate(toInts(input_ids), max_new_tokens);
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
        MNN_PRINT("[MNNLLM] response: is_embedding\n");
        Py_RETURN_NONE;
    }
    
    PyObject* content = nullptr;
    int stream = 0;
    int max_new_tokens = 2048;
    
    if (!PyArg_ParseTuple(args, "O|ii", &content, &stream, &max_new_tokens)) {
        MNN_PRINT("[MNNLLM] response: PyArg_ParseTuple failed\n");
        Py_RETURN_NONE;
    }
    
    std::ostringstream null_os;
    std::ostream* output_stream = stream ? &std::cout : &null_os;
    
    if (isString(content)) {
        std::string text = object2String(content);
        MNN_PRINT("[MNNLLM] response: text=%s, stream=%d, max_new_tokens=%d\n", text.c_str(), stream, max_new_tokens);
        self->llm->response(text, output_stream, nullptr, max_new_tokens);
    } else if (isPyDict(content)) {
        MNN::Transformer::MultimodalPrompt multimodal_input;
        PyObject* text_obj = PyDict_GetItemString(content, "text");
        if (text_obj && isString(text_obj)) {
            multimodal_input.prompt_template = object2String(text_obj);
        }
        PyObject* images_obj = PyDict_GetItemString(content, "images");
        if (images_obj && PyList_Check(images_obj)) {
            Py_ssize_t img_count = PyList_Size(images_obj);
            for (Py_ssize_t i = 0; i < img_count; i++) {
                PyObject* img_dict = PyList_GetItem(images_obj, i);
                if (isPyDict(img_dict)) {
                    PyObject* data_obj = PyDict_GetItemString(img_dict, "data");
                    PyObject* width_obj = PyDict_GetItemString(img_dict, "width");
                    PyObject* height_obj = PyDict_GetItemString(img_dict, "height");
                    
                    if (data_obj && width_obj && height_obj) {
                        MNN::Transformer::PromptImagePart image_part;
                        image_part.image_data = toVar(data_obj);
                        image_part.width = PyLong_AsLong(width_obj);
                        image_part.height = PyLong_AsLong(height_obj);
                        
                        std::string key = "image_" + std::to_string(i);
                        multimodal_input.images[key] = image_part;
                    }
                }
            }
        }
        
        PyObject* audios_obj = PyDict_GetItemString(content, "audios");
        if (audios_obj && PyList_Check(audios_obj)) {
            Py_ssize_t audio_count = PyList_Size(audios_obj);
            for (Py_ssize_t i = 0; i < audio_count; i++) {
                PyObject* audio_dict = PyList_GetItem(audios_obj, i);
                if (isPyDict(audio_dict)) {
                    MNN::Transformer::PromptAudioPart audio_part;
                    
                    PyObject* file_path_obj = PyDict_GetItemString(audio_dict, "file_path");
                    if (file_path_obj && isString(file_path_obj)) {
                        audio_part.file_path = object2String(file_path_obj);
                    }
                    
                    PyObject* waveform_obj = PyDict_GetItemString(audio_dict, "waveform");
                    if (waveform_obj) {
                        audio_part.waveform = toVar(waveform_obj);
                    }
                    
                    std::string key = "audio_" + std::to_string(i);
                    multimodal_input.audios[key] = audio_part;
                }
            }
        }
        MNN_PRINT("[MNNLLM] response: multimodal, stream=%d, max_new_tokens=%d\n", stream, max_new_tokens);
        self->llm->response(multimodal_input, output_stream, nullptr, max_new_tokens);
    } else {
        PyMNN_ERROR("content must be str or dict");
    }
    std::string response_str = null_os.str();
    MNN_PRINT("[MNNLLM] response: %s\n", response_str.c_str());
    return string2Object(response_str);
}
static PyObject* PyMNNLLM_tokenizer_encode(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* prompt = NULL;
    if (!PyArg_ParseTuple(args, "s", &prompt)) {
        Py_RETURN_NONE;
    }
    auto ids = self->llm->tokenizer_encode(prompt);
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
static PyObject* PyMNNLLM_create_lora(LLM *self, PyObject *args);

static PyObject* PyMNNLLM_set_config(LLM *self, PyObject *args) {
    const char* config = NULL;
    if (!PyArg_ParseTuple(args, "s", &config)) {
        Py_RETURN_NONE;
    }
    bool res = self->llm->set_config(config);
    return toPyObj(res);
}

static PyObject* PyMNNLLM_reset(LLM *self, PyObject *args) {
    self->llm->reset();
    Py_RETURN_NONE;
}

static PyObject* PyMNNLLM_get_statistics(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    auto statistics = self->llm->get_statistics();
    return string2Object(statistics);
}

#ifdef PYMNN_LLM_COLLECTION
static PyObject* PyMNNLLM_enable_collection_mode(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    
    int mode = 0;
    const char* output_file = NULL;
    float target_sparsity = 0.5;  
    
    if (!PyArg_ParseTuple(args, "i|sf", &mode, &output_file, &target_sparsity)) {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments. Usage: enable_collection_mode(mode, output_file=None, target_sparsity=0.5)");
        Py_RETURN_NONE;
    }
    
    std::string filename;
    
    switch (mode) {
        case 1: {
            // Threshold mode 
            if (output_file == NULL) {
                filename = "thresholds.json";
            } else {
                filename = std::string(output_file);
            }
            
            MNN::LinearInput::initGetThreshold(filename, target_sparsity);
            MNN_PRINT("Enabled threshold collection mode. Output: %s, Sparsity: %.2f\n", 
                        filename.c_str(), target_sparsity);
           
            break;
        }
        
        case 2: {
            // MaxValue mode
            if (output_file == NULL) {
                filename = "max_values.json";
            } else {
                filename = std::string(output_file);
            }
            
            MNN::LinearInput::initGetMaxValue(filename);
            MNN_PRINT("Enabled max value collection mode. Output: %s\n", filename.c_str());
           
            break;
        }
        
        default: {
            PyErr_SetString(PyExc_ValueError, "Invalid mode. Use 1 for threshold collection, 2 for max value collection");
            Py_RETURN_NONE;
        }
    }
    
    return toPyObj(true);  
}
#endif
static PyMethodDef PyMNNLLM_methods[] = {
    {"load", (PyCFunction)PyMNNLLM_load, METH_VARARGS, "load model."},
    {"forward", (PyCFunction)PyMNNLLM_forward, METH_VARARGS, "forward `logits` by `input_ids`."},
    {"generate", (PyCFunction)PyMNNLLM_generate, METH_VARARGS, "generate `output_ids` by `input_ids`."},
    {"response", (PyCFunction)PyMNNLLM_response, METH_VARARGS, "response `query` - supports both text and multimodal input."},
    {"get_current_history", (PyCFunction)PyMNNLLM_getCurrentHistory, METH_VARARGS, "Get Current History."},
    {"erase_history", (PyCFunction)PyMNNLLM_eraseHistory, METH_VARARGS, "Erase History."},
    {"tokenizer_encode", (PyCFunction)PyMNNLLM_tokenizer_encode, METH_VARARGS, "tokenizer encode."},
    {"tokenizer_decode", (PyCFunction)PyMNNLLM_tokenizer_decode, METH_VARARGS, "tokenizer decode."},
    {"txt_embedding", (PyCFunction)PyMNNLLM_txt_embedding, METH_VARARGS, "txt embedding."},
    {"create_lora", (PyCFunction)PyMNNLLM_create_lora, METH_VARARGS, "create_lora."},
    {"set_config", (PyCFunction)PyMNNLLM_set_config, METH_VARARGS, "set_config."},
    {"reset", (PyCFunction)PyMNNLLM_reset, METH_VARARGS, "reset."},
    {"get_statistics", (PyCFunction)PyMNNLLM_get_statistics, METH_VARARGS, "get performance statistics."},
#ifdef PYMNN_LLM_COLLECTION
    {"enable_collection_mode", (PyCFunction)PyMNNLLM_enable_collection_mode, METH_VARARGS, "Enable data collection mode."},
#endif
    {NULL}  /* Sentinel */
};

static PyTypeObject PyMNNLLM = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "LLM",                                    /*tp_name*/
    sizeof(LLM),                              /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNLLM_dealloc,             /*tp_dealloc*/
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
static PyObject* PyMNNLLM_create_lora(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    const char* path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        Py_RETURN_NONE;
    }
    auto lora = self->llm->create_lora(path);
    LLM *llm = (LLM *)PyObject_Call((PyObject*)PyType_FindTLSType(&PyMNNLLM), PyTuple_New(0), NULL);
    if (!llm) {
        return NULL;
    }
    llm->llm = lora;;
    return (PyObject*)llm;
}

static PyObject* PyMNNLLM_create(PyObject *self, PyObject *args) {
    if (!PyTuple_Size(args)) {
        return NULL;
    }
    const char* path = NULL;
    int embedding_model = 0;
    if (!PyArg_ParseTuple(args, "s|i", &path, &embedding_model)) {
        PyMNN_ERROR_LOG("Invalid arguments. Usage: create(path, embedding_model=False)");
        return NULL;
    }
    LLM *llm = (LLM *)PyObject_Call((PyObject*)PyType_FindTLSType(&PyMNNLLM), PyTuple_New(0), NULL);
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
