#include <sstream>
#include <iostream>
#include "common.h"
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
    bool is_borrowed = false;  // Flag to indicate if this is a borrowed reference
} LLM;

static PyObject* PyMNNLLM_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    LLM* self = (LLM *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}
static void PyMNNLLM_dealloc(LLM *self) {
    // Only destroy if we own the LLM pointer (not borrowed)
    if (!self->is_borrowed && nullptr != self->llm) {
        MNN::Transformer::Llm::destroy(self->llm);
        self->llm = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMNNLLM_str(PyObject *self) {
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
    *(logits->var) = self->llm->forward(toInts(input_ids));
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

MNN::Transformer::MultimodalPrompt parse_multimodal_input(PyObject* content) {
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
    return multimodal_input;
}

MNN::Transformer::ChatMessages parse_chat_messages(PyObject* messages_obj) {
    MNN::Transformer::ChatMessages chat_messages;
    if (isVec<isPyDict>(messages_obj)) {
        Py_ssize_t msg_count = PyList_Size(messages_obj);
        for (Py_ssize_t i = 0; i < msg_count; i++) {
            PyObject* message_obj = PyList_GetItem(messages_obj, i);
            if (isPyDict(message_obj)) {
                PyObject* role_obj = PyDict_GetItemString(message_obj, "role");
                PyObject* content_obj = PyDict_GetItemString(message_obj, "content");
                if (role_obj && content_obj) {
                    MNN::Transformer::ChatMessage chat_message;
                    chat_message.first = object2String(role_obj);
                    chat_message.second = object2String(content_obj);
                    chat_messages.push_back(chat_message);
                }
            }
        }
    }
    return chat_messages;
}

static PyObject* PyMNNLLM_response(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        MNN_PRINT("[MNNLLM] response: is_embedding\n");
        Py_RETURN_NONE;
    }

    PyObject* content = nullptr;
    int stream = 0;
    int max_new_tokens = -1;

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
        auto multimodal_input = parse_multimodal_input(content);
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
    PyObject* content = nullptr;
    if (!PyArg_ParseTuple(args, "O", &content)) {
        Py_RETURN_NONE;
    }
    std::vector<int> ids;
    if (isString(content)) {
        std::string prompt = object2String(content);
        ids = self->llm->tokenizer_encode(prompt);
    } else if (isPyDict(content)) {
        auto multimodal_input = parse_multimodal_input(content);
        ids = self->llm->tokenizer_encode(multimodal_input);
    } else {
        PyMNN_ERROR("content must be str or dict");
    }
    return toPyObj<int, toPyObj>(ids);
}

static PyObject* PyMNNLLM_apply_chat_template(LLM *self, PyObject *args) {
    PyObject* content = nullptr;
    if (!PyArg_ParseTuple(args, "O", &content)) {
        Py_RETURN_NONE;
    }
    std::string prompt;
    if (isString(content)) {
        std::string query = object2String(content);
        prompt = self->llm->apply_chat_template(query);
    } else if (isVec<isPyDict>(content)) {
        auto messages = parse_chat_messages(content);
        prompt = self->llm->apply_chat_template(messages);
    } else {
        PyMNN_ERROR("content must be str or dict");
    }
    return string2Object(prompt);
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

static PyObject* PyMNNLLM_dump_config(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    auto config = self->llm->dump_config();
    return toPyObj(config);
}

static PyObject* PyMNNLLM_reset(LLM *self, PyObject *args) {
    self->llm->reset();
    Py_RETURN_NONE;
}

static PyObject* PyMNNLLM_stoped(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }
    bool stopped = self->llm->stoped();
    return toPyObj(stopped);
}

static PyObject* PyMNNLLM_generate_init(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }

    PyObject* stream_obj = nullptr;
    PyObject* end_with_obj = nullptr;

    // Parse arguments: optional stream object and end_with object
    if (!PyArg_ParseTuple(args, "|OO", &stream_obj, &end_with_obj)) {
        Py_RETURN_NONE;
    }

    // Convert end_with object to string if provided
    const char* end_with = nullptr;
    std::string end_with_str;
    if (end_with_obj != nullptr && end_with_obj != Py_None) {
        if (!PyUnicode_Check(end_with_obj)) {
            PyErr_SetString(PyExc_TypeError, "end_with must be a string or None");
            Py_RETURN_NONE;
        }
        end_with_str = object2String(end_with_obj);
        end_with = end_with_str.c_str();
    }

    // For now, we pass nullptr for ostream since we don't directly handle stream objects from Python
    // In the future, we could implement a proper stream wrapper if needed
    std::ostream* os = nullptr;

    self->llm->generate_init(os, end_with);
    Py_RETURN_NONE;
}

static PyObject* PyMNNLLM_get_context(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }

    auto context = self->llm->getContext();
    if (!context) {
        Py_RETURN_NONE;
    }

    PyObject* dict = PyDict_New();

    // Forward parameters
    PyDict_SetItemString(dict, "prompt_len", PyLong_FromLong(context->prompt_len));
    PyDict_SetItemString(dict, "gen_seq_len", PyLong_FromLong(context->gen_seq_len));
    PyDict_SetItemString(dict, "all_seq_len", PyLong_FromLong(context->all_seq_len));
    PyDict_SetItemString(dict, "end_with", string2Object(context->end_with));

    // Performance metrics
    PyDict_SetItemString(dict, "load_us", PyLong_FromLongLong(context->load_us));
    PyDict_SetItemString(dict, "vision_us", PyLong_FromLongLong(context->vision_us));
    PyDict_SetItemString(dict, "audio_us", PyLong_FromLongLong(context->audio_us));
    PyDict_SetItemString(dict, "prefill_us", PyLong_FromLongLong(context->prefill_us));
    PyDict_SetItemString(dict, "decode_us", PyLong_FromLongLong(context->decode_us));
    PyDict_SetItemString(dict, "sample_us", PyLong_FromLongLong(context->sample_us));
    PyDict_SetItemString(dict, "pixels_mp", PyFloat_FromDouble(context->pixels_mp));
    PyDict_SetItemString(dict, "audio_input_s", PyFloat_FromDouble(context->audio_input_s));

    // Tokens
    PyDict_SetItemString(dict, "current_token", PyLong_FromLong(context->current_token));

    PyObject* history_tokens = toPyObj<int, toPyObj>(context->history_tokens);
    PyDict_SetItemString(dict, "history_tokens", history_tokens);

    PyObject* output_tokens = toPyObj<int, toPyObj>(context->output_tokens);
    PyDict_SetItemString(dict, "output_tokens", output_tokens);

    PyDict_SetItemString(dict, "generate_str", string2Object(context->generate_str));

    // llm status
    PyDict_SetItemString(dict, "status", PyLong_FromLong((int)context->status));

    return dict;
}

static PyObject* PyMNNLLM_set_context(LLM *self, PyObject *args) {
    if (self->is_embedding) {
        Py_RETURN_NONE;
    }

    PyObject* dict = nullptr;
    if (!PyArg_ParseTuple(args, "O", &dict) || !PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        Py_RETURN_NONE;
    }

    auto context = const_cast<MNN::Transformer::LlmContext*>(self->llm->getContext());
    if (!context) {
        Py_RETURN_NONE;
    }

    // Forward parameters
    PyObject* prompt_len = PyDict_GetItemString(dict, "prompt_len");
    if (prompt_len && PyLong_Check(prompt_len)) {
        context->prompt_len = PyLong_AsLong(prompt_len);
    }

    PyObject* gen_seq_len = PyDict_GetItemString(dict, "gen_seq_len");
    if (gen_seq_len && PyLong_Check(gen_seq_len)) {
        context->gen_seq_len = PyLong_AsLong(gen_seq_len);
    }

    PyObject* all_seq_len = PyDict_GetItemString(dict, "all_seq_len");
    if (all_seq_len && PyLong_Check(all_seq_len)) {
        context->all_seq_len = PyLong_AsLong(all_seq_len);
    }

    PyObject* end_with = PyDict_GetItemString(dict, "end_with");
    if (end_with && PyUnicode_Check(end_with)) {
        context->end_with = object2String(end_with);
    }

    // Tokens
    PyObject* current_token = PyDict_GetItemString(dict, "current_token");
    if (current_token && PyLong_Check(current_token)) {
        context->current_token = PyLong_AsLong(current_token);
    }

    PyObject* history_tokens = PyDict_GetItemString(dict, "history_tokens");
    if (history_tokens && PyList_Check(history_tokens)) {
        context->history_tokens = toInts(history_tokens);
    }

    PyObject* output_tokens = PyDict_GetItemString(dict, "output_tokens");
    if (output_tokens && PyList_Check(output_tokens)) {
        context->output_tokens = toInts(output_tokens);
    }

    PyObject* generate_str = PyDict_GetItemString(dict, "generate_str");
    if (generate_str && PyUnicode_Check(generate_str)) {
        context->generate_str = object2String(generate_str);
    }

    PyObject* status = PyDict_GetItemString(dict, "status");
    if (status && PyLong_Check(status)) {
        context->status = (MNN::Transformer::LlmStatus)PyLong_AsLong(status);
    }

    Py_RETURN_NONE;
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
    {"apply_chat_template", (PyCFunction)PyMNNLLM_apply_chat_template, METH_VARARGS, "apply chat template."},
    {"tokenizer_encode", (PyCFunction)PyMNNLLM_tokenizer_encode, METH_VARARGS, "tokenizer encode."},
    {"tokenizer_decode", (PyCFunction)PyMNNLLM_tokenizer_decode, METH_VARARGS, "tokenizer decode."},
    {"txt_embedding", (PyCFunction)PyMNNLLM_txt_embedding, METH_VARARGS, "txt embedding."},
    {"create_lora", (PyCFunction)PyMNNLLM_create_lora, METH_VARARGS, "create_lora."},
    {"set_config", (PyCFunction)PyMNNLLM_set_config, METH_VARARGS, "set_config."},
    {"dump_config", (PyCFunction)PyMNNLLM_dump_config, METH_VARARGS, "dump_config."},
    {"reset", (PyCFunction)PyMNNLLM_reset, METH_VARARGS, "reset."},
#ifdef PYMNN_LLM_COLLECTION
    {"enable_collection_mode", (PyCFunction)PyMNNLLM_enable_collection_mode, METH_VARARGS, "Enable data collection mode."},
#endif
    {"get_context", (PyCFunction)PyMNNLLM_get_context, METH_VARARGS, "Get LlmContext data."},
    {"set_context", (PyCFunction)PyMNNLLM_set_context, METH_VARARGS, "Set LlmContext data."},
    {"generate_init", (PyCFunction)PyMNNLLM_generate_init, METH_VARARGS, "Initialize generation with optional stream and end_with parameters."},
    {"stoped", (PyCFunction)PyMNNLLM_stoped, METH_NOARGS, "Check if the generation has stopped."},
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
    PyMNNLLM_str,                             /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    PyMNNLLM_str,                             /*tp_str*/
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
