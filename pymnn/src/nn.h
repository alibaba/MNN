#include "util.h"
#ifdef PYMNN_INTERNAL_SERVING
#include <MNN/AutoTime.hpp>
#include <MNN/MNNForwardType.h>
#include "internal/monitor_service.h"
#include "internal/verify_service.h"
#endif

// NN Module Start
def_class_start(_Module, Module)
def_class_getset(
    _Module,
    name, 0,
    is_training, 0,
    parameters, 0
)
def_class_methods(_Module,
    forward, "forward",
    onForward, "onForward",
    set_name, "set name",
    train, "set is_training",
    load_parameters, "load parameters",
    clear_cache, "clear cache",
    _register_submodules, "register submodules",
    _add_parameter, "add parameter"
)
def_class_end(_Module, Module)

static PyObject* load_module(PyObject *inputs, PyObject *outputs, PyObject *backend, PyObject *memory_mode,
                             PyObject *power_mode, PyObject *precision_mode, const char* file_name, int dynamic,
                             int shape_mutable, int rearrange, int thread_num) {

    BackendConfig backend_config;
    backend_config.memory = toEnum<MemoryMode>(memory_mode);
    backend_config.power = toEnum<PowerMode>(power_mode);
    backend_config.precision = toEnum<PrecisionMode>(precision_mode);

    Module::BackendInfo backend_info;
    backend_info.type = toEnum<MNNForwardType>(backend);
    backend_info.config = &backend_config;

    Module::Config config;
    config.dynamic = dynamic;
    config.shapeMutable = shape_mutable;
    config.rearrange = rearrange;
    config.backend = &backend_info;

    auto converted_file_name = convertBytesEncodeIfNeed(file_name);
    auto m_ptr = Module::load(toStrings(inputs), toStrings(outputs), converted_file_name.data(), &config);
    if (m_ptr == nullptr) {
        std::string mnn_errno = "load_module_from_file failed ";
        mnn_errno = mnn_errno + std::string(file_name);
        PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
    }

    return toPyObj(m_ptr);
}

static PyObject* PyMNN_Module_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyMNN_Module *self = (PyMNN_Module *)type->tp_alloc(type, 0);
    self->ptr = Module::createEmpty({});
    return (PyObject*)self;
}
// PyMNN_Module getter/setter impl
static PyObject* PyMNN_Module_getname(PyMNN_Module *self, void *closure) {
    if (self->ptr) {
        return toPyObj(self->ptr->name());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_getis_training(PyMNN_Module *self, void *closure) {
    if (self->ptr) {
        return toPyObj(self->ptr->getIsTraining());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_getparameters(PyMNN_Module *self, void *closure) {
    if (self->ptr) {
        return toPyObj<VARP, toPyObj>(self->ptr->parameters());
    }
    Py_RETURN_NONE;
}
// PyMNN_Module methods impl
static PyObject* PyMNN_Module_forward(PyMNN_Module *self, PyObject *args) {
    PyObject *input;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        Py_RETURN_NONE;
    }
    if (isVars(input)) {
#ifdef PYMNN_INTERNAL_SERVING
        int status = 0;
        Timer timer;
        auto vars = self->ptr->onForward(toVars(input));
        if (vars.empty()) {
            PyMNN_ERROR("module onForward occur error.");
            status = -1;
        }

        (void) MonitorService::GetInstance().EventTrack(self->ptr, timer, status, "PyMNN_Module_forward");
        return toPyObj<VARP, toPyObj>(vars);
#else
        return toPyObj<VARP, toPyObj>(self->ptr->onForward(toVars(input)));
#endif
    }
    if (isVar(input)) {
#ifdef PYMNN_INTERNAL_SERVING
        int status = 0;
        Timer timer;
        auto var = self->ptr->forward(toVar(input));
        (void) MonitorService::GetInstance().EventTrack(self->ptr, timer, status, "PyMNN_Module_forward");
        return toPyObj(var);
#else
        return toPyObj(self->ptr->forward(toVar(input)));
#endif
    }
    PyMNN_ERROR("PyMNN_Module_forward: args must be Var/[Var].");
}
static PyObject* PyMNN_Module_onForward(PyMNN_Module *self, PyObject *args) {
    PyObject *inputs;
    if (!PyArg_ParseTuple(args, "O", &inputs)) {
        Py_RETURN_NONE;
    }
#ifdef PYMNN_INTERNAL_SERVING
    int status = 0;
    Timer timer;
    auto vars = self->ptr->onForward(toVars(inputs));
    if (vars.empty()) {
        PyMNN_ERROR("module onForward occur error.");
        status = -1;
    }

    (void) MonitorService::GetInstance().EventTrack(self->ptr, timer, status, "PyMNN_Module_onForward");
    return toPyObj<VARP, toPyObj>(vars);
#else
    return toPyObj<VARP, toPyObj>(self->ptr->onForward(toVars(inputs)));
#endif
}

static PyObject* PyMNN_Module_set_name(PyMNN_Module *self, PyObject *args) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        Py_RETURN_NONE;
    }
    self->ptr->setName(name);
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_train(PyMNN_Module *self, PyObject *args) {
    int isTraining;
    if (!PyArg_ParseTuple(args, "i", &isTraining)) {
        Py_RETURN_NONE;
    }
    self->ptr->setIsTraining(isTraining);
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_load_parameters(PyMNN_Module *self, PyObject *args) {
    PyObject* parameters;
    if (!PyArg_ParseTuple(args, "O", &parameters)) {
        Py_RETURN_NONE;
    }
    return toPyObj(self->ptr->loadParameters(toVars(parameters)));
}
static PyObject* PyMNN_Module_clear_cache(PyMNN_Module *self, PyObject *args) {
    self->ptr->clearCache();
    Py_RETURN_NONE;
}
std::shared_ptr<Module> toSharedModule(PyObject* obj)  {
    return std::shared_ptr<Module>(to_Module(obj), [](Module*){});
}
static PyObject* PyMNN_Module__register_submodules(PyMNN_Module *self, PyObject *args) {
    PyObject *children;
    if (!PyArg_ParseTuple(args, "O", &children)) {
        Py_RETURN_NONE;
    }
    self->ptr->registerModel(toVec<std::shared_ptr<Module>, toSharedModule>(children));
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module__add_parameter(PyMNN_Module *self, PyObject *args) {
    PyObject *parameter;
    if (!PyArg_ParseTuple(args, "O", &parameter)) {
        Py_RETURN_NONE;
    }
    return toPyObj(self->ptr->addParameter(toVar(parameter)));
}
// NN methods
static PyObject* PyMNNNN_load_module(PyObject *self, PyObject *args) {
    PyObject *inputs, *outputs;
    int fortrain;
    if (!PyArg_ParseTuple(args, "OOi", &inputs, &outputs, &fortrain)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
#ifdef PYMNN_TRAIN_API
    auto m = NN::extract(toVars(inputs), toVars(outputs), fortrain);
#else
    auto m = Module::extract(toVars(inputs), toVars(outputs), fortrain);
#endif
    return toPyObj(m);
}
static PyObject* PyMNNNN_load_module_from_file(PyObject *self, PyObject *args) {
#ifdef PYMNN_INTERNAL_SERVING
    PyErr_SetString(PyExc_Exception,
                        "PyMNNNN_load_module_from_file: unsupported interface, should use load_module_from_file_with_token.");
    return NULL;
#endif
    PyObject *inputs, *outputs, *backend, *memory_mode, *power_mode, *precision_mode;
    const char* file_name;
    int dynamic, shape_mutable, rearrange;
    int thread_num;
    if (!PyArg_ParseTuple(args, "OOsiiiOOOOi", &inputs, &outputs, &file_name, &dynamic,
                          &shape_mutable, &rearrange, &backend, &memory_mode,
                          &power_mode, &precision_mode, &thread_num)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }

    return load_module(inputs, outputs, backend, memory_mode, power_mode, precision_mode, file_name, dynamic,
     shape_mutable,  rearrange,  thread_num);
}

#ifdef PYMNN_INTERNAL_SERVING
static PyObject* PyMNNNN_load_module_from_file_with_token(PyObject *self, PyObject *args) {
    PyObject *inputs, *outputs;
    const char* file_name;
    PyObject *backend = toPyObj(MNN_FORWARD_CPU);
    PyObject *memory_mode = toPyObj(MemoryMode::Memory_Normal);
    PyObject *power_mode = toPyObj(PowerMode::Power_Normal);;
    PyObject *precision_mode = toPyObj(PrecisionMode::Precision_Normal);;
    int dynamic = 0;
    int shape_mutable = 0;
    int rearrange = 0;
    char *token = NULL;
    char *scene = NULL;
    char *app_key = NULL;
    int thread_num = 1;
    if (!PyArg_ParseTuple(args, "OOssss|iiiOOOOi", &inputs, &outputs, &file_name, &token, &scene, &app_key, &dynamic,
                          &shape_mutable, &rearrange, &backend, &memory_mode, &power_mode, &precision_mode,
                          &thread_num)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }

    if (!token || !scene || !app_key) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNNN_load_module_from_file_with_token: input invalid, token, scene or app_key is null.");
        return NULL;
    }

    MonitorService::GetInstance().Start();
    VerifyService::GetInstance().Start();
    bool ret = VerifyService::GetInstance().VerifyToken(std::string(token), std::string(scene), std::string(app_key));
    if (!ret) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNNN_load_module_from_file_with_token: check token failed, return null module.");
        return NULL;
    }

    return load_module(inputs, outputs, backend, memory_mode, power_mode, precision_mode, file_name, dynamic,
     shape_mutable,  rearrange,  thread_num);

}
#endif

#ifdef PYMNN_TRAIN_API
static PyObject* PyMNNNN_conv(PyObject *self, PyObject *args) {
    INTS default_1 = {1, 1}, default_0 = {0, 0};
    int in_channel, out_channel;
    PyObject *kernel_size,
             *stride = toPyObj(default_1),
             *padding = toPyObj(default_0),
             *dilation = toPyObj(default_1),
             *padding_mode = toPyObj(PaddingMode::VALID);
    int depthwise = 0, bias = 1;
    if (!PyArg_ParseTuple(args, "iiO|OOOiiO", &in_channel, &out_channel, &kernel_size,
                          &stride, &padding, &dilation, &depthwise, &bias, &padding_mode)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
    NN::ConvOption option;
    option.channel = {in_channel, out_channel};
    option.kernelSize = toInts(kernel_size);
    auto stride_ = toInts(stride);
    auto padding_ = toInts(padding);
    auto dilation_ = toInts(dilation);
    if (!stride_.empty()) {
        option.stride = stride_;
    }
    option.padMode = toEnum<PaddingMode>(padding_mode);
    if (!padding_.empty()) {
        option.pads = padding_;
    }
    if (!dilation_.empty()) {
        option.dilate = dilation_;
    }
    option.depthwise = depthwise;
    return toPyObj(NN::Conv(std::move(option), bias));
}
static PyObject* PyMNNNN_linear(PyObject *self, PyObject *args) {
    int in_channel, out_channel;
    int bias = 1;
    if (!PyArg_ParseTuple(args, "ii|i", &in_channel, &out_channel, &bias)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
    return toPyObj(NN::Linear(in_channel, out_channel, bias));
}
static PyObject* PyMNNNN_batch_norm(PyObject *self, PyObject *args) {
    int channels, dims = 4;
    float momentum = 0.99, epsilon = 1e-5;
    if (!PyArg_ParseTuple(args, "i|iff", &channels, &dims, &momentum, &epsilon)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
    return toPyObj(NN::BatchNorm(channels, dims, momentum, epsilon));
}
static PyObject* PyMNNNN_dropout(PyObject *self, PyObject *args) {
    float dropout_ratio;
    if (!PyArg_ParseTuple(args, "f", &dropout_ratio)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }
    return toPyObj(NN::Dropout(dropout_ratio));
}
#endif
static PyMethodDef PyMNNNN_methods[] = {
#ifdef PYMNN_INTERNAL_SERVING
    register_methods(NN,
        load_module, "load_module([Var], [Var], bool)",
        load_module_from_file_with_token, "load_module_from_file_with_token([string], [string], filename, bool, ...)",
        load_module_from_file, "load_module_from_file([string], [string], filename, bool, ...)"
    )
#else
    register_methods(NN,
        load_module, "load_module([Var], [Var], bool)",
        load_module_from_file, "load_module_from_file([string], [string], filename, bool, ...)"
    )
#endif
#ifdef PYMNN_TRAIN_API
    register_methods(NN,        
        conv, "conv Module",
        linear, "linear Module",
        batch_norm, "batch_norm Module",
        dropout, "dropout module"
    )
#endif
};
// NN Module End
