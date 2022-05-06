#include "util.h"
#ifdef PYMNN_INTERNAL_SERVING
#include <MNN/AutoTime.hpp>
#include <MNN/MNNForwardType.h>
#include "internal/monitor_service.h"
#include "internal/verify_service.h"
#endif

#if defined(PYMNN_INTERNAL_SERVING) || defined(PYMNN_USE_ALINNPYTHON)
#include "internal/PythonAuthByPass.hpp"
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

// NN RuntimeManager Start
def_class_smart_start(RuntimeManager, Executor::RuntimeManager)
def_class_methods(RuntimeManager,
    set_cache, "set cache",
    update_cache, "update cache",
    set_mode, "set mode",
    set_hint, "set hint"
)
def_class_without_getset(RuntimeManager)
def_class_smart_end(RuntimeManager, Executor::RuntimeManager)

static PyObject* load_module(PyObject *runtime_manager, PyObject *inputs, PyObject *outputs, PyObject *backend, PyObject *memory_mode,
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
    std::shared_ptr<Executor::RuntimeManager> rt_mgr(nullptr);
    if(Py_TYPE(runtime_manager) == PyType_FindTLSType(&PyMNNRuntimeManagerType)) {
        rt_mgr = *(toRuntimeManager(runtime_manager));
    }


#if defined(PYMNN_INTERNAL_SERVING) || defined(PYMNN_USE_ALINNPYTHON)
    Module* m_ptr = PythonAuthByPass::loadModuleWithoutAuth(toStrings(inputs), toStrings(outputs), converted_file_name.data(), rt_mgr, &config);
#else
    Module* m_ptr = Module::load(toStrings(inputs), toStrings(outputs), converted_file_name.data(), rt_mgr, &config);
#endif

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

static PyObject* PyMNNRuntimeManager_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyMNNRuntimeManager *self = (PyMNNRuntimeManager *)type->tp_alloc(type, 0);
    self->ptr = new std::shared_ptr<Executor::RuntimeManager>(nullptr);
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
    PyObject *inputs, *outputs, *backend, *memory_mode, *power_mode, *precision_mode, *runtime_manager;
    const char* file_name;
    int dynamic, shape_mutable, rearrange;
    int thread_num;
    if (!PyArg_ParseTuple(args, "OOOsiiiOOOOi", &runtime_manager, &inputs, &outputs, &file_name, &dynamic,
                          &shape_mutable, &rearrange, &backend, &memory_mode,
                          &power_mode, &precision_mode, &thread_num)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }

    return load_module(runtime_manager, inputs, outputs, backend, memory_mode, power_mode, precision_mode, file_name, dynamic,
     shape_mutable,  rearrange,  thread_num);
}


static PyObject* PyMNNRuntimeManager_set_cache(PyMNNRuntimeManager *self, PyObject *args) {
    char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNRuntimeManager_set_cache: Not string input");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    std::string cachePath = path;
    (*(self->ptr))->setCache(cachePath);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static PyObject* PyMNNRuntimeManager_update_cache(PyMNNRuntimeManager *self, PyObject *args) {
    (*(self->ptr))->updateCache();
    Py_RETURN_NONE;
}

static PyObject* PyMNNRuntimeManager_set_mode(PyMNNRuntimeManager *self, PyObject *args) {
    int session_val;
    if (!PyArg_ParseTuple(args, "i", &session_val)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNRuntimeManager_set_mode: Not interger input");
        return NULL;
    }

    auto mode = (MNN::Interpreter::SessionMode)session_val;

    (*(self->ptr))->setMode(mode);
    Py_RETURN_NONE;
}
static PyObject* PyMNNRuntimeManager_set_hint(PyMNNRuntimeManager *self, PyObject *args) {
    int type_val = 0;
    int num_val = 0;
    if (!PyArg_ParseTuple(args, "ii", &type_val, &num_val)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNRuntimeManager_set_hint: Not interger input and interger input");
        return NULL;
    }

    auto type = (MNN::Interpreter::HintMode)type_val;
    (*(self->ptr))->setHint(type, num_val);
    Py_RETURN_NONE;
}

static std::pair<bool, std::pair<ScheduleConfig, std::shared_ptr<BackendConfig>>> getScheduleConfig(PyObject* dict) {
    std::pair<bool, std::pair<ScheduleConfig, std::shared_ptr<BackendConfig>>> result;
    result.first = false;
    auto& config = result.second.first;
    auto& backendConfig = result.second.second;
    backendConfig.reset(new BackendConfig);
    config.backendConfig = backendConfig.get();

    if (dict) {
        PyObject *backend = PyDict_GetItemString(dict, "backend");
        config.type = MNN_FORWARD_CPU;
        if (backend && checkString(backend)) {
            auto backend_name = object2String(backend);
            // Avoid misusing backend not supported by the bridge and corresponding MNN library on python level,
            // then user will ask for right version bridge library to us, same like MNN.expr.Backend.* python enum
            std::unordered_map<std::string, MNNForwardType> backend_map = {
                // Don't care whether MNN library support corresponding backend, all backend type are usable by user,
                // which make MNN.whl setup.py easy
                {"CPU", MNN_FORWARD_CPU},
                {"OPENCL", MNN_FORWARD_OPENCL},
                {"OPENGL", MNN_FORWARD_OPENGL},
                {"VULKAN", MNN_FORWARD_VULKAN},
                {"METAL", MNN_FORWARD_METAL},
                {"TRT", MNN_FORWARD_USER_1},
                {"CUDA", MNN_FORWARD_CUDA},
                {"HIAI", MNN_FORWARD_USER_0},
                {"AUTO", MNN_FORWARD_AUTO}
            };
            auto iter = backend_map.find(backend_name);
            if (iter == backend_map.end()) {
                // backend not support, issue on python level when development
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: backend not support");
                return result;
            }
            config.type = iter->second;
        } else if (backend && PyLong_Check(backend)) {
            config.type = (MNNForwardType)PyLong_AsLong(backend); // {'backend': 1L} for example
        }
        PyObject *numThread = PyDict_GetItemString(dict, "numThread");
        if (numThread) {
            if (!PyLong_Check(numThread)) {
                PyErr_SetString(PyExc_Exception,
                                "PyMNNInterpreter_createSession: numThread must be a integer");
                return result;
            }
            config.numThread = (int)PyLong_AsLong(numThread);
        }

        {
            //precision
            PyObject *obj = PyDict_GetItemString(dict, "precision");
            if (obj) {
                auto obj_name = object2String(obj);
                if (!obj_name.compare("low")) {
                    MNN_PRINT("MNN use low precision\n");
                    backendConfig->precision = MNN::BackendConfig::Precision_Low;
                }

                if (!obj_name.compare("high")) {
                    MNN_PRINT("MNN use high precision\n");
                    backendConfig->precision = MNN::BackendConfig::Precision_High;
                }
            }
        }
    }
    result.first = true;
    return result;
}

static PyObject* PyMNNNN_create_runtime_manager(PyObject *self, PyObject *args) {
    PyObject* dicts = NULL;
    if (!PyArg_ParseTuple(args, "O", &dicts)) {
        std::string mnn_errno = "create_runtime_manager failed 0";
        PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
        return NULL;
    }
    if (!PySequence_Check(dicts)) {
        std::string mnn_errno = "create_runtime_manager failed 1";
        PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
        return Py_None;
    }
    // BackendConfig lifetime management
    std::vector<ScheduleConfig> configs;
    for (auto i = 0; i < PySequence_Size(dicts); ++i) {
        auto config = getScheduleConfig(PySequence_GetItem(dicts, i));
        if (!config.first) {
            return Py_None;
        }
        configs.push_back(config.second.first);
    }

    Executor::RuntimeManager* m_ptr;
    if(configs.size() == 1) {
        m_ptr = Executor::RuntimeManager::createRuntimeManager(configs[0]);
    } else {
        m_ptr = Executor::RuntimeManager::createRuntimeManager(configs);
    }

    if (m_ptr == nullptr) {
        printf("config size:%d\n", configs.size());
        std::string mnn_errno = "create_runtime_manager failed ";
        PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
    }

    auto res = toPyObj(m_ptr);
    return res;
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
        load_module_from_file, "load_module_from_file([string], [string], filename, bool, ...)",
        create_runtime_manager, "create_runtime_manager(dict...)"
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
