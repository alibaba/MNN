#include "util.h"
#ifdef PYMNN_INTERNAL_SERVING
#include <MNN/AutoTime.hpp>
#include <MNN/MNNForwardType.h>
#include "internal/monitor_service.h"
#include "internal/verify_service.h"
#endif

// NN Module Start
def_class_smart_start(_Module, Module)
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
    get_info, "get module info",
    train, "set is_training",
    load_parameters, "load parameters",
    clear_cache, "clear cache",
    _register_submodules, "register submodules",
    _add_parameter, "add parameter",
    clone, "clone module"
)
def_class_smart_end(_Module, Module)

// NN RuntimeManager Start
def_class_smart_start(RuntimeManager, Executor::RuntimeManager)
def_class_methods(RuntimeManager,
    set_cache, "set cache",
    set_external, "set external",
    update_cache, "update cache",
    set_mode, "set mode",
    set_hint, "set hint"
)
def_class_without_getset(RuntimeManager)
def_class_smart_end(RuntimeManager, Executor::RuntimeManager)
class_basic_call_impl(RuntimeManager)

static PyObject* load_module(PyObject *runtime_manager, PyObject *inputs, PyObject *outputs,
                             MNNForwardType backend, MemoryMode memory_mode, PowerMode power_mode, PrecisionMode precision_mode,
                             const char* file_name, int dynamic,
                             int shape_mutable, int rearrange, int thread_num) {

    BackendConfig backend_config;
    backend_config.memory = memory_mode;
    backend_config.power = power_mode;
    backend_config.precision = precision_mode;

    Module::BackendInfo backend_info;
    backend_info.type = backend;
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

    Module* m_ptr = Module::load(toStrings(inputs), toStrings(outputs), converted_file_name.data(), rt_mgr, &config);
    if (m_ptr == nullptr) {
        std::string mnn_errno = "load_module_from_file failed ";
        mnn_errno = mnn_errno + std::string(file_name);
        PyErr_SetString(PyExc_Exception, mnn_errno.c_str());
        Py_RETURN_NONE;
    }

    return toPyObj(m_ptr);
}

class_basic_init_impl(_Module)
static PyObject* PyMNN_Module_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyMNN_Module *self = (PyMNN_Module *)type->tp_alloc(type, 0);
    self->ptr = new std::shared_ptr<Module>(Module::createEmpty({}));
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
        return toPyObj((*(self->ptr))->name());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_getis_training(PyMNN_Module *self, void *closure) {
    if (self->ptr) {
        return toPyObj((*(self->ptr))->getIsTraining());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_getparameters(PyMNN_Module *self, void *closure) {
    if (self->ptr) {
        return toPyObj<VARP, toPyObj>((*(self->ptr))->parameters());
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
        return toPyObj<VARP, toPyObj>((*(self->ptr))->onForward(toVars(input)));
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
        return toPyObj((*(self->ptr))->forward(toVar(input)));
#endif
    }
    PyMNN_ERROR("PyMNN_Module_forward: args must be Var/[Var].");
}
static PyObject* PyMNN_Module_get_info(PyMNN_Module *self, PyObject *args) {
    auto m = (*(self->ptr));
    auto info = m->getInfo();
    if (nullptr == info) {
        PyMNN_ERROR("The module can't get info");
        Py_RETURN_NONE;
    }
    auto res = PyDict_New();
    PyDict_SetItemString(res, "version", char2Object(info->version.c_str()));
    {
        auto names = PyList_New(info->inputNames.size());
        for (int i=0; i<info->inputNames.size(); ++i) {
            PyList_SetItem(names, i, char2Object(info->inputNames[i].c_str()));
        }
        PyDict_SetItemString(res, "inputNames", names);
    }
    {
        auto names = PyList_New(info->outputNames.size());
        for (int i=0; i<info->outputNames.size(); ++i) {
            PyList_SetItem(names, i, char2Object(info->outputNames[i].c_str()));
        }
        PyDict_SetItemString(res, "outputNames", names);
    }
    {
        auto inputs = PyList_New(info->inputs.size());
        for (int i=0; i<info->inputs.size(); ++i) {
            auto& v = info->inputs[i];
            auto var = MNN::Express::_Input(v.dim, v.order, v.type);
            PyList_SetItem(inputs, i, toPyObj(var));
        }
        PyDict_SetItemString(res, "inputs", inputs);
    }
    return res;
}
static PyObject* PyMNN_Module_onForward(PyMNN_Module *self, PyObject *args) {
    PyObject *inputs;
    if (!PyArg_ParseTuple(args, "O", &inputs)) {
        Py_RETURN_NONE;
    }
#ifdef PYMNN_INTERNAL_SERVING
    int status = 0;
    Timer timer;
    auto vars = (*(self->ptr))->onForward(toVars(inputs));
    if (vars.empty()) {
        PyMNN_ERROR("module onForward occur error.");
        status = -1;
    }

    (void) MonitorService::GetInstance().EventTrack(self->ptr->get(), timer, status, "PyMNN_Module_onForward");
    return toPyObj<VARP, toPyObj>(vars);
#else
    return toPyObj<VARP, toPyObj>((*(self->ptr))->onForward(toVars(inputs)));
#endif
}

static PyObject* PyMNN_Module_call(PyObject *self, PyObject *args, PyObject *kwds) {
    return PyMNN_Module_forward((PyMNN_Module*)self, args);
}

static PyObject* PyMNN_Module_set_name(PyMNN_Module *self, PyObject *args) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        Py_RETURN_NONE;
    }
    (*(self->ptr))->setName(name);
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_train(PyMNN_Module *self, PyObject *args) {
    int isTraining;
    if (!PyArg_ParseTuple(args, "i", &isTraining)) {
        Py_RETURN_NONE;
    }
    (*(self->ptr))->setIsTraining(isTraining);
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module_load_parameters(PyMNN_Module *self, PyObject *args) {
    PyObject* parameters;
    if (!PyArg_ParseTuple(args, "O", &parameters)) {
        Py_RETURN_NONE;
    }
    return toPyObj((*(self->ptr))->loadParameters(toVars(parameters)));
}
static PyObject* PyMNN_Module_clear_cache(PyMNN_Module *self, PyObject *args) {
    (*(self->ptr))->clearCache();
    Py_RETURN_NONE;
}
std::shared_ptr<Module> toSharedModule(PyObject* obj)  {
    return *to_Module(obj);
}
static PyObject* PyMNN_Module__register_submodules(PyMNN_Module *self, PyObject *args) {
    PyObject *children;
    if (!PyArg_ParseTuple(args, "O", &children)) {
        Py_RETURN_NONE;
    }
    (*(self->ptr))->registerModel(toVec<std::shared_ptr<Module>, toSharedModule>(children));
    Py_RETURN_NONE;
}
static PyObject* PyMNN_Module__add_parameter(PyMNN_Module *self, PyObject *args) {
    PyObject *parameter;
    if (!PyArg_ParseTuple(args, "O", &parameter)) {
        Py_RETURN_NONE;
    }
    return toPyObj((*(self->ptr))->addParameter(toVar(parameter)));
}
static PyObject* PyMNN_Module_clone(PyMNN_Module *self, PyObject *args) {
    return toPyObj((*(self->ptr))->clone((*(self->ptr)).get()));
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
    PyObject *inputs, *outputs, *runtime_manager,
             *backend = nullptr /* MNN_FORWARD_CPU */,
             *memory_mode = nullptr /* MemoryMode::Memory_Normal */,
             *power_mode = nullptr /* PowerMode::Power_Normal */,
             *precision_mode = nullptr /* PrecisionMode::Precision_Normal */;
    const char* file_name;
    int dynamic, shape_mutable, rearrange;
    int thread_num;
    if (!PyArg_ParseTuple(args, "OOOsiiiOOOOi", &runtime_manager, &inputs, &outputs, &file_name, &dynamic,
                          &shape_mutable, &rearrange, &backend, &memory_mode,
                          &power_mode, &precision_mode, &thread_num)) {
        printf("PyArg_ParseTuple Error\n");
        return NULL;
    }

    return load_module(runtime_manager, inputs, outputs,
                           PARSE(backend, MNN_FORWARD_CPU, toEnum<MNNForwardType>),
                           PARSE(memory_mode, MemoryMode::Memory_Normal, toEnum<MemoryMode>),
                           PARSE(power_mode, PowerMode::Power_Normal, toEnum<PowerMode>),
                           PARSE(precision_mode, PrecisionMode::Precision_Normal, toEnum<PrecisionMode>),
                           file_name, dynamic,
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
static PyObject* PyMNNRuntimeManager_set_external(PyMNNRuntimeManager *self, PyObject *args) {
    char *path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_Exception,
                        "PyMNNRuntimeManager_set_external: Not string input");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    std::string externalPath = path;
    (*(self->ptr))->setExternalFile(externalPath);
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
    ScheduleConfig config[MAX_CONFIG_SIZE];
    BackendConfig backendConfig[MAX_CONFIG_SIZE];

    if(PySequence_Size(dicts) > MAX_CONFIG_SIZE) {
        MNN_PRINT("Error: MNN support max ScheduleConfig size is %d\n", MAX_CONFIG_SIZE);
        return Py_None;
    }
    for (auto i = 0; i < PySequence_Size(dicts); ++i) {
        backendConfig[i].sharedContext = nullptr;
        config[i].backendConfig = &backendConfig[i];
        bool ret = getScheduleConfig(PySequence_GetItem(dicts, i), config[i]);
        if (!ret) {
            return Py_None;
        }
        configs.push_back(config[i]);
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
    PyObject *backend = nullptr /* MNN_FORWARD_CPU */,
             *memory_mode = nullptr /* MemoryMode::Memory_Normal */,
             *power_mode = nullptr /* PowerMode::Power_Normal */,
             *precision_mode = nullptr /* PrecisionMode::Precision_Normal */;
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

    return load_module(inputs, outputs,
                           PARSE(backend, MNN_FORWARD_CPU, toEnum<MNNForwardType>),
                           PARSE(memory_mode, MemoryMode::Memory_Normal, toEnum<MemoryMode>),
                           PARSE(power_mode, PowerMode::Power_Normal, toEnum<PowerMode>),
                           PARSE(precision_mode, PrecisionMode::Precision_Normal, toEnum<PrecisionMode>),
                           file_name, dynamic,
                           shape_mutable,  rearrange,  thread_num);

}
#endif

#ifdef PYMNN_TRAIN_API
static PyObject* PyMNNNN_conv(PyObject *self, PyObject *args, PyObject* kwargs) {
    INTS default_1 = {1, 1}, default_0 = {0, 0};
    int in_channel, out_channel;
    PyObject *kernel_size,
             *stride = nullptr /* default_1 */,
             *padding = nullptr /* default_0 */,
             *dilation = nullptr /* default_1 */,
             *padding_mode = nullptr /* PaddingMode::VALID */;
    int depthwise = 0, bias = 1;
    static char *kwlist[] = { "in_channels", "out_channels", "kernel_size", "stride", "padding",
                              "dilation", "depthwise", "bias", "padding_mode", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO|OOOiiO", kwlist, &in_channel, &out_channel, &kernel_size,
                          &stride, &padding, &dilation, &depthwise, &bias, &padding_mode)) {
        PyMNN_ERROR("conv require args: int, int, [int], |[int], [int], [int], bool, bool, PaddingMode)");
    }
    NN::ConvOption option;
    option.channel = {in_channel, out_channel};
    option.kernelSize = toInts(kernel_size);
    auto stride_ = PARSE(stride, default_1, toInts);
    auto padding_ = PARSE(padding, default_0, toInts);
    auto dilation_ = PARSE(dilation, default_1, toInts);
    if (!stride_.empty()) {
        option.stride = stride_;
    }
    option.padMode = PARSE(padding_mode, PaddingMode::VALID, toEnum<PaddingMode>);
    if (!padding_.empty()) {
        option.pads = padding_;
    }
    if (!dilation_.empty()) {
        option.dilate = dilation_;
    }
    option.depthwise = depthwise;
    return toPyObj(NN::Conv(std::move(option), bias));
}
static PyObject* PyMNNNN_linear(PyObject *self, PyObject *args, PyObject* kwargs) {
    int in_channel, out_channel;
    int bias = 1;
    static char *kwlist[] = { "in_channels", "out_channels", "bias", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|i", kwlist, &in_channel, &out_channel, &bias)) {
        PyMNN_ERROR("linear require args: int, int, |bool)");
    }
    return toPyObj(NN::Linear(in_channel, out_channel, bias));
}
static PyObject* PyMNNNN_batch_norm(PyObject *self, PyObject *args, PyObject* kwargs) {
    int channels, dims = 4;
    float momentum = 0.99, epsilon = 1e-5;
    static char *kwlist[] = { "channels", "dims", "momentum", "epsilon", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|iff", kwlist, &channels, &dims, &momentum, &epsilon)) {
        PyMNN_ERROR("batch_norm require args: int, |int, float, float)");
    }
    return toPyObj(NN::BatchNorm(channels, dims, momentum, epsilon));
}
static PyObject* PyMNNNN_dropout(PyObject *self, PyObject *args, PyObject* kwargs) {
    float dropout_ratio;
    static char *kwlist[] = { "dropout_ratio", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f", kwlist, &dropout_ratio)) {
        PyMNN_ERROR("dropout require args: float)");
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
    register_methods_kw(NN,
        conv, "conv Module",
        linear, "linear Module",
        batch_norm, "batch_norm Module",
        dropout, "dropout module"
    )
#endif
};
// NN Module End
