#include "util.h"
#include "OpGrad.hpp"

// Optim Module
def_enum(Regularization_Method, ParameterOptimizer::RegularizationMethod,
        RegularizationMethod::L1, "L1",
        RegularizationMethod::L2, "L2",
        RegularizationMethod::L1L2, "L1L2"
        )
// define
def_class_start(Optimizer, ParameterOptimizer)
def_class_getset(
    Optimizer,
    learning_rate, 1,
    momentum, 1,
    momentum2, 1,
    weight_decay, 1,
    eps, 1,
    regularization_method, 1
)
def_class_methods(Optimizer,
    step, "Optimizer step",
    grad, "Grad for variables",
    get_update_graph, "Get Update Graph for parameters",
    save_graph, "Save Update Graph to MNN File"
)
def_class_end(Optimizer, ParameterOptimizer)
// impl
class_basic_new_impl(Optimizer)
class_basic_init_impl(Optimizer)
class_basic_call_impl(Optimizer)
// PyMNNOptimizer getter/setter functions impl
static PyObject* PyMNNOptimizer_getlearning_rate(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<SGD*>(self->ptr)->currentLearningRate();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNOptimizer_getmomentum(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<SGD*>(self->ptr)->getMomentum();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNOptimizer_getmomentum2(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<ADAM*>(self->ptr)->getMomentum2();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNOptimizer_getweight_decay(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<SGD*>(self->ptr)->getWeightDecay();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNOptimizer_geteps(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<ADAM*>(self->ptr)->getEps();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNOptimizer_getregularization_method(PyMNNOptimizer *self, void *closure) {
    if (self->ptr) {
        auto ret = static_cast<SGD*>(self->ptr)->getRegularizationMethod();
        return toPyObj(ret);
    }
    Py_RETURN_NONE;
}
static int PyMNNOptimizer_setlearning_rate(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<SGD*>(self->ptr)->setLearningRate(toFloat(value));
    }
    return 0;
}
static int PyMNNOptimizer_setmomentum(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<SGD*>(self->ptr)->setMomentum(toFloat(value));
    }
    return 0;  
}
static int PyMNNOptimizer_setmomentum2(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<ADAM*>(self->ptr)->setMomentum2(toFloat(value));
    }
    return 0;  
}
static int PyMNNOptimizer_setweight_decay(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<SGD*>(self->ptr)->setWeightDecay(toFloat(value));
    }
    return 0;  
}
static int PyMNNOptimizer_seteps(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<ADAM*>(self->ptr)->setEps(toFloat(value));
    }
    return 0;  
}
static int PyMNNOptimizer_setregularization_method(PyMNNOptimizer *self, PyObject *value, void *closure) {
    if (self->ptr) {
        static_cast<SGD*>(self->ptr)->setRegularizationMethod(toEnum<RegularizationMethod>(value));
    }
    return 0;  
}

static PyObject* _makeTupleFromPairVector(const std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>& values) {
    PyObject* obj0 = PyList_New(values.first.size());
    for (int i = 0; i < values.first.size(); i++) {
        PyList_SetItem(obj0, i, toPyObj(values.first[i]));
    }
    PyObject* obj1 = PyList_New(values.second.size());
    for (int i = 0; i < values.second.size(); i++) {
        PyList_SetItem(obj1, i, toPyObj(values.second[i]));
    }
    PyObject* obj = PyTuple_New(2);
    PyTuple_SetItem(obj, 0, obj0);
    PyTuple_SetItem(obj, 1, obj1);
    return obj;
}
static PyObject* PyMNNOptimizer_grad(PyMNNOptimizer *self, PyObject *args) {
    PyObject* outputs;
    PyObject* outputDiffs;
    PyObject* parameters;
    if (PyArg_ParseTuple(args, "OOO", &outputs, &outputDiffs, &parameters)) {
        if (isVars(outputs) && isVals(outputDiffs) && isVars(parameters)) {
            auto values = OpGrad::gradCommon(toVars(outputs), toVars(outputDiffs), toVars(parameters));
            return _makeTupleFromPairVector(values);
        }
    }
    PyMNN_ERROR("grad require args: ([Var](outputs),[Var](output Diff), [Var](parameters))");
    return Py_None;
}
static PyObject* PyMNNOptimizer_get_update_graph(PyMNNOptimizer *self, PyObject *args) {
    PyObject* parameter;
    PyObject* parameterGrad;
    PyObject* learningRate;
    if (PyArg_ParseTuple(args, "OOO", &parameter, &parameterGrad, &learningRate)) {
        if (isVars(parameter) && isVals(parameterGrad) && isVars(learningRate)) {
            if (self->ptr) {
                auto p = toVars(parameter);
                auto pd = toVars(parameterGrad);
                auto lr = toVars(learningRate);
                auto values = static_cast<ParameterOptimizer*>(self->ptr)->makeParameterUpdateGraphByGrad(p, pd, lr);
                return _makeTupleFromPairVector(values);
            }
        }
    }
    PyMNN_ERROR("get_update_graph require args: ([Var](parameter),[Var](parameter grad), [Var](learningRate))");
    return Py_None;
}
static PyObject* PyMNNOptimizer_save_graph(PyMNNOptimizer *self, PyObject *args) {
    const char* modelFile      = NULL;
    PyObject* outputs;
    PyObject* parameter;
    PyObject* parameterUpdate;
    if (PyArg_ParseTuple(args, "sOOO", &modelFile, &outputs, &parameter, &parameterUpdate)) {
        if (isVars(parameter) && isVals(parameterUpdate) && isVars(outputs)) {
            auto o = toVars(outputs);
            auto p = toVars(parameter);
            auto pu = toVars(parameterUpdate);
            ParameterOptimizer::makeLoopModel(modelFile, o, std::make_pair(p, pu));
            return Py_None;
        }
    }
    PyMNN_ERROR("save_graph require args: ([string](outputPath),[Var](outputs), [Var](parameter),  [Var](parameterUpdate))");
    return Py_None;
}

// PyMNNOptimizer methods impl
static PyObject* PyMNNOptimizer_step(PyMNNOptimizer *self, PyObject *args) {
    PyObject *loss;
    if (!PyArg_ParseTuple(args, "O", &loss)) {
        Py_RETURN_NONE;
    }
    return toPyObj(self->ptr->step(toVar(loss)));
}
static PyObject* PyMNNOptim_Grad(PyObject *self, PyObject *args, PyObject *kwargs) {
    float learning_rate = 1e-3, momentum = 0.9, weight_decay = 0.0;
    std::shared_ptr<Module> m;
    return toPyObj(ParameterOptimizer::createSGD(m, learning_rate, momentum,
                                                 weight_decay, RegularizationMethod::L2));
}
static PyObject* PyMNNOptim_SGD(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *module = nullptr, *method = nullptr /* L2 */;
    float learning_rate = 1e-3, momentum = 0.9, weight_decay = 0.0;
    static char *kwlist[] = { "module", "learning_rate", "momentum", "weight_decay", "regularization_method", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fffO", kwlist, &module, &learning_rate,
                          &momentum, &weight_decay, &method)) {
        PyMNN_ERROR("SGD require args: Module, |float, float, float, RegularizationMethod)");
    }
    auto method_ = method == nullptr ? RegularizationMethod::L2 : toEnum<RegularizationMethod>(method);
    std::shared_ptr<Module> m = *to_Module(module);
    return toPyObj(ParameterOptimizer::createSGD(m, learning_rate, momentum,
                                                 weight_decay, method_));
}
static PyObject* PyMNNOptim_ADAM(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *module = nullptr, *method = nullptr /* L2 */;
    float learning_rate = 1e-3, momentum = 0.9, momentum2 = 0.999,
          weight_decay = 0.0, eps = 1e-8;
    static char *kwlist[] = { "module", "learning_rate", "momentum", "momentum2", "weight_decay", "eps", "regularization_method", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fffffO", kwlist, &module, &learning_rate, &momentum,
                          &momentum2, &weight_decay, &eps, &method)) {
        PyMNN_ERROR("ADAM require args: Module, |float, float, float, float, float, RegularizationMethod)");
    }
    auto method_ = method == nullptr ? RegularizationMethod::L2 : toEnum<RegularizationMethod>(method);
    std::shared_ptr<Module> m = *to_Module(module);
    return toPyObj(ParameterOptimizer::createADAM(m, learning_rate, momentum, momentum2,
                                                  weight_decay, eps, method_));
}
static PyMethodDef PyMNNOptim_methods[] = {
    register_methods_kw(Optim,
        Grad, "Grad Only",
        SGD, "SGD Optimizer",
        ADAM, "ADAM Optimizer"
    )
};
// Optim Module End