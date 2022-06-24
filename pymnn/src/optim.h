#include "util.h"

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
    step, "Optimizer step"
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
// PyMNNOptimizer methods impl
static PyObject* PyMNNOptimizer_step(PyMNNOptimizer *self, PyObject *args) {
    PyObject *loss;
    if (!PyArg_ParseTuple(args, "O", &loss)) {
        Py_RETURN_NONE;
    }
    return toPyObj(self->ptr->step(toVar(loss)));
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
        SGD, "SGD Optimizer",
        ADAM, "ADAM Optimizer"
    )
};
// Optim Module End