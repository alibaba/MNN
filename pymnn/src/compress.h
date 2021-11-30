// compress Module Start
def_enum(Feature_Scale_Method, NN::FeatureScaleStatMethod,
    NN::PerTensor, "PER_TENSOR",
    NN::PerChannel, "PER_CHANNEL"
)
def_enum(Scale_Update_Method, NN::ScaleUpdateMethod,
    NN::Maximum, "MAXIMUM",
    NN::MovingAverage, "MOVING_AVERAGE"
)
static PyObject* PyMNNCompress_train_quant(PyMNNOptimizer *self, PyObject *args) {
    PyObject *module,
             *feature_scale_method = toPyObj(NN::PerTensor),
             *scale_update_method = toPyObj(NN::MovingAverage);
    int quant_bits = 8;
    if (!PyArg_ParseTuple(args, "O|iOO", &module, &quant_bits,
                          &feature_scale_method, &scale_update_method)) {
        Py_RETURN_NONE;
    }
    auto feature_scale_method_ = toEnum<NN::FeatureScaleStatMethod>(feature_scale_method);
    auto scale_update_method_ = toEnum<NN::ScaleUpdateMethod>(scale_update_method);
    return toPyObj(NN::turnQuantize(to_Module(module), quant_bits, feature_scale_method_, scale_update_method_));
}
static PyMethodDef PyMNNCompress_methods[] = {
    register_methods(Compress,
        train_quant, "train_quant"
    )
};
// compress Module End