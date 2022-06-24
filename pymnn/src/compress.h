// compress Module Start
def_enum(Feature_Scale_Method, NN::FeatureScaleStatMethod,
    NN::PerTensor, "PER_TENSOR",
    NN::PerChannel, "PER_CHANNEL"
)
def_enum(Scale_Update_Method, NN::ScaleUpdateMethod,
    NN::Maximum, "MAXIMUM",
    NN::MovingAverage, "MOVING_AVERAGE"
)
static PyObject* PyMNNCompress_train_quant(PyMNNOptimizer *self, PyObject *args, PyObject *kwargs) {
    PyObject *module = nullptr,
             *feature_scale_method = nullptr /* PerTensor */,
             *scale_update_method = nullptr /* MovingAverage */;
    int quant_bits = 8;
    static char *kwlist[] = { "module", "quant_bits", "feature_scale_method", "scale_update_method", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iOO", kwlist, &module, &quant_bits, &feature_scale_method, &scale_update_method)) {
        PyMNN_ERROR("train_quant require args: (Module, |int, Feature_Scale_Method, Scale_Update_Method)");
    }
    auto feature_scale_method_ = feature_scale_method == nullptr ? NN::PerTensor :
                                 toEnum<NN::FeatureScaleStatMethod>(feature_scale_method);
    auto scale_update_method_ = scale_update_method == nullptr ? NN::MovingAverage :
                                toEnum<NN::ScaleUpdateMethod>(scale_update_method);
    return toPyObj(NN::turnQuantize(to_Module(module)->get(), quant_bits, feature_scale_method_, scale_update_method_));
}
static PyMethodDef PyMNNCompress_methods[] = {
    register_methods_kw(Compress,
        train_quant, "train_quant"
    )
};
// compress Module End