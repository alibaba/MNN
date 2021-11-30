// loss Module Start
def_binary(Loss,
    cross_entropy, _CrossEntropy,
    kl, _KLDivergence,
    mse, _MSE,
    mae, _MAE,
    hinge, _Hinge
)
static PyMethodDef PyMNNLoss_methods[] = {
    register_methods(Loss,
        cross_entropy, "cross_entropy loss",
        kl, "kl loss",
        mse, "mse loss",
        mae, "mae loss",
        hinge, "hinge loss"
    )
};
// loss Module End