// class DataSet def
static PyObject* toPyObj(Example example) {
    // typedef std::pair<std::vector<VARP>, std::vector<VARP>> Example;
    // Example ==> ([Var], [Var])
    PyObject *ret = PyList_New(2);
    PyList_SetItem(ret, 0, toPyObj<VARP, toPyObj>(example.first));
    PyList_SetItem(ret, 1, toPyObj<VARP, toPyObj>(example.second));
    return ret;
}
def_class_start(Dataset, Dataset)
def_class_without_getset(Dataset)
def_class_methods(Dataset,
    __getitem__, "get item: []",
    __len__, "get length: len()"
)
def_class_end(Dataset, Dataset)
// class DataSet impl
class_basic_new_impl(Dataset)
static PyObject* PyMNNDataset___getitem__(PyMNNDataset *self, PyObject *args) {
    int index;
    if (!PyArg_ParseTuple(args, "i", &index)) {
        Py_RETURN_NONE;
    }
    return toPyObj(self->ptr->get(index));
}
static PyObject* PyMNNDataset___len__(PyMNNDataset *self, PyObject *args) {
    return toPyObj((int)self->ptr->size());
}

// class DataLoader def
def_class_start(DataLoader, DataLoader)
def_class_getset(
    DataLoader,
    iter_number, 0,
    size, 0   
)
def_class_methods(DataLoader,
    reset, "reset DataLoader",
    next, "get next DataLoader"
)
def_class_end(DataLoader, DataLoader)
// class DataLoader impl
static PyObject* PyMNNDataLoader_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject* dataset;
    int batch_size, num_workers = 0;
    int shuffle = 1;
    if (!PyArg_ParseTuple(args, "Oi|ii", &dataset, &batch_size, &shuffle, &num_workers)) {
        Py_RETURN_NONE;
    }
    std::shared_ptr<Dataset> dataset_(toDataset(dataset));
    PyMNNDataLoader *self = (PyMNNDataLoader *)type->tp_alloc(type, 0);
    self->ptr = DataLoader::makeDataLoader(dataset_, batch_size, true, shuffle, num_workers);
    return (PyObject*)self;
}
static PyObject* PyMNNDataLoader_getiter_number(PyMNNDataLoader *self, void *closure) {
    if (self->ptr) {
        return toPyObj((int)self->ptr->iterNumber());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNDataLoader_getsize(PyMNNDataLoader *self, void *closure) {
    if (self->ptr) {
        return toPyObj((int)self->ptr->size());
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNDataLoader_reset(PyMNNDataLoader *self, PyObject *args) {
    self->ptr->reset();
    Py_RETURN_NONE;
}
static PyObject* PyMNNDataLoader_next(PyMNNDataLoader *self, PyObject *args) {
    return toPyObj(self->ptr->next()[0]);
}