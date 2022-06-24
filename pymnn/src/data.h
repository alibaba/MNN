// class DataSet def
static PyObject* toPyObj(Example example) {
    // typedef std::pair<std::vector<VARP>, std::vector<VARP>> Example;
    // Example ==> ([Var], [Var])
    PyObject *ret = PyList_New(2);
    PyList_SetItem(ret, 0, toPyObj<VARP, toPyObj>(example.first));
    PyList_SetItem(ret, 1, toPyObj<VARP, toPyObj>(example.second));
    return ret;
}
class DatasetWrapper : public Dataset {
public:
    using Dataset::Dataset;
    DatasetWrapper(PyObject* py_dataset) {
        Py_INCREF(py_dataset);
        this->py_dataset = py_dataset;
    }
    ~DatasetWrapper() {
        if (py_dataset) {
            Py_DECREF(py_dataset);
        }
    }
    Example get(size_t index) override {
        auto getfunc = PyObject_GetAttrString(py_dataset, "__getitem__");
        auto arg = PyTuple_New(1);
        PyTuple_SetItem(arg, 0, PyLong_FromLong(index));
        auto res = PyEval_CallObject(getfunc, arg);
        Py_DECREF(arg);
        Py_DECREF(getfunc);
        // res to Example
        auto py_example = PyTuple_GetItem(res, 0);
        auto py_example_second = PyTuple_GetItem(res, 1);
        auto example = std::make_pair(
            toVars(py_example),
            toVars(py_example_second)
        );
        Py_DECREF(res);
        return example;
    }
    size_t size() override {
        auto sizefunc = PyObject_GetAttrString(py_dataset, "__len__");
        auto res = PyEval_CallObject(sizefunc, NULL);
        Py_DECREF(sizefunc);
        auto size = toInt(res);
        Py_DECREF(res);
        return size;
    }
private:
    PyObject *py_dataset = nullptr;
};

typedef struct {
    PyObject_HEAD
    std::shared_ptr<Dataset>* ptr;
} PyMNNDataset;

static PyObject* PyMNNDataset_new(struct _typeobject *type, PyObject *args, PyObject *kwds) {
    PyMNNDataset* self = (PyMNNDataset *)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int PyMNNDataset_init(PyMNNDataset *self, PyObject *args, PyObject *kwds) {
    self->ptr = new std::shared_ptr<Dataset>(new DatasetWrapper((PyObject*)self));
    return 0;
}

static void PyMNNDataset_dealloc(PyMNNDataset *self) {
    if (self->ptr) {
        // delete self->ptr;
        self->ptr->reset();
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject PyMNNDatasetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Dataset",                                /*tp_name*/
    sizeof(PyMNNDataset),                     /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNDataset_dealloc,         /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    // Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE, /*tp_flags*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Dataset",                                /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)PyMNNDataset_init,              /* tp_init */
    0,                                        /* tp_alloc */
    PyMNNDataset_new,                         /* tp_new */
};

static std::shared_ptr<Dataset> toDataset(PyObject* m) {
    return *((PyMNNDataset*)m)->ptr;
}

def_class_register(Dataset)

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
class_basic_call_impl(DataLoader)
class_basic_init_impl(DataLoader)
static PyObject* PyMNNDataLoader_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject* dataset = nullptr;
    int batch_size, num_workers = 0;
    int shuffle = 1;
    static char *kwlist[] = { "dataset", "batch_size", "shuffle", "num_workers", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", kwlist, &dataset, &batch_size, &shuffle, &num_workers)) {
        PyMNN_ERROR("DataLoader require args: Dataset, int, |int, int)");
    }
    std::shared_ptr<Dataset> dataset_ = std::move(toDataset(dataset));
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