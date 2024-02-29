#ifdef PYMNN_EXPR_API

// Expr Module Start
def_enum(data_format, Dimensionformat,
        NHWC, "NHWC",
        NC4HW4, "NC4HW4",
        NCHW, "NCHW"
        )
def_enum(dtype, DType,
        DType_FLOAT, "float",
        DType_DOUBLE, "double",
        DType_INT32, "int",
        DType_INT64, "int64",
        DType_UINT8, "uint8",
        DType_INT8, "int8"
        )
def_enum(Padding_Mode, PaddingMode,
        CAFFE, "CAFFE",
        VALID, "VALID",
        SAME, "SAME"
        )
def_enum(PadValue_Mode, MNN::Express::PadValueMode,
        CONSTANT, "CONSTANT",
        REFLECT, "REFLECT",
        SYMMETRIC, "SYMMETRIC"
        )
def_enum(Pooling_Mode, PoolingMode,
        MAXPOOL, "MAXPOOL",
        AVEPOOL, "AVEPOOL"
        )
def_enum(Interp_Method, InterpolationMethod,
        BILINEAR, "BILINEAR",
        NEAREST, "NEAREST"
        )
def_enum(Backend, MNNForwardType,
        MNN_FORWARD_CPU, "CPU",
        MNN_FORWARD_OPENCL, "OPENCL",
        MNN_FORWARD_OPENGL, "OPENGL",
        MNN_FORWARD_VULKAN, "VULKAN",
        MNN_FORWARD_METAL, "METAL",
        MNN_FORWARD_USER_1, "TRT",
        MNN_FORWARD_CUDA, "CUDA",
        MNN_FORWARD_USER_0, "HIAI"
        )
using MemoryMode = BackendConfig::MemoryMode;
using PowerMode = BackendConfig::PowerMode;
using PrecisionMode = BackendConfig::PrecisionMode;
def_enum(MemoryMode, MemoryMode,
        MemoryMode::Memory_Normal, "Normal",
        MemoryMode::Memory_High, "High",
        MemoryMode::Memory_Low, "Low"
        )
def_enum(PowerMode, PowerMode,
        PowerMode::Power_Normal, "Normal",
        PowerMode::Power_High, "High",
        PowerMode::Power_Low, "Low"
        )
def_enum(PrecisionMode, PrecisionMode,
        PrecisionMode::Precision_Normal, "Normal",
        PrecisionMode::Precision_High, "High",
        PrecisionMode::Precision_Low, "Low",
        PrecisionMode::Precision_Low_BF16, "Low_BF16"
        )
// class Var
typedef struct {
    PyObject_HEAD
    VARP* var;
    int iter_index;
} PyMNNVar;
static PyObject* PyMNNVar_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void PyMNNVar_dealloc(PyMNNVar *self);
static PyObject* PyMNNVar_repr(PyObject *self);
// property: getter & setter
static PyObject* PyMNNVar_getshape(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getvalid(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getdata_format(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getdtype(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getsize(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getname(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getndim(PyMNNVar *self, void *closure);
static PyObject* PyMNNVar_getptr(PyMNNVar *self, void *closure);
static int PyMNNVar_setname(PyMNNVar *self, PyObject *value, void *closure);
#ifdef BUILD_OPTYPE
static PyObject* PyMNNVar_getop_type(PyMNNVar *self, void *closure);
#endif
static PyObject* PyMNNVar_getinputs(PyMNNVar *self, void *closure);
// method
static PyObject* PyMNNVar_fix_as_placeholder(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_fix_as_const(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_fix_as_trainable(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_close(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_copy_from(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_set_inputs(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_replace(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_reorder(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_resize(PyMNNVar *self, PyObject *args);
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNVar_read(PyMNNVar *self, PyObject *args);
#endif
static PyObject* PyMNNVar_read_as_tuple(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_write(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_sync(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_set_device_ptr(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_copy_to_device_ptr(PyMNNVar *self, PyObject *args);
static PyObject* PyMNNVar_add(PyMNNVar *self, PyObject *args);
static PyGetSetDef PyMNNVar_getsetters[] = {
    {"shape", (getter)PyMNNVar_getshape, NULL, "shape", NULL},
    {"valid", (getter)PyMNNVar_getvalid, NULL, "valid", NULL},
    {"data_format", (getter)PyMNNVar_getdata_format, NULL, "data_format", NULL},
    {"dtype", (getter)PyMNNVar_getdtype, NULL, "dtype", NULL},
    {"size", (getter)PyMNNVar_getsize, NULL, "size", NULL},
    {"name", (getter)PyMNNVar_getname, (setter)PyMNNVar_setname, "name", NULL},
#ifdef BUILD_OPTYPE
    {"op_type", (getter)PyMNNVar_getop_type, NULL, "op_type", NULL},
#endif
    {"inputs", (getter)PyMNNVar_getinputs, NULL, "inputs", NULL},
    {"ndim", (getter)PyMNNVar_getndim, NULL, "ndim", NULL},
    {"ptr", (getter)PyMNNVar_getptr, NULL, "ptr", NULL},
    {NULL}  /* Sentinel */
};
static PyMethodDef PyMNNVar_methods[] = {
    {"fix_as_placeholder", (PyCFunction)PyMNNVar_fix_as_placeholder, METH_VARARGS, "fix as input"},
    {"fix_as_const", (PyCFunction)PyMNNVar_fix_as_const, METH_VARARGS, "fix as const"},
    {"fix_as_trainable", (PyCFunction)PyMNNVar_fix_as_trainable, METH_VARARGS, "fix as trainable"},
    {"close", (PyCFunction)PyMNNVar_close, METH_VARARGS, "close"},
    {"copy_from", (PyCFunction)PyMNNVar_copy_from, METH_VARARGS, "copy from arg"},
    {"set_inputs", (PyCFunction)PyMNNVar_set_inputs, METH_VARARGS, "set inputs"},
    {"replace", (PyCFunction)PyMNNVar_replace, METH_VARARGS, "replace with arg"},
    {"reorder", (PyCFunction)PyMNNVar_reorder, METH_VARARGS, "reorder with arg"},
    {"resize", (PyCFunction)PyMNNVar_resize, METH_VARARGS, "resize with arg shape"},
#ifdef PYMNN_NUMPY_USABLE
    {"read", (PyCFunction)PyMNNVar_read, METH_VARARGS, "read data(numpy)"},
#endif
    {"read_as_tuple", (PyCFunction)PyMNNVar_read_as_tuple, METH_VARARGS, "read data(tuple)"},
    {"write", (PyCFunction)PyMNNVar_write, METH_VARARGS, "write data"},
    {"sync", (PyCFunction)PyMNNVar_sync, METH_VARARGS, "sync var data"},
    {"set_device_ptr", (PyCFunction)PyMNNVar_set_device_ptr, METH_VARARGS, "set_device_ptr data"},
    {"copy_to_device_ptr", (PyCFunction)PyMNNVar_copy_to_device_ptr, METH_VARARGS, "copy_to_device_ptr data"},

    
    {NULL}  /* Sentinel */
};
static PyObject* PyMNNVar_add(PyObject*, PyObject*);
static PyObject* PyMNNVar_subtract(PyObject*, PyObject*);
static PyObject* PyMNNVar_multiply(PyObject*, PyObject*);
static PyObject* PyMNNVar_true_divide(PyObject*, PyObject*);
static PyObject* PyMNNVar_floor_divide(PyObject*, PyObject*);
static PyObject* PyMNNVar_power(PyObject*, PyObject*, PyObject*);
static PyObject* PyMNNVar_negative(PyObject*);
static PyObject* PyMNNVar_absolute(PyObject*);
static Py_ssize_t PyMNNVar_length(PyObject*);
static PyObject* PyMNNVar_subscript(PyObject*, PyObject*);
static int PyMNNVar_ass_subscript(PyObject*, PyObject*, PyObject*);
static PyObject* PyMNNVar_iter(PyObject*);
static PyObject* PyMNNVar_iternext(PyObject*);
#if PY_MAJOR_VERSION >= 3
static PyNumberMethods PyMNNVar_as_number = {
    PyMNNVar_add,           /*nb_add*/
    PyMNNVar_subtract,      /*nb_subtract*/
    PyMNNVar_multiply,      /*nb_multiply*/
    0,                      /*nb_remainder*/
    0,                      /*nb_divmod*/
    PyMNNVar_power,         /*nb_power*/
    PyMNNVar_negative,      /*nb_negative*/
    0,                      /*nb_positive*/
    PyMNNVar_absolute,      /*nb_absolute*/
    0,                      /*nb_bool*/
    0,                      /*nb_invert*/
    0,                      /*nb_lshift*/
    0,                      /*nb_rshift*/
    0,                      /*nb_and*/
    0,                      /*nb_xor*/
    0,                      /*nb_or*/
    0,                      /*nb_int*/
    0,                      /*nb_reserved*/
    0,                      /*nb_float*/
    0,                      /*nb_inplace_add*/
    0,                      /*nb_inplace_subtract*/
    0,                      /*nb_inplace_multiply*/
    0,                      /*nb_inplace_remainder*/
    0,                      /*nb_inplace_power*/
    0,                      /*nb_inplace_lshift*/
    0,                      /*nb_inplace_rshift*/
    0,                      /*nb_inplace_and*/
    0,                      /*nb_inplace_xor*/
    0,                      /*nb_inplace_or*/
    PyMNNVar_floor_divide,  /*nb_floor_divide*/
    PyMNNVar_true_divide,   /*nb_true_divide*/
    0,                      /*nb_inplace_floor_divide*/
    0,                      /*nb_inplace_true_divide*/
    0,                      /*nb_index*/
    0,                      /*nb_matrix_multiply*/
    0,                      /*nb_inplace_matrix_multiply*/
};
#else
static PyNumberMethods PyMNNVar_as_number = {
    PyMNNVar_add,           /*nb_add*/
    PyMNNVar_subtract,      /*nb_subtract*/
    PyMNNVar_multiply,      /*nb_multiply*/
    PyMNNVar_true_divide,   /*nb_divide*/
    0,                      /*nb_remainder*/
    0,                      /*nb_divmod*/
    PyMNNVar_power,         /*nb_power*/
    PyMNNVar_negative,      /*nb_negative*/
    0,                      /*nb_positive*/
    PyMNNVar_absolute,      /*nb_absolute*/
    0,                      /*nb_nonzero*/
    0,                      /*nb_invert*/
    0,                      /*nb_lshift*/
    0,                      /*nb_rshift*/
    0,                      /*nb_and*/
    0,                      /*nb_xor*/
    0,                      /*nb_or*/
    0,                      /*nb_coerce*/
    0,                      /*nb_int*/
    0,                      /*nb_long*/
    0,                      /*nb_float*/
    0,                      /*nb_oct*/
    0,                      /*nb_hex*/
    0,                      /*nb_inplace_add*/
    0,                      /*nb_inplace_subtract*/
    0,                      /*nb_inplace_multiply*/
    0,                      /*nb_inplace_divide*/
    0,                      /*nb_inplace_remainder*/
    0,                      /*nb_inplace_power*/
    0,                      /*nb_inplace_lshift*/
    0,                      /*nb_inplace_rshift*/
    0,                      /*nb_inplace_and*/
    0,                      /*nb_inplace_xor*/
    0,                      /*nb_inplace_or*/
    PyMNNVar_floor_divide,  /*nb_floor_divide*/
    PyMNNVar_true_divide,   /*nb_true_divide*/
    0,                      /*nb_inplace_floor_divide*/
    0,                      /*nb_inplace_true_divide*/
    0,                      /*nb_index*/
};
#endif
static PyMappingMethods PyMNNVar_as_mapping = {
    PyMNNVar_length,        /*mp_length*/
    PyMNNVar_subscript,     /*mp_subscript*/
    PyMNNVar_ass_subscript, /*mp_ass_subscript*/
};
PyObject *PyMNNVar_richcompare(PyObject *self, PyObject *other, int op);
static PyTypeObject PyMNNVarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Var",                                    /*tp_name*/
    sizeof(PyMNNVar),                         /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)PyMNNVar_dealloc,             /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    PyMNNVar_repr,                            /*tp_repr*/
    &PyMNNVar_as_number,                      /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    &PyMNNVar_as_mapping,                     /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    PyMNNVar_repr,                            /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
#if PY_MAJOR_VERSION < 3
    // this flag `tp_as_number` accept arguments of arbitrary object types in py2
    | Py_TPFLAGS_CHECKTYPES
#endif
    ,                                         /*tp_flags*/
    "MNN Var objects",                        /*tp_doc*/
    0,                                        /*tp_traverse*/
    0,                                        /*tp_clear*/
    &PyMNNVar_richcompare,                    /*tp_richcompare*/
    0,                                        /*tp_weaklistoffset*/
    &PyMNNVar_iter,                           /*tp_iter*/
    &PyMNNVar_iternext,                       /*tp_iternext*/
    PyMNNVar_methods,                         /*tp_methods*/
    0,                                        /*tp_members*/
    PyMNNVar_getsetters,                      /*tp_getset*/
    0,                                        /*tp_base*/
    0,                                        /*tp_dict*/
    0,                                        /*tp_descr_get*/
    0,                                        /*tp_descr_set*/
    0,                                        /*tp_dictoffset*/
    0,                                        /*tp_init*/
    0,                                        /*tp_alloc*/
    PyMNNVar_new,                             /*tp_new*/
};
// helper functions
static PyMNNVar* getVar() {
    PyMNNVar *var = (PyMNNVar *)PyObject_CallObject((PyObject*)PyType_FindTLSType(&PyMNNVarType), NULL);
    var->var = new VARP;
    return var;
}
static PyObject* toPyObj(VARP var) {
    auto ret = getVar();
    *(ret->var) = var;
    return (PyObject*)ret;
}
static bool isVar(PyObject* var) {
    return isInt(var) || isInts(var) ||
           isFloat(var) || isFloats(var) ||
           Py_TYPE(var) == PyType_FindTLSType(&PyMNNVarType);
}
static bool isVars(PyObject* var) {
    return isVec<isVar>(var);
}
static VARP toVar(PyObject* var) {
    // accept int/[int]/float/[float]/var
    if (isInt(var)) {
        std::unique_ptr<int[]> ptr(new int[1]);
        ptr[0] = toInt(var);
        return _Const(ptr.get(), {}, NCHW, halide_type_of<int32_t>());
    }
    if (isInts(var)) {
        auto ints = toInts(var);
        return _Const(ints.data(), {static_cast<int>(ints.size())}, NCHW, halide_type_of<int32_t>());
    }
    if (isFloat(var)) {
        std::unique_ptr<float[]> ptr(new float[1]);
        ptr[0] = toFloat(var);
        return _Const(ptr.get(), {}, NCHW, halide_type_of<float>());
    }
    if (isFloats(var)) {
        auto floats = toFloats(var);
        return _Const(floats.data(), {static_cast<int>(floats.size())}, NCHW, halide_type_of<float>());
    }
    return *(((PyMNNVar*)var)->var);
}
static VARPS toVars(PyObject* vars) {
    VARPS varps;
    if (PyList_Check(vars)) {
        size_t size = PyList_Size(vars);
        varps.resize(size);
        for (int i = 0; i < size; i++) {
            varps[i] = toVar(PyList_GetItem(vars, i));
        }
    }
    return varps;
}
std::pair<VARP, VARP> toVarPair(PyObject* l, PyObject* r, bool fp = false) {
    if (!isVar(l) || !isVar(r)) {
        PyMNN_ERROR_LOG("binary lhs and rhs must be Var.");
    }
    auto varl = toVar(l);
    auto varr = toVar(r);
    auto dtypel = varl->getInfo()->type;
    auto dtyper = varr->getInfo()->type;
    if (fp) {
        if (dtypel != halide_type_of<float>()) {
            varl = Express::_Cast(varl, dtyper);
        }
        if (dtyper != halide_type_of<float>()) {
            varr = Express::_Cast(varr, dtypel);
        }
    } else if (dtypel != dtyper) {
        if (dtypel.code == halide_type_float) {
            varr = Express::_Cast(varr, dtypel);
        } else if (dtyper.code == halide_type_float) {
            varl = Express::_Cast(varl, dtyper);
        } else if (dtypel.bits > dtyper.bits) {
            varr = Express::_Cast(varr, dtypel);
        } else {
            varl = Express::_Cast(varl, dtyper);
        }
    }
    return std::make_pair(varl, varr);
}
PyObject *PyMNNVar_richcompare(PyObject *l, PyObject *r, int op) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    VARP res;
    switch (op) {
        case Py_LT:
            res = Express::_Less(vl, vr);
            break;
        case Py_LE:
            res = Express::_LessEqual(vl, vr);
            break;
        case Py_EQ:
            res = Express::_Equal(vl, vr);
            break;
        case Py_NE:
            res = Express::_NotEqual(vl, vr);
            break;
        case Py_GT:
            res = Express::_Greater(vl, vr);
            break;
        case Py_GE:
            res = Express::_GreaterEqual(vl, vr);
            break;
        default:
            Py_RETURN_NONE;
    }
    return toPyObj(res);
}
static PyObject* PyMNNVar_add(PyObject* l, PyObject* r) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_Add(vl, vr));
}
static PyObject* PyMNNVar_subtract(PyObject* l, PyObject* r) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_Subtract(vl, vr));
}
static PyObject* PyMNNVar_multiply(PyObject* l, PyObject* r) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_Multiply(vl, vr));
}
static PyObject* PyMNNVar_true_divide(PyObject* l, PyObject* r) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_Divide(vl, vr));
}
static PyObject* PyMNNVar_floor_divide(PyObject* l, PyObject* r) {
    auto lr = toVarPair(l, r);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_FloorDiv(vl, vr));
}
static PyObject* PyMNNVar_power(PyObject* x, PyObject* y, PyObject* z) {
    auto lr = toVarPair(x, y, true);
    auto vl = lr.first, vr = lr.second;
    return toPyObj(Express::_Pow(vl, vr));
}
static PyObject* PyMNNVar_absolute(PyObject* x) {
    return toPyObj(Express::_Abs(toVar(x)));
}
static PyObject* PyMNNVar_negative(PyObject* x) {
    return toPyObj(Express::_Negative(toVar(x)));
}
static Py_ssize_t PyMNNVar_length(PyObject* x) {
    Py_ssize_t size = 0;
    auto info = toVar(x)->getInfo();
    if(info && !info->dim.empty()) {
        size = info->dim[0];
    }
    return size;
}

static void dealSlice(PyObject* slice, std::vector<int>& begin, std::vector<int>& end, std::vector<int>& strides,
                      int& new_axis_mask, int& shrink_axis_mask, int& begin_mask, int& end_mask, int& ellipsis_mask) {
    int index = 0;
    auto dealItem = [&](PyObject* item) {
        if (PySlice_Check(item)) {
            Py_ssize_t startl = 0, stopl = 0, stepl = 1;
            auto slice_res = PySlice_Unpack(item, &startl, &stopl, &stepl);
            // py2 don't check return value.
            int start = static_cast<int>(startl);
            int stop = static_cast<int>(stopl);
            int step = static_cast<int>(stepl);
            begin.push_back(start);
            end.push_back(stop);
            strides.push_back(step);
            if ((step == 1 && start == 0) || (step == -1 && start == -1)) {
                begin_mask |= (1 << index);
            }
            if ((step == -1 && stop == 0) || PY_SSIZE_T_MAX == stopl) {
                end_mask |= (1 << index);
            }
        }
        if (PyObject_IsInstance(item, (PyObject*)&PyEllipsis_Type)) {
            begin.push_back(0);
            end.push_back(0);
            strides.push_back(1);
            ellipsis_mask |= (1 << index);
        }
        if (item == Py_None) {
            begin.push_back(0);
            end.push_back(0);
            strides.push_back(1);
            new_axis_mask |= (1 << index);
        }
        if (isInt(item)) {
            int axis = toInt(item);
            begin.push_back(axis);
            end.push_back(axis + 1);
            strides.push_back(1);
            shrink_axis_mask |= (1 << index);
        }
        index++;
    };
    if (PyTuple_Check(slice)) {
        size_t size = PyTuple_Size(slice);
        for (int i = 0; i < size; i++) {
            auto item = PyTuple_GetItem(slice, i);
            dealItem(item);
        }
    } else {
        dealItem(slice);
    }
}
static inline bool isIdx(PyObject* slice) {
    return Py_TYPE(slice) == PyType_FindTLSType(&PyMNNVarType) || (PyList_Check(slice) && isInts(slice));
}
static bool isBoolIdx(VARP idx, int reqSize) {
    auto size = idx->getInfo()->size;
    bool isbool = (size == reqSize);
    if (isbool) {
        auto ptr = idx->readMap<int>();
        for (int i = 0; i < size; i++) {
            if (ptr[i] != 0 && ptr[i] != 1) {
                return false;
            }
        }
    }
    return isbool;
}
static PyObject* PyMNNVar_subscript(PyObject* x, PyObject* slice) {
    // gather: 1. 0-1 gather; 2. idx gather;
    if (isIdx(slice)) {
        auto val = toVar(x);
        auto idx = toVar(slice);
        if (val->getInfo()->size > 1 && isBoolIdx(idx, val->getInfo()->size)) {
            // 0-1 gather -> idx gather
            idx = Express::_Where(idx);
            val = Express::_GatherND(val, idx);
            val = Express::_Reshape(val, {-1});
            return toPyObj(val);
        }
        auto r = Express::_Gather(val, idx);
        r->readMap<void>();
        return toPyObj(r);
    }

    std::vector<int> begin, end, strides;
    int new_axis_mask = 0, shrink_axis_mask = 0, begin_mask = 0, end_mask = 0, ellipsis_mask = 0;
    dealSlice(slice, begin, end, strides, new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask);
    int size_ = static_cast<int>(begin.size());
    auto begin_ = Express::_Const(begin.data(), {size_}, NHWC, halide_type_of<int>());
    auto end_ = Express::_Const(end.data(), {size_}, NHWC, halide_type_of<int>());
    auto strides_ = Express::_Const(strides.data(), {size_}, NHWC, halide_type_of<int>());
    auto res = Express::_StridedSlice(toVar(x), begin_, end_, strides_, begin_mask, end_mask,
                                      ellipsis_mask, new_axis_mask, shrink_axis_mask);
    auto info = res->getInfo();
    if (!info) {
        PyMNN_ERROR("subscript: unable to get variable info");
    }
    // to scalar
    if (info->dim.empty()) {
        auto dtype = info->type;
        if (dtype == halide_type_of<float>()) {
            return toPyObj(res->readMap<float>()[0]);
        }
        if (dtype == halide_type_of<int>()) {
            return toPyObj(res->readMap<int>()[0]);
        }
        if (dtype == halide_type_of<uint8_t>()) {
            return toPyObj(res->readMap<uint8_t>()[0]);
        }
        if (dtype == halide_type_of<double>()) {
            return toPyObj((float)res->readMap<double>()[0]);
        }
    }
    return toPyObj(res);
}

static int PyMNNVar_ass_subscript(PyObject* x, PyObject* slice, PyObject* y) {
    if (!isVar(x) || !isVar(y)) {
        PyMNN_ERROR_LOG("ass_subscript require args: (Var, int/Var, int/float/Var)");
        return -1;
    }
    auto var = toVar(x);
    auto val = toVar(y);
    auto varInfo = var->getInfo();
    if (isIdx(slice)) {
        auto idx = toVar(slice);
        if (isBoolIdx(idx, varInfo->size)) {
            idx = Express::_Where(idx);
        }
        auto idxDim = idx->getInfo()->dim;
        int scatterNum = idxDim[0], scatterDim = 1;
        if (idxDim.size() < 2) {
            idx = Express::_Unsqueeze(idx, {-1});
        } else {
            scatterDim = idxDim[1];
        }
        // val broadcast_to [scatterNum, (scatterDim < varDim.size() ? varDim[scatterDim:] : 1)]
        auto varDim = varInfo->dim;
        std::vector<int> valDim(1, scatterNum);
        if (scatterDim >= varDim.size()) {
            valDim.push_back(1);
        } else {
            for (int i = scatterDim; i < varDim.size(); i++) {
                valDim.push_back(varDim[i]);
            }
        }
        val = Express::_BroadcastTo(val, _Const(valDim.data(), {static_cast<int>(valDim.size())}, NCHW, halide_type_of<int32_t>()));
        *(((PyMNNVar*)x)->var) = Express::_ScatterNd(idx, val, Express::_Shape(var), var);
        return 0;
    }
    std::vector<int> begin, end, strides;
    int new_axis_mask = 0, shrink_axis_mask = 0, begin_mask = 0, end_mask = 0, ellipsis_mask = 0;
    dealSlice(slice, begin, end, strides, new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask);
    int size_ = static_cast<int>(begin.size());
    auto begin_ = Express::_Const(begin.data(), {size_}, NHWC, halide_type_of<int>());
    auto end_ = Express::_Const(end.data(), {size_}, NHWC, halide_type_of<int>());
    auto strides_ = Express::_Const(strides.data(), {size_}, NHWC, halide_type_of<int>());
    *(((PyMNNVar*)x)->var) = Express::_StridedSliceWrite(var, begin_, end_, strides_, val, begin_mask, end_mask,
                                                         ellipsis_mask, new_axis_mask, shrink_axis_mask);
    return 0;
}
static PyObject* PyMNNVar_iter(PyObject *self) {
    auto var = toVar(self);
    if (var->getInfo()->dim.empty()) {
        PyMNN_ERROR("iteration over a 0-d array");
    }
    Py_INCREF(self);
    return self;
}
static PyObject* PyMNNVar_iternext(PyObject *self) {
    auto idx = ((PyMNNVar*)self)->iter_index++;
    auto var = toVar(self);
    auto conut = var->getInfo()->dim[0];
    if (idx >= conut) return NULL;
    return toPyObj(Express::_Gather(var, Express::_Scalar<int>(idx)));
}
// PyMNNVar basic functions impl
static PyObject* PyMNNVar_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyMNNVar* self = (PyMNNVar *)type->tp_alloc(type, 0);
    self->iter_index = 0;
    self->var = nullptr;
    return (PyObject*)self;
}
static void PyMNNVar_dealloc(PyMNNVar *self) {
    if (self->var) {
        delete self->var;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}
static PyObject* PyMNNVar_repr(PyObject *self) {
    PyMNNVar* var = (PyMNNVar*)self;
    if (var->var == nullptr) {
        return toPyObj("None Var");
    }
    auto info = (*(var->var))->getInfo();
    const void* ptr = (*(var->var))->readMap<void>();
    if (info == nullptr || ptr == nullptr) {
        return toPyObj((*(var->var))->name());
    }
#ifdef PYMNN_NUMPY_USABLE
    auto content = PyMNNVar_read((PyMNNVar*)self, NULL);
#else
    auto content = PyMNNVar_read_as_tuple((PyMNNVar*)self, NULL);
#endif
    auto reprfunc = PyObject_GetAttrString(content, "__repr__");
    auto str = PyEval_CallObject(reprfunc, NULL);
    Py_DECREF(content);
    Py_DECREF(reprfunc);
    return str;
}
// PyMNNVar getter/setter functions impl
static PyObject* PyMNNVar_getshape(PyMNNVar *self, void *closure) {
    PyObject *shape = NULL;
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr == info) {
            PyMNN_ERROR("getshape: unable to get variable info");
        }
        shape = toPyObj(info->dim);
    }
    return shape;
}
static PyObject* PyMNNVar_getvalid(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr != info) {
            Py_RETURN_TRUE;
        }
    }
    Py_RETURN_FALSE;
}
static PyObject* PyMNNVar_getdata_format(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr == info) {
            PyMNN_ERROR("getdata_format: unable to get variable info");
        }
        return toPyObj(info->order);
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_getdtype(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr == info) {
            PyMNN_ERROR("getdtype: unable to get variable info");
        }
        return toPyObj(htype2dtype(info->type));
    }
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_getsize(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr == info) {
            PyMNN_ERROR("getsize: unable to get variable info");
        }
        return toPyObj((int)info->size);
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNVar_getname(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto name = (*(self->var))->name();
        return toPyObj(name.c_str());
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNVar_getndim(PyMNNVar *self, void *closure) {
    PyObject *ndim = NULL;
    if (self->var) {
        auto info = (*(self->var))->getInfo();
        if(nullptr == info) {
            PyMNN_ERROR("getndim: unable to get variable info");
        }
        ndim = toPyObj((int)info->dim.size());
    }
    return ndim;
    Py_RETURN_NONE;
}

static PyObject* PyMNNVar_getptr(PyMNNVar *self, void *closure) {
    if (self->var) {
        const void* ptr = (*(self->var))->readMap<void>();
        if(nullptr != ptr) {
            return PyCapsule_New(const_cast<void*>(ptr), NULL, NULL);
        }
    }
    PyMNN_ERROR("getptr: unable to get data ptr.");
}

static int PyMNNVar_setname(PyMNNVar *self, PyObject *value, void *closure) {
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The name must be a string");
        return -1;
    }
    if (self->var) {
        (*(self->var))->setName(toString(value));
    }
    return 0;
}
#ifdef BUILD_OPTYPE
static PyObject* PyMNNVar_getop_type(PyMNNVar *self, void *closure) {
    if (self->var) {
        auto op = (*(self->var))->expr().first->get();
        if (nullptr == op) {
            switch ((*(self->var))->expr().first->inputType()) {
                case VARP::INPUT:
                    return toPyObj("Input");
                case VARP::CONSTANT:
                    return toPyObj("Const");
                case VARP::TRAINABLE:
                    return toPyObj("Trainable");
            }
        }
        auto type = op->type();
        if (type == OpType_BinaryOp) {
            return toPyObj(MNN::EnumNameBinaryOpOperation((BinaryOpOperation)op->main_as_BinaryOp()->opType()));
        }
        if (type == OpType_UnaryOp) {
            return toPyObj(MNN::EnumNameUnaryOpOperation((UnaryOpOperation)op->main_as_UnaryOp()->opType()));
        }
        return toPyObj(MNN::EnumNameOpType(type));
    }
    Py_RETURN_NONE;
}
#endif
static PyObject* PyMNNVar_getinputs(PyMNNVar *self, void *closure) {
    auto inputs = (*(self->var))->expr().first->inputs();
    return toPyObj<VARP, toPyObj>(inputs);
}
// PyMNNVar methods impl
static PyObject* PyMNNVar_fix_as_placeholder(PyMNNVar *self, PyObject *args) {
    (*(self->var)).fix(VARP::INPUT);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_fix_as_const(PyMNNVar *self, PyObject *args) {
    (*(self->var)).fix(VARP::CONSTANT);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_fix_as_trainable(PyMNNVar *self, PyObject *args) {
    (*(self->var)).fix(VARP::TRAINABLE);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_close(PyMNNVar *self, PyObject *args) {
    (*(self->var))->input(VARP(nullptr));
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_copy_from(PyMNNVar *self, PyObject *args) {
    PyMNNVar* src = NULL;
    if (!PyArg_ParseTuple(args, "O", &src)) {
        return NULL;
    }
    if (!src->var || !self->var) {
        PyMNN_ERROR("PyMNNVar_copy_from: source or destination var is NULL!");
    }
    (*(self->var))->input(*(src->var));
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_set_inputs(PyMNNVar *self, PyObject *args) {
    PyObject* inputs = NULL;
    if (!PyArg_ParseTuple(args, "O", &inputs)) {
        Py_RETURN_NONE;
    }
    auto source = toVars(inputs);
    auto expr = (*(self->var))->expr();
    auto newExpr = Expr::create(expr.first->extra(), std::move(source), expr.first->outputSize());
    Expr::replace(expr.first, newExpr);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_replace(PyMNNVar *self, PyObject *args) {
    PyObject* src = NULL;
    if (!PyArg_ParseTuple(args, "O", &src)) {
        Py_RETURN_NONE;
    }
    VARP source = toVar(src);
    if (!self->var) {
        PyMNN_ERROR("PyMNNVar_replace: destination var is NULL!");
    }
    Variable::replace(*(self->var), source);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_reorder(PyMNNVar *self, PyObject *args) {
    PyObject* order;
    if (!PyArg_ParseTuple(args, "O", &order)) {
        Py_RETURN_NONE;
    }
    auto newInput = _ChangeInputFormat(*(self->var), toEnum<Dimensionformat>(order));
    *(self->var) = newInput;
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_resize(PyMNNVar *self, PyObject *args) {
    PyObject* shape = NULL;
    if (!PyArg_ParseTuple(args, "O", &shape)) {
        Py_RETURN_NONE;
    }
    (*(self->var))->resize(toInts(shape));
    Py_RETURN_NONE;
}
#ifdef PYMNN_NUMPY_USABLE
static PyObject* PyMNNVar_read(PyMNNVar *self, PyObject *args) {
    auto info = (*(self->var))->getInfo();
    if(nullptr == info) {
        PyMNN_ERROR("read: unable to get variable info");
    }
    auto dtype = htype2dtype(info->type);
    auto shape = info->dim;
    int64_t total_length = info->size;
    auto readptr = [self](DType dtype, INTS shape, int64_t total_length) {
        void *dataPtr = (void *) (*(self->var))->readMap<void>();
        if (nullptr == dataPtr) {
            PyMNN_ERROR("call to readMap meet a error");
        }
        std::vector<npy_intp> npy_dims;
        for(const auto dim : shape) {
            npy_dims.push_back(dim);
        }
        switch(dtype) {
            case DType_FLOAT:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_FLOAT, dataPtr);
            case DType_DOUBLE:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_DOUBLE, dataPtr);
            case DType_INT32:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT32, dataPtr);
            case DType_INT64:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT64, dataPtr);
            case DType_UINT8:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_UINT8, dataPtr);
            case DType_INT8:
                return PyArray_SimpleNewFromData(npy_dims.size(), npy_dims.data(), NPY_INT8, dataPtr);
            default:
                PyMNN_ERROR("does not support this dtype");
        }
    };
    auto data = readptr(dtype, shape, total_length);
    (*(self->var))->unMap();
    return (PyObject*)data;
}
#endif
static PyObject* PyMNNVar_read_as_tuple(PyMNNVar *self, PyObject *args) {
    auto info = (*(self->var))->getInfo();
    if(nullptr == info) {
        PyMNN_ERROR("read_as_tuple: unable to get variable info");
    }
    auto dtype = htype2dtype(info->type);
    auto shape = info->dim;
    size_t total_length = info->size;
    auto readptr = [self](DType dtype, INTS shape, size_t total_length) {
        void *dataPtr = (void *) (*(self->var))->readMap<void>();
        if (nullptr == dataPtr) {
            PyMNN_ERROR("call to readMap meet a error");
        }
        auto obj = PyTuple_New(total_length);
        if(DType_FLOAT == dtype) {
            auto data = (float*)dataPtr;
            for(size_t i = 0; i < total_length; i++) {
                PyTuple_SetItem(obj, i, PyFloat_FromDouble(data[i]));
            }
        } else if(DType_INT32 == dtype) {
            auto data = (int32_t*)dataPtr;
            for(size_t i = 0; i < total_length; i++) {
                PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
            }
        } else if(DType_UINT8 == dtype) {
            auto data = (uint8_t*)dataPtr;
            for(size_t i = 0; i < total_length; i++) {
                PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
            }
        } else if(DType_INT8 == dtype) {
            auto data = (int8_t*)dataPtr;
            for(size_t i = 0; i < total_length; i++) {
                PyTuple_SetItem(obj, i, PyLong_FromLong(data[i]));
            }
        } else {
            PyMNN_ERROR("Don't support data type");
        }
        return obj;
    };
    auto data = readptr(dtype, shape, total_length);
    (*(self->var))->unMap();
    return (PyObject*)data;
}
static PyObject* PyMNNVar_write(PyMNNVar *self, PyObject *args) {
    PyObject* data = NULL;
    if (!PyArg_ParseTuple(args, "O", &data)) {
        Py_RETURN_NONE;
    }
    auto info = (*(self->var))->getInfo();
    if(nullptr == info) {
        PyMNN_ERROR("write: unable to get variable info");
    }
    auto dtype = htype2dtype(info->type);
    int64_t total_length = info->size;
    toPtr(data, dtype, total_length, (*(self->var))->writeMap<void>());
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_sync(PyMNNVar *self, PyObject *args) {
    ((MNN::Tensor*)(*(self->var))->getTensor())->wait(MNN::Tensor::MAP_TENSOR_READ, true);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_set_device_ptr(PyMNNVar *self, PyObject *args) {
    uint64_t devicePtr;
    int memoryType;
    if (!PyArg_ParseTuple(args, "Ki", &devicePtr, &memoryType)) {
        Py_RETURN_NONE;
    }

    (*(self->var))->setDevicePtr((const void*)devicePtr, memoryType);
    Py_RETURN_NONE;
}
static PyObject* PyMNNVar_copy_to_device_ptr(PyMNNVar *self, PyObject *args) {
    uint64_t devicePtr;
    int memoryType;
    if (!PyArg_ParseTuple(args, "Ki", &devicePtr, &memoryType)) {
        Py_RETURN_NONE;
    }

    (*(self->var))->copyToDevicePtr((void*)devicePtr, memoryType);
    Py_RETURN_NONE;
}
// Expr methods
static PyObject* PyMNNExpr_set_thread_number(PyObject *self, PyObject *args) {
    int numberThread;
    if (!PyArg_ParseTuple(args, "i", &numberThread)) {
        Py_RETURN_NONE;
    }
    if (numberThread < 1) {
        numberThread = 1;
    }
    if (numberThread > 8) {
        numberThread = 8;
    }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, numberThread);
    Py_RETURN_NONE;
}
static PyObject* PyMNNExpr_load_as_list(PyObject *self, PyObject *args) {
    const char *fileName;
    if (!PyArg_ParseTuple(args, "s", &fileName)) {
        Py_RETURN_NONE;
    }
    return toPyObj<VARP, toPyObj>(Variable::load(fileName));
}
static PyObject* PyMNNExpr_save(PyObject *self, PyObject *args) {
    PyObject* vars = NULL;
    const char *fileName = NULL;
    int forInference = 1;
    if (!PyArg_ParseTuple(args, "Os|i", &vars, &fileName, &forInference)) {
        return NULL;
    }
    auto newVars = toVars(vars);
#ifdef PYMNN_TRAIN_API
    if (forInference) {
        Transformer::turnModelToInfer()->onExecute(newVars);
    }
#endif
    Variable::save(newVars, fileName);
#ifdef PYMNN_TRAIN_API
    ConvertToFullQuant::convert(fileName);
#endif
    Py_RETURN_NONE;
}
static PyObject* PyMNNExpr_load_as_dict(PyObject *self, PyObject *args) {
    const char *fileName = NULL;
    if (!PyArg_ParseTuple(args, "s", &fileName)) {
        Py_RETURN_NONE;
    }
    return toPyObj<string, toPyObj, VARP, toPyObj>(Variable::loadMap(fileName));
}
static PyObject* PyMNNExpr_get_inputs_and_outputs(PyObject *self, PyObject *args) {
    PyObject *allVariable;
    if (!PyArg_ParseTuple(args, "O", &allVariable)) {
        Py_RETURN_NONE;
    }
    auto arg = toMap<string, toString, VARP, toVar>(allVariable);
    return toPyObj<std::map<std::string, VARP>,
                    toPyObj<string, toPyObj, VARP, toPyObj>,
                    std::map<std::string, VARP>,
                    toPyObj<string, toPyObj, VARP, toPyObj>
                    >(Variable::getInputAndOutput(arg));
}
static PyObject* PyMNNExpr_gc(PyObject *self, PyObject *args) {
    int full;
    if (!PyArg_ParseTuple(args, "i", &full)) {
        return NULL;
    }
    auto exe = ExecutorScope::Current();
    if (full) {
        exe->gc(Executor::FULL);
    } else {
        exe->gc(Executor::PART);
    }
    Py_RETURN_NONE;
}

static PyObject* PyMNNExpr_lazy_eval(PyObject *self, PyObject *args) {
    int lazy = 0;
    if (!PyArg_ParseTuple(args, "i", &lazy)) {
        return NULL;
    }
    ExecutorScope::Current()->lazyEval = lazy;
    Py_RETURN_NONE;
}

static PyObject* PyMNNExpr_set_lazy_mode(PyObject *self, PyObject *args) {
    int lazy = 0;
    if (!PyArg_ParseTuple(args, "i", &lazy)) {
        return NULL;
    }
    ExecutorScope::Current()->setLazyComputeMode((Executor::LazyMode)lazy);
    Py_RETURN_NONE;
}
static PyObject* PyMNNExpr_set_global_executor_config(PyObject *self, PyObject *args) {
    int numberThread, backendType, precisionType;
    if (!PyArg_ParseTuple(args, "iii", &backendType, &precisionType, &numberThread)) {
        Py_RETURN_NONE;
    }

    auto exe = ExecutorScope::Current();
    BackendConfig config;
    config.precision = (BackendConfig::PrecisionMode)precisionType;
    exe->setGlobalExecutorConfig((MNNForwardType)backendType, config, numberThread);
    Py_RETURN_NONE;
}

def_unary(Expr,
    sign, Express::_Sign,
    abs, Express::_Abs,
    negative, Express::_Negative,
    floor, Express::_Floor,
    round, Express::_Round,
    ceil, Express::_Ceil,
    square, Express::_Square,
    sqrt, Express::_Sqrt,
    rsqrt, Express::_Rsqrt,
    exp, Express::_Exp,
    log, Express::_Log,
    sin, Express::_Sin,
    sinh, Express::_Sinh,
    cos, Express::_Cos,
    cosh, Express::_Cosh,
    tan, Express::_Tan,
    tanh, Express::_Tanh,
    asin, Express::_Asin,
    asinh, Express::_Asinh,
    acos, Express::_Acos,
    acosh, Express::_Acosh,
    atan, Express::_Atan,
    atanh, Express::_Atanh,
    reciprocal, Express::_Reciprocal,
    log1p, Express::_Log1p,
    gelu, Express::_Gelu,
    sigmoid, Express::_Sigmoid,
    erf, Express::_Erf,
    erfc, Express::_Erfc,
    erfinv, Express::_Erfinv,
    expm1, Express::_Expm1,
    // other unary-like
    softplus, Express::_Softplus,
    softsign, Express::_Softsign,
    size, Express::_Size,
    zeros_like, Express::_ZerosLike,
    where, Express::_Where,
    rank, Express::_Rank
)
def_binary(Expr,
    add, Express::_Add,
    subtract, Express::_Subtract,
    multiply, Express::_Multiply,
    divide, Express::_Divide,
    pow, Express::_Pow,
    minimum, Express::_Minimum,
    maximum, Express::_Maximum,
    greater, Express::_Greater,
    greater_equal, Express::_GreaterEqual,
    less, Express::_Less,
    floordiv, Express::_FloorDiv,
    squared_difference, Express::_SquaredDifference,
    less_equal, Express::_LessEqual,
    floormod, Express::_FloorMod,
    equal, Express::_Equal,
    mod, Express::_Mod,
    atan2, Express::_Atan2,
    logical_or, Express::_LogicalOr,
    not_equal, Express::_NotEqual,
    bias_add, Express::_BiasAdd,
    bitwise_and, Express::_BitwiseAnd,
    bitwise_or, Express::_BitwiseOr,
    bitwise_xor, Express::_BitwiseXor,
    // other binary-like
    fill, Express::_Fill,
    tile, Express::_Tile,
    gather, Express::_Gather,
    gather_nd, Express::_GatherND,
    setdiff1d, Express::_SetDiff1D,
    unravel_index, Express::_UnravelIndex
)
def_reduce(Expr,
    reduce_sum, Express::_ReduceSum,
    reduce_mean, Express::_ReduceMean,
    reduce_max, Express::_ReduceMax,
    reduce_min, Express::_ReduceMin,
    reduce_prod, Express::_ReduceProd,
    reduce_any, Express::_ReduceAny,
    reduce_all, Express::_ReduceAll
)
def_eltwise(Expr,
    eltwise_prod, Express::_Prod,
    eltwise_sum, Express::_Sum,
    eltwise_max, Express::_Max,
    eltwise_sub, Express::_Sub
)
def_axis_op(Expr,
    channel_shuffle, Express::_ChannelShuffle,
    space_to_depth, Express::_SpaceToDepth,
    depth_to_space, Express::_DepthToSpace
)
def_triple(Expr,
    slice, Express::_Slice,
    select, Express::_Select,
    batch_to_space_nd, Express::_BatchToSpaceND,
    matrix_band_part, Express::_MatrixBandPart,
    space_to_batch_nd, Express::_SpaceToBatchND,
    range, Express::_Range,
    scatter_nd, Express::_ScatterNd
)
def_axiss_op(Expr,
    squeeze, Express::_Squeeze,
    unsqueeze, Express::_Unsqueeze
)
// binary
// other ops
static PyObject* PyMNNExpr_cast(PyObject *self, PyObject *args) {
    PyObject *x, *dtype;
    if (PyArg_ParseTuple(args, "OO", &x, &dtype) && isVar(x) && isdtype(dtype)) {
        return toPyObj(Express::_Cast(toVar(x), dtype2htype(toEnum<DType>(dtype))));
    }
    PyMNN_ERROR("cast require args: (Var, dtype)");
}
static PyObject* PyMNNExpr_matmul(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    int transposeA = false;
    int transposeB = false;
    if (PyArg_ParseTuple(args, "OOii", &a, &b, &transposeA, &transposeB) && isVar(a) && isVar(b)) {
        return toPyObj(Express::_MatMul(toVar(a), toVar(b), transposeA, transposeB));
    }
    PyMNN_ERROR("matmul require args: (Var, Var, bool, bool)");
}
static PyObject* PyMNNExpr_normalize(PyObject *self, PyObject *args) {
    PyObject *x, *scale;
    int acrossSpatial, channelShared;
    float eps;
    if (PyArg_ParseTuple(args, "OiifO", &x, &acrossSpatial, &channelShared, &eps, &scale)
        && isVar(x) && isFloats(scale)) {
        return toPyObj(Express::_Normalize(toVar(x), acrossSpatial, channelShared, eps, toFloats(scale)));
    }
    PyMNN_ERROR("normalize require args: (Var, int, int, float, [float])");
}
static PyObject* PyMNNExpr_argmax(PyObject *self, PyObject *args) {
    PyObject *input;
    int axis = 0;
    if (PyArg_ParseTuple(args, "O|i", &input, &axis) && isVar(input)) {
        return toPyObj(Express::_ArgMax(toVar(input), axis));
    }
    PyMNN_ERROR("argmax require args: (Var, |int)");
}
static PyObject* PyMNNExpr_argmin(PyObject *self, PyObject *args) {
    PyObject *input;
    int axis = 0;
    if (PyArg_ParseTuple(args, "O|i", &input, &axis) && isVar(input)) {
        return toPyObj(Express::_ArgMin(toVar(input), axis));
    }
    PyMNN_ERROR("argmin require args: (Var, |int)");
}
static PyObject* PyMNNExpr_cumsum(PyObject *self, PyObject *args) {
    PyObject *input;
    int axis = 0;
    if (PyArg_ParseTuple(args, "O|i", &input, &axis) && isVar(input)) {
        return toPyObj(Express::_CumSum(toVar(input), axis));
    }
    PyMNN_ERROR("cumsum require args: (Var, |int)");
}
static PyObject* PyMNNExpr_cumprod(PyObject *self, PyObject *args) {
    PyObject *input;
    int axis = 0;
    if (PyArg_ParseTuple(args, "O|i", &input, &axis) && isVar(input)) {
        return toPyObj(Express::_CumProd(toVar(input), axis));
    }
    PyMNN_ERROR("cumprod require args: (Var, |int)");
}
static PyObject* PyMNNExpr_svd(PyObject *self, PyObject *args) {
    PyObject *input;
    if (PyArg_ParseTuple(args, "O", &input) && isVar(input)) {
        return toPyObj<VARP, toPyObj>(Express::_Svd(toVar(input)));
    }
    PyMNN_ERROR("svd require args: (Var)");
}
static PyObject* PyMNNExpr_histogram(PyObject *self, PyObject *args) {
    PyObject *input;
    int binNum, minVal, maxVal;
    if (PyArg_ParseTuple(args, "Oiii", &input, &binNum, &minVal, &maxVal) && isVar(input)) {
        return toPyObj(Express::_Histogram(toVar(input), binNum, minVal, maxVal));
    }
    PyMNN_ERROR("histogram require args: (Var, int, int, int)");
}
static PyObject* PyMNNExpr_one_hot(PyObject *self, PyObject *args) {
    PyObject *indices;
    int depth, axis = -1;
    float onValue = 1, offValue = 0;
    if (PyArg_ParseTuple(args, "Oi|ffi", &indices, &depth, &onValue, &offValue, &axis) && isVar(indices)) {
        return toPyObj(Express::_OneHot(toVar(indices), Express::_Scalar<int>(depth),
                                        Express::_Scalar<float>(onValue),
                                        Express::_Scalar<float>(offValue), axis));
    }
    PyMNN_ERROR("one_hot require args: (Var, int, |float, float, int)");
}
static PyObject* PyMNNExpr_broadcast_to(PyObject *self, PyObject *args) {
    PyObject *input, *shape;
    if (PyArg_ParseTuple(args, "OO", &input, &shape) && isVar(input) && isVar(shape)) {
        return toPyObj(Express::_BroadcastTo(toVar(input), toVar(shape)));
    }
    PyMNN_ERROR("broadcast_to require args: (Var, Var)");
}
// NN ops
static PyObject* PyMNNExpr_placeholder(PyObject *self, PyObject *args) {
    INTS default_shape = {};
    PyObject *shape = nullptr /* default_shape */ ,
             *format = nullptr /* NCHW */ ,
             *type = nullptr /* DType_FLOAT */ ;
    if (PyArg_ParseTuple(args, "|OOO", &shape, &format, &type)
        && (shape == nullptr || isInts(shape))
        && (format == nullptr || isdata_format(format))
        && (type == nullptr || isdtype(type))) {
            auto data_format = PARSE(format, NCHW, toEnum<Dimensionformat>);
            auto dtype = PARSE(type, DType_FLOAT, toEnum<DType>);
            return toPyObj(Express::_Input(PARSE(shape, default_shape, toInts), data_format, dtype2htype(dtype)));
    }
    PyMNN_ERROR("placeholder require args: (|[int], data_format, dtype)");
}
static PyObject* PyMNNExpr_clone(PyObject *self, PyObject *args) {
    PyObject *source;
    int deepCopy = 0;
    if (PyArg_ParseTuple(args, "O|i", &source, &deepCopy) && isVar(source)) {
        return toPyObj(Express::_Clone(toVar(source), deepCopy));
    }
    PyMNN_ERROR("clone require args: (Var, |bool)");
}
static PyObject* PyMNNExpr_const(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *value, *shapes, *format = nullptr /* NCHW */, *type = nullptr /* DType_FLOAT */;
    static char *kwlist[] = { "value_list", "shape", "data_format", "dtype", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &value, &shapes, &format, &type)) {
        PyMNN_ERROR("const require args: (ndarray/list/tuple/bytes/PyCapsule/int_addr, [ints], |data_format, dtype)");
    }
    if ((!isVals(value) && !isInt(value)) || !isInts(shapes) || (format != nullptr && !isdata_format(format)) || (type != nullptr && !isdtype(type))) {
        PyMNN_ERROR("const require args: (ndarray/list/tuple/bytes/PyCapsule/int_addr, [ints], |data_format, dtype)");
    }
    auto data_format = (format == nullptr ? NCHW : toEnum<Dimensionformat>(format));
    auto dtype = (type == nullptr ? DType_FLOAT : toEnum<DType>(type));
    auto shape = toInts(shapes);
    int64_t total_length = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        if (data_format == NC4HW4 && 1 == i) {
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#endif
            total_length *= ROUND_UP(shape[i], 4);
        } else {
            total_length *= shape[i];
        }
    }
    auto ret = getVar();
    if (total_length > 0) {
        void* data = nullptr;
        bool need_free = false;
        if (PyCapsule_CheckExact(value)) {
            data = PyCapsule_GetPointer(value, NULL);
        } else if (isInt(value)) {
            data = PyLong_AsVoidPtr(value);
        } else if (PyBytes_Check(value)) {
            int64_t bytesize = PyBytes_Size(value);
            data = toPtr(value, DType_UINT8, bytesize);
            need_free = true;
        } else {
            data = toPtr(value, dtype, total_length);
            need_free = true;
        }
        if(data) {
            *(ret->var) = _Const((const void*)data, shape, data_format, dtype2htype(dtype));
            if (need_free) {
                free(data);
            }
        }
    } else {
        *(ret->var) = _Const(nullptr, shape, data_format, dtype2htype(dtype));
    }
    return (PyObject *)ret;
}
static PyObject* PyMNNExpr_conv2d(PyObject *self, PyObject *args) {
    INTS default_stride = {1, 1};
    INTS default_pads = {0, 0};
    PyObject *input, *weight, *bias,
             *stride = nullptr /* default_stride */,
             *padding = nullptr /* default_pads */,
             *dilate = nullptr /* default_stride */,
             *padding_mode = nullptr /* VALID */;
    int group = 1;
    if (PyArg_ParseTuple(args, "OOO|OOOiO", &input, &weight, &bias,
        &stride, &padding, &dilate, &group, &padding_mode)
        && isVar(input) && isVar(weight) && isVar(bias)
        && (stride == nullptr || isInts(stride))
        && (padding_mode == nullptr || isPadding_Mode(padding_mode))
        && (padding == nullptr || isInts(padding))) {
        return toPyObj(Express::_Conv(toVar(weight), toVar(bias), toVar(input),
                                    PARSE(padding_mode, VALID, toEnum<PaddingMode>),
                                    PARSE(stride, default_stride, toInts),
                                    PARSE(dilate, default_stride, toInts),
                                    group,
                                    PARSE(padding, default_pads, toInts)));
    }
    PyMNN_ERROR("conv2d require args: (Var, Var, Var, |Padding_Mode, [int], [int], int, [int])");
}
static PyObject* PyMNNExpr_conv2d_transpose(PyObject *self, PyObject *args) {
    INTS default_stride = {1, 1};
    INTS default_pads = {0, 0};
    PyObject *input, *weight, *bias,
             *stride = nullptr /* default_stride */,
             *padding = nullptr /* default_pads */,
             *dilate = nullptr /* default_stride */,
             *padding_mode = nullptr /* VALID */;
    int group = 1;
    if (PyArg_ParseTuple(args, "OOO|OOOiO", &input, &weight, &bias,
        &stride, &padding, &dilate, &group, &padding_mode)
        && isVar(input) && isVar(weight) && isVar(bias)
        && (stride == nullptr || isInts(stride))
        && (padding_mode == nullptr || isPadding_Mode(padding_mode))
        && (padding == nullptr || isInts(padding))) {
        return toPyObj(Express::_Deconv(toVar(weight), toVar(bias), toVar(input),
                                    PARSE(padding_mode, VALID, toEnum<PaddingMode>),
                                    PARSE(stride, default_stride, toInts),
                                    PARSE(dilate, default_stride, toInts),
                                    group,
                                    PARSE(padding, default_pads, toInts)));
    }
    PyMNN_ERROR("conv2d_transpose require args: (Var, Var, Var, |Padding_Mode, [int], [int], int, [int])");
}
static PyObject* PyMNNExpr_max_pool(PyObject *self, PyObject *args) {
    INTS default_pads = {0, 0};
    PyObject *x, *kernel, *stride,
             *padding_mode = nullptr /* VALID */,
             *pads = nullptr /* default_pads */;
    if (PyArg_ParseTuple(args, "OOO|OO", &x, &kernel, &stride, &padding_mode, &pads)
        && isVar(x) && isInts(kernel) && isInts(stride)
        && (padding_mode == nullptr || isPadding_Mode(padding_mode))
        && (pads == nullptr || isInts(pads))) {
        return toPyObj(Express::_MaxPool(toVar(x), toInts(kernel), toInts(stride),
                                    PARSE(padding_mode, VALID, toEnum<PaddingMode>),
                                    PARSE(pads, default_pads, toInts)));
    }
    PyMNN_ERROR("max_pool require args: (Var, [int], [int], |Padding_Mode, [int])");
}
static PyObject* PyMNNExpr_avg_pool(PyObject *self, PyObject *args) {
    INTS default_pads = {0, 0};
    PyObject *x, *kernel, *stride,
             *padding_mode = nullptr /* VALID */,
             *pads = nullptr /* default_pads */;
    if (PyArg_ParseTuple(args, "OOO|OO", &x, &kernel, &stride, &padding_mode, &pads)
        && isVar(x) && isInts(kernel) && isInts(stride)
        && (padding_mode == nullptr || isPadding_Mode(padding_mode))
        && (pads == nullptr || isInts(pads))) {
        return toPyObj(Express::_AvePool(toVar(x), toInts(kernel), toInts(stride),
                                    PARSE(padding_mode, VALID, toEnum<PaddingMode>),
                                    PARSE(pads, default_pads, toInts)));
    }
    PyMNN_ERROR("avg_pool require args: (Var, [int], [int], |Padding_Mode, [int])");
}
static PyObject* PyMNNExpr_reshape(PyObject *self, PyObject *args) {
    PyObject *x, *shape, *original_format = nullptr /* NCHW */;
    if (PyArg_ParseTuple(args, "OO|O", &x, &shape, &original_format)
        && isVar(x) && isInts(shape)
        && (original_format == nullptr || isdata_format(original_format))) {
        return toPyObj(Express::_Reshape(toVar(x), toInts(shape),
                            PARSE(original_format, NCHW, toEnum<Dimensionformat>)));
    }
    PyMNN_ERROR("reshape require args: (Var, [int], |data_format)");
}
static PyObject* PyMNNExpr_scale(PyObject *self, PyObject *args) {
    PyObject *x, *scale, *bias;
    int channels;
    if (PyArg_ParseTuple(args, "OiOO", &x, &channels, &scale, &bias)
        && isVar(x) && isFloats(scale) && isFloats(bias)) {
        return toPyObj(Express::_Scale(toVar(x), channels, toFloats(scale), toFloats(bias)));
    }
    PyMNN_ERROR("scale require args: (Var, int, [float], [float])");
}
static PyObject* PyMNNExpr_relu(PyObject *self, PyObject *args) {
    PyObject *x;
    float slope = 0.0f;
    if (PyArg_ParseTuple(args, "O|f", &x, &slope) && isVar(x)) {
        return toPyObj(Express::_Relu(toVar(x), slope));
    }
    PyMNN_ERROR("relu require args: (Var, |float)");
}
static PyObject* PyMNNExpr_relu6(PyObject *self, PyObject *args) {
    PyObject *x;
    float min = 0.0f, max = 6.0f;
    if (PyArg_ParseTuple(args, "O|ff", &x, &min, &max) && isVar(x)) {
        return toPyObj(Express::_Relu6(toVar(x), min, max));
    }
    PyMNN_ERROR("relu6 require args: (Var, |float, float)");
}
static PyObject* PyMNNExpr_prelu(PyObject *self, PyObject *args) {
    PyObject *x, *slopes;
    if (PyArg_ParseTuple(args, "OO", &x, &slopes) && isVar(x) && isFloats(slopes)) {
        return toPyObj(Express::_PRelu(toVar(x), toFloats(slopes)));
    }
    PyMNN_ERROR("prelu require args: (Var, [float])");
}
static PyObject* PyMNNExpr_softmax(PyObject *self, PyObject *args) {
    PyObject *logits;
    int axis = -1;
    if (PyArg_ParseTuple(args, "O|i", &logits, &axis) && isVar(logits)) {
        return toPyObj(Express::_Softmax(toVar(logits), axis));
    }
    PyMNN_ERROR("softmax require args: (Var, |int)");
}
static PyObject* PyMNNExpr_split(PyObject *self, PyObject *args) {
    PyObject *input, *size_splits;
    int axis;
    if (PyArg_ParseTuple(args, "OOi", &input, &size_splits, &axis)
        && isVar(input) && isInts(size_splits)) {
        return toPyObj<VARP, toPyObj>(Express::_Split(toVar(input), toInts(size_splits), axis));
    }
    PyMNN_ERROR("split require args: (Var, [int], int)");
}
static PyObject* PyMNNExpr_strided_slice(PyObject *self, PyObject *args) {
    PyObject *input, *begin, *end, *strides;
    int begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask;
    if (PyArg_ParseTuple(args, "OOOOiiiii", &input, &begin, &end,
                          &strides, &begin_mask, &end_mask, &ellipsis_mask,
                          &new_axis_mask, &shrink_axis_mask)
        && isVar(input) && isVar(begin) && isVar(end) && isVar(strides)) {
        return toPyObj(Express::_StridedSlice(toVar(input), toVar(begin), toVar(end),
                                        toVar(strides), begin_mask, end_mask,
                                        ellipsis_mask, new_axis_mask, shrink_axis_mask));
    }
    PyMNN_ERROR("strided_slice require args: (Var, Var, Var, Var, int, int, int, int, int)");
}
static PyObject* PyMNNExpr_concat(PyObject *self, PyObject *args) {
    PyObject *values;
    int axis;
    if (PyArg_ParseTuple(args, "Oi", &values, &axis) && isVars(values)) {
        return toPyObj(Express::_Concat(toVars(values), axis));
    }
    PyMNN_ERROR("concat require args: ([Var], int)");
}
static PyObject* PyMNNExpr_convert(PyObject *self, PyObject *args) {
    PyObject *input, *format;
    if (PyArg_ParseTuple(args, "OO", &input, &format)
        && isVar(input) && isdata_format(format)) {
        return toPyObj(Express::_Convert(toVar(input), toEnum<Dimensionformat>(format)));
    }
    PyMNN_ERROR("convert require args: (Var, data_format)");
}
static PyObject* PyMNNExpr_expand_dims(PyObject *self, PyObject *args) {
    PyObject *x, *axis;
    if (PyArg_ParseTuple(args, "OO", &x, &axis) && isVar(x)) {
        if (isInt(axis)) {
            return toPyObj(Express::_ExpandDims(toVar(x), toInt(axis)));
        }
        if (isVar(axis)) {
            return toPyObj(Express::_ExpandDims(toVar(x), toVar(axis)));
        }
    }
    PyMNN_ERROR("expand_dims require args: (Var, int|Var)");
}
static PyObject* PyMNNExpr_transpose(PyObject *self, PyObject *args) {
    PyObject *x, *perm;
    if (PyArg_ParseTuple(args, "OO", &x, &perm) && isVar(x)) {
        if (isInts(perm)) {
            return toPyObj(Express::_Transpose(toVar(x), toInts(perm)));
        }
        if (isVar(perm)) {
            return toPyObj(Express::_Transpose(toVar(x), toVar(perm)));
        }
    }
    PyMNN_ERROR("transpose require args: (Var, [int]|Var)");
}
static PyObject* PyMNNExpr_reverse(PyObject *self, PyObject *args) {
    PyObject *x, *y;
    if (PyArg_ParseTuple(args, "OO", &x, &y) && isVar(x) && isVar(y)) {
        return toPyObj(Express::_Reverse(toVar(x), toVar(y)));
    }
    PyMNN_ERROR("reverse require args: (Var, Var)");
}
static PyObject* PyMNNExpr_reverse_sequence(PyObject *self, PyObject *args) {
    PyObject *x, *y;
    int batchDim, seqDim;
    if (PyArg_ParseTuple(args, "OOii", &x, &y, &batchDim, &seqDim) && isVar(x) && isVar(y)) {
        return toPyObj(Express::_ReverseSequence(toVar(x), toVar(y), batchDim, seqDim));
    }
    PyMNN_ERROR("reverse_sequence require args: (Var, Var, int, int)");
}
static PyObject* PyMNNExpr_crop(PyObject *self, PyObject *args) {
    PyObject *images, *size, *offset;
    int axis;
    if (PyArg_ParseTuple(args, "OOiO", &images, &size, &axis, &offset)
        && isVar(images) && isVar(size) && isInts(offset)) {
        return toPyObj(Express::_Crop(toVar(images), toVar(size), axis, toInts(offset)));
    }
    PyMNN_ERROR("crop require args: (Var, Var, int, [int])");
}
static PyObject* PyMNNExpr_resize(PyObject *self, PyObject *args) {
    PyObject *images;
    float x_scale, y_scale;
    if (PyArg_ParseTuple(args, "Off", &images, &x_scale, &y_scale) && isVar(images)) {
        return toPyObj(Express::_Resize(toVar(images), x_scale, y_scale));
    }
    PyMNN_ERROR("resize require args: (Var, float, float)");
}
static PyObject* PyMNNExpr_pad(PyObject *self, PyObject *args) {
    PyObject *x, *paddings, *mode = nullptr /* CONSTANT */;
    if (PyArg_ParseTuple(args, "OO|O", &x, &paddings, &mode)
        && isVar(x) && isVar(paddings)
        && (mode == nullptr || isPadValue_Mode(mode))) {
        return toPyObj(Express::_Pad(toVar(x), toVar(paddings),
                       PARSE(mode, CONSTANT, toEnum<MNN::Express::PadValueMode>)));
    }
    PyMNN_ERROR("pad require args: (Var, Var, |Padding_Mode)");
}
static PyObject* PyMNNExpr_shape(PyObject *self, PyObject *args) {
    PyObject *input;
    if (PyArg_ParseTuple(args, "O", &input) && isVar(input)) {
        return toPyObj(Express::_Shape(toVar(input), false));
    }
    PyMNN_ERROR("shape require args: (Var)");
}
static PyObject* PyMNNExpr_stack(PyObject *self, PyObject *args) {
    PyObject *values;
    int axis;
    if (PyArg_ParseTuple(args, "Oi", &values, &axis) && isVars(values)) {
        return toPyObj(Express::_Stack(toVars(values), axis));
    }
    PyMNN_ERROR("stack require args: ([Var], int)");
}
static PyObject* PyMNNExpr_crop_and_resize(PyObject *self, PyObject *args) {
    PyObject *image, *boxes, *box_ind, *crop_size,
             *method = nullptr /* BILINEAR */;
    float extrapolation_value = 0.0f;
    if (PyArg_ParseTuple(args, "OOOO|Of", &image, &boxes, &box_ind,
                         &crop_size, &method, &extrapolation_value)
        && isVar(image) && isVar(boxes) && isVar(box_ind)
        && isVar(crop_size)
        && (method == nullptr || isInterp_Method(method))) {
        return toPyObj(Express::_CropAndResize(toVar(image), toVar(boxes),
                                            toVar(box_ind), toVar(crop_size),
                                            PARSE(method, BILINEAR, toEnum<InterpolationMethod>),
                                            extrapolation_value));
    }
    PyMNN_ERROR("crop_and_resize require args: (Var, Var, Var, Var, |Interp_Method, float)");
}
static PyObject* PyMNNExpr_selu(PyObject *self, PyObject *args) {
    PyObject *features;
    float scale, alpha;
    if (PyArg_ParseTuple(args, "Off", &features, &scale, &alpha) && isVar(features)) {
        return toPyObj(Express::_Selu(toVar(features), scale, alpha));
    }
    PyMNN_ERROR("selu require args: (Var, float, float)");
}
static PyObject* PyMNNExpr_elu(PyObject *self, PyObject *args) {
    PyObject *features;
    float alpha = 1.0;
    if (PyArg_ParseTuple(args, "O|f", &features, &alpha) && isVar(features)) {
        return toPyObj(Express::_Elu(toVar(features), alpha));
    }
    PyMNN_ERROR("elu require args: (Var, |float)");
}
static PyObject* PyMNNExpr_moments(PyObject *self, PyObject *args) {
    PyObject *x, *axis, *shift;
    int keep_dims;
    if (PyArg_ParseTuple(args, "OOOi", &x, &axis, &shift, &keep_dims)
        && isVar(x) && isInts(axis) && isVar(shift)) {
        return toPyObj<VARP, toPyObj>(Express::_Moments(toVar(x), toInts(axis), toVar(shift), keep_dims));
    }
    PyMNN_ERROR("moments require args: (Var, [int], Var, bool)");
}
static PyObject* PyMNNExpr_unstack(PyObject *self, PyObject *args) {
    PyObject *value;
    int axis = 0;
    if (PyArg_ParseTuple(args, "O|i", &value, &axis) && isVar(value)) {
        return toPyObj<VARP, toPyObj>(Express::_Unstack(toVar(value), axis));
    }
    PyMNN_ERROR("unstack require args: (Var, |int)");
}
static PyObject* PyMNNExpr_randomuniform(PyObject *self, PyObject *args) {
    PyObject *shape, *dtype;
    float low = 0.f, high = 1.f;
    int seed0 = 0, seed1 = 0;
    if (PyArg_ParseTuple(args, "OO|ffii", &shape, &dtype, &low, &high, &seed0, &seed1) &&
        isVar(shape) && isdtype(dtype)) {
        return toPyObj(Express::_RandomUnifom(toVar(shape),
                                dtype2htype(toEnum<DType>(dtype)),
                                low, high, seed0, seed1));
    }
    PyMNN_ERROR("randomuniform require args: (Var, dtype, |float, float, int, int)");
}
static PyObject* PyMNNExpr_sort(PyObject *self, PyObject *args) {
    PyObject *x;
    int axis = -1, arg = 0, descend = 0, bykey = -1;
    if (PyArg_ParseTuple(args, "O|iii", &x, &axis, &arg, &descend) && isVar(x)) {
        return toPyObj(Express::_Sort(toVar(x), axis, arg, descend));
    }
    PyMNN_ERROR("sort require args: (Var, |int, bool, bool)");
}
static PyObject* PyMNNExpr_raster(PyObject *self, PyObject *args) {
    PyObject *var, *region, *shape;
    if (PyArg_ParseTuple(args, "OOO", &var, &region, &shape) &&
        isVars(var) && isInts(region) && isInts(shape)) {
        return toPyObj(Express::_Raster(toVars(var), toInts(region), toInts(shape)));
    }
    PyMNN_ERROR("raster require args: ([Var], [int], [int])");
}
static PyObject* PyMNNExpr_quant(PyObject *self, PyObject *args) {
    PyObject *var, *scale;
    int min = -128, max = 127, zero = 0;
    if (PyArg_ParseTuple(args, "OO|ii", &var, &scale, &min, &max, &zero) && isVar(var) && isVar(scale)) {
        return toPyObj(Express::_FloatToInt8(toVar(var), toVar(scale), min, max, zero));
    }
    PyMNN_ERROR("quant require args: (Var, Var, |int, int)");
}
static PyObject* PyMNNExpr_dequant(PyObject *self, PyObject *args) {
    PyObject *var, *scale;
    int zero;
    if (PyArg_ParseTuple(args, "OOi", &var, &scale, &zero) && isVar(var) && isVar(scale)) {
        return toPyObj(Express::_Int8ToFloat(toVar(var), toVar(scale), zero));
    }
    PyMNN_ERROR("dequant require args: (Var, Var, int)");
}
static PyObject* PyMNNExpr_nms(PyObject *self, PyObject *args) {
    PyObject *boxes, *scores;
    int max_detections;
    float iou_threshold = -1.0, score_threshold = -1.0;
    if (PyArg_ParseTuple(args, "OOi|ff", &boxes, &scores, &max_detections, &iou_threshold, &score_threshold) &&
        isVar(boxes) && isVar(scores)) {
        return toPyObj(Express::_Nms(toVar(boxes), toVar(scores), max_detections, iou_threshold, score_threshold));
    }
    PyMNN_ERROR("nms require args: (Var, Var, |float, float)");
}
static PyObject* PyMNNExpr_detection_post_process(PyObject *self, PyObject *args) {
    PyObject *encode_boxes, *class_predictions, *anchors, *centersize_encoding;
    int num_classes, max_detections, max_class_per_detection, detections_per_class;
    float nms_threshold, iou_threshold;
    int use_regular_nms = 0;
    if (PyArg_ParseTuple(args, "OOOiiiiffpO", &encode_boxes, &class_predictions,
        &anchors, &num_classes, &max_detections, &max_class_per_detection, &detections_per_class,
        &nms_threshold, &iou_threshold, &use_regular_nms, &centersize_encoding)
        && isVar(encode_boxes) && isVar(class_predictions) && isVar(anchors) && isFloats(centersize_encoding)) {
        auto res = Express::_DetectionPostProcess(toVar(encode_boxes), toVar(class_predictions),
                                                  toVar(anchors), num_classes, max_detections,
                                                  max_class_per_detection, detections_per_class,
                                                  nms_threshold, iou_threshold, use_regular_nms,
                                                  toFloats(centersize_encoding));
        return toPyObj<VARP, toPyObj>(res);
    }
    PyMNN_ERROR("detection_post_process require args: (Var, Var, Var, int, int, int, int, float, float, bool, [float])");
}
static PyObject* PyMNNExpr_roi_pooling(PyObject *self, PyObject *args) {
    PyObject *input, *roi;
    int pooledHeight, pooledWidth;
    float spatialScale;
    int outputGrad = 0;
    PyObject *backwardDiff = nullptr;
    if (PyArg_ParseTuple(args, "OOiifpO", &input, &roi, &pooledHeight, &pooledWidth,
        &spatialScale, &outputGrad, &backwardDiff) && isVar(input) && isVar(roi) && isVar(backwardDiff)) {
        auto res = Express::_ROIPooling(toVar(input), toVar(roi), pooledHeight, pooledWidth, spatialScale, outputGrad, toVar(backwardDiff));
        return toPyObj(res);
    }
    PyMNN_ERROR("roi_pooling require args: (Var, Var, int, int, float, [bool, Var])");
}
static PyObject* PyMNNExpr_roi_align(PyObject *self, PyObject *args) {
    PyObject *input, *roi;
    int pooledHeight, pooledWidth;
    float spatialScale;
    int samplingRatio;
    int aligned;
    PyObject *poolType;
    int outputGrad = 0;
    PyObject *backwardDiff = nullptr;
    if (PyArg_ParseTuple(args, "OOiifipOpO", &input, &roi, &pooledHeight, &pooledWidth,
        &spatialScale, &samplingRatio, &aligned, &poolType, &outputGrad, &backwardDiff)
        && isVar(input) && isVar(roi) && isPooling_Mode(poolType) && isVar(backwardDiff)) {
        auto res = Express::_ROIAlign(toVar(input), toVar(roi), pooledHeight, pooledWidth, spatialScale,
                                    samplingRatio, aligned, toEnum<PoolingMode>(poolType),
                                    outputGrad, toVar(backwardDiff));
        return toPyObj(res);
    }
    PyMNN_ERROR("roi_align require args: (Var, Var, int, int, float, int, bool, PoolingMode, [bool, Var])");
}
static PyMethodDef PyMNNExpr_methods[] = {
    register_methods_kw(Expr,
        const, "build const var."
    )
    register_methods(Expr,
        // Var methods
        set_thread_number, "set thread number of expr.",
        load_as_list, "load file as var list.",
        save, "save vars to file.",
        load_as_dict, "load file as var dict.",
        get_inputs_and_outputs, "get input and output of var dict.",
        gc, "do gc full or part.",
        lazy_eval, "expr do lazy evaluation or not.",
        set_lazy_mode, "set lazy compute mode, content: 0 or full: 1.",
        set_global_executor_config, "set global executor config for expr."
    )
    register_methods(Expr,
        // unary expr
        sign, "build unary:sign expr.",
        abs, "build unary:abs expr.",
        negative, "build unary:negative expr.",
        floor, "build unary:floor expr.",
        round, "build unary:round expr.",
        ceil, "build unary:ceil expr.",
        square, "build unary:square expr.",
        sqrt, "build unary:sqrt expr.",
        rsqrt, "build unary:rsqrt expr.",
        exp, "build unary:exp expr.",
        log, "build unary:log expr.",
        sin, "build unary:sin expr.",
        sinh, "build unary:sinh expr.",
        cos, "build unary:cos expr.",
        cosh, "build unary:cosh expr.",
        tan, "build unary:tan expr.",
        tanh, "build unary:tanh expr.",
        asin, "build unary:asin expr.",
        asinh, "build unary:asinh expr.",
        acos, "build unary:acos expr.",
        acosh, "build unary:acosh expr.",
        atan, "build unary:atan expr.",
        atanh, "build unary:atanh expr.",
        reciprocal, "build unary:reciprocal expr.",
        log1p, "build unary:log1p expr.",
        gelu, "build unary:gelu expr.",
        sigmoid, "build unary:sigmoid expr.",
        erf, "build unary:erf expr.",
        erfc, "build unary:erfc expr.",
        erfinv, "build unary:erfinv expr.",
        expm1, "build unary:expm1 expr."
    )
    register_methods(Expr,
        // binary expr
        add, "build add expr.",
        subtract, "build substract expr.",
        multiply, "build multiply expr.",
        divide, "build divide expr.",
        floordiv, "build floordiv expr.",
        mod, "build mod expr.",
        floormod, "build floormod expr.",
        pow, "build pow expr.",
        minimum, "build minimum expr.",
        maximum, "build maximum expr.",
        equal, "build equal expr.",
        not_equal, "build not_equal expr.",
        greater, "build greater expr.",
        greater_equal, "build greater_equal expr.",
        less, "build less expr.",
        less_equal, "build less_equal expr.",
        squared_difference, "build squared_difference expr.",
        atan2, "build atan2 expr.",
        logical_or, "build logical_or expr.",
        bias_add, "build bias_add expr.",
        bitwise_and, "build bitwise_and expr.",
        bitwise_or, "build bitwise_or expr.",
        bitwise_xor, "build bitwise_xor expr."
    )
    register_methods(Expr,
        // reduce expr
        reduce_sum, "build reduce_sum expr.",
        reduce_mean, "build reduce_mean expr.",
        reduce_max, "build reduce_max expr.",
        reduce_min, "build reduce_min expr.",
        reduce_prod, "build reduce_prod expr.",
        reduce_any, "build reduce_any expr.",
        reduce_all, "build reduce_all expr.",
        // eltwise expr
        eltwise_prod, "build eltwise_prod expr.",
        eltwise_sum, "build eltwise_sum expr.",
        eltwise_max, "build eltwise_max expr.",
        eltwise_sub, "build eltwise_sub expr."
    )
    register_methods(Expr,
        // other math ops
        cast, "build cast expr: cast(Var x, dtype dst_type)",
        matmul, "build matmul expr: matmul(Var a, Var b, bool transposeA, bool transposeB)",
        normalize, "build normalize expr:",
        argmax, "build argmax expr:",
        argmin, "build argmin expr:",
        cumsum, "build cumsum expr:",
        cumprod, "build cumprod expr:",
        svd, "build svd expr:",
        histogram, "build histogram expr:",
        unravel_index, "build unravel_index expr:",
        scatter_nd, "build scatter_nd expr:",
        one_hot, "build one_hot expr:",
        broadcast_to, "build broadcast_to expr:"
    )
    register_methods(Expr,
        // NN ops
        placeholder, "build placeholder var.",
        clone, "clone var.",
        conv2d, "build conv2d expr",
        conv2d_transpose, "build conv2d_transpose expr",
        max_pool, "build max_pool expr",
        avg_pool, "build avg_pool expr",
        quant, "build quant expr",
        dequant, "build dequant expr"
    )
    {"reshape",  PyMNNExpr_reshape, METH_VARARGS, "build reshape: (Var, [int], |data_format)"},
    register_methods(Expr,
        scale, "build scale expr",
        relu, "build relu expr",
        relu6, "build relu6 expr",
        prelu, "build prelu expr",
        softmax, "build softmax expr",
        softplus, "build softplus expr",
        softsign, "build softsign expr",
        slice, "build slice expr",
        split, "build split expr",
        strided_slice, "build strided_slice expr",
        concat, "build concat expr",
        where, "build where expr",
        convert, "build convert expr"
    )
    {"transpose",  PyMNNExpr_transpose, METH_VARARGS, "build transpose: (Var, [int]/Var)"},
    register_methods(Expr,
        channel_shuffle, "build channel_shuffle expr",
        reverse, "build reverse expr",
        reverse_sequence, "build reverse_sequence expr",
        crop, "build crop expr",
        resize, "build resize expr",
        crop_and_resize, "build crop_and_resize expr",
        pad, "build pad expr",
        randomuniform, "build randomuniform expr"
    )
    {"expand_dims",  PyMNNExpr_expand_dims, METH_VARARGS, "build expand_dims: (Var, int/Var)"},
    register_methods(Expr,
        rank, "build rank expr",
        size, "build size expr",
        shape, "build shape expr",
        stack, "build stack expr",
        fill, "build fill expr",
        tile, "build tile expr",
        gather, "build gather expr",
        gather_nd, "build gather_nd expr",
        select, "build select expr",
        squeeze, "build squeeze expr",
        unsqueeze, "build unsqueeze expr",
        depth_to_space, "build depth_to_space expr",
        space_to_depth, "build space_to_depth expr",
        batch_to_space_nd, "build batch_to_space_nd expr",
        space_to_batch_nd, "build space_to_batch_nd expr",
        elu, "build elu expr",
        selu, "build selu expr",
        matrix_band_part, "build matrix_band_part expr",
        moments, "build moments expr",
        setdiff1d, "build setdiff1d expr",
        zeros_like, "build zeros_like expr",
        unstack, "build unstack expr",
        range, "build range expr",
        sort, "build sort expr",
        raster, "build raster expr",
        nms, "build nms expr",
        detection_post_process, "build detection_post_process expr",
        roi_pooling, "build roi_pooling expr",
        roi_align, "build roi_align expr"
    )
};
// Expr Module End
#endif
