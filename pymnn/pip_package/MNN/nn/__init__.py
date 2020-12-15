# TODO: avoid import everything from _mnncengine._nn for visable control
from _mnncengine._nn import *

import _mnncengine._expr as _F
import _mnncengine._nn as _nn

# old call: load_module_from_file(file_name, for_training)
# new call: load_module_from_file(file_name, dynamic=False, shape_mutable=True)
# support two by args and kwargs
def load_module_from_file(file_name, *args, **kwargs):
    old_call = len(args) > 0 #for_training
    
    m = _F.load_as_dict(file_name)
    inputs_outputs = _F.get_inputs_and_outputs(m)

    inputs = []
    for key in inputs_outputs[0].keys():
        inputs.append(inputs_outputs[0][key] if old_call else key)

    outputs = []
    for key in inputs_outputs[1].keys():
        outputs.append(inputs_outputs[1][key] if old_call else key)

    if old_call:
        for_training = args[0]
        module = _nn.load_module(inputs, outputs, for_training)
    else:
        dynamic = kwargs.get('dynamic', False)
        shape_mutable = kwargs.get('shape_mutable', True)
        module = _nn.load_module_from_file(inputs, outputs, file_name, dynamic, shape_mutable)
    return module


class Module(_nn._Module):
    def __init__(self):
        super(Module, self).__init__()
        self._children = {}
        self._vars = {}
    
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError("__call__ not implemented, please use 'forward' method in subclasses")

    def __setattr__(self, name, value):
        self.__dict__[name] = value

        def remove_from(dicts):
            if name in dicts:
                del dicts[name]

        if isinstance(value, (Module, _nn._Module)):
            remove_from(self._children)
            value.set_name(name)
            self._children[name] = value
            self._register_submodules([value])
            return
        if isinstance(value, _F.Var):
            value.name = name
            if name in self._vars:
                self._vars[name].replace(value)
            else:
                self._vars[name] = value
                self._add_parameter(value)
