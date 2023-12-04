# TODO: avoid import everything from _mnncengine._nn for visable control
from _mnncengine._nn import *

import _mnncengine._expr as _F
import _mnncengine._nn as _nn

def load_module_from_file(file_name, input_names, output_names, **kwargs):
    runtime_manager = kwargs.get('runtime_manager', None)
    dynamic = kwargs.get('dynamic', False)
    shape_mutable = kwargs.get('shape_mutable', True)
    rearrange = kwargs.get('rearrange', False)
    backend = kwargs.get('backend', _F.Backend.CPU)
    memory_mode = kwargs.get('memory_mode', _F.MemoryMode.Normal)
    power_mode = kwargs.get('power_mode', _F.PowerMode.Normal)
    precision_mode = kwargs.get('precision_mode', _F.PrecisionMode.Normal)
    thread_num = kwargs.get('thread_num', 4)

    module = _nn.load_module_from_file(runtime_manager, input_names, output_names, file_name, dynamic, shape_mutable, rearrange,
                                       backend, memory_mode, power_mode, precision_mode, thread_num)
    
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


class EmptyModule(_nn._Module):
    def __init(self):
        super(EmptyModule, self).__init__()
    def forward(self):
        return None
dummy = EmptyModule()