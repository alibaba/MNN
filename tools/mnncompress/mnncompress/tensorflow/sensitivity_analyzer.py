from __future__ import print_function
import numpy as np
import tensorflow as tf
from .helpers import get_op_name_of_type, is_weight_tensor, get_variable_by_tensor

_MNN_compress_scope = 'MNN_pruner_'
_Prune_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']


class SensitivityAnalyzer(object):
    def __init__(self, gradient_var_ops, graph=None):
        self._gradient_var_ops = gradient_var_ops
        self._graph = graph
        if graph is None:
            self._graph = tf.get_default_graph()
        
        self._op_names = get_op_name_of_type(self._graph, _Prune_Support_Ops)
        self._ops = self._graph.get_operations()
        self._name_to_ops = {op.name : op for op in self._ops}

        self._prune_weight_variable_op_names = {}
        self._prune_var_mask_abs_grads = {}
        self.analyze_steps = 10

        self._get_prune_weight_variable_op_names()

    def analyze(self, gradient_var_numpy):
        print("analyzing weight sensitivity...")
        for i in range(len(gradient_var_numpy)):
            (g, v) = self._gradient_var_ops[i]
            if g is None:
                continue

            (gn, vn) = gradient_var_numpy[i]
            if v in self._prune_weight_variable_op_names.keys():
                mask_abs_grad = np.abs(vn * gn)
                self._prune_var_mask_abs_grads[v.name] = mask_abs_grad + self._prune_var_mask_abs_grads[v.name]

    def save_sensitivity_npy(self, filename):
        np.save(filename, self._prune_var_mask_abs_grads)
        print("sensitivity info saved to npy file:", filename)

    def _get_prune_weight_variable_op_names(self):
        for op_name in self._op_names:
            op = self._name_to_ops[op_name]
            if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                weight_variable = get_variable_by_tensor(op.inputs[1])
                if len(op.outputs[0].consumers()) == 0:
                    continue
                self._prune_weight_variable_op_names[weight_variable] = op.name
                self._prune_var_mask_abs_grads[weight_variable.name] = np.zeros(weight_variable.shape.as_list())
            
            if op.type == "MatMul":
                if "gradients" in op.name and "_grad" in op.name:
                    continue
                
                for t in op.inputs:
                    if is_weight_tensor(t):
                        weight_variable = get_variable_by_tensor(t)
                        if len(op.outputs[0].consumers()) == 0:
                            continue
                        self._prune_weight_variable_op_names[weight_variable] = op.name
                        self._prune_var_mask_abs_grads[weight_variable.name] = np.zeros(weight_variable.shape.as_list())
