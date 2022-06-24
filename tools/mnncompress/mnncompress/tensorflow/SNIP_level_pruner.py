from __future__ import print_function
import tensorflow as tf
import yaml
import numpy as np

from ..common import MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
import uuid
from .helpers import get_op_name_of_type, get_input_tensor_index, is_weight_tensor, get_variable_by_tensor

_MNN_compress_scope = 'MNN_pruner_'
_Prune_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']


class SNIPLevelPruner(object):
    def __init__(self, sensitivity_npy_file, graph=None, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, prune_ratio_result_file="SNIPLevel_found_prune_ratios.yml"):
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph
        
        self._sensitivities = np.load(sensitivity_npy_file, allow_pickle=True).item()
        self._target_sparsity = sparsity
        self._max_prune_ratio = max_prune_ratio
        self._prune_ratio_result_file = prune_ratio_result_file

        self._op_names = get_op_name_of_type(self._graph, _Prune_Support_Ops)
        self._config_dict = {}
        
        self._debug_info = debug_info

        self._ops = self._graph.get_operations()
        self._name_to_ops = {op.name : op for op in self._ops}
        self._prune_weight_variable_op_names = {}
        self._mask_dict = {}
        self._config_file = config_file
        self._generated_config_dict = {}

        if (total_pruning_iterations < 0) and (not isinstance(total_pruning_iterations, int)):
            raise ValueError("total_pruning_iterations must be a integer >= 0")
        self._total_pruning_iterations = total_pruning_iterations
        self._prune_finetune_iterations = prune_finetune_iterations
        self._finetune_step = 0
        
        self._step = tf.get_variable(name='MNN_pruner_internal_step', initializer=0, trainable=False)
        self._step_init_op = tf.variables_initializer([self._step])
        self._control_prune_op = self._step < self._total_pruning_iterations

        self._update_step_op = tf.assign(self._step, self._step + 1)
        tf.add_to_collection("update_internal_step", self._update_step_op)

        self._op_name_weight_tensor_name = {}
        self._current_accumulate_step = 0
        self._total_accumulate_steps = 10
        self._prune_var_mask_abs_grads = {}
        self._mask_abs_grads = []
        self._initialized = False
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0

        self._get_prune_weight_variable_op_names()
        self._generate_prune_ratios()
        self._generate_all_constants()
        self._insert_pruning_ops()
        self._update_pruning_ops = tf.get_collection("update_pruning_masks")
        self._op_prune_ratio_dict = {}
        for op_name in self._mask_dict.keys():
            ratio = 1.0 - tf.reduce_mean(self._mask_dict[op_name])
            self._op_prune_ratio_dict[op_name] = ratio

    def do_pruning(self, sess):
        if not self._initialized:
            sess.run(self._step_init_op)
            self._initialized = True

        if sess.run(self._control_prune_op):
            if self._prune_finetune_iterations == 0:
                sess.run(self._update_step_op)
            else:
                if self._finetune_step == 0:
                    sess.run(self._update_step_op)
                    self._finetune_step += 1
                else:
                    self._finetune_step = (self._finetune_step + 1) % self._prune_finetune_iterations

        sess.run(self._update_pruning_ops)

    def save_compress_params(self, filename, sess, append=False):
        compress_proto = compress_pb.Pipeline()

        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

            pop_index = []
            for i in range(len(compress_proto.algo)):
                if compress_proto.algo[i].type == compress_pb.CompressionAlgo.CompressionType.PRUNE:
                    pop_index.append(i)
            for i in pop_index:
                compress_proto.algo.pop(i)

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid
        algorithm = compress_proto.algo.add()
        algorithm.type = compress_pb.CompressionAlgo.CompressionType.PRUNE
        algorithm.prune_params.type = compress_pb.PruneParams.RANDOM
        weight_tensor_names = algorithm.prune_params.level_pruner_params.weight_tensor_names
        layer_names = algorithm.prune_params.level_pruner_params.layer_names
        prune_ratios = algorithm.prune_params.level_pruner_params.prune_ratios

        for op_name in self._mask_dict.keys():
            tensor_name = self._op_name_weight_tensor_name[op_name]
            layer_names.append(op_name)
            weight_tensor_names.append(tensor_name)
            ratio = sess.run(self._op_prune_ratio_dict[op_name])
            self._total_weight_num += sess.run(tensor_name).size
            # / 4.0, for we always expect using weight quant or full quant after prune
            self._remain_weight_num += (sess.run(tensor_name).size * (1 - ratio) / 4.0)
            prune_ratios.append(ratio)
            print(op_name, "pruning_ratio:", ratio)

        if not self._reported:
            detail = {"algorithm": "SNIP_Level", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"overall_sparsity": self._target_sparsity, "total_pruning_iterations": self._total_pruning_iterations, "prune_ratios": self._config_dict}}
            self._reported = mnn_logger.on_done("tensorflow", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

    def _get_prune_weight_variable_op_names(self):
        for op_name in self._op_names:
            op = self._name_to_ops[op_name]
            if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                weight_variable = get_variable_by_tensor(op.inputs[1])
                if len(op.outputs[0].consumers()) == 0:
                    continue
                self._prune_weight_variable_op_names[weight_variable] = op.name
                self._prune_var_mask_abs_grads[weight_variable] = self._sensitivities[weight_variable.name].astype(np.float32)
            
            if op.type == "MatMul":
                if "gradients" in op.name and "_grad" in op.name:
                    continue
                
                for t in op.inputs:
                    if is_weight_tensor(t):
                        weight_variable = get_variable_by_tensor(t)
                        if len(op.outputs[0].consumers()) == 0:
                            continue
                        self._prune_weight_variable_op_names[weight_variable] = op.name
                        self._prune_var_mask_abs_grads[weight_variable] = self._sensitivities[weight_variable.name].astype(np.float32)

    def _generate_prune_ratios(self):
        config_dict = {}
        if self._config_file is not None:
            f = open(self._config_file, 'r')
            config_dict = yaml.safe_load(f)
        
        print("generating pruning ratios...")
        for v, m in self._prune_var_mask_abs_grads.items():
                self._mask_abs_grads.append(m)
        self._generate_prune_ratios_per_layer(config_dict)
        
        f = open(self._prune_ratio_result_file, "w")
        yaml.dump(self._config_dict, f)
        f.close()
        print("config_dict saved to file:", self._prune_ratio_result_file)

    def _generate_prune_ratios_per_layer(self, config_dict = {}):
        all_weights_flatten = []
        for m in self._mask_abs_grads:
            all_weights_flatten = np.concatenate([all_weights_flatten, m.flatten()])
        all_weights_flatten = np.sort(all_weights_flatten)
        threshold_index = int(all_weights_flatten.size * self._target_sparsity)
        threshold = all_weights_flatten[threshold_index]

        total_weight_num = 0
        total_pruned_weight_num = 0
        for v, m in self._prune_var_mask_abs_grads.items():
            op_name = self._prune_weight_variable_op_names[v]
            op = self._name_to_ops[op_name]
            mask = m >= threshold
            total_weight_num += mask.size
            sparsity = 1.0 - np.mean(mask)
            op_name = self._prune_weight_variable_op_names[v]
            self._generated_config_dict[op_name] = sparsity.tolist()
            if op_name in config_dict.keys():
                self._config_dict[op_name] = config_dict[op_name]
            else:
                self._config_dict[op_name] = sparsity.tolist()
            
            if self._config_dict[op_name] > self._max_prune_ratio:
                self._config_dict[op_name] = self._max_prune_ratio
            
            if op.type == 'DepthwiseConv2dNative':
                self._config_dict[op_name] = 0.0

            total_pruned_weight_num += int(mask.size * self._config_dict[op_name])

        self._target_sparsity = float(total_pruned_weight_num) / total_weight_num
        print("overall prune ratio:", self._target_sparsity)
        print("pruning config:")
        for key, value in self._config_dict.items():
            print(key, ":", value)

    def _insert_pruning_ops(self):
        scope_index = -1
        for layer_name in self._config_dict.keys():
            scope_index += 1
            self._prune_one_layer(layer_name, _MNN_compress_scope + str(scope_index))

    def _prune_one_layer(self, layer_name, scope):
        if self._name_to_ops[layer_name].type == 'Conv2D' or self._name_to_ops[layer_name].type == 'DepthwiseConv2dNative':
            self._prune_conv(self._name_to_ops[layer_name], scope)

        if self._name_to_ops[layer_name].type == 'MatMul':
            self._prune_matmul(self._name_to_ops[layer_name], scope)

    def _get_weight_variable_name(self, tensor):
        return tensor.op.inputs[0].op.name

    def _get_variable_by_name(self, name):
        return [v for v in tf.global_variables() if v.op.name == name][0]

    def _prune_conv(self, conv_op, scope):
        weight_tensor = conv_op.inputs[1]
        self._op_name_weight_tensor_name[conv_op.name] = weight_tensor.name
        new_weight_tensor = self._prune_weight(conv_op, weight_tensor, scope)

    def _prune_matmul(self, matmul_op, scope):
        if "gradients" in matmul_op.name and "_grad" in matmul_op.name:
            return
        
        input_1 = matmul_op.inputs[0]
        input_2 = matmul_op.inputs[1]

        for input_tensor in [input_1, input_2]:
            index = get_input_tensor_index(matmul_op, input_tensor)[0]
            if is_weight_tensor(input_tensor):
                self._op_name_weight_tensor_name[matmul_op.name] = input_tensor.name
                new_weight_tensor = self._prune_weight(matmul_op, input_tensor, scope)

    def _generate_all_constants(self):
        self._pname_sens_constants = {}
        for v, m in self._prune_var_mask_abs_grads.items():
            self._pname_sens_constants[v] = tf.constant(m)

    def _prune_weight(self, op, weight_tensor, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weight_variable = get_variable_by_tensor(weight_tensor)
            m = self._prune_var_mask_abs_grads[weight_variable]
            m_const = self._pname_sens_constants[weight_variable]
            m_sorted = tf.contrib.framework.sort(tf.reshape(m_const, [-1]))
            num = m_sorted.shape.num_elements() * self._config_dict[op.name] / self._total_pruning_iterations * tf.cast(self._step, tf.float32)
            num = tf.cast(num, tf.int64)
            threshold = m_sorted[num]
            mask = tf.cast(m_const >= threshold, tf.float32)
            self._mask_dict[op.name] = mask

            if self._debug_info:
                pruning_ratio = 1.0 - tf.reduce_mean(mask)
                print_op = tf.print("pruning iter:", self._step, "/", self._total_pruning_iterations, op.name, "pruning_ratio:", pruning_ratio)

                with tf.control_dependencies([print_op]):
                    pruned_weight_tensor = weight_variable * mask
            else:
                pruned_weight_tensor = weight_variable * mask

            new_weight_tensor_assign = tf.assign(weight_variable, pruned_weight_tensor)
            tf.add_to_collection("update_pruning_masks", new_weight_tensor_assign)

            return new_weight_tensor_assign
