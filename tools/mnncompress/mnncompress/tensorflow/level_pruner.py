from __future__ import print_function
import tensorflow as tf
import yaml

from ..common import MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
import uuid
from .helpers import get_op_name_of_type, get_input_tensor_index, is_weight_tensor
from .graph_checker import check_for_grad_ops

_MNN_compress_scope = 'MNN_pruner_'
_Prune_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']


class LevelPruner(object):
    def __init__(self, graph=None, config_dict={}, total_pruning_iterations=50000, skip_prune_layers = [], default_prune_ratio=0.0, debug_info= False):
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph

        op_names = get_op_name_of_type(self._graph, _Prune_Support_Ops)
        self._config_dict = {}
        self._skip_prune_layers = skip_prune_layers
        for op_name in op_names:
            if op_name not in skip_prune_layers:
                self._config_dict[op_name] = default_prune_ratio

        if isinstance(config_dict, str):
            f = open(config_dict, "r")
            config_dict = yaml.load(f)
            f.close()

        for key, value in config_dict.items():
            if key not in op_names:
                raise ValueError("op name:", key, "not found in the model, ")
            if key not in skip_prune_layers:
                self._config_dict[key] = value

        skip_names = []
        for key, value in self._config_dict.items():
            if value < 1e-8:
                skip_names.append(key)
                self._skip_prune_layers.append(key)

        for name in skip_names:
            self._config_dict.pop(name)
        
        self._debug_info = debug_info

        if debug_info:
            print("pruning config:")
            for key, value in self._config_dict.items():
                print(key, ":", value)
            
            print("skip pruning layers:")
            for layer in self._skip_prune_layers:
                if layer in op_names:
                    print(layer)

        self._ops = self._graph.get_operations()
        self._name_to_ops = {op.name : op for op in self._ops}
        self._mask_dict = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0

        if (total_pruning_iterations < 0) and (not isinstance(total_pruning_iterations, int)):
            raise ValueError("total_pruning_iterations must be a integer >= 0")
        self._total_pruning_iterations = total_pruning_iterations
        
        self._step = tf.get_variable(name='MNN_pruner_internal_step', initializer=0, trainable=False)

        self._update_step_op = tf.assign(self._step, self._step + 1)
        tf.add_to_collection("update_internal_step", self._update_step_op)

        self._op_name_weight_tensor_name = {}
        self._insert_pruning_ops()
        self._update_pruning_ops = tf.get_collection("update_pruning_masks")
        self._control_prune_op = self._step < self._total_pruning_iterations
        self._initialized = False

        self._op_prune_ratio_dict = {}
        for op_name in self._mask_dict.keys():
            ratio = 1.0 - tf.reduce_mean(self._mask_dict[op_name])
            self._op_prune_ratio_dict[op_name] = ratio
        
    def do_pruning(self, sess):
        if not self._initialized:
            sess.run(tf.variables_initializer([self._step]))
            self._initialized = True
        
        if sess.run(self._control_prune_op):
            sess.run(self._update_step_op)
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

        self._target_sparsity = 1.0 - self._remain_weight_num / self._total_weight_num

        if not self._reported:
            detail = {"algorithm": "Level", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"overall_sparsity": self._target_sparsity, "total_pruning_iterations": self._total_pruning_iterations, "prune_ratios": self._config_dict, \
                    "skip_prune_layers": self._skip_prune_layers}}
            self._reported = mnn_logger.on_done("tensorflow", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

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

    def _prune_weight(self, op, weight_tensor, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weight_abs_tensor = tf.abs(weight_tensor)
            weight_sorted_abs_tensor = tf.contrib.framework.sort(tf.reshape(weight_abs_tensor, [-1]))
            num = weight_sorted_abs_tensor.shape.num_elements() * self._config_dict[op.name] / self._total_pruning_iterations * tf.cast(self._step, tf.float32)
            num = tf.cast(num, tf.int64)
            threshold = weight_sorted_abs_tensor[num]
            mask = tf.cast(weight_abs_tensor > threshold, tf.float32)
            self._mask_dict[op.name] = mask

            if self._debug_info:
                pruning_ratio = 1.0 - tf.reduce_mean(mask)
                print_op = tf.print("pruning iter:", self._step, "/", self._total_pruning_iterations, op.name, "pruning_ratio:", pruning_ratio)

                with tf.control_dependencies([print_op]):
                    pruned_weight_tensor = weight_tensor * mask
            else:
                pruned_weight_tensor = weight_tensor * mask

            weight_variable_name = self._get_weight_variable_name(weight_tensor)
            weight_variable = self._get_variable_by_name(weight_variable_name)
            new_weight_tensor_assign = tf.assign(weight_variable, pruned_weight_tensor)

            tf.add_to_collection("update_pruning_masks", new_weight_tensor_assign)

            return new_weight_tensor_assign
