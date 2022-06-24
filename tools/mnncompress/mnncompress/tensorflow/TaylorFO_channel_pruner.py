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


class TaylorFOChannelPruner(object):
    def __init__(self, sensitivity_npy_file, graph=None, sparsity=0.5, total_pruning_iterations=1, config_file=None, debug_info= False, prune_finetune_iterations=0, max_prune_ratio=0.99, prune_ratio_result_file="TaylorFOChannel_found_prune_ratios.yml", align_channels=4):
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
        self._generated_config_dict = {}
        
        self._debug_info = debug_info

        self._ops = self._graph.get_operations()
        self._name_to_ops = {op.name : op for op in self._ops}
        self._prune_weight_variable_op_names = {}
        self._mask_dict = {}
        self._config_file = config_file

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
        self._initialized = False
        self._prune_var_mask_abs_grads = {}
        self._all_means_flatten = []
        self._pname_sens = {}
        self._pname_bias = {}
        self._pname_bn = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0
        self._align_channels = align_channels

        self._get_prune_weight_variable_op_names()
        self._generate_prune_ratios()
        self._generate_all_constants()
        self._insert_pruning_ops()
        self._init_mask_op = tf.variables_initializer(tf.get_collection("init_masks"))
        self._update_pruning_ops = tf.get_collection("update_pruning_masks")
        self._op_prune_ratio_dict = {}
        for op_name in self._mask_dict.keys():
            ratio = 1.0 - tf.reduce_mean(self._mask_dict[op_name])
            self._op_prune_ratio_dict[op_name] = ratio
        
    def do_pruning(self, sess):
        if not self._initialized:
            sess.run(self._step_init_op)
            sess.run(self._init_mask_op)
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
        algorithm.prune_params.type = compress_pb.PruneParams.FILTER
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
            detail = {"algorithm": "TaylorFOChannel", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
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
        self._generate_prune_ratios_per_layer(config_dict)
        
        f = open(self._prune_ratio_result_file, "w")
        yaml.dump(self._config_dict, f)
        f.close()
        print("config_dict saved to file:", self._prune_ratio_result_file)

    def _generate_prune_ratios_per_layer(self, config_dict = {}):
        for v, gn in self._prune_var_mask_abs_grads.items():
            op_name = self._prune_weight_variable_op_names[v]
            op = self._name_to_ops[op_name]
            if op.type == 'DepthwiseConv2dNative':
                gn = gn.transpose((0, 1, 3, 2))
            
            self._pname_sens[v] = np.sum(gn*gn, tuple([i for i in range(0, len(gn.shape)-1)]), keepdims=True)
            self._all_means_flatten.append(self._pname_sens[v].flatten())

        all_means_flatten = np.concatenate(self._all_means_flatten)
        all_means_flatten = np.sort(all_means_flatten)
        threshold_index = int(all_means_flatten.size * self._target_sparsity)
        threshold = all_means_flatten[threshold_index]

        total_weight_num = 0
        total_pruned_weight_num = 0
        for v, m in self._prune_var_mask_abs_grads.items():
            op_name = self._prune_weight_variable_op_names[v]
            op = self._name_to_ops[op_name]
            if op.type == 'DepthwiseConv2dNative':
                m = m.transpose((0, 1, 3, 2))

            sens = self._pname_sens[v]
            mask = sens > threshold

            sparsity = 1 - np.mean(mask)
            total_weight_num += m.size
            op_name = self._prune_weight_variable_op_names[v]
            self._generated_config_dict[op_name] = sparsity
            if op_name in config_dict.keys():
                self._config_dict[op_name] = config_dict[op_name]
            else:
                self._config_dict[op_name] = sparsity
            
            if self._config_dict[op_name] > self._max_prune_ratio:
                self._config_dict[op_name] = self._max_prune_ratio

            if op.type == 'DepthwiseConv2dNative':
                self._config_dict[op_name] = 0.0

            remain_channels = int(mask.size * (1.0 - self._config_dict[op_name]))
            if remain_channels != mask.size:
                remain_channels = remain_channels - remain_channels % self._align_channels
                if remain_channels <= 0:
                    remain_channels = min(self._align_channels, mask.size)

            self._config_dict[op_name] = (mask.size - remain_channels + 0.0) / mask.size

            total_pruned_weight_num += (m.size * self._config_dict[op_name])

        self._target_sparsity = float(total_pruned_weight_num) / total_weight_num
        print("overall prune ratio:", self._target_sparsity)
        print("pruning config:")
        for key, value in self._config_dict.items():
            print(key, ":", value)

    def _insert_pruning_ops(self):
        scope_index = -1
        for layer_name in self._prune_weight_variable_op_names.values():
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

        if self._config_dict[conv_op.name] == 0.0:
            self._mask_dict[conv_op.name] = tf.ones_like(weight_tensor)
        elif conv_op.type == 'Conv2D':
            self._get_bias_and_batch_norm_statistics(conv_op)
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
                if self._config_dict[matmul_op.name] == 0.0:
                    self._mask_dict[matmul_op.name] = tf.ones_like(input_tensor)
                else:
                    new_weight_tensor = self._prune_weight(matmul_op, input_tensor, scope)

    def _generate_all_constants(self):
        self._pname_sens_constants = {}
        for v, _ in self._prune_var_mask_abs_grads.items():
            self._pname_sens_constants[v] = tf.constant(self._pname_sens[v])

    def _prune_weight(self, op, weight_tensor, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weight_variable = get_variable_by_tensor(weight_tensor)
            m = self._prune_var_mask_abs_grads[weight_variable]

            if op.type == 'DepthwiseConv2dNative':
                m = m.transpose((0, 1, 3, 2))
            
            sens_constant = self._pname_sens_constants[weight_variable]
            m_sorted = tf.contrib.framework.sort(tf.reshape(sens_constant, [-1]))
            num = m_sorted.shape.num_elements() * self._config_dict[op.name] / self._total_pruning_iterations * tf.cast(self._step, tf.float32)
            num = tf.cast(num, tf.int64)
            threshold = m_sorted[num]
            mask = tf.cast(sens_constant >= threshold, tf.float32)

            if op.type == 'DepthwiseConv2dNative':
                mask = tf.transpose(mask, (0, 1, 3, 2))

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

            if weight_variable in self._pname_bias.keys():
                bias_variable = self._pname_bias[weight_variable]
                pruned_bias_tensor = bias_variable * tf.reshape(mask, bias_variable.shape)
                new_bias_assign = tf.assign(bias_variable, pruned_bias_tensor)
                tf.add_to_collection("update_pruning_masks", new_bias_assign)

            if weight_variable in self._pname_bn.keys():
                for variable in self._pname_bn[weight_variable]:
                    pruned_variable_tensor = variable * tf.reshape(mask, variable.shape)
                    new_variable_tensor = variable * pruned_variable_tensor
                    new_variable_assign = tf.assign(variable, new_variable_tensor)
                    tf.add_to_collection("update_pruning_masks", new_variable_assign)
            
            return new_weight_tensor_assign

    def _check_return_cond(self, output_tensor):
        output_no_grad_consumers = [c for c in output_tensor.consumers() if 'gradients/' not in c.name]
        if len(output_no_grad_consumers) != 1:
            if len(output_no_grad_consumers) != 2:
                return True
            else:
                c1 = output_tensor.consumers()[0]
                c2 = output_tensor.consumers()[1]
                if (c1.type != 'Switch') or (c2.type != 'Switch'):
                    return True
        return False

    def _skip_identity(self, tensor):
        while (len(tensor.consumers()) == 1) and (tensor.consumers()[0].type == 'Identity'):
            tensor = tensor.consumers()[0].outputs[0]
        return tensor

    def _get_bias_and_batch_norm_statistics(self, conv2d_op):
        weight_variable = get_variable_by_tensor(conv2d_op.inputs[1])

        output_tensor = conv2d_op.outputs[0]
        if self._check_return_cond(output_tensor):
            return

        output_tensor = self._skip_identity(output_tensor)
        if self._check_return_cond(output_tensor):
            return
        
        # bias add
        if output_tensor.consumers()[0].type in ['Add', 'AddV2', 'BiasAdd']:
            bias_add_op = output_tensor.consumers()[0]
            bias_variable = get_variable_by_tensor(bias_add_op.inputs[1])
            self._pname_bias[weight_variable] = bias_variable
            output_tensor = bias_add_op.outputs[0]

            output_tensor = self._skip_identity(output_tensor)
            if self._check_return_cond(output_tensor):
                return

        def _get_switch_index(switch_op):
            for i in range(len(switch_op.outputs)):
                if len(switch_op.outputs[i].consumers()) > 0:
                    return i
            raise ValueError("switch op's outputs don't have consumers")

        # batch norm with tf.cond
        output_no_grad_consumers = [c for c in output_tensor.consumers() if 'gradients/' not in c.name]

        if (len(output_no_grad_consumers) == 2) and (output_tensor.consumers()[0].type == 'Switch') and (output_tensor.consumers()[1].type == 'Switch'):
            switch_op1 = output_tensor.consumers()[0]
            switch_op2 = output_tensor.consumers()[1]

            index = _get_switch_index(switch_op1)
            no_grad_consumers = [c for c in switch_op1.outputs[index].consumers() if 'gradients/' not in c.name]
            cond1 = (len(no_grad_consumers) == 1)
            c1 = switch_op1.outputs[index].consumers()[0]
            cond2 = ('FusedBatchNorm' in c1.type)
            merge_op = c1.outputs[0].consumers()[0]
            cond3 = (merge_op.type == 'Merge')

            index = _get_switch_index(switch_op2)
            no_grad_consumers = [c for c in switch_op2.outputs[index].consumers() if 'gradients/' not in c.name]
            cond4 = (len(no_grad_consumers) == 1)
            c2 = switch_op2.outputs[index].consumers()[0]
            cond5 = ('FusedBatchNorm' in c2.type)
            cond6 = (len(merge_op.inputs) == 2)

            sub_cond1 = ((merge_op.inputs[0] == c1.outputs[0]) and (merge_op.inputs[1] == c2.outputs[0]))
            sub_cond2 = ((merge_op.inputs[0] == c2.outputs[0]) and (merge_op.inputs[1] == c1.outputs[0]))

            cond7 = (sub_cond1 or sub_cond2)

            final_cond = (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7)

            if final_cond is True:
                bn_scope = merge_op.name[:-10]
                gamma_op = self._get_variable_by_name(bn_scope + "gamma")
                beta_op = self._get_variable_by_name(bn_scope + "beta")
                moving_mean_op = self._get_variable_by_name(bn_scope + "moving_mean")
                moving_variance_op = self._get_variable_by_name(bn_scope + "moving_variance")

                statistics = []
                statistics.append(gamma_op)
                statistics.append(beta_op)
                statistics.append(moving_mean_op)
                # statistics.append(moving_variance_op)
                # statistics.append(c1.get_attr("epsilon"))

                self._pname_bn[weight_variable] = statistics

                return

        if 'FusedBatchNorm' in output_tensor.consumers()[0].type:
            c = output_tensor.consumers()[0]
            gamma_op = None
            beta_op = None
            bn_scope = None
            input_names = [t.op.name for t in c.inputs]
            for name in input_names:
                if name.endswith("gamma/read"):
                    gamma_op = self._graph.get_operation_by_name(name)
                if name.endswith("beta/read"):
                    beta_op = self._graph.get_operation_by_name(name)
                    bn_scope = name.rsplit("beta/read", 1)[0]
            
            assert bn_scope is not None
            moving_mean_op = self._graph.get_operation_by_name(bn_scope + "moving_mean/read")
            moving_variance_op = self._graph.get_operation_by_name(bn_scope + "moving_variance/read")

            statistics = []
            if gamma_op is not None:
                statistics.append(self._get_variable_by_name(gamma_op.inputs[0].name[:-2]))
            # else:
            #     statistics.append(gamma_op)
            statistics.append(self._get_variable_by_name(beta_op.inputs[0].name[:-2]))
            statistics.append(self._get_variable_by_name(moving_mean_op.inputs[0].name[:-2]))
            # statistics.append(moving_variance_op.inputs[0])
            # statistics.append(c.get_attr("epsilon"))

            self._pname_bn[weight_variable] = statistics
            
            return
