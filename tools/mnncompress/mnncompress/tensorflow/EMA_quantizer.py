from __future__ import print_function
import tensorflow as tf
import numpy as np

from ..common import MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
import uuid
from .helpers import get_input_tensor_index, is_weight_tensor, get_variable_by_tensor
from .helpers import find_quant_output_tensor, find_bias_add_op, get_batch_norm_statistics, get_variable_by_name
from .graph_checker import check_for_grad_ops


_MNN_compress_scope = 'MNN_QAT_'

_MNN_mark_start_name = 'MNN_QAT_MARK_START'
_MNN_mark_end_name = 'MNN_QAT_MARK_END'

_MNN_variable_collection_name = 'MNN_QAT_variables'

def MNN_QAT_variables():
    all_vars = tf.global_variables()
    return [v for v in all_vars if 'MNN_QAT' in v.name]

_Quant_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']

_Variable_types = ['Variable', 'VariableV2']

_MNN_QAT_DEBUG = False


def strip_MNN_QAT_ops(graph):
    '''only invoke this when you need to save the model for inference'''
    def get_MNN_QAT_scope(name, match):
        name_split = name.split('/')
        for i in range(len(name_split)):
            if name_split[i] == match:
                assert i > 0, "get MNN QAT scope error"
                return name_split[i-1]
        return ''

    all_ops = graph.get_operations()
    start_op_names = [op.name for op in all_ops if op.name.endswith(_MNN_mark_start_name) and op.type in ['Abs', 'RealDiv'] and 'gradients/' not in op.name]
    end_op_names = [op.name for op in all_ops if op.name.endswith(_MNN_mark_end_name+'/Merge') and op.type == 'Merge' and 'gradients/' not in op.name]

    for start_name in start_op_names:
        start_scope = get_MNN_QAT_scope(start_name, _MNN_mark_start_name)
        end_name = ''
        for end in end_op_names:
            if start_scope in end:
                end_name = end
                break
        assert end_name != '', "error: end op name not found"

        start_op = graph.get_operation_by_name(start_name)
        start_input_tensor = start_op.inputs[0]
        end_tensor = graph.get_tensor_by_name(end_name+':0')
        end_consumers = end_tensor.consumers()

        for c in end_consumers:
            indices = get_input_tensor_index(c, end_tensor)
            for id in indices:
                c._update_input(id, start_input_tensor)

def debug_print(*argv):
    if _MNN_QAT_DEBUG:
        print("debug info: ", end="")
        for arg in argv:
            print(arg, end=' ')
        print()

def get_qat_state_placeholder(graph=None):
    if graph is None:
        graph = tf.get_default_graph()

    all_tensor_names = [tensor.name for op in graph.get_operations() for tensor in op.values()]
    for name in all_tensor_names:
        if 'MNN_QAT_is_training:0' in name:
            return graph.get_tensor_by_name(name)

    raise ValueError('Could not find QAT state placeholder')


class EMAQuantizer(object):
    def __init__(self, graph=None, is_training = None, skip_quant_layers = [], skip_quant_flag = [], bits = 8, debug_info = False, retain_sparsity=False):
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph
        self._ops = self._graph.get_operations()
        self.skip_quant_layers = skip_quant_layers
        self.skip_quant_flag = skip_quant_flag
        self._all_conv_and_matmul_layers = [l for l in self._ops if l.type in _Quant_Support_Ops]
        self.quant_layer_names = [l.name for l in self._all_conv_and_matmul_layers]

        skip_flag_usable = False
        for sf in self.skip_quant_flag:
            for item in self.quant_layer_names:
                if sf in item:
                    self.skip_quant_layers.append(item)
                    skip_flag_usable = True
        
        if skip_quant_flag != [] and skip_flag_usable == False:
            raise ValueError("skip_quant_flag invalid")

        for item in self.skip_quant_layers:
            if item in [l.name for l in self._all_conv_and_matmul_layers]:
                self.quant_layer_names.remove(item)
            else:
                raise ValueError(item+" not found in graph ops, availables are: "+str([l.name for l in self._all_conv_and_matmul_layers]))

        if bits < 2 or bits > 8:
            raise ValueError("bits must be a integer in [2, 8]")
        self.bits = bits
        self._clamp_value = float(pow(2, bits-1) - 1)

        global _MNN_QAT_DEBUG
        _MNN_QAT_DEBUG = debug_info

        self._retain_sparsity = retain_sparsity
        self._momentum = 0.99
        if is_training is not None:
            if not isinstance(is_training, bool):
                raise ValueError("is_training should be None or python bool")
        if isinstance(is_training, bool):
            self._is_training = tf.constant(is_training)
        else:
            print("is_training set as: tf.placeholder(tf.bool, name='MNN_QAT_is_training')")
            self._is_training = tf.placeholder(tf.bool, name='MNN_QAT_is_training')
        self._original_tensor_consumer_index_map = {}
        self._quant_tensor_consumer_index_map = {}
        self._feature_scale_original_tensor_dims_map = {}
        self._weight_scale_original_tensor_map = {}
        self._model_per_layer_scale_info = []
        self._prune_weight_ops = []
        self._mask = {}
        self._all_weight_tensors = []
        self._feature_scale_init_op = {}
        self._op_scale_initialized = {}
        self._initialized = False
        self._layer_input_clamp_value = {}
        self._eps = 1e-9
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0
        self._init_prune_ratios = {}
        self._op_name_weight_tensor = {}
        self._skip_quant_weight_tensors = []
        self._find_all_variable_tensors()
        self._insert_quant_ops()

    @property
    def is_training(self):
        return self._is_training

    def init(self, sess):
        self._initialized = True

        for name in self.skip_quant_layers:
            if name in self._op_name_weight_tensor.keys():
                weight_tensor = self._op_name_weight_tensor[name]
                self._skip_quant_weight_tensors.append(weight_tensor)
        
        for v in self._all_weight_tensors:
            self._mask[v] = sess.run(self._mask[v])
            self._init_prune_ratios[v.name] = 1. - np.mean(self._mask[v])
            self._total_weight_num += self._mask[v].size
            if v in self._skip_quant_weight_tensors:
                self._remain_weight_num += self._mask[v].size * (1 - self._init_prune_ratios[v.name]) / 4.0
            else:
                self._remain_weight_num += self._mask[v].size * (1 - self._init_prune_ratios[v.name]) / (32.0 / self.bits)

    def update(self, sess):
        if not self._initialized:
            raise RuntimeError("EMAQuantizer is not initialized, please call 'init(sess)' method before training loop")
        if self._retain_sparsity:
            sess.run(self._prune_weight_ops)
        if _MNN_QAT_DEBUG:
            for v in self._all_weight_tensors:
                weight = sess.run(v)
                prune_ratio = 1 - np.mean(np.abs(weight) > self._eps)
                print(v, "prune_ratio:", prune_ratio)

    def _generate_mask_and_prune_ops(self, v):
        if self._retain_sparsity:
            th = self._eps
        else:
            th = -1.0
        
        self._mask[v] = tf.cast(tf.abs(v) > th, tf.float32)

        if isinstance(v, tf.Tensor):
            return
        
        if self._retain_sparsity:
            prune_weight_op = tf.assign(v, v * self._mask[v])
            self._prune_weight_ops.append(prune_weight_op)

    def _find_all_variable_tensors(self):
        for op in self._all_conv_and_matmul_layers:
            if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                weight_tensor = get_variable_by_tensor(op.inputs[1])
                self._all_weight_tensors.append(weight_tensor)
                self._op_name_weight_tensor[op.name] = weight_tensor
                self._generate_mask_and_prune_ops(weight_tensor)

            if op.type == 'MatMul':
                find_weight = False
                for i in range(len(op.inputs)):
                    input_tensor = op.inputs[i]
                    if is_weight_tensor(input_tensor):
                        find_weight = True
                        weight_tensor = get_variable_by_tensor(input_tensor)
                        self._all_weight_tensors.append(weight_tensor)
                        self._op_name_weight_tensor[op.name] = weight_tensor
                        self._generate_mask_and_prune_ops(weight_tensor)
                
                if not find_weight:
                    if op.name in self.quant_layer_names:
                        self.quant_layer_names.remove(op.name)
                        print("no weight find for {}, skip".format(op.name))

    def strip_qat_ops(self):
        '''only invoke this when you need to save the model for inference'''
        for tensor in self._original_tensor_consumer_index_map.keys():
            for index, consumer in self._original_tensor_consumer_index_map[tensor]:
                consumer._update_input(index, tensor)
                debug_print("on save:", tensor.name, consumer.name, consumer.type, index)

    def save_compress_params(self, filename, sess, append=False):
        def get_save_tensor_name(name):
            if name.endswith(':0'):
                return name[:-2]
            else:
                return name

        compress_proto = compress_pb.Pipeline()

        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

            pop_index = []
            for i in range(len(compress_proto.algo)):
                if compress_proto.algo[i].type == compress_pb.CompressionAlgo.CompressionType.QUANTIZE:
                    pop_index.append(i)
            for i in pop_index:
                compress_proto.algo.pop(i)

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid
        quant_algorithm = compress_proto.algo.add()
        for layer_info in self._model_per_layer_scale_info:
            l = quant_algorithm.quant_params.layer.add()
            # {"weight_info": weight_tensor_scale, "input_info": input_scale_and_dim, "output_info": output_scale_and_dim}
            weight_info = layer_info["weight_info"]
            input_info = layer_info["input_info"]
            output_info = layer_info["output_info"]

            if weight_info is not []:
                weight_params = compress_pb.LayerQuantizeParams.WeightParams()
                weight_params.name = get_save_tensor_name(get_variable_by_tensor(weight_info[0]).name)
                scales = sess.run(weight_info[1])
                scales = scales.reshape((scales.size))
                bn_stat = layer_info["bn_stat"]

                gamma = np.ones_like(scales)
                if bn_stat[0] is not None:
                    gamma = sess.run(bn_stat[0])
                    gamma = gamma.reshape((gamma.size))
                rstd = np.ones_like(scales)
                if bn_stat[3] is not None:
                    bn_var = sess.run(bn_stat[3])
                    bn_var = bn_var.reshape((bn_var.size))
                    rstd = 1. / np.sqrt(bn_var + bn_stat[4])
                    scales = scales * gamma * rstd

                for s in scales:
                    weight_params.scales.append(abs(s))
                weight_params.bits = self.bits
                weight_params.clamp_min = -int(self._clamp_value)
                weight_params.clamp_max = int(self._clamp_value)
                l.weight.append(weight_params)

            input_params = compress_pb.LayerQuantizeParams.ActivationParams()
            input_params.name = get_save_tensor_name(input_info[0].name)
            scale = sess.run(input_info[1]).tolist()
            debug_print(input_info[1].name, scale)
            # for count in range(input_info[2]):
            input_params.scales.append(scale)
            if len(input_info) == 4:
                clamp = sess.run(input_info[3]).tolist()
                input_params.clamp_min = -int(clamp)
                input_params.clamp_max = int(clamp)
            l.input.append(input_params)

            if len(input_info) == 6:
                input_params2 = compress_pb.LayerQuantizeParams.ActivationParams()
                input_params2.name = get_save_tensor_name(input_info[3].name)
                scale = sess.run(input_info[4]).tolist()
                debug_print(input_info[4].name, scale)
                for count in range(input_info[5]):
                    input_params2.scales.append(scale)
                l.input.append(input_params2)
            
            output_params = compress_pb.LayerQuantizeParams.ActivationParams()
            output_params.name = get_save_tensor_name(output_info[0].name)
            scale = sess.run(output_info[1])
            debug_print(output_info[1].name, scale)
            # for count in range(output_info[2]):
            output_params.scales.append(scale)
            if len(output_info) == 4:
                clamp = sess.run(output_info[3]).tolist()
                output_params.clamp_min = -int(clamp)
                output_params.clamp_max = int(clamp)
            l.output.append(output_params)

        if not self._reported:
            detail = {"algorithm": "EMAQ", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"bits": self.bits, "skip_quant_layers": self.skip_quant_layers, "init_prune_ratios": self._init_prune_ratios}}
            self._reported = mnn_logger.on_done("tensorflow", self._guid, detail)

        with tf.gfile.Open(filename, mode="wb") as f:
            f.write(compress_proto.SerializeToString())

        print("compress proto saved to:", filename)

    def _recover_train_quant_graph(self):
        if self._quant_tensor_consumer_index_map == {}:
            return
            
        for tensor in self._quant_tensor_consumer_index_map.keys():
            for index, consumer in self._quant_tensor_consumer_index_map[tensor]:
                consumer._update_input(index, tensor)
                debug_print("on recover:", consumer.name, consumer.type, index, tensor.name)

    def _insert_quant_ops(self):
        grad_ops = check_for_grad_ops(self._graph)
        if grad_ops:
            raise ValueError('gradient op found in graph, exiting %s\nplease invoke with inference graph only. create quantizer before construct model optimizer\n' % grad_ops)
        scope_index = -1
        for layer_name in self.quant_layer_names:
            if 'gradients/' in layer_name and '_grad' in layer_name:
                continue
            scope_index += 1
            self._quant_one_layer(layer_name, _MNN_compress_scope + str(scope_index))

    def _quant_one_layer(self, layer_name, scope):
        all_op_names_ops = {op.name : op for op in self._ops}
        if layer_name not in all_op_names_ops.keys():
            raise ValueError("%s is not in the graph." % layer_name)

        if all_op_names_ops[layer_name].type not in _Quant_Support_Ops:
            raise ValueError("op name: %s, type = %s is not supported." % (layer_name, all_op_names_ops[layer_name].type))

        if layer_name not in self._layer_input_clamp_value.keys():
            self._layer_input_clamp_value[layer_name] = None

        if all_op_names_ops[layer_name].type == 'Conv2D' or all_op_names_ops[layer_name].type == 'DepthwiseConv2dNative':
            print("fake quant", layer_name, all_op_names_ops[layer_name].type)
            depthwise = False
            if all_op_names_ops[layer_name].type == 'DepthwiseConv2dNative':
                depthwise = True
            self._op_scale_initialized[all_op_names_ops[layer_name]] = {}
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self._op_scale_initialized[all_op_names_ops[layer_name]]['input'] = tf.get_variable(name=layer_name+'_input_scale_initialized', initializer=0, trainable=False)
                self._op_scale_initialized[all_op_names_ops[layer_name]]['output'] = tf.get_variable(name=layer_name+'_output_scale_initialized', initializer=0, trainable=False)
            tf.add_to_collection(_MNN_variable_collection_name, self._op_scale_initialized[all_op_names_ops[layer_name]]['input'])
            tf.add_to_collection(_MNN_variable_collection_name, self._op_scale_initialized[all_op_names_ops[layer_name]]['output'])
            res = self._quant_conv(all_op_names_ops[layer_name], scope, depthwise)
            self._model_per_layer_scale_info.append(res)

        if all_op_names_ops[layer_name].type == 'MatMul':
            print("fake quant", layer_name, all_op_names_ops[layer_name].type)
            self._op_scale_initialized[all_op_names_ops[layer_name]] = {}
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self._op_scale_initialized[all_op_names_ops[layer_name]]['input'] = tf.get_variable(name=layer_name+'_input_scale_initialized', initializer=0, trainable=False)
                self._op_scale_initialized[all_op_names_ops[layer_name]]['output'] = tf.get_variable(name=layer_name+'_output_scale_initialized', initializer=0, trainable=False)
            tf.add_to_collection(_MNN_variable_collection_name, self._op_scale_initialized[all_op_names_ops[layer_name]]['input'])
            tf.add_to_collection(_MNN_variable_collection_name, self._op_scale_initialized[all_op_names_ops[layer_name]]['output'])
            res = self._quant_matmul(all_op_names_ops[layer_name], scope)
            self._model_per_layer_scale_info.append(res)

    def _get_weight_variable_name(self, tensor):
        return tensor.op.inputs[0].op.name

    def _save_original_tensor_consumer_info(self, input_tensor, index, consumer_op):
        if input_tensor in self._original_tensor_consumer_index_map.keys():
            self._original_tensor_consumer_index_map[input_tensor].append([index, consumer_op])
        else:
            self._original_tensor_consumer_index_map[input_tensor] = [[index, consumer_op]]

    def _save_quant_tensor_consumer_info(self, input_tensor, index, consumer_op):
        if input_tensor in self._quant_tensor_consumer_index_map.keys():
            self._quant_tensor_consumer_index_map[input_tensor].append([index, consumer_op])
        else:
            self._quant_tensor_consumer_index_map[input_tensor] = [[index, consumer_op]]

    def _get_quant_scope(self, name, match):
        name_split = name.split("/")
        return_scope = ''
        for i in range(len(name_split)):
            cond1 = _MNN_compress_scope in name_split[i]
            cond2 = match in name_split[i]
            if cond1 and cond2:
                for j in range(i+1):
                    return_scope = return_scope + name_split[j] + '/'
                
                return return_scope[:-1]

        return return_scope

    def _quant_conv(self, conv2d_op, scope, depthwise):
        input_feature = conv2d_op.inputs[0]
        weight_tensor = conv2d_op.inputs[1]

        reduce_dims = [0, 1, 2]
        if depthwise:
            reduce_dims = [0, 1, 3]

        weight_tensor_scale = []
        input_scale_and_dim = []
        output_scale_and_dim = []

        input_scale = None
        weight_scales = None
        output_scale = None
        init_moving_scale_op = None

        if _MNN_compress_scope not in input_feature.name:
            conv2d_op_input, input_scale, dims, init_moving_scale_op = self._fake_quant_feature(input_feature, scope + '_input', conv2d_op, 'input')
            input_scale_and_dim.append(input_feature)
            input_scale_and_dim.append(input_scale)
            input_scale_and_dim.append(dims)
            conv2d_op._update_input(0, conv2d_op_input)
            self._save_quant_tensor_consumer_info(conv2d_op_input, 0, conv2d_op)
            self._save_original_tensor_consumer_info(input_feature, 0, conv2d_op)
        else:
            pre_output_scope = self._get_quant_scope(input_feature.name, '_output')
            pre_output_scale_name = pre_output_scope + '/moving_average_scale'
            pre_output_scale = get_variable_by_name(pre_output_scale_name)
            input_scale = pre_output_scale
            init_moving_scale_op = self._feature_scale_init_op[pre_output_scale]
            assert pre_output_scale in self._feature_scale_original_tensor_dims_map.keys()
            input_scale_and_dim.append(self._feature_scale_original_tensor_dims_map[pre_output_scale][0])
            input_scale_and_dim.append(pre_output_scale)
            input_scale_and_dim.append(self._feature_scale_original_tensor_dims_map[pre_output_scale][1])

            debug_print("pre_output_scale:", pre_output_scale, pre_output_scale.name)
            debug_print(pre_output_scale in self._feature_scale_original_tensor_dims_map)
            debug_print(self._feature_scale_original_tensor_dims_map[pre_output_scale])
            debug_print(input_feature.name, "feature already quant")

        # if _MNN_compress_scope not in weight_tensor.name:
        new_weight_tensor, weight_scales = self._fake_quant_weight(weight_tensor, reduce_dims, scope + '_weight')
        weight_tensor_scale = [weight_tensor, weight_scales]
        conv2d_op._update_input(1, new_weight_tensor)
        self._save_quant_tensor_consumer_info(new_weight_tensor, 1, conv2d_op)
        self._save_original_tensor_consumer_info(weight_tensor, 1, conv2d_op)

        bias_add_op = find_bias_add_op(conv2d_op)
        if bias_add_op is not None:
            for index in range(len(bias_add_op.inputs)):
                input_tensor = bias_add_op.inputs[index]
                if is_weight_tensor(input_tensor):
                    debug_print("found bias:", input_tensor.name, "index:", index)
                    new_bias = self._fake_quant_bias(input_tensor, input_scale, weight_scales, scope + '_bias_' + str(index), init_moving_scale_op)
                    bias_add_op._update_input(index, new_bias)
                    self._save_quant_tensor_consumer_info(new_bias, index, bias_add_op)
                    self._save_original_tensor_consumer_info(input_tensor, index, bias_add_op)

        output_tensor = find_quant_output_tensor(conv2d_op)
        output_consumers = output_tensor.consumers()
        # if _MNN_compress_scope not in output_consumers[0].name:
        new_output_tensor, output_scale, dims, init_moving_scale_op = self._fake_quant_feature(output_tensor, scope + '_output', conv2d_op, 'output')
        output_scale_and_dim.append(output_tensor)
        output_scale_and_dim.append(output_scale)
        output_scale_and_dim.append(dims)
        for c in output_consumers:
            index = get_input_tensor_index(c, output_tensor)
            for id in index:
                c._update_input(id, new_output_tensor)
                self._save_quant_tensor_consumer_info(new_output_tensor, id, c)
                self._save_original_tensor_consumer_info(output_tensor, id, c)

        input_scale_and_dim.append(tf.constant(self._clamp_value))
        output_scale_and_dim.append(tf.constant(self._clamp_value))

        bn_stat = get_batch_norm_statistics(conv2d_op)

        return {"weight_info": weight_tensor_scale, "input_info": input_scale_and_dim, "output_info": output_scale_and_dim, "bn_stat": bn_stat}
        
    def _quant_matmul(self, matmul_op, scope):
        weight_tensor_scale = []
        input_scale_and_dim = []
        output_scale_and_dim = []

        input_scale = None
        weight_scales = None
        output_scale = None
        init_moving_scale_op = None

        for index in range(len(matmul_op.inputs)):
            input_tensor = matmul_op.inputs[index]
            if is_weight_tensor(input_tensor):
                debug_print(input_tensor.name, "is weight tensor")
                # if _MNN_compress_scope not in input_tensor.name:
                reduce_dims = None
                trans_a = matmul_op.get_attr("transpose_a")
                trans_b = matmul_op.get_attr("transpose_b")
                if index == 0:
                    if not trans_a:
                        reduce_dims = [1]
                    else:
                        reduce_dims = [0]
                if index == 1:
                    if not trans_b:
                        reduce_dims = [0]
                    else:
                        reduce_dims = [1]

                new_weight_tensor, weight_scales = self._fake_quant_weight(input_tensor, reduce_dims, scope + '_weight_' + str(index))
                weight_tensor_scale = [input_tensor, weight_scales]
                matmul_op._update_input(index, new_weight_tensor)
                self._save_quant_tensor_consumer_info(new_weight_tensor, index, matmul_op)
                self._save_original_tensor_consumer_info(input_tensor, index, matmul_op)
                debug_print("weight quant updated")
            else:
                debug_print(input_tensor.name, "is input tensor")
                if _MNN_compress_scope not in input_tensor.name:
                    new_op_input, input_scale, dims, init_moving_scale_op = self._fake_quant_feature(input_tensor, scope + '_input_' + str(index), matmul_op, 'input')
                    input_scale_and_dim.append(input_tensor)
                    input_scale_and_dim.append(input_scale)
                    input_scale_and_dim.append(dims)
                    matmul_op._update_input(index, new_op_input)
                    self._save_quant_tensor_consumer_info(new_op_input, index, matmul_op)
                    self._save_original_tensor_consumer_info(input_tensor, index, matmul_op)
                    debug_print("feature quant updated")
                else:
                    pre_output_scope = self._get_quant_scope(input_tensor.name, '_output')
                    pre_output_scale_name = pre_output_scope + '/moving_average_scale'
                    pre_output_scale = get_variable_by_name(pre_output_scale_name)
                    input_scale = pre_output_scale
                    init_moving_scale_op = self._feature_scale_init_op[pre_output_scale]
                    assert pre_output_scale in self._feature_scale_original_tensor_dims_map.keys()
                    input_scale_and_dim.append(self._feature_scale_original_tensor_dims_map[pre_output_scale][0])
                    input_scale_and_dim.append(pre_output_scale)
                    input_scale_and_dim.append(self._feature_scale_original_tensor_dims_map[pre_output_scale][1])

                    debug_print("pre_output_scale:", pre_output_scale, pre_output_scale.name)
                    debug_print(pre_output_scale in self._feature_scale_original_tensor_dims_map)
                    debug_print(self._feature_scale_original_tensor_dims_map[pre_output_scale])
                    debug_print(input_tensor.name, "feature already quant")

        bias_add_op = find_bias_add_op(matmul_op)
        if bias_add_op is not None:
            for index in range(len(bias_add_op.inputs)):
                input_tensor = bias_add_op.inputs[index]
                if is_weight_tensor(input_tensor):
                    debug_print("found bias:", input_tensor.name, "index:", index)
                    new_bias = self._fake_quant_bias(input_tensor, input_scale, weight_scales, scope + '_bias_' + str(index), init_moving_scale_op)
                    bias_add_op._update_input(index, new_bias)
                    self._save_quant_tensor_consumer_info(new_bias, index, bias_add_op)
                    self._save_original_tensor_consumer_info(input_tensor, index, bias_add_op)

        output_tensor = find_quant_output_tensor(matmul_op)
        output_consumers = output_tensor.consumers()
        # if _MNN_compress_scope not in output_consumers[0].name:
        new_output_tensor, output_scale, dims, init_moving_scale_op = self._fake_quant_feature(output_tensor, scope + '_output', matmul_op, 'output')
        output_scale_and_dim.append(output_tensor)
        output_scale_and_dim.append(output_scale)
        output_scale_and_dim.append(dims)
        for c in output_consumers:
            index = get_input_tensor_index(c, output_tensor)
            for id in index:
                c._update_input(id, new_output_tensor)
                self._save_quant_tensor_consumer_info(new_output_tensor, id, c)
                self._save_original_tensor_consumer_info(output_tensor, id, c)

        input_scale_and_dim.append(tf.constant(self._clamp_value))
        output_scale_and_dim.append(tf.constant(self._clamp_value))

        bn_stat = get_batch_norm_statistics(matmul_op)

        return {"weight_info": weight_tensor_scale, "input_info": input_scale_and_dim, "output_info": output_scale_and_dim, "bn_stat": bn_stat}

    def _fake_quant_bias(self, bias_tensor, input_scale, weight_scales, scope, init_moving_scale_op):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with self._graph.gradient_override_map({'Round': 'Identity'}):
                with tf.control_dependencies([init_moving_scale_op]):
                    weight_scales_reshape = tf.reshape(weight_scales, bias_tensor.shape)
                    bias_scale = input_scale * weight_scales_reshape
                    bias_scale = bias_scale + tf.cast(tf.less(tf.abs(bias_scale), 1e-6), tf.float32) * self._eps
                    before_round = tf.divide(bias_tensor, bias_scale, name=_MNN_mark_start_name)
                    before_round = before_round + 1e-3 * tf.sign(before_round)
                    quant_bias = tf.round(before_round)
                    new_bias = quant_bias * bias_scale
                    new_bias_tensor = tf.cond(tf.constant(True), lambda: new_bias, lambda: bias_tensor, name=_MNN_mark_end_name)
                    
                    return new_bias_tensor

    def _fake_quant_weight(self, weight_tensor, reduce_dims, scope, clamp_value=None):
        if clamp_value is None:
            clamp_value = self._clamp_value

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with self._graph.gradient_override_map({'Round': 'Identity'}):
                scales = tf.reduce_max(tf.abs(weight_tensor, name=_MNN_mark_start_name), axis=reduce_dims, keep_dims=True) / clamp_value + self._eps
                weight_scales = tf.get_variable(name='weight_scales', initializer=tf.zeros_like(scales), trainable=False)
                tf.add_to_collection(_MNN_variable_collection_name, weight_scales)
                scales_assign = tf.assign(weight_scales, scales)
                with tf.control_dependencies([scales_assign]):
                    quant_w = tf.clip_by_value(tf.round(weight_tensor / scales), -clamp_value, clamp_value)
                    fake_quant_w = quant_w * scales

            if _MNN_QAT_DEBUG:
                print_op = tf.print(scope, "weight clamp value:", clamp_value)
                with tf.control_dependencies([print_op]):
                    new_weight_tensor = tf.cond(tf.constant(True), lambda: fake_quant_w, lambda: weight_tensor, name=_MNN_mark_end_name)
            else:
                new_weight_tensor = tf.cond(tf.constant(True), lambda: fake_quant_w, lambda: weight_tensor, name=_MNN_mark_end_name)

        self._weight_scale_original_tensor_map[weight_scales] = weight_tensor

        return new_weight_tensor, weight_scales

    def _fake_quant_feature(self, feature, scope, op, type, clamp_value=None):
        if clamp_value is None:
            clamp_value = self._clamp_value
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with self._graph.gradient_override_map({'Round': 'Identity'}):
                feature_abs_max = tf.reduce_max(tf.abs(feature, name=_MNN_mark_start_name))
                if _MNN_QAT_DEBUG:
                    print_op = tf.print("op:", op.name, type, "feature_abs_max, clamp_value:", feature_abs_max, clamp_value)
                    with tf.control_dependencies([print_op]):
                        instance_scale = feature_abs_max / clamp_value
                else:
                    instance_scale = feature_abs_max / clamp_value
                moving_average_scale = tf.get_variable(name='moving_average_scale', initializer=0., trainable=False)
                tf.add_to_collection(_MNN_variable_collection_name, moving_average_scale)

                def init():
                    init_assign = tf.assign(moving_average_scale, instance_scale)
                    initialize_op = tf.assign(self._op_scale_initialized[op][type], 1)
                    if _MNN_QAT_DEBUG:
                        print_op = tf.print("init", moving_average_scale.name, init_assign)
                        with tf.control_dependencies([print_op, initialize_op]):
                            return tf.identity(init_assign)
                    else:
                        with tf.control_dependencies([initialize_op]):
                            return tf.identity(init_assign)

                def do_nothing():
                    return moving_average_scale

                init_moving_scale = tf.cond(tf.equal(self._op_scale_initialized[op][type], 0), init, do_nothing)
                self._feature_scale_init_op[moving_average_scale] = init_moving_scale

                def fake_quant():
                    def moving_average_scale_update():
                        if _MNN_QAT_DEBUG:
                            with tf.control_dependencies([init_moving_scale]):
                                assign_scale = tf.assign(moving_average_scale, moving_average_scale * self._momentum + instance_scale * (1 - self._momentum))
                                with tf.control_dependencies([assign_scale]):
                                    print_op = tf.print("train:", scope + "_scale:", instance_scale, moving_average_scale)
                            with tf.control_dependencies([assign_scale, print_op]):
                                return tf.identity(instance_scale)
                        else:
                            with tf.control_dependencies([init_moving_scale]):
                                assign_scale = tf.assign(moving_average_scale, moving_average_scale * self._momentum + instance_scale * (1 - self._momentum))
                            with tf.control_dependencies([assign_scale]):
                                return tf.identity(instance_scale)

                    def return_average_scale():
                        if _MNN_QAT_DEBUG:
                            print_op = tf.print("test:", moving_average_scale.name, moving_average_scale)
                            with tf.control_dependencies([print_op]):
                                return tf.identity(moving_average_scale)
                        else:
                            return tf.identity(moving_average_scale)

                    scale = tf.cond(self._is_training, moving_average_scale_update, return_average_scale)
                    quant_x = tf.clip_by_value(tf.round(feature / scale), -clamp_value, clamp_value)
                    fake_quant_x = quant_x * scale
                    
                    return fake_quant_x

                new_feature = tf.cond(tf.constant(True), fake_quant, lambda:feature, name=_MNN_mark_end_name)
        
        self._feature_scale_original_tensor_dims_map[moving_average_scale] = [feature, feature.shape.as_list()[-1]]
        
        return new_feature, moving_average_scale, feature.shape.as_list()[-1], init_moving_scale
