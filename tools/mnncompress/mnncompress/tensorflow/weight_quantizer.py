from __future__ import print_function
import tensorflow as tf
import numpy as np

from .helpers import is_weight_tensor
from .graph_checker import check_for_grad_ops
import mnncompress.common.MNN_compression_pb2 as compress_pb
import uuid
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods

_Weight_Quant_Support_Ops = ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']
_WQ_Mark_name = "_ori_weight_MNN_wq_"

def strip_wq_ops():
    graph = tf.get_default_graph()
    all_ops = graph.get_operations()
    for op in all_ops:
        if _WQ_Mark_name in op.name:
            ori_op_name = op.name.split(_WQ_Mark_name)[0]
            ori_index = int(op.name.split(_WQ_Mark_name)[1])
            ori_op = graph.get_operation_by_name(ori_op_name)
            ori_op._update_input(ori_index, op.outputs[0])

class WeightQuantizer(object):
    def __init__(self, graph=None, bits=8, debug_info=False):
        if graph is None:
            self._graph = tf.get_default_graph()
        else:
            self._graph = graph
        self._ops = self._graph.get_operations()

        if bits < 2 or bits > 8:
            raise ValueError("bits must be in [2, 8]")
        self._bits = bits
        self._clamp_value = pow(2.0, bits-1) - 1.0

        self._all_conv_and_matmul_layers = [l for l in self._ops if l.type in _Weight_Quant_Support_Ops]
        self._debug_info = debug_info
        self._prune_weight_ops = []
        self._mask = {}
        self._all_weight_tensors = []
        self._all_weight_tensor_op_map = {}
        self._fake_quant_weight_ops = []
        self._initialized = False
        self._eps = 1e-9
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0
        self._init_prune_ratios = {}
        self._find_all_variable_tensors()
        self._insert_wq_ops()

    def save_compress_params(self, filename, append=False):
        compress_proto = compress_pb.Pipeline()
        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid

        if not self._reported:
            detail = {"algorithm": "WQ", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"bits": self._bits, "init_prune_ratios": self._init_prune_ratios}}
            self._reported = mnn_logger.on_done("tensorflow", self._guid, detail)

        with tf.gfile.Open(filename, mode="wb") as f:
            f.write(compress_proto.SerializeToString())

        print("compress proto saved to:", filename)

    def _strip_wq_ops(self):
        for ori_weight_tensor, op_index in self._all_weight_tensor_op_map.items():
            op = op_index[0]
            index = op_index[1]
            op._update_input(index, ori_weight_tensor)

    def init(self, sess):
        if self._initialized is True:
            raise RuntimeError("you should only initialize weight quantizer once")

        for v in self._all_weight_tensors:
            mask = tf.cast(tf.abs(v) > 1e-9, tf.float32)
            prune_ratio = 1.0 - tf.reduce_mean(mask)
            self._mask[v] = sess.run(mask)
            prune_weight_op = tf.assign(v, v * self._mask[v])
            self._prune_weight_ops.append(prune_weight_op)

            prune_ratio = 1 - np.mean(sess.run(mask))
            self._init_prune_ratios[v.name] = prune_ratio
            self._remain_weight_num += self._mask[v].size * (1 - prune_ratio) / (32.0 / self._bits)
            print(v, "initial prune ratio:", prune_ratio)
        
        self._initialized = True

    def update(self, sess):
        if self._initialized is False:
            raise RuntimeError("please initialize the quantizer by 'weight_quantizer.init(sess)', before training the model")
        
        sess.run(self._prune_weight_ops)
        if self._debug_info:
            for v in self._all_weight_tensors:
                weight = sess.run(v)
                prune_ratio = 1 - np.mean(np.abs(weight) > 1e-9)
                print(v, "prune_ratio:", prune_ratio, "bits:", self._bits, "clamp_value:", self._clamp_value)

    def _insert_wq_ops(self):
        grad_ops = check_for_grad_ops(self._graph)
        if grad_ops:
            raise ValueError('gradient op found in graph, exiting %s\nplease invoke with inference graph only. create quantizer before construct model optimizer\n' % grad_ops)

        for v in self._all_weight_tensors:
            op = self._all_weight_tensor_op_map[v][0]
            index = self._all_weight_tensor_op_map[v][1]

            reduce_dims = []
            if op.type == "Conv2D":
                reduce_dims = [0, 1, 2]
            if op.type == "DepthwiseConv2dNative":
                reduce_dims = [0, 1, 3]
            if op.type == "MatMul":
                reduce_dims = None
                trans_a = op.get_attr("transpose_a")
                trans_b = op.get_attr("transpose_b")
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
            reduce_dims = tuple(reduce_dims)
            
            with tf.variable_scope(op.name + "_weight_quant", reuse=tf.AUTO_REUSE):
                with self._graph.gradient_override_map({'Round': 'Identity'}):
                    scales = tf.reduce_max(tf.abs(v), axis=reduce_dims, keep_dims=True) / self._clamp_value + self._eps
                    quant_w = tf.clip_by_value(tf.round(v / scales), -self._clamp_value, self._clamp_value)
                    fake_quant_w = quant_w * scales

                    op._update_input(index, fake_quant_w)

            id = tf.identity(v, name=op.name+_WQ_Mark_name+str(index))

    def _find_all_variable_tensors(self):
        for op in self._all_conv_and_matmul_layers:
            if op.type in _Weight_Quant_Support_Ops:
                if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                    weight_tensor = op.inputs[1].op.inputs[0]
                    self._total_weight_num += weight_tensor.shape.num_elements()
                    self._all_weight_tensors.append(weight_tensor)
                    self._all_weight_tensor_op_map[weight_tensor] = [op, 1]
                    if self._debug_info:
                        print("found weight tensor:", weight_tensor)

                if op.type == 'MatMul':
                    for i in range(len(op.inputs)):
                        input_tensor = op.inputs[i]
                        if is_weight_tensor(input_tensor):
                            weight_tensor = input_tensor.op.inputs[0]
                            self._total_weight_num += weight_tensor.shape.num_elements()
                            self._all_weight_tensors.append(weight_tensor)
                            self._all_weight_tensor_op_map[weight_tensor] = [op, i]
                            if self._debug_info:
                                print("found weight tensor:", weight_tensor)
