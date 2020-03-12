# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import os
import sys
import argparse
from .mnn_fb import Net, Op, Blob
from .mnn_fb import TensorDescribe, OpType
from .mnn_fb import Convolution2DCommon, Convolution2D
from .mnn_fb import OpParameter
from .utils  import opname as OpName
try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None


def check_pydot():
    """check whehter pydot is ready or not"""
    if pydot is None:
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except OSError:
        return False

def add_edge(dot, src_name, dst_name):
    "find the nodes with src_name/dst_name and add a edge"
    if not src_name == dst_name:
        src_node = dot.get_node(src_name)
        dst_node = dot.get_node(dst_name)
        if src_node and dst_node and not dot.get_edge(src_name, dst_name):
            dot.add_edge(pydot.Edge(src_name, dst_name))

def mnn_to_dot(mnn_file):
    "load a mnn file and create a dot file"
    if not os.path.exists(mnn_file):
        return None
    with open(mnn_file, 'rb') as f:
        buf = f.read()
        f.close()
    buf = bytearray(buf)
    net = Net.Net.GetRootAsNet(buf, 0)
    op_num = net.OplistsLength()
    if op_num > 1400:
        print("This graph contains a large number of nodes,abort the visual operation")
        return None
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    for idx in range(op_num):
        op = net.Oplists(idx)
        name = "op_" + str(idx)
        op_type = op.Type()
        label = OpName.optype_to_name(op_type)
        main_type = op.MainType()

        if main_type == OpParameter.OpParameter.Convolution2D:
            union_conv2d = Convolution2D.Convolution2D()
            union_conv2d.Init(op.Main().Bytes, op.Main().Pos)
            common = union_conv2d.Common()
            KernelX = common.KernelX()
            KernelY = common.KernelY()
            input_channels = common.InputCount()
            output_channels = common.OutputCount()
            if input_channels == 0:
                input_channels = -1
            if output_channels == 0:
                output_channels = -1
            label += "\nshape:" + str(input_channels) + "*" + str(output_channels) \
                     + "*" + str(KernelX) + "*" + str(KernelY)
        if op_type != OpType.OpType.Input:
            node = pydot.Node(name, label=label, shape="egg", color='blue')
            dot.add_node(node)
    ts_num = net.TensorNameLength()
    for idx in range(ts_num):
        ts_name = net.TensorName(idx)
        name = "tensor_" + str(idx)
        label = ts_name
        if isinstance(label, bytes):
            label = label.decode()
        if label == '':
            print("tensor name is invalid")
            label = '?'
        node = dot.get_node(name)
        node = pydot.Node(name, label=label, shape="record", color='red')
        dot.add_node(node)

    for idx in range(op_num):
        op = net.Oplists(idx)
        name_op = "op_" + str(idx)
        input_len = op.InputIndexesLength()
        output_len = op.OutputIndexesLength()
        for idx_ts in range(input_len):
            name_tensor = "tensor_" + str(op.InputIndexes(idx_ts))
            add_edge(dot, name_tensor, name_op)
        for idx_ts in range(output_len):
            name_tensor = "tensor_" + str(op.OutputIndexes(idx_ts))
            add_edge(dot, name_op, name_tensor)

    return dot


def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("target_mnn", type=str,\
        help="target mnn file, for example:target.mnn")
    parser.add_argument("target_pic", type=str,\
        help="target picture file, for example:target.png, or target.jpeg")
    args = parser.parse_args()
    target_mnn = args.target_mnn
    target_pic = args.target_pic
    if not os.path.exists(target_mnn):
        return -1
    if target_pic is None:
        return -1
    if not check_pydot():
        raise ImportError('You must install pydot and graphviz')
        return -1
    _, extension = os.path.splitext(target_pic)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot = mnn_to_dot(target_mnn)
    if dot is not None:
        dot.write(target_pic, format=extension)
    return 0


if __name__ == "__main__":
    main()
