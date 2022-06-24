from __future__ import print_function
import mnncompress.common.MNN_compression_pb2 as compress_pb

def get_pipeline_methods(compress_proto):
    compress_types = compress_pb.CompressionAlgo.CompressionType.keys()
    used_compress_types = [compress_types[algo.type] for algo in compress_proto.algo]
    quant_types = compress_pb.LayerQuantizeParams.QuantMethod.keys()
    prune_types = compress_pb.PruneParams.PruneType.keys()

    methods = []
    for i in range(len(used_compress_types)):
        type = used_compress_types[i]
        algo = compress_proto.algo[i]

        if type == "QUANTIZE":
            method = quant_types[algo.quant_params.layer[0].method]
            for layer in algo.quant_params.layer:
                if quant_types[layer.method] == "OverflowAware":
                    method = quant_types[layer.method]
                if quant_types[layer.method] == "WinogradAware":
                    method = quant_types[layer.method]

        if type == "PRUNE":
            method = prune_types[algo.prune_params.type]

        methods.append(type + "." + method)

    return methods

def get_align_channels(value, max_value, align_channels, minimal_ratio=0.0):
    res = value // align_channels * align_channels
    if res == 0:
        if align_channels <= max_value:
            res = align_channels
        else:
            res = max_value

    if (res / float(max_value)) < minimal_ratio:
        res = int(max_value * minimal_ratio)
        res = res // align_channels * align_channels
        if res == 0:
            if align_channels <= max_value:
                res = align_channels
            else:
                res = max_value

    return res
