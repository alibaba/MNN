import argparse
import os
import numpy as np
import mnncompress.common.MNN_compression_pb2 as compression

parser = argparse.ArgumentParser(
    description="fix asr QAT quant table.")
parser.add_argument("--qat_quant_table", required=True, type=str,
                    help="qat_quant_table binary filename.")
parser.add_argument("--quant_weight_io_map_npy", required=True, type=str,
                    help="quant_weight_io_map_npy filename.")

args = parser.parse_args()


quant_weight_io_map = np.load(args.quant_weight_io_map_npy, allow_pickle=True).item()

qat_pipeline = compression.Pipeline()
with open(args.qat_quant_table, "rb") as f:
    qat_pipeline.ParseFromString(f.read())


for algo in qat_pipeline.algo:
    if compression.CompressionAlgo.CompressionType.Name(algo.type) == 'QUANTIZE':
        for layer in algo.quant_params.layer:
            qat_weight_name = layer.weight[0].name
            qat_input_name = layer.input[0].name
            qat_output_name = layer.output[0].name

            if qat_weight_name not in quant_weight_io_map.keys():
                raise ValueError(qat_weight_name+" not found in quant_weight_io_map_npy file")

            fixed_input_name = quant_weight_io_map[qat_weight_name][0]
            fixed_output_name = quant_weight_io_map[qat_weight_name][1]
            op_name = quant_weight_io_map[qat_weight_name][2]

            if qat_input_name == fixed_input_name and qat_output_name == fixed_output_name:
                print("no need to fix:", op_name, qat_weight_name, qat_input_name, qat_output_name)
                continue

            layer.input[0].name = fixed_input_name
            layer.output[0].name = fixed_output_name
            print("fixed:", op_name, qat_weight_name, qat_input_name, qat_output_name)
        
        break


with open("qat_quant_table_fixed.bin", "wb") as f:
    f.write(qat_pipeline.SerializeToString())

print("fixed quant params saved to: qat_quant_table_fixed.bin")
