import os
import argparse
from MNN.tools import mnnconvert

def convert_json(json_path, output_name):

    if not os.path.exists(json_path):
        print(f"错误: 输入文件未找到 '{json_path}'")
        return

    print(f"开始转换: {json_path} -> {output_name}")

    convert_args = [
        '',
        '-f',
        'JSON',
        '--modelFile',
        json_path,
        '--MNNModel',
        output_name,
    ]

    try:
        mnnconvert.convert(convert_args)
        print("转换成功！")
        print(f"二进制模型已保存至: {output_name}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")

if __name__ == '__main__':

    input_json_file = './model/llm.mnn.json'
    
    output_mnn_file = './model/llm.mnn'

    
    convert_json(input_json_file, output_mnn_file)