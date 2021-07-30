import os
from os import makedirs
from os.path import join, basename, exists
from shutil import copy, rmtree
import argparse

parser = argparse.ArgumentParser(description='Pull Test model and data from AliNNModel')
parser.add_argument('--alinnmodel_path', dest='src_path', required=True, help='AliNNModel project path')
parser.add_argument('--playground_path', dest='dest_path', required=True, help='Test Playground path')
parser.add_argument('--models', dest='models', type=str, nargs='+', help='target models')
args = parser.parse_args()

def main():
    src_path = join(args.src_path, 'TestResource')
    dest_path = join(args.dest_path, 'models')
    if exists(dest_path):
        rmtree(dest_path)
    makedirs(dest_path)
    if args.models is not None and len(args.models) > 0:
        model_dirs = [join(src_path, m) for m in args.models]
    else:
        model_dirs = [f.path for f in os.scandir(src_path) if f.is_dir()]
    model_names_record_path = join(args.dest_path, 'model_names.txt')
    with open(model_names_record_path, 'w') as f:
        for model_dir in model_dirs:
            model_name = basename(model_dir)
            f.write(model_name + '\n')
            dest_dir = join(dest_path, model_name)
            makedirs(dest_dir)
            copy(join(model_dir, 'temp.bin'), dest_dir)
            copy(join(model_dir, 'input_0.txt'), dest_dir)
            copy(join(model_dir, 'output.txt'), dest_dir)

if __name__ == '__main__':
    main()
