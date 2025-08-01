import os
import argparse
from tqdm import tqdm
import MNN.llm as mnnllm
from datasets import load_dataset
import torch
import copy

def main(args):

    model = mnnllm.create(args.mnn_path)
    
    model.set_config({'all_logits': True, 'use_template': False})
    
    model.set_config({'enable_debug': True})
    model.load()
    
    model.enable_collection_mode(2, args.output_path)

    eval_dataset = args.eval_dataset
    dataset_name = eval_dataset.split("/")[0]
    dataset_dir = eval_dataset.split("/")[1]

    dataset = load_dataset(dataset_name, dataset_dir, split="test")
    input_ids = model.tokenizer_encode("\n\n".join(dataset["text"]))
    input_ids = input_ids[:args.length]

    _ = model.forward(input_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get max values from MNN model.")
    parser.add_argument(
        "-m",
        "--mnn-path",
        type=str,
        required=True,
        help="mnn model path",
    )

    parser.add_argument(
        "-d", "--eval_dataset", type=str, default='wikitext/wikitext-2-raw-v1', help="dataset, default is `wikitext/wikitext-2-raw-v1`."
    )

    parser.add_argument(
        "-o", "--output-path", type=str, default='max_values.json', help="output path, default is `max_values.json`."
    )

    parser.add_argument(
        "-l", "--length", type=int, default=512, help="length of samples, default is 512."
    )

    args = parser.parse_args()

    main(args)