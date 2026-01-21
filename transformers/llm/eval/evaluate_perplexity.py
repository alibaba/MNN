import os
import argparse
from tqdm import tqdm
import MNN.llm as mnnllm
from datasets import load_dataset, load_from_disk
import torch
import copy

def main(args):
    # load model
    model = mnnllm.create(args.mnn_path)
    model.set_config({"attention_mode": args.attention_mode})
    model.set_config({'all_logits': True})
    model.load()

    model.generate_init()

    # load dataset
    eval_dataset = args.eval_dataset
    if os.path.exists(eval_dataset):
        print("Loading dataset from disk: {}".format(eval_dataset))
        dataset = load_from_disk(eval_dataset)
    else:
        dataset_name = eval_dataset.split("/")[0]
        dataset_dir = eval_dataset.split("/")[1]
        dataset = load_dataset(dataset_name, dataset_dir, split="test")

    input_ids = model.tokenizer_encode("\n\n".join(dataset["text"]))
    stride = 512
    context_length = stride + stride // 2
    seq_len = len(input_ids)
    # seq_len = 10240

    nlls = []
    prev_end_loc = 0
    criterion = torch.nn.CrossEntropyLoss()
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + context_length, seq_len)
        chunk_ids = input_ids[begin_loc:end_loc]
        model.reset()
        logits = model.forward(chunk_ids)
        npy_logits = copy.deepcopy(logits.read())
        logits = torch.from_numpy(npy_logits).squeeze(0)
        # logits = torch.from_numpy(logits.read()).squeeze(0) # crash when opencl

        target_ids = torch.tensor(chunk_ids)
        trg_len = end_loc - prev_end_loc
        target_ids[:-trg_len] = -100
        neg_log_likelihood = criterion(logits[:-1, :], target_ids[1:])
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    perplexity = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {perplexity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mnn perplexity.")
    parser.add_argument(
        "-m",
        "--mnn-path",
        type=str,
        required=True,
        help="mnn model path",
    )

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", "--eval_dataset", type=str, default='wikitext/wikitext-2-raw-v1', help="Evaluation dataset, default is `wikitext/wikitext-2-raw-v1`."
    )
    group.add_argument(
        "--attention_mode",
        type=int,
        default=8,
        choices=[0, 1, 2, 8, 9, 10],
        help="""Quantization option for query, key, value in CPU attention operator. Options: 0, 1, 2, 8, 9, 10. Default: 8.
        0: No Flash Attention, no quantization for query, key, value;
        1: No Flash Attention, 8-bit asymmetric quantization for query and key, no quantization for value;
        2: No Flash Attention, 8-bit asymmetric quantization for query, key, and value;
        8: Flash Attention enabled, no quantization for query, key, value;
        9: Flash Attention enabled, 8-bit asymmetric quantization for query and key, no quantization for value;
        10: Flash Attention enabled, 8-bit asymmetric quantization for query, key, and value.""",
    )
    args = parser.parse_args()

    main(args)
