import os
import argparse
from tqdm import tqdm
import MNN.llm as mnnllm
from datasets import load_dataset
import torch
import copy

def main(args):
    # load model
    model = mnnllm.create(args.mnn_path)
    model.load()

    # load dataset
    eval_dataset = args.eval_dataset
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

    args = parser.parse_args()

    main(args)
