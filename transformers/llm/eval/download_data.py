from datasets import load_dataset
import argparse
import os
def main(args):
    output_path = args.output_path
    # load dataset
    eval_dataset = args.eval_dataset
    dataset_name = eval_dataset.split("/")[0]
    dataset_dir = eval_dataset.split("/")[1]

    dataset = load_dataset(dataset_name, dataset_dir, split="test")
    os.makedirs(output_path, exist_ok = True)
    with open(os.path.join(output_path, "prompt.txt"), 'w') as f:
        f.write("\n\n".join(dataset["text"]))
    with open(os.path.join(output_path, "describe.txt"), 'w') as f:
        f.write(dataset_name + '\n')
        f.write(dataset_dir + '\n')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset and extract to string.")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="target dataset path",
    )

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", "--eval_dataset", type=str, default='wikitext/wikitext-2-raw-v1', help="Evaluation dataset, default is `wikitext/wikitext-2-raw-v1`."
    )

    args = parser.parse_args()

    main(args)
