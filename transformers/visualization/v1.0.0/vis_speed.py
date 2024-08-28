import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm

def readLog(log_names: List[str]) -> Dict[str, str]:
    data_dict = {}
    logs = os.listdir("data")
    assert len(log_names) == len(logs)
    for name, log in zip(log_names, logs):
        data_dict[name] = open(os.path.join("data",log), "rt").read()
    return data_dict


def processData(data_dict: Dict[str, str]) -> None:
    def processExpr(data_dict, key, identifier):
        string = data_dict[key].split(identifier)[1]
        lines = string.split("\n")
        expr_line_num = int(lines[0].split(" ")[1])
        expr_title = lines[1].split(" ")
        expr_dict = {title : [] for title in expr_title}
        for idx in range(expr_line_num):
            for j, d in enumerate(lines[idx+2].split(" ")):
                expr_dict[expr_title[j]].append(float(d))
        return expr_dict

    for key in data_dict.keys():
        prefill_dict = processExpr(data_dict, key, "prefill")
        decode_dict = processExpr(data_dict, key, "decode")
        data_dict[key] = [prefill_dict, decode_dict]
    return prefill_dict, decode_dict

def visPrefill(data_dict, out_path: str = os.path.join("pic", "prefill.png")) -> None:
    return

def visDecode(data_dict, out_path: str = os.path.join("pic", "decode.png")) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(10,5))
    min_token = np.inf
    for name, data in data_dict.items():
        data = data[1]
        min_token = min(len(data[list(data.keys())[0]]), min_token)
    for name, data in data_dict.items():
        data = data[1]
        plt.plot(data[list(data.keys())[0]][:min_token], data[list(data.keys())[1]][:min_token], lw=1, label=name, marker='.', ms=3)
    ax.set_title("decode speed")
    ax.set_xlabel(list(data_dict[list(data_dict.keys())[0]][1].keys())[0])
    ax.set_ylabel(list(data_dict[list(data_dict.keys())[0]][1].keys())[1])
    ax.legend(loc='upper right', ncols=len(data_dict))
    ax.grid(True)
    plt.savefig(out_path, dpi=800)
    plt.close()
    return

if __name__=="__main__":
    names = ["ours-cont", "ours-sep", "ori-MNN"]
    data_dict = readLog(names)
    processData(data_dict)
    visPrefill(data_dict)
    visDecode(data_dict)