import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cbook
from matplotlib.axes import Axes
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import argparse
import os
import re
from io import StringIO

def split_by_turns(id: str, content: str) -> List[pd.DataFrame]:
    pattern = "<{id}>\n(.*?)</{id}>\n".format(id=id)
    return [pd.read_csv(StringIO(item)) for item in re.findall(pattern, content, flags=re.DOTALL)]
def preprocess(file_path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    content = open(file_path, "rt").read()
    return split_by_turns("prefill", content), split_by_turns("decode", content)
def get_max_turn(no_reuse_prefill_record):
    return max(10, max([len(record) for record in no_reuse_prefill_record]))
def draw_history_len(ax: Axes, no_reuse_prefill_record:  List[pd.DataFrame]):
    max_round = get_max_turn(no_reuse_prefill_record)
    history_len = [0 for _ in range(0, max_round)]
    for turn in range(0, max_round):
        history_len[turn] = np.median([record["input_token"][turn] - record["prompt_token"][turn]
                                     for record in no_reuse_prefill_record if len(record)>=turn+1]).item()
    plt.plot(np.arange(1, max_round+1), history_len, label="median history len", marker=".", markersize=8)
    return
def draw_prefill_bar_chat(ax: Axes, no_reuse, reuse):
    offset = 0.2
    max_round = len(no_reuse)
    no_reuse_med = [np.median(turn) for turn in no_reuse]
    rects = ax.bar(np.arange(1,max_round+1) + offset, no_reuse_med, offset*2, label="no reuse kv", color="tomato")
    ax.bar_label(rects, fmt="{:.2f}", padding=4, fontsize=6)
    reuse_med = [np.median(turn) for turn in reuse]
    rects = ax.bar(np.arange(1,max_round+1) - offset, reuse_med, offset*2, label="reuse kv", color="springgreen")
    ax.bar_label(rects, fmt="{:.2f}", padding=4, fontsize=6)
    return
def compare_prefill_reuse_kv(no_reuse_prefill_record: List[pd.DataFrame],
                             reuse_prefill_record: List[pd.DataFrame]):
    plt.close()
    _,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plot history_len
    draw_history_len(ax2, no_reuse_prefill_record)
    # calculate per turn 
    max_round = get_max_turn(no_reuse_prefill_record)
    no_reuse = [[] for _ in range(0, max_round)]
    for turn in range(0, max_round):
        no_reuse[turn] = [record["response_speed"][turn] for record in no_reuse_prefill_record if len(record)>=turn+1]
    reuse = [[] for _ in range(0, max_round)]
    for turn in range(0, max_round):
        reuse[turn] = [record["response_speed"][turn] for record in reuse_prefill_record if len(record)>=turn+1]
    # plot the bar chat (with error bar)
    draw_prefill_bar_chat(ax1, no_reuse, reuse)
    ax1.set_xticks(np.arange(1,max_round+1),np.arange(1,max_round+1),fontsize=9)
    ax1.set_ylim(0,100)
    ax2.set_ylim(0,1000)
    ax1.legend(loc='upper left', title="prefill response speed")
    ax2.legend(loc='upper right')
    ax1.set_ylabel("prefill\nresponse\nspeed", rotation=0, labelpad=12)
    ax2.set_ylabel("history\nlen", rotation=0, labelpad=8)
    ax1.set_xlabel("round")
    plt.title("KV cache reuse for multi-turn chat\neffects on ShareGPT")
    plt.tight_layout() 
    plt.savefig("./pic/fig.png",dpi=1200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--no_reuse", type=str, default="shareGPT_common_en_70k_noreuse.txt")
    parser.add_argument("--reuse", type=str, default="shareGPT_common_en_70k_reuse.txt")
    args = parser.parse_args()

    no_reuse_file_path = os.path.join(args.root, args.no_reuse)
    reuse_file_path = os.path.join(args.root, args.reuse)
    no_reuse_prefill_record, no_reuse_decode_record = preprocess(no_reuse_file_path)
    reuse_prefill_record, reuse_decode_record = preprocess(reuse_file_path)
    # visualize prefill
    compare_prefill_reuse_kv(no_reuse_prefill_record, reuse_prefill_record)
