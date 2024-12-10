import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib import cbook
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import argparse
import os

vis_root = "pic"

def remove_blanks(df: pd.DataFrame) -> pd.DataFrame:
    # Removing unnamed columns using drop function
    df.drop(df.columns[df.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    return df
def add_turns(df: pd.DataFrame) -> pd.DataFrame:
    df["turns"] = (1-df.isnull()).sum(axis=1) // 2
    return df
def get_max_turn(df: pd.DataFrame) -> int:
    keys = list(df.keys())
    return max([int(key.replace("decode", "")) for key in keys if "decode" in key]) + 1
def add_pd_ratio(df: pd.DataFrame) -> pd.DataFrame:
    max_turns = get_max_turn(df)
    for i in range(max_turns):
        df["pd_ratio{}".format(i)] = df["prefill{}".format(i)] / df["decode{}".format(i)]
    return df 
def preprocess(file_path: str) -> pd.DataFrame:
    table = pd.read_csv(file_path)
    table = remove_blanks(table)
    table = add_turns(table)
    table = add_pd_ratio(table)
    print(table)
    return table

def draw_distribution(df: pd.DataFrame, file_path: str):
    turns_bin = df.value_counts(subset=["turns"], sort=False)
    print(turns_bin)
    plt.close()
    plt.rcParams['font.size'] = 10
    _, ax = plt.subplots()
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(df["turns"], bins=get_max_turn(df), density=True, align="left", label=True)
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # Now we format the y-axis to display percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlim((0.5, get_max_turn(df)-0.5))
    ax.set_xticks(np.arange(1,get_max_turn(df)+1),np.arange(1,get_max_turn(df)+1),rotation=60, fontsize=9)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_xlabel("num of turns", fontsize=14)
    plt.savefig(file_path, dpi=600)
    plt.close()

def draw_prefill(df: pd.DataFrame, ax: Axes):
    stats = [cbook.boxplot_stats(df[df["prefill{}".format(i)].notna()]["prefill{}".format(i)], labels=[i+1])[0]
                 for i in range(get_max_turn(df))]
    print(stats)
    ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'}, flierprops=dict(marker='o', markersize=2))
    ax.set_ylim(0,600)
    ax.set_yticks(np.arange(0,700,100), np.arange(0,700,100), fontsize=9)
    ax.set_ylabel("prefill", fontsize=12, rotation=90)
    return
def draw_decode(df: pd.DataFrame, ax: Axes):
    stats = [cbook.boxplot_stats(df[df["decode{}".format(i)].notna()]["decode{}".format(i)], labels=[i+1])[0]
                 for i in range(get_max_turn(df))]
    print(stats)
    ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'}, flierprops=dict(marker='o', markersize=2))
    ax.set_ylim(0,600)
    ax.set_yticks(np.arange(0,700,100), np.arange(0,700,100), fontsize=9)
    ax.set_ylabel("decode", fontsize=12, rotation=90)
    return
def draw_pd_ratio(df: pd.DataFrame, ax: Axes):
    stats = [cbook.boxplot_stats(df[df["pd_ratio{}".format(i)].notna()]["pd_ratio{}".format(i)], labels=[i+1])[0]
                 for i in range(get_max_turn(df))]
    print(stats)
    ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'}, flierprops=dict(marker='o', markersize=2))
    ax.plot(np.arange(0,get_max_turn(df)+2), np.ones_like(np.arange(0,get_max_turn(df)+2),dtype=float))
    ax.set_xlim(0, get_max_turn(df)+1)
    ax.set_ylim(0, 2.)
    ax.set_xticks(np.arange(1,get_max_turn(df)), np.arange(1,get_max_turn(df)), rotation=60, fontsize=9)
    ax.set_yticks([0,0.5,1,2], [0,0.5,1,2], fontsize=9)
    ax.set_xlabel("round", fontsize=12)
    ax.set_ylabel("prefill/decode", fontsize=12, rotation=90)
    return
def draw_reuse_kv(df: pd.DataFrame, file_path: str):
    plt.close()
    _, axs = plt.subplots(3,1,sharex="col")
    draw_prefill(df, axs[0])
    draw_decode(df, axs[1])
    draw_pd_ratio(df, axs[2])
    plt.savefig(file_path, dpi=1200)
    plt.close()
    return
def draw_no_reuse_kv():
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--name", type=str, default="shareGPT_dialog_stats_common_en.csv")
    args = parser.parse_args()

    file_path = os.path.join(args.root, args.name)
    dist_path = os.path.join(vis_root, args.name.split('.')[0]+"_dist.png")
    pd_dist_path = os.path.join(vis_root, args.name.split('.')[0]+"_pd_dist.png")
    table = preprocess(file_path)
    draw_distribution(table, dist_path)
    draw_reuse_kv(table, pd_dist_path)