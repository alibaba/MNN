import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict

color_num = 10
colors = sns.color_palette(palette="bright",n_colors=color_num)[:color_num]

# fig, ax = plt.subplots()

# fruits = ['apple', 'blueberry', 'cherry', 'orange']
# counts = [40, 100, 30, 55]
# bar_labels = ['red', 'blue', '_red', 'orange']
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

# ax.set_ylabel('fruit supply')
# ax.set_title('Fruit supply by kind and color')
# ax.legend(title='Fruit color')

# plt.show()

def read_op_mem(path: str) -> List[dict]:
    # ["object", "size", "start", "end", "ptr"]
    # must minus the base_offset of static memory!!
    blocks = []
    data = open(path, "rt").read().split("\n")
    for d in data:
        if d == "":
            continue
        ptr, size = d.split(' ')
        ptr, size = int(ptr), int(size)

def visualizeDynamicMemAllocation(op,
                                  job_allocation_list: List[dict],
                                  base_offset: int,
                                  pic_path="pic.png"):
        # ["object", "size", "start", "end", "ptr"]
        # must minus the base_offset of static memory!!
        plt.close()
        # plot
        fig, ax = plt.subplots()
        max_h = 0
        max_x = 0
        for idx, t in enumerate(job_allocation_list):
            _,s,r,c,alpha = t.values()
            ax.add_patch(Rectangle((r,alpha-base_offset),c-r,s,
                                #    edgecolor="white",
                                facecolor=colors[idx % color_num],
                                fill=True))
            max_x = max(c,max_x)
            max_h = max(alpha+s,max_h)
        ax.set_xlim(0,max_x)
        ax.set_ylim(0,max_h-base_offset)
        ax.set_xlabel("step")
        ax.set_ylabel("memory (MB)")
        ax.set_title("{}dynamic memory allocation".format(op))
        plt.savefig(pic_path,dpi=1000)
        plt.close()
        return

if __name__=="__main__":
    read_op_mem()