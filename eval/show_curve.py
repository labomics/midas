import sys
sys.path.append("modules")
from os import path
from os.path import join as pj
import argparse
import toml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils


# args
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='dogma_full')
parser.add_argument('--exps', type=str, nargs='+', default=['e0'])
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--action', type=str, default='train',
    choices=['train', 'test', 'foscttm'],
    help="Choose an action to run")
parser.add_argument('--smooth_size', type=int, default=1)
parser.add_argument('--step', type=int, default=1,
    help="epoch sampling step")
o = parser.parse_args()


# global variables
colors = sns.color_palette("Set2", 10)
lines = {
    'e0':         ['e0', '', colors[6]],
    'e1':         ['e1', '', colors[5]],
    'e2':         ['e2', '', colors[2]],
    'e3':         ['e3', '', colors[1]],
    'e4':         ['e4', '', colors[6]],
    'e5':         ['e5', '', colors[5]],
    'e6':         ['e6', '', colors[2]],
    'e7':         ['e7', '', colors[1]],
    # 'semi_5':     ['semi_5',       '.', colors[5]],
    # 'semi_10':     ['semi_10', 'v', colors[2]],
    # 'semi_50':     ['semi_50', 'o', colors[1]],
    # 'xx3':         ['TBAc-noMem', '^', colors[3]],
    # 'xx4':         ['TBAc-noRep', 's', colors[4]],
    # 'xx5':         ['xx1',        '',  colors[7]],
    # 'xx6':         ['xx2',        '',  colors[8]],
}
plt.rc('lines', linewidth=2, markersize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10.5)
plt.rc('ytick', labelsize=10.5)
plt.rc('legend', fontsize=12)

ylabels = {
    "train": "train_loss",
    "test": "test_loss",
    "foscttm": "foscttm",
}
result_dir = pj('result', o.task)
fig_dir = pj(result_dir, "training_curve")
utils.mkdirs(fig_dir, remove_old=False)
smooth_pad = o.smooth_size // 2


def main():
    results = load_multi_result()
    num = min([len(v[0]) for v in results.values()])

    _, ax = plt.subplots()
    # plt.figure(1)
    handles = []
    for exp, result in results.items():
        x = [result[0][p] for p in range(0, num, o.step)]
        y = result[1][0:num]
        if o.smooth_size > 1:
            y = [y[0]]*smooth_pad + y + [y[-1]]*smooth_pad
            y = [sum(y[i-smooth_pad:i+smooth_pad+1])/(smooth_pad*2+1) for i in 
                range(smooth_pad, smooth_pad+num)]
        y = [y[p] for p in range(0, num, o.step)]
        h, = plt.plot(x, y, label=lines[exp][0], marker=lines[exp][1], color=lines[exp][2])
        # handles = [h] + handles
        handles.append(h)
    plt.legend(handles = handles)
    plt.xlabel("epoch")
    plt.ylabel(ylabels[o.action])
    if o.action in ["train", "test"]:
        plt.ylim(bottom=0)
    elif o.action == "foscttm":
        # plt.ylim(0, 0.53)
        plt.yticks(np.arange(0, 0.53, step=0.05))
    ax.grid(linestyle='dashed', linewidth='0.5')
    plt.show()
    plt.savefig(pj(fig_dir, "%s_%s_%s_%s.png" % (o.task, str(o.exps), o.model, o.action)))


def load_single_result(exp):
    result = utils.load_toml(pj(result_dir, exp, o.model, "train", 'sp_latest.toml'))
    loss_pairs = result['benchmark'][ylabels[o.action]]
    return utils.transpose_list(loss_pairs)


def load_multi_result():
    results = {}
    for exp in o.exps:
        results[exp] = load_single_result(exp)
    return results


main()
