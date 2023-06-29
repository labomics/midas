import os
from os import path
from os.path import join as pj

import shutil
import torch as th
import cv2 as cv
import json
import toml
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import csv
import copy
import math
import itertools
from tqdm import tqdm
import re


# IO

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_toml(data, filename):
    with open(filename, 'w') as f:
        toml.dump(data, f)
        

def load_toml(filename):
    with open(filename, 'r') as f:
        data = toml.load(f)
    return data


def rmdir(directory):
    if path.exists(directory):
        print('Removing directory "%s"' % directory)
        shutil.rmtree(directory)


def mkdir(directory, remove_old=False):
    if remove_old:
        rmdir(directory)
    if not path.exists(directory):
        os.makedirs(directory)


def mkdirs(directories, remove_old=False):
    """
    Make directories recursively
    """
    t = type(directories)
    if t in [tuple, list]:
        for d in directories:
            mkdirs(d, remove_old=remove_old)
    elif t is dict:
        for d in directories.values():
            mkdirs(d, remove_old=remove_old)
    else:
        mkdir(directories, remove_old=remove_old)
        

def get_filenames(directory, extension):
    filenames = glob(pj(directory, "*."+extension))
    filenames = [path.basename(filename) for filename in filenames]
    filenames.sort()
    return filenames


def load_csv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def load_tsv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        data = list(reader)
    return data


def convert_tensor_to_list(data):
    """
    Covert a 2D tensor `data` to a 2D list
    """
    if th.is_tensor(data):
        return [list(line) for line in list(data.cpu().detach().numpy())]
    else:
        return [list(line) for line in list(data)]


def save_list_to_csv(data, filename, delimiter=','):
    """
    Save a 2D list `data` into a `.csv` file named as `filename`
    """
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)
        

def save_tensor_to_csv(data, filename, delimiter=','):
    """
    Save a 2D tensor `data` into a `.csv` file named as `filename`
    """
    data_list = convert_tensor_to_list(data)
    save_list_to_csv(data_list, filename, delimiter)


def get_name_fmt(file_num):
    """
    Get the format of the filename with a minimum string lenth when each file \
    are named by its ID
    - `file_num`: the number of files to be named
    """
    return "%0" + str(math.floor(math.log10(file_num))+1) + "d"
    

def gen_data_dirs(base_dir, integ_dir):
    dirs = {
        "base":   pj("data", base_dir),
        "prepr":  pj("data", base_dir, "preprocessed"),
        "integ":  pj("data", base_dir, "preprocessed", integ_dir),
        "mat":    pj("data", base_dir, "preprocessed", integ_dir, "mat"),
        "vec":    pj("data", base_dir, "preprocessed", integ_dir, "vec"),
        "name":   pj("data", base_dir, "preprocessed", integ_dir, "name"),
        "fig":    pj("data", base_dir, "preprocessed", integ_dir, "fig"),
        "seurat": pj("data", base_dir, "preprocessed", "seurat")
    }
    return dirs


# Visualization

def imshow(img, height=None, width=None, name='img', delay=1):
    # img: H * W * D
    if th.is_tensor(img):
        img = img.cpu().numpy()
    h = img.shape[0] if height == None else height
    w = img.shape[1] if width == None else width
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.imshow(name, img)
    cv.waitKey(delay)


def imwrite(img, name='img'):
    # img: H * W * 
    img = (img * 255).byte().cpu().numpy()
    cv.imwrite(name + '.jpg', img)


def imresize(img, height, width):
    # img: H * W * D
    is_torch = False
    if th.is_tensor(img):
        img = img.cpu().numpy()
        is_torch = True
    img_resized = cv.resize(img, (width, height))
    if is_torch:
        img_resized = th.from_numpy(img_resized)
    return img_resized


def heatmap(img, cmap='hot'):
    cm = plt.get_cmap(cmap)
    cimg = cm(img.cpu().numpy())
    cimg = th.from_numpy(cimg[:, :, :3])
    cimg = th.index_select(cimg, 2, th.LongTensor([2, 1, 0])) # convert to BGR for opencv
    return cimg


def plot_figure(handles,
                legend = False,
                xlabel = False, ylabel = False,
                xlim = False, ylim = False,
                title = False,
                save_path = False,
                show = False
               ):
    
    print("Plotting figure: " + (title if title else "(no name)"))

    plt.subplots()

    if legend:
        plt.legend(handles = handles)
        
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()


# Data structure conversion

def copy_dict(src_dict):
    return copy.deepcopy(src_dict)
    
    
def get_num(s):
    # return int(path.splitext(s)[0])
    return int(''.join(filter(str.isdigit, s)))


def get_dict(keys, values):
    """
    Construct a dictionary with a key list `keys` and a corresponding value list `values`
    """
    return dict(zip(keys, values))


def get_sub_dict(src_dict, keys):
    return {k: src_dict[k] for k in keys}


def convert_tensors_to_cuda(x):
    """
    Recursively converts tensors to cuda
    """
    y = {}
    for kw, arg in x.items():
        y[kw] = arg.cuda() if th.is_tensor(arg) else convert_tensors_to_cuda(arg)
    return y


def detach_tensors(x):
    """
    Recursively detach tensors
    """
    y = {}
    for kw, arg in x.items():
        y[kw] = arg.detach() if th.is_tensor(arg) else detach_tensors(arg)
    return y


def list_intersection(x):
    """
    - `x`: lists
    """
    return list(set.intersection(*map(set, x)))


def flat_lists(x):
    """
    - `x`: a list of lists
    """
    return list(itertools.chain.from_iterable(x))


def transpose_list(x):
    """
    - `x`: a 2D list
    """
    # return list(zip(*lists))
    return list(map(list, zip(*x)))


def gen_all_batch_ids(s_joint, combs):
    
    s_joint = flat_lists(s_joint)
    combs = flat_lists(combs)
    s = []

    for subset, comb in enumerate(combs):
        s_subset = {}
        for m in comb+["joint"]:
            s_subset[m] = s_joint[subset]
        s.append(s_subset)

    dims_s = {}
    for m in list(np.unique(flat_lists(combs)))+["joint"]:
        s_m = []
        for s_subset in s:
            if m in s_subset.keys():
                s_m.append(s_subset[m])

        sorted, _ = th.tensor(np.unique(s_m)).sort()
        sorted = sorted.tolist()
        dims_s[m] = len(sorted)

        for s_subset in s:
            if m in s_subset.keys():
                s_subset[m] = sorted.index(s_subset[m])

    return s_joint, combs, s, dims_s


def ref_sort(x, ref):
    """
    - Sort `x` with order `ref`
    """
    y = []
    for v in ref:
        if v in x:
            y += [v]
    return y


# Debugging

def get_nan_mask(x):
    mask = th.isinf(x) + th.isnan(x)
    is_nan = mask.sum() > 0
    return mask, is_nan


# Math computations

def sample_gaussian(mu, logvar):
    std = (0.5*logvar).exp()
    eps = th.randn_like(std)
    return mu + std*eps


def extract_values(x):
    """
    Recursively extracting all values in a tuple/list/dictionary
    """
    values = []
    t = type(x)
    if t in [tuple, list]:
        for v in x:
            values += extract_values(v)
    elif t is dict:
        for v in x.values():
            values += extract_values(v)
    else:
        values += [x]
    return values


def extract_tria_values(x):
    """
    Extract, vectorize, and sort matrix values of the upper triangular part.
    Note it contains in-place operations.
    - `x`: the 2D input matrix of size N * N
    - `y`: the 1D output vector of size (N-1)*N/2
    """
    N = x.size(0)
    x_triu = x.triu_(diagonal=1)
    y, _ = x_triu.view(-1).sort(descending=True)
    y = y[:(N-1)*N//2]
    return y


def block_compute(func, block_size, *args):
    """
    - `args`: the args of function `func`
    """
    assert len(args) % 2 == 0, "The number of args must be even!"
    para_num = len(args)//2
    args = [arg.split(block_size, dim=0) for arg in args]
    I = len(args[0])
    J = len(args[para_num])
    z_rows = []
    for i in range(I):
        z_row = []
        for j in range(J):
            x = [args[k][i] for k in range(para_num)]
            y = [args[k+para_num][j] for k in range(para_num)]
            z = func(*(x+y))  # B * B
            z_row.append(z)
        z_row = th.cat(z_row, dim=1)  # B * JB
        z_rows.append(z_row)
    z_rows = th.cat(z_rows, dim=0)  # IB * JB
    return z_rows
    
    
def calc_squared_dist(x, y):
    """
    Squared Euclidian distances between two sets of variables
    - `x`: N1 * D
    - `y`: N2 * D
    """
    return th.cdist(x, y) ** 2
    

def calc_bhat_dist(mu1, logvar1, mu2, logvar2, mem_limit=1e9):
    """
    Bhattacharyya distances between two sets of Gaussian distributions
    - `mu1`, `logvar1`: N1 * D
    - `mu2`, `logvar2`: N2 * D
    - `mem_limit`: the maximal memory allowed for computaion
    """
 
    def calc_bhat_dist_(mu1, logvar1, mu2, logvar2):
        N1, N2 = mu1.size(0), mu2.size(0)
        mu1 = mu1.unsqueeze(1)          # N1 * 1 * D
        logvar1 = logvar1.unsqueeze(1)  # N1 * 1 * D
        mu2 = mu2.unsqueeze(0)          # 1 * N2 * D
        logvar2 = logvar2.unsqueeze(0)  # 1 * N2 * D
        
        var1 = logvar1.exp()  # N1 * 1 * D
        var2 = logvar2.exp()  # 1 * N2 * D
        var = (var1 + var2) / 2  # N1 * N2 * D
        inv_var = 1 / var  # N1 * N2 * D
        inv_covar = inv_var.diag_embed()  # N1 * N2 * D * D
        
        ln_det_covar = var.log().sum(-1)  # N1 * N2
        ln_sqrt_det_covar12 = 0.5*(logvar1.sum(-1) + logvar2.sum(-1))  # N1 * N2
        
        mu_diff = mu1 - mu2  # N1 * N2 * D
        mu_diff_h = mu_diff.unsqueeze(-2)  # N1 * N2 * 1 * D
        mu_diff_v = mu_diff.unsqueeze(-1)  # N1 * N2 * D * 1
        
        dist = 1./8 * mu_diff_h.matmul(inv_covar).matmul(mu_diff_v).reshape(N1, N2) +\
               1./2 * (ln_det_covar - ln_sqrt_det_covar12)  # N1 * N2
        return dist
 
    block_size = int(math.sqrt(mem_limit / (mu1.size(1) * mu2.size(1))))
    return block_compute(calc_bhat_dist_, block_size, mu1, logvar1, mu2, logvar2)


# Evaluation metrics

def calc_foscttm(mu1, mu2, logvar1=None, logvar2=None):
    """
    Fraction Of Samples Closer Than the True Match
    - `mu1`, `mu2`, `logvar1`, `logvar2`: N * D
    """
    N = mu1.size(0)
    if logvar1 is None:
        dists = th.cdist(mu1, mu2)  # N * N
    else: 
        dists = calc_bhat_dist(mu1, logvar1, mu2, logvar2)  # N * N
    
    true_match_dists = dists.diagonal().unsqueeze(-1).expand_as(dists)  # N * N
    foscttms = dists.lt(true_match_dists).sum(-1).float() / (N - 1)  # N
    return foscttms.mean().item()
    
    # fracs = []
    # for n in range(N):
    #     dist = dists[n]  # N
    #     # if n == 0:
    #     #     for i in range(dist.size(0)//50):
    #     #         print(dist[i].item())
    #     true_match_dist = dist[n]
    #     fracs += [dist.lt(true_match_dist).sum().float().item() / (N - 1)]
    #     # if n == 0:
    #     #     exit()
    # foscttm = sum(fracs) / len(fracs)
    # return foscttm


def calc_subset_foscttm(o, model, data_loader):
    mods = data_loader.dataset.comb
    z_mus = get_dict(mods, [[] for _ in mods])
    z_logvars = get_dict(mods, [[] for _ in mods])
    with th.no_grad():
        for data in data_loader:
            data = convert_tensors_to_cuda(data)
            for m in data["x"].keys():
                input_data = {
                    "x": {m: data["x"][m]},
                    "s": data["s"], 
                    "e": {}
                }
                if m in data["e"].keys():
                    input_data["e"][m] = data["e"][m]
                _, _, z_mu, z_logvar, *_ = model(input_data)  # N * Z
                z_mus[m].append(z_mu)
                z_logvars[m].append(z_logvar)

    for m in mods:
        z_mus[m] = th.cat(z_mus[m], dim=0)  # SN * Z
        z_logvars[m] = th.cat(z_logvars[m], dim=0)  # SN * Z

    foscttm = {}
    for m in mods:
        for m_ in set(mods) - {m}:
            foscttm[m+"_to_"+m_] = calc_foscttm(z_mus[m][:, :o.dim_c], z_mus[m_][:, :o.dim_c])
    return foscttm


def calc_subsets_foscttm(o, model, data_loaders, foscttm_list, split, epoch_id=0):
    model.eval()
    foscttm_sums, foscttm_nums = [], []
    for subset, data_loader in data_loaders.items():
        if len(data_loader) > 0 and len(data_loader.dataset.comb) > 1:
            foscttm = calc_subset_foscttm(o, model, data_loader)
            for k, v in foscttm.items():
                print('Epoch: %d/%d, subset: %d, split: %s, %s foscttm: %.4f' %
                (epoch_id+1, o.epoch_num, subset, split, k, v))
            foscttm_sums.append(sum(foscttm.values()))
            foscttm_nums.append(len(foscttm))
    if len(foscttm_sums) > 0:
        foscttm_avg = sum(foscttm_sums) / sum(foscttm_nums)
        print('Epoch: %d/%d, %s foscttm: %.4f\n' % (epoch_id+1, o.epoch_num, split, foscttm_avg))
        foscttm_list.append((float(epoch_id), float(foscttm_avg)))


def load_predicted(o, joint_latent=True, mod_latent=False, impute=False, batch_correct=False, 
                   translate=False, input=False, group_by="modality"):
    print("Loading predicted variables ...")
    dirs = get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input)
    data = {}
    for subset_id in dirs.keys():
        data[subset_id] = {"s": {}}
        for varible in dirs[subset_id].keys():
            data[subset_id][varible] = {}
            for m, dir in dirs[subset_id][varible].items():
                print("Loading subset %d: %s, %s" % (subset_id, varible, m))
                data[subset_id][varible][m] = []
                if varible == "z":
                    data[subset_id]["s"][m] = []
                filenames = get_filenames(dir, "csv")
                for filename in tqdm(filenames):
                    v = load_csv(pj(dir, filename))
                    data[subset_id][varible][m] += v
                    if varible == "z":
                        data[subset_id]["s"][m] += [subset_id]*len(v)
    
    print("Converting to numpy ...")
    for subset_id in data.keys():
        for varible in data[subset_id].keys():
            for m, v in data[subset_id][varible].items():
                print("Converting subset %d: %s, %s" % (subset_id, varible, m))
                if varible in ["z", "x_trans", "x_impt"]:
                    data[subset_id][varible][m] = np.array(v, dtype=np.float32)
                elif varible in ["s", "x", "x_bc"]:
                    data[subset_id][varible][m] = np.array(v, dtype=np.int16)
    
    if group_by == "subset":
        return data
    elif group_by == "modality":
        data_m = {}
        for subset_id in data.keys():
            for varible in data[subset_id].keys():
                if varible not in data_m.keys():
                    data_m[varible] = {}
                for m, v in data[subset_id][varible].items():
                    if m not in data_m[varible].keys():
                        data_m[varible][m] = {}
                    data_m[varible][m][subset_id] = data[subset_id][varible][m]

        for varible in data_m.keys():
            for m in data_m[varible].keys():
                data_m[varible][m] = np.concatenate(tuple(data_m[varible][m].values()), axis=0)
    
        return data_m
    

def get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input):
    dirs = {}
    for subset_id in range(len(o.s_joint)):
        subset_dir = pj(o.pred_dir, "subset_"+str(subset_id))
        dirs[subset_id] = {}
        if joint_latent or mod_latent:
            dirs[subset_id]["z"] = {}
            if joint_latent:
                dirs[subset_id]["z"]["joint"] = pj(subset_dir, "z", "joint")
            if mod_latent:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["z"][m] = pj(subset_dir, "z", m)
        if impute:
            dirs[subset_id]["x_impt"] = {}
            for m in o.mods:
                dirs[subset_id]["x_impt"][m] = pj(subset_dir, "x_impt", m)
        if batch_correct:
            dirs[subset_id]["x_bc"] = {}
            for m in o.mods:
                dirs[subset_id]["x_bc"][m] = pj(subset_dir, "x_bc", m)
        if translate:
            dirs[subset_id]["x_trans"] = {}
            for m in o.combs[subset_id]: # single to double
                for m_ in set(o.mods) - {m}:
                    dirs[subset_id]["x_trans"][m+"_to_"+m_] = pj(subset_dir, "x_trans", m+"_to_"+m_)
            for mods in itertools.combinations(o.combs[subset_id], 2): # double to single
                m1, m2 = ref_sort(mods, ref=o.mods)
                m_ = list(set(o.mods) - set(mods))[0]
                dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_] = pj(subset_dir, "x_trans", m1+"_"+m2+"_to_"+m_)
        if input:
            dirs[subset_id]["x"] = {}
            for m in o.combs[subset_id]:
                dirs[subset_id]["x"][m] = pj(subset_dir, "x", m)
    return dirs


def gen_data_config(task, reference=''):
    if "continual" in task:
        assert reference != '', "Reference must be specified!"
        data_config = load_toml("configs/data.toml")[re.sub("_continual", "", task)]
        data_config_ref = load_toml("configs/data.toml")[reference]
        data_config["raw_data_dirs"] += data_config_ref["raw_data_dirs"]
        data_config["raw_data_frags"] += data_config_ref["raw_data_frags"]
        data_config["combs"] += data_config_ref["combs"]
        data_config["comb_ratios"] += data_config_ref["comb_ratios"]
        data_config["s_joint"] += [[v[0]+len(data_config["s_joint"])] for v in data_config_ref["s_joint"]]
    else:
        cfg_task = re.sub("_vd.*|_vt.*|_atlas|_generalize|_transfer|_ref_.*", "", task)
        data_config = load_toml("configs/data.toml")[cfg_task]
    return data_config


def rename_label(label_list):
    return [re.sub(" cell.*", "", l) for l in label_list]