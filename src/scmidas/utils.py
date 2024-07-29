import os
from os import path
from os.path import join as pj

import shutil
import torch as th
# import cv2 as cv
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
import pandas as pd

import scanpy as sc
from scanpy import AnnData

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
    with open(filename, "w", newline='') as file:
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


def convert_tensors_to_cuda(x, device):
    """
    Recursively converts tensors to cuda
    """
    y = {}
    for kw, arg in x.items():
        y[kw] = arg.to(device) if th.is_tensor(arg) else convert_tensors_to_cuda(arg, device)
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


def calc_subset_foscttm(o, model, data_loader, device="cuda:0"):
    mods = data_loader.dataset.comb
    z_mus = get_dict(mods, [[] for _ in mods])
    z_logvars = get_dict(mods, [[] for _ in mods])
    with th.no_grad():
        for data in data_loader:
            data = convert_tensors_to_cuda(data, device)
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
                # print("Loading subset %d: %s, %s" % (subset_id, varible, m))
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
            if len(o.mods) == 3:
                for mods in itertools.combinations(o.combs[subset_id], 2): # double to single
                    m1, m2 = ref_sort(mods, ref=o.mods)
                    m_ = list(set(o.mods) - set(mods))[0]
                    dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_] = pj(subset_dir, "x_trans", m1+"_"+m2+"_to_"+m_)
        if input:
            dirs[subset_id]["x"] = {}
            for m in o.combs[subset_id]:
                dirs[subset_id]["x"][m] = pj(subset_dir, "x", m)
    return dirs

def gen_data_config(task):
    if "continual" in task:
        data_config = load_toml("configs/data.toml")[re.sub("_continual", "", task)]
        data_config_ref = load_toml("configs/data.toml")["atlas_no_dogma"]
        data_config["raw_data_dirs"] += data_config_ref["raw_data_dirs"]
        data_config["raw_data_frags"] += data_config_ref["raw_data_frags"]
        data_config["combs"] += data_config_ref["combs"]
        data_config["comb_ratios"] += data_config_ref["comb_ratios"]
        data_config["s_joint"] += [[v[0]+len(data_config["s_joint"])] for v in data_config_ref["s_joint"]]
    else:
        cfg_task = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", task)
        data_config = load_toml("configs/data.toml")[cfg_task]
    return data_config


def detect_batches(path, task):
    dir_list = os.listdir(pj(path, task))
    n = 0
    mods_all = []
    for i in dir_list:
        if 'subset_' in i:
            n += 1
    for i in range(n):
        mods = []
        for j in os.listdir(pj(path, task, 'subset_'+str(i), 'vec')):
            mods.append(j)
        mods = ref_sort(mods, ["atac", "rna", "adt"])
        mods_all.append(mods)
    return n, mods_all

def detect_modalities(dir):
    if not os.path.exists(dir):
        print('error, no directory %s in the dir' % dir)
    else:
        file = pd.read_csv(pj(dir, 'feat_dims.csv'), index_col=0)
        mods = list(file.columns.values)
        mods = ref_sort(mods, ["atac", "rna", "adt"])
        dims = {}
        features = {}
        dims_chr = []
        for c in file.columns:
            if c == 'atac':
                dims['atac'] = file[['atac']].sum().values[0]
                features['atac'] = pd.read_csv(pj(dir, 'feat_names_atac.csv'), index_col=0).values.flatten().tolist()
                dims_chr = file[['atac']].values.flatten().tolist()
            else:
                dims[c] = file[[c]].values[0][0]
                features[c] = pd.read_csv(pj(dir, 'feat_names_%s.csv'%c), index_col=0).values.flatten().tolist()
        return mods, dims, features, dims_chr
            
class simple_obj():
    def __init__(self, args):
        self.__dict__ = args


def merge_features(f1, f2, only_f1=False):
    # if only_f1, use only f1 features.
    assert len(f1) == len(set(f1)), 'Duplicate values in f1'
    assert len(f2) == len(set(f2)), 'Duplicate values in f2'
    if not only_f1:
        f1.extend((set(f2) - set(f1)))
    p1 = []
    p2 = []
    for i,j in enumerate(f1):
        if j in f2:
            p2.append(f2.index(j))
            p1.append(i)
    p = [p1, p2]
    return f1, p

def update_model(savepoint, dims_h_past, dims_h, curr_model):
    past_model = savepoint['net_states']
    temp_model = curr_model.state_dict()
    for k, v in temp_model.items():
        # print(k, v.shape)
        if k not in past_model.keys():
            pass
            # print('not in new, pass')
        elif k == 'sct.x_dec.net.4.weight' or k == 'sct.x_dec.net.4.bias':
            # print('last layer')
            param_dict_last = dict(zip(dims_h_past.keys(), past_model[k].split(list(dims_h_past.values()), dim=0)))
            param_dict_len = [i.shape[0] for i in temp_model[k].split(list(dims_h.values()), dim=0)]
            shape_list = dict(zip(dims_h.keys(), param_dict_len))
            start_id = 0
            for m in dims_h_past.keys():
                temp_model[k][start_id:start_id+param_dict_last[m].shape[0]] = param_dict_last[m]
                start_id += shape_list[m]
        elif v.shape == past_model[k].shape:
            # print('same shape layer')
            temp_model[k] = past_model[k]
        elif len(v.shape)==2 and v.shape[0] == past_model[k].shape[0]:
            # print('padding')
            temp_model[k][:, :past_model[k].shape[1]] = past_model[k]
        elif len(v.shape)==2 and v.shape[1] == past_model[k].shape[1]:
            # print('padding')
            temp_model[k][:past_model[k].shape[0], :] = past_model[k]
        else:
            # print('padding')
            temp_model[k][:past_model[k].shape[0]] = past_model[k]
        curr_model.load_state_dict(temp_model)
    return curr_model


    
def combine_mod(mods):
    return ref_sort(np.unique(np.concatenate(list(mods.values()))), ref=['atac', 'rna', 'adt'])


def split_list_by_prefix(input_list):
    result_dict = {}
    for item in input_list:
        prefix = item.split('-')[0]
        if prefix not in result_dict:
            result_dict[prefix] = []
        result_dict[prefix].append(item)
    return result_dict

def lists_are_identical(lists):
    set_of_tuples = set(tuple(lst) for lst in lists)
    return len(set_of_tuples) == 1

def viz_mod_latent(emb:dict, label:list, h:int = 2, w:int = 2, legend:bool = True, legend_loc="right"):
    """Visualize modality-specific embeddings.

    Args:
        emb (dict): A dictionary containing modality-specific embeddings.
        label (list): Labels for different batches.
        h (int): Height of the subfigure.
        w (int): Width of the subfigure.
        legend (bool): Whether to plot the legend.
        legend_loc (str): Location of the legend.
    """
    adata_list = []
    for m in emb["s"].keys():
        adata = sc.AnnData(emb["z"][m][:, :32])
        adata.obs["modality"] = m
        adata.obs["batch"] = emb["s"][m]
        labels = np.concatenate([label[label["s"]==i]["x"].values for i in sorted(np.unique(adata.obs["batch"]))])
        adata.obs["label"] = labels
        adata_list.append(adata)
        adata = sc.concat(adata_list)
        
        # sc.pp.subsample(adata, fraction=1) # shuffle for better visualization
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    nrows = len(np.unique(adata.obs["modality"]))
    ncols = len(np.unique(adata.obs["batch"]))
    mods = np.unique(adata.obs["modality"])
    mods_ = []
    for m in ["rna", "adt", "atac", "joint"]:
        if m in mods:
            mods_.append(m)
    f, ax = plt.subplots(nrows, ncols, figsize=[ncols*h, nrows*w])
    if np.unique(adata.obs["batch"]).shape[0] == 1:
        ax = ax[:, np.newaxis]
    for i, b in enumerate(np.unique(adata.obs["batch"])):
        for j, m in enumerate(mods_):
            if len(adata[(adata.obs['modality']==m) & (adata.obs['batch']==b)]):
                sc.pl.umap(adata, ax=ax[j,i], show=False, s=0.5)
                sc.pl.umap(adata[(adata.obs['modality']==m) & (adata.obs['batch']==b)], color='label', ax=ax[j,i], show=False, s=2)
                ax[j,i].get_legend().set_visible(False)
            ax[j,i].set_xticks([])
            ax[j,i].set_yticks([])
            ax[j,i].set_xlabel("")

            if i==0:
                ax[j,i].set_ylabel(m)
            else:
                ax[j,i].set_ylabel("")
            if j==0:
                ax[j,i].set_title(b)
            else:
                ax[j,i].set_title("")
    handles, labels = ax[j,i].get_legend_handles_labels()
    if legend:
        f.legend(handles, labels, loc=legend_loc, ncol=1) 
    plt.show()
    
def viz_diff(adata:AnnData, group:str = "label", emb:str = "emb", n:int = 2, viz_rank:bool = True, viz_heat:bool = True, show_log:bool = True):
    """Visualize differential features.

    Args:
        adata (AnnData): The input dataset.
        group (str): The key for grouping data in the differential features analysis, e.g., 'label'.
        emb (str): The key for UMAP embeddings.
        n (int): The number of features to visualize.
        viz_rank (bool): Whether to run rank_genes_groups_dotplot().
        viz_heat (bool): Whether to visualize each feature individually.
        show_log (bool): Whether to log-transform adata.X for visualization.
    """
    adata = adata.copy()
    if "distances" in adata.obsp:
        print("skipping sc.pp.neighbors")
    else:
        sc.pp.neighbors(adata, use_rep=emb)

    if "X_umap" in adata.obsm:
        print("skipping sc.tl.umap")
    else:
        sc.tl.umap(adata)

    if show_log:
        sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=group, method="wilcoxon")

    if viz_rank:
        sc.pl.rank_genes_groups_dotplot(adata, groupby=group, standard_scale="var", n_genes=n)

    if viz_heat:
        celltypes = np.unique(adata.obs[group])
        for i in range(len(celltypes)):
            print("group:", celltypes[i])
            dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=celltypes[i]).head(n)["names"]
            sc.pl.umap(
                adata,
                color=[*dc_cluster_genes],
                frameon=False,
                ncols=3,
            )