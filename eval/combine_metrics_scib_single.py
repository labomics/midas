#!/usr/bin/env python
# coding: utf-8

# In[97]:


import os
os.chdir("/root/workspace/code/midas/")
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import numpy as np
import anndata as ad
import pandas as pd
import copy
import re


# In[98]:


parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, nargs='+',  default=["dogma_single_rna"])
parser.add_argument('--method', type=str, default='midas_embed')
# parser.add_argument('--method', type=str, default='scmomat')
# parser.add_argument('--method', type=str, default='scvaeit')
# parser.add_argument('--method', type=str, default='stabmap')
# parser.add_argument('--method', type=str, default='multigrate')
parser.add_argument('--mosaic', type=int, default=1)
parser.add_argument('--sota', type=int, default=1)
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='l_2')
parser.add_argument('--init_model', type=str, default='sp_00001999')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[99]:


# mosaic results
df_batch_bio_embed = {}
if o.mosaic == 0:
    o.tasks = [o.tasks[0]]
for task in o.tasks:
    if o.method == "midas_embed":
        fp = pj("result", "comparison", task, o.method, o.experiment, o.model, o.init_model, "metrics_batch_bio.xlsx")
    else:
        fp = pj("result", "comparison", task, o.method, "metrics_batch_bio.xlsx")
    df_batch_bio_embed[task] = pd.read_excel(fp)
    df_batch_bio_embed[task].rename(index={0: task}, inplace=True)
df_batch_bio_embed_cat = pd.concat(df_batch_bio_embed.values(), axis=0)

df_batch_bio_embed_cat["Task"] = df_batch_bio_embed_cat.index
df_batch_bio_embed_cat.rename(index={i: o.method for i in df_batch_bio_embed_cat.index}, inplace=True)
df_batch_bio_embed_cat


# In[100]:


# sota results
if o.sota == 1:
    methods = [
        "seurat_rpca+wnn",
        "seurat_cca+wnn",
    ]

    df_sota = {}
    for method in methods:
        if "midas" in method:
            df_sota[method] = pd.read_excel(pj("result", "comparison", o.tasks[0], method, o.experiment, o.model, o.init_model, "metrics_batch_bio.xlsx"))
        else:
            df_sota[method] = pd.read_excel(pj("result", "comparison", re.sub("_transfer", "", o.tasks[0]), method, "metrics_batch_bio.xlsx"))
        df_sota[method].rename(index={0: method}, inplace=True)
    df_sota_cat = pd.concat(df_sota.values(), axis=0)

    df_sota_cat[["Task"]] = o.tasks[0]
    df_sota_cat
    df_cat = pd.concat([df_batch_bio_embed_cat, df_sota_cat], axis=0)
else:
    df_cat = df_batch_bio_embed_cat


# In[101]:


df_mean_cat = copy.deepcopy(df_cat)
df_mean_cat["batch_score"] = df_cat[["iLISI", "graph_conn", "kBET"]].mean(axis=1)
df_mean_cat["bio_score"] = df_cat[["NMI", "ARI", "il_score_f1", "cLISI"]].mean(axis=1)
df_mean_cat["overall_score"] = 0.4 * df_mean_cat["batch_score"] + 0.6 * df_mean_cat["bio_score"]
df_mean_cat = df_mean_cat[["Task", "iLISI", "graph_conn", "kBET", "batch_score", "NMI", "ARI", "il_score_f1", "cLISI", "bio_score", "overall_score"]]
df_mean_cat_sorted = df_mean_cat.sort_values("overall_score", ascending=False, inplace=False)
df_mean_cat_sorted


# In[102]:


# df_norm = copy.deepcopy(df)
# for metric in metrics:
#     v = [df[method][metric].values[0] for method in methods]
#     v_min = min(v)
#     v_max = max(v)
#     for method in methods:
#         df_norm[method][metric] = (df[method][metric] - v_min) / (v_max - v_min)
# df_norm_cat = pd.concat(df_norm.values(), axis=0)
# df_norm_cat["batch_score"] = (df_norm_cat["iLISI"] + df_norm_cat["graph_conn"] + df_norm_cat["kBET"]) / 3
# df_norm_cat["bio_score"] = (df_norm_cat["NMI"] + df_norm_cat["ARI"] + df_norm_cat["il_score_f1"] + df_norm_cat["cLISI"]) / 4
# df_norm_cat["overall_score"] = 0.4 * df_norm_cat["batch_score"] + 0.6 * df_norm_cat["bio_score"]
# df_norm_cat = df_norm_cat[["iLISI", "graph_conn", "kBET", "batch_score", "NMI", "ARI", "il_score_f1", "cLISI", "bio_score", "overall_score"]]
# df_norm_cat


# In[103]:


out_dir = pj("eval", "plot", "data")
utils.mkdir(out_dir, remove_old=False)
if o.mosaic == 0:
    ms = "_sota_"
elif o.sota == 0:
    ms = "_mosaic_"
else:
    ms = "_sota+mosaic_"
df_mean_cat_sorted.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0]+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_sorted_R1_12.xlsx"))
# df_mean_cat.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0]+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_unsorted_R1_12.xlsx"))

