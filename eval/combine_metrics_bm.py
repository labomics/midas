#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
os.chdir("/root/workspace/code/midas/")
from os import path
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


# In[15]:


parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, nargs='+',  default=["bm"])
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='la_1')
# parser.add_argument('--tasks', type=str, nargs='+',  default=["bm_transfer"])
# parser.add_argument('--experiment', type=str, default='no_map_ref')
# parser.add_argument('--model', type=str, default='default')
parser.add_argument('--methods', type=str, nargs='+', default=["midas_embed", "scmomat", "scvaeit", "stabmap", "multigrate"])
# parser.add_argument('--init_models', type=str, nargs='+', default=["sp_00001899"])
parser.add_argument('--init_models', type=str, nargs='+', default=["sp_00001299", "sp_00001399", "sp_00001499", "sp_00001599",
    "sp_00001699", "sp_00001799", "sp_00001899", "sp_00001999", "sp_00002099", "sp_00002199", "sp_00002299", "sp_00002399",
    "sp_00002499", "sp_00002599", "sp_00002699", "sp_00002799", "sp_00002899", "sp_00002999", "sp_00003099", "sp_00003199",
    "sp_00003299", "sp_00003399", "sp_00003499", "sp_00003599", "sp_00003699", "sp_00003799", "sp_00003899", "sp_00003999"])
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[16]:


# parser = argparse.ArgumentParser()
# parser.add_argument('--tasks', type=str, nargs='+',  default=["dogma_full"])
# parser.add_argument('--experiment', type=str, default='e0')
# parser.add_argument('--model', type=str, default='')
# parser.add_argument('--methods', type=str, nargs='+', default=["midas_embed"])
# parser.add_argument('--init_models', type=str, nargs='+', default=["sp_00000999", "sp_00001099", "sp_00001199", "sp_00001299", "sp_00001399", "sp_00001499", "sp_00001599",
#     "sp_00001699", "sp_00001799", "sp_00001899", "sp_00001999", ])
# o, _ = parser.parse_known_args()  # for python interactive
# # o = parser.parse_args()


# In[17]:


# mosaic results
df_batch_bio_embed = []
for task in o.tasks:
    for method in o.methods:
        if method == "midas_embed":
            for init_model in o.init_models:
                fp = pj("result", "comparison", task, method, o.experiment, o.model, init_model, "metrics_batch_bio.xlsx")
                if path.exists(fp):
                    batch_bio_embed = pd.read_excel(fp)
                    batch_bio_embed["Task"] = task
                    batch_bio_embed["Method"] = method
                    batch_bio_embed["Init model"] = init_model
                    df_batch_bio_embed.append(batch_bio_embed)
        else:
            fp = pj("result", "comparison", task, method, "metrics_batch_bio.xlsx")
            if path.exists(fp):
                batch_bio_embed = pd.read_excel(fp)
                batch_bio_embed["Task"] = task
                batch_bio_embed["Method"] = method
                batch_bio_embed["Init model"] = "-"
                df_batch_bio_embed.append(batch_bio_embed)
df_batch_bio_embed_cat = pd.concat(df_batch_bio_embed, axis=0)
df_batch_bio_embed_cat


# In[18]:


df_cat = df_batch_bio_embed_cat
df_mean_cat = copy.deepcopy(df_cat)
df_mean_cat["batch_score"] = df_cat[["iLISI", "graph_conn", "kBET"]].mean(axis=1)
df_mean_cat["bio_score"] = df_cat[["NMI", "ARI", "il_score_f1", "cLISI"]].mean(axis=1)
df_mean_cat["overall_score"] = 0.4 * df_mean_cat["batch_score"] + 0.6 * df_mean_cat["bio_score"]
df_mean_cat = df_mean_cat[["Task", "Method", "Init model", "iLISI", "graph_conn", "kBET", "batch_score", "NMI", "ARI", "il_score_f1", "cLISI", "bio_score", "overall_score"]]
df_mean_cat_sorted = df_mean_cat.sort_values("overall_score", ascending=False, inplace=False)
df_mean_cat_sorted


# In[19]:


# out_dir = pj("eval", "plot", "data")
# utils.mkdir(out_dir, remove_old=False)
# if o.mosaic == 0:
#     ms = "_sota_"
# elif o.sota == 0:
#     ms = "_mosaic_"
# else:
#     ms = "_sota+mosaic_"
# if o.method == "midas_embed":
#     df_mean_cat_sorted.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0].split("_")[0]+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_sorted.xlsx"))
#     df_mean_cat.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0].split("_")[0]+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_unsorted.xlsx"))
# else:
#     df_mean_cat_sorted.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0].split("_")[0]+"_"+o.method+"_sorted.xlsx"))
#     df_mean_cat.to_excel(pj(out_dir, "scib_metrics"+ms+o.tasks[0].split("_")[0]+"_"+o.method+"_unsorted.xlsx"))

