#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, nargs='+',  default=["dogma_full", "dogma_paired_full", 
    "dogma_paired_abc", "dogma_paired_ab",  "dogma_paired_ac", "dogma_paired_bc",
    "dogma_single_full", "dogma_single"])
parser.add_argument('--method', type=str, default='midas_embed')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_00001899')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[ ]:


df_batch_bio_embed = {}
for task in o.tasks:
    df_batch_bio_embed[task] = pd.read_excel(pj("result", "comparison", task, o.method, o.experiment, o.model, o.init_model, "metrics_batch_bio.xlsx"))
    df_batch_bio_embed[task].rename(index={0: task}, inplace=True)
df_batch_bio_embed_cat = pd.concat(df_batch_bio_embed.values(), axis=0)
df_batch_bio_embed_cat.rename(columns={c: c+"_embed" for c in df_batch_bio_embed_cat.columns}, inplace=True)
df_batch_bio_embed_cat

df_batch_bio_feat = {}
for task in o.tasks:
    df_batch_bio_feat[task] = pd.read_excel(pj("result", "comparison", task, "midas_feat+wnn", o.experiment, o.model, o.init_model, "metrics_batch_bio.xlsx"))
    df_batch_bio_feat[task].rename(index={0: task}, inplace=True)
df_batch_bio_feat_cat = pd.concat(df_batch_bio_feat.values(), axis=0)
df_batch_bio_feat_cat.rename(columns={c: c+"_feat" for c in df_batch_bio_feat_cat.columns}, inplace=True)
df_batch_bio_feat_cat

df_mod = {}
for task in o.tasks:
    df_mod[task] = pd.read_excel(pj("result", "comparison", task, o.method, o.experiment, o.model, o.init_model, "metrics_mod.xlsx"))
    df_mod[task].rename(index={0: task}, inplace=True)
df_mod_cat = pd.concat(df_mod.values(), axis=0)
df_mod_cat

df_cat = pd.concat([df_batch_bio_embed_cat, df_batch_bio_feat_cat, df_mod_cat], axis=1)


# In[ ]:



df_mean_cat = copy.deepcopy(df_cat)
df_mean_cat["batch_score"] = df_cat[["iLISI_feat",  "graph_conn_feat",  "kBET_feat",
                                     "iLISI_embed", "graph_conn_embed", "kBET_embed"]].mean(axis=1)

df_mean_cat["mod_score"] = df_cat[["ASW_mod", "FOSCTTM", "Label_transfer",
                                   "AUROC", "Pearson_RNA", "Pearson_ADT"]].mean(axis=1)

df_mean_cat["bio_score"] = df_cat[["NMI_feat",  "ARI_feat",  "il_score_f1_feat",  "cLISI_feat",
                                   "NMI_embed", "ARI_embed", "il_score_f1_embed", "cLISI_embed"]].mean(axis=1)

df_mean_cat["overall_score"] =  0.3 * df_mean_cat["batch_score"] +                                 0.3 * df_mean_cat["mod_score"] +                                 0.4 * df_mean_cat["bio_score"]

df_mean_cat = df_mean_cat[["iLISI_feat",  "graph_conn_feat",  "kBET_feat", "iLISI_embed", "graph_conn_embed", "kBET_embed", "batch_score",
                           "ASW_mod", "FOSCTTM", "Label_transfer", "AUROC", "Pearson_RNA", "Pearson_ADT", "mod_score",
                           "NMI_feat",  "ARI_feat",  "il_score_f1_feat",  "cLISI_feat", "NMI_embed", "ARI_embed", "il_score_f1_embed", "cLISI_embed", "bio_score",
                           "overall_score"]]
df_mean_cat_sorted = df_mean_cat.sort_values("overall_score", ascending=False, inplace=False)
df_mean_cat_sorted

# df_mean_cat[["iLISI_feat",  "graph_conn_feat",  "kBET_feat", "iLISI_embed", "graph_conn_embed", "kBET_embed", "batch_score"]]
# df_mean_cat[["ASW_mod", "FOSCTTM", "Label_transfer", "AUROC", "Pearson_RNA", "Pearson_ADT", "mod_score"]]
# df_mean_cat[["NMI_feat",  "ARI_feat",  "il_score_f1_feat",  "cLISI_feat", "NMI_embed", "ARI_embed", "il_score_f1_embed", "cLISI_embed", "bio_score"]]
# df_mean_cat[["batch_score", "mod_score", "bio_score", "overall_score"]]


# In[ ]:


out_dir = pj("eval", "plot", "data")
utils.mkdir(out_dir, remove_old=False)

tname = o.tasks[0].split("_")[0]
if "vd" in o.tasks[0] or "vt" in o.tasks[0]:
    tname = tname + "_" + o.tasks[0].split("_")[-1]
df_mean_cat_sorted.to_excel(pj(out_dir, "scmib_metrics_"+tname+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_sorted.xlsx"))
df_mean_cat.to_excel(pj(out_dir, "scmib_metrics_"+tname+"_"+o.experiment+"_"+o.model+"_"+o.init_model+"_unsorted.xlsx"))

