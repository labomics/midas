#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.chdir("/root/workspace/code/sc-transformer/")
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import numpy as np
import torch as th
import scib
import scib.metrics as me
import anndata as ad
import scipy
import pandas as pd
import re
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.stats import pearsonr
import copy
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='dogma_single_rna_continual')
parser.add_argument('--reference', type=str, default='atlas_no_dogma')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_00002199')
parser.add_argument('--init_model_ref', type=str, default='sp_latest')
parser.add_argument('--method', type=str, default='midas_embed')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[ ]:


result_dir = pj("result", "comparison", o.task, o.method, o.experiment, o.init_model)

# Load latent variables
data_config = utils.gen_data_config(o.task)
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)

o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", o.init_model)
pred = utils.load_predicted(o, group_by="subset")

c = [v["z"]["joint"][:, :o.dim_c] for v in pred.values()]
subset_num = 4
c_query = np.concatenate(c[:subset_num], axis=0)
c_ref = np.concatenate(c[subset_num:], axis=0)


# In[ ]:


# load labels
label_atlas = utils.load_csv(pj("result", "downstream", "labels", "labels2.atlas.csv"))
label_gt = np.array(utils.transpose_list(label_atlas)[1][1:])[:len(c_query)]
label_ref = np.array(utils.transpose_list(label_atlas)[1][1:])[len(c_query):]

# label_atlas_no_dogma = utils.load_csv(pj("result", "downstream", "labels", "labels2."+o.reference+".csv"))
# label_ref = np.array(utils.transpose_list(label_atlas_no_dogma)[1][1:])


# In[ ]:


# transfer labels via knn
knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
knn.fit(c_ref, label_ref)
label_pred = knn.predict(c_query)


# In[ ]:


utils.mkdirs(result_dir, remove_old=False)
utils.save_list_to_csv([list(line) for line in list(label_pred.reshape(-1, 1))], pj(result_dir, "label_transferred.csv"))
utils.save_list_to_csv([list(line) for line in list(label_gt.reshape(-1, 1))], pj(result_dir, "label_gt.csv"))


# In[ ]:


label_gt_keys = np.unique(label_gt)
label_pred_keys = np.unique(label_pred)


# In[ ]:


results = {}
results["confusion"] = confusion_matrix(label_gt, label_pred, labels=label_gt_keys)
results["f1"] = f1_score(label_gt, label_pred, average='micro')
print(o.task, o.init_model, " f1: ", results["f1"])


# In[ ]:



# plt.figure(figsize=(20,17))
# sns.set(font_scale=1.5)
# cm = results["confusion"].astype('float') / results["confusion"].sum(axis=1)[:, np.newaxis]
# ax = sns.heatmap(cm, annot=True, annot_kws={"size": 16})
# ax.xaxis.set_ticklabels(label_gt_keys, rotation=45)
# ax.yaxis.set_ticklabels(label_gt_keys, rotation=45)
# plt.title(o.task)
# # plt.savefig(pj(fig_dir, "confusion_"+o.data+"_"+task+".png"))


# In[ ]:


# label_gt = utils.load_csv(pj("result", "downstream", "labels", "labels2.atlas.csv"))
# label_gt = np.array(utils.transpose_list(label_gt)[1][1:])[len(c):]
# label_gt_keys = np.unique(label_gt)
# results["confusion"] = confusion_matrix(label_gt, label_ref, labels=label_gt_keys)


# In[ ]:




# plt.figure(figsize=(20,17))
# sns.set(font_scale=1.5)
# cm = results["confusion"].astype('float') / results["confusion"].sum(axis=1)[:, np.newaxis]
# ax = sns.heatmap(cm, annot=True, annot_kws={"size": 16})
# ax.xaxis.set_ticklabels(label_gt_keys, rotation=45)
# ax.yaxis.set_ticklabels(label_gt_keys, rotation=45)
# # plt.savefig(pj(fig_dir, "confusion_"+o.data+"_"+task+".png"))


# In[ ]:





# for batch_id in pred.keys():
    
#     print("Processing batch: ", batch_id)
#     z = pred[batch_id]["z"]


#     c = {m: v[:, :o.dim_c] for m, v in z.items()}
#     c_cat = np.concatenate((c["atac"], c["rna"], c["adt"]), axis=0)
#     mods_cat = ["atac"]*len(c["atac"]) + ["rna"]*len(c["rna"]) + ["adt"]*len(c["adt"])
    
#     label = utils.load_csv(pj(o.raw_data_dirs[batch_id], "label_seurat", "l1.csv"))
#     label = np.array(utils.transpose_list(label)[1][1:])
#     label_cat = np.tile(label, 3)
    
#     assert len(c_cat) == len(mods_cat) == len(label_cat), "Inconsistent lengths!"
    
#     batch = str(batch_id) # toml dict key must be str


#     results["f1"][batch] = {}

#     print("Computing f1")
#     knn.fit(c_ref["joint"], label_ref["joint"])
#     label_pred = knn.predict(c["joint"])
#     # cm = confusion_matrix(label_gt, label_pred, labels=knn.classes_)
#     results["f1"][batch] = f1_score(label_gt, label_pred, average='micro')
#     # f1_weighted = f1_score(label_gt, label_pred, average='weighted')
    

# utils.mkdirs(result_dir, remove_old=False)
# utils.save_toml(results, pj(result_dir, "metrics_mod_detailed.toml"))


# In[ ]:


# results_avg = {metric: np.mean(utils.extract_values(v)) for metric, v in results.items()}
# df = pd.DataFrame({
#     'Label_transfer':   [results_avg['f1']],
# })
# print(df)
# df.to_excel(pj(result_dir, "metrics_mod.xlsx"), index=False)

