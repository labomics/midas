#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='dogma_paired_full')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_latest')
parser.add_argument('--method', type=str, default='midas_embed')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[7]:


if "midas" in o.method:
    result_dir = pj("result", "comparison", o.task, o.method, o.experiment, o.init_model)
else:
    result_dir = pj("result", "comparison", o.task, o.method)
t = o.task.split("_")[0] # dogma
o.task = re.sub(t, t+"_full_ref", o.task) # dogma_full_ref_paired_full
# data_dir = pj("data", "processed", re.sub("_generalize", "_transfer", o.task))
data_dir = pj("data", "processed", o.task)
cfg_task = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", o.task) # dogma_full
data_config = utils.load_toml("configs/data.toml")[cfg_task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)


# In[5]:


# Load predicted latent variables
o.mods = ["atac", "rna", "adt"]
o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", o.init_model)
pred = utils.load_predicted(o, mod_latent=True, translate=True, input=True, group_by="subset")


# In[6]:


output_type = "embed"
embed = "X_emb"
batch_key = "batch"
label_key = "label"
mod_key = "modality"
cluster_key = "cluster"
si_metric = "euclidean"
subsample = 0.5
verbose = False

results = {
    "asw_mod": {},
    "foscttm": {},
    "f1": {},
    "auroc": {},
    "pearson_rna": {},
    "pearson_adt": {},
}


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

for batch_id in pred.keys():
    
    print("Processing batch: ", batch_id)
    z = pred[batch_id]["z"]
    x = pred[batch_id]["x"]
    x_trans = pred[batch_id]["x_trans"]
    mask_dir = pj(data_dir, "subset_"+str(batch_id), "mask")

    c = {m: v[:, :o.dim_c] for m, v in z.items()}
    c_cat = np.concatenate((c["atac"], c["rna"], c["adt"]), axis=0)
    mods_cat = ["atac"]*len(c["atac"]) + ["rna"]*len(c["rna"]) + ["adt"]*len(c["adt"])
    
    label = utils.load_csv(pj(o.raw_data_dirs[batch_id], "label_seurat", "l1.csv"))
    label = np.array(utils.transpose_list(label)[1][1:])
    label_cat = np.tile(label, 3)
    
    assert len(c_cat) == len(mods_cat) == len(label_cat), "Inconsistent lengths!"
    
    batch = str(batch_id) # toml dict key must be str
    print("Computing asw_mod")
    adata = ad.AnnData(c_cat)
    adata.obsm[embed] = c_cat
    adata.obs[mod_key] = mods_cat
    adata.obs[mod_key] = adata.obs[mod_key].astype("category")
    adata.obs[label_key] = label_cat
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    results["asw_mod"][batch] = me.silhouette_batch(adata, batch_key=mod_key,
        group_key=label_key, embed=embed, metric=si_metric, verbose=verbose)
    
    results["foscttm"][batch] = {}
    results["f1"][batch] = {}
    results["auroc"][batch] = {}
    results["pearson_rna"][batch] = {}
    results["pearson_adt"][batch] = {}
    for m in c.keys() - {"joint"}:
        for m_ in set(c.keys()) - {m, "joint"}:
            k = m+"_to_"+m_
            print(k+":")
            print("Computing foscttm")
            results["foscttm"][batch][k] = 1 - utils.calc_foscttm(th.from_numpy(c[m]), th.from_numpy(c[m_]))
            
            print("Computing f1")
            knn.fit(c[m], label)
            label_pred = knn.predict(c[m_])
            # cm = confusion_matrix(label, label_pred, labels=knn.classes_)
            results["f1"][batch][k] = f1_score(label, label_pred, average='micro')
            # f1_weighted = f1_score(label, label_pred, average='weighted')
            
            if m_ in ["atac"]:
                x_gt = x[m_].reshape(-1)
                x_pred = x_trans[k].reshape(-1)
                print("Computing auroc")
                results["auroc"][batch][k] = roc_auc_score(x_gt, x_pred)
            elif m_ in ["rna", "adt"]:
                mask = np.array(utils.load_csv(pj(mask_dir, m_+".csv"))[1][1:]).astype(bool)
                x_gt = x[m_][:, mask].reshape(-1)
                x_pred = x_trans[k][:, mask].reshape(-1)
                print("Computing pearson_"+m_)
                results["pearson_"+m_][batch][k] = pearsonr(x_gt, x_pred)[0]
                
    for mods in itertools.combinations(c.keys() - {"joint"}, 2): # double to single
        m1, m2 = utils.ref_sort(mods, ref=o.mods)
        m_ = list(set(o.mods) - set(mods))[0]
        k = m1+"_"+m2+"_to_"+m_
        print(k+":")
        
        if m_ in ["atac"]:
            x_gt = x[m_].reshape(-1)
            x_pred = x_trans[k].reshape(-1)
            print("Computing auroc")
            results["auroc"][batch][k] = roc_auc_score(x_gt, x_pred)
        elif m_ in ["rna", "adt"]:
            mask = np.array(utils.load_csv(pj(mask_dir, m_+".csv"))[1][1:]).astype(bool)
            x_gt = x[m_][:, mask].reshape(-1)
            x_pred = x_trans[k][:, mask].reshape(-1)
            print("Computing pearson_"+m_)
            results["pearson_"+m_][batch][k] = pearsonr(x_gt, x_pred)[0]

utils.mkdirs(result_dir, remove_old=False)
utils.save_toml(results, pj(result_dir, "metrics_mod_detailed.toml"))


# In[ ]:


results_avg = {metric: np.mean(utils.extract_values(v)) for metric, v in results.items()}
df = pd.DataFrame({
    'ASW_mod':          [results_avg['asw_mod']],
    'FOSCTTM':          [results_avg['foscttm']],
    'Label_transfer':   [results_avg['f1']],
    'AUROC':            [results_avg['auroc']],
    'Pearson_RNA':      [results_avg['pearson_rna']],
    'Pearson_ADT':      [results_avg['pearson_adt']],
})
print(df)
df.to_excel(pj(result_dir, "metrics_mod.xlsx"), index=False)

