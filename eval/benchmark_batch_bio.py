#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
os.chdir("/root/workspace/code/midas/")
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import numpy as np
import scib
import scib.metrics as me
import anndata as ad
import scipy
import pandas as pd
import re


# In[11]:


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='teadog_single')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_00001899')
# parser.add_argument('--method', type=str, default='stabmap')
parser.add_argument('--method', type=str, default='midas_embed')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()


# In[12]:


if "midas" in o.method:
    result_dir = pj("result", "comparison", o.task, o.method, o.experiment, o.model, o.init_model)
else:
    result_dir = pj("result", "comparison", o.task, o.method)
cfg_task = re.sub("_vd.*|_vt.*|_atlas|_generalize|_transfer|_ref_.*", "", o.task)
data_config = utils.load_toml("configs/data.toml")[cfg_task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)


# In[13]:


# Load cell type labels
labels = []
label_file = "l1_" + o.task.split("_")[-1] + ".csv" if "_vt" in o.task else "l1.csv"
for raw_data_dir in o.raw_data_dirs:
    label = utils.load_csv(pj(raw_data_dir, "label_seurat", label_file))
    labels += utils.transpose_list(label)[1][1:]
labels = np.array(labels)
print(np.unique(labels))


# In[14]:


# Load predicted latent variables
o.mods = ["atac", "rna", "adt"]
o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", o.init_model)
pred = utils.load_predicted(o)


# In[15]:


if o.method in ["midas_embed", "mofa", "scmomat", "stabmap", "scvaeit", "multigrate", "glue"]:
    output_type = "embed"
elif o.method in [
    "midas_feat+wnn", 
    "harmony+wnn", 
    "pca+wnn",
    "seurat_cca+wnn",
    "seurat_rpca+wnn",
    "scanorama_embed+wnn",
    "scanorama_feat+wnn",
    "liger+wnn",
    "bbknn",
    ]:
    output_type = "graph"
else:
    assert False, o.method+": invalid method!"


# In[16]:


embed = "X_emb"
batch_key = "batch"
label_key = "label"
cluster_key = "cluster"
si_metric = "euclidean"
subsample = 0.5
verbose = False


# In[17]:


c = pred["z"]["joint"][:, :o.dim_c]
s = pred["s"]["joint"]

if o.method == "midas_embed":
    adata = ad.AnnData(c)
    adata.obsm[embed] = c
    adata.obs[batch_key] = s.astype(str)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = labels
    adata.obs[label_key] = adata.obs[label_key].astype("category")
elif o.method in ["mofa", "stabmap", "multigrate", "glue"]:
    adata = ad.AnnData(c*0)
    embeddings = utils.load_csv(pj(result_dir, "embeddings.csv"))
    adata.obsm[embed] = np.array(embeddings)[1:, 1:].astype(np.float32)
    adata.obs[batch_key] = s.astype(str)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = labels
    adata.obs[label_key] = adata.obs[label_key].astype("category")
elif o.method in ["scmomat", "scvaeit"]:
    adata = ad.AnnData(c*0)
    embeddings = utils.load_csv(pj(result_dir, "embeddings.csv"))
    adata.obsm[embed] = np.array(embeddings).astype(np.float32)
    adata.obs[batch_key] = s.astype(str)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = labels
    adata.obs[label_key] = adata.obs[label_key].astype("category")
elif o.method in [
    "midas_feat+wnn", 
    "harmony+wnn", 
    "pca+wnn",
    "seurat_cca+wnn",
    "seurat_rpca+wnn",
    "scanorama_embed+wnn",
    "scanorama_feat+wnn",
    "liger+wnn",
    "bbknn",
    ]:
    adata = ad.AnnData(c*0)
    adata.obs[batch_key] = s.astype(str)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = labels
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    adata.obsp["connectivities"] = scipy.io.mmread(pj(result_dir, "connectivities.mtx")).tocsr()
    adata.uns["neighbors"] = {'connectivities_key': 'connectivities'}


# In[ ]:


results = {}

print('clustering...')
res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata, label_key=label_key,
    cluster_key=cluster_key, function=me.nmi, use_rep=embed, verbose=verbose, inplace=True)

results['NMI'] = me.nmi(adata, group1=cluster_key, group2=label_key, method='arithmetic')
print("NMI: " + str(results['NMI']))

results['ARI'] = me.ari(adata, group1=cluster_key, group2=label_key)
print("ARI: " + str(results['ARI']))

type_ = "knn" if output_type == "graph" else None
results['kBET'] = me.kBET(adata, batch_key=batch_key, label_key=label_key, embed=embed, 
    type_=type_, verbose=verbose)
print("kBET: " + str(results['kBET']))

results['il_score_f1'] = me.isolated_labels(adata, label_key=label_key, batch_key=batch_key,
    embed=embed, cluster=True, verbose=verbose)
print("il_score_f1: " + str(results['il_score_f1']))

results['graph_conn'] = me.graph_connectivity(adata, label_key=label_key)
print("graph_conn: " + str(results['graph_conn']))

results['cLISI'] = me.clisi_graph(adata, batch_key=batch_key, label_key=label_key, type_="knn",
    subsample=subsample*100, n_cores=1, verbose=verbose)
print("cLISI: " + str(results['cLISI']))

results['iLISI'] = me.ilisi_graph(adata, batch_key=batch_key, type_="knn",
    subsample=subsample*100, n_cores=1, verbose=verbose)
print("iLISI: " + str(results['iLISI']))

results = {k: float(v) for k, v in results.items()}
# results['batch_score'] = np.nanmean([results['iLISI'], results['graph_conn'], results['kBET']])
# results['bio_score'] = np.nanmean([results['NMI'], results['ARI'], results['il_score_f1'], results['cLISI']])
# results["overall_score"] = float(0.4 * results['batch_score'] + 0.6 * results['bio_score'])

df = pd.DataFrame({
    'iLISI':          [results['iLISI']],
    'graph_conn':     [results['graph_conn']],
    'kBET':           [results['kBET']],
    # 'batch_score':    [results['batch_score']],
    'NMI':            [results['NMI']],
    'ARI':            [results['ARI']],
    'il_score_f1':    [results['il_score_f1']],
    'cLISI':          [results['cLISI']],
    # 'bio_score':      [results['bio_score']],
    # 'overall_score':  [results['overall_score']]
})
print(df)
utils.mkdirs(result_dir, remove_old=False)
df.to_excel(pj(result_dir, "metrics_batch_bio.xlsx"), index=False)

