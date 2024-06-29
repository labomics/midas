from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import pearsonr
import itertools
import scib
import scib.metrics as me
import os
import torch
import numpy as np
import scmidas.utils as utils
import scanpy as sc

def eval_scib(pred, label, dim_c=32, subsample=0.5, verbose=True):
    """ Evaluation the results for MIDAS.

    Args:
        pred (dict): A dictionary generated by MIDAS.read_embeddings(group_by="subset"). Joint embedding must be contained.
        label (list): A list like [label_for_batch1, label_for_batch2, ...].
        dim_c (int): Dimension of biological variables.
        subsample (float): The fraction of samples to be evaluated. Use this when the memory is not enough or to shorten the time for calculating.
        verbose (bool): Wether to print intermediate results.
    Return:
    
    Example:
        >>> MIDAS.predict(joint_latent=True)
        >>> pred = MIDAS.read_preds(joint_latent=True, group_by="subset")
        >>> evaluation = eval_scib(pred, label)

    """
    embed = "X_emb"
    batch_key = "batch"
    label_key = "label"
    cluster_key = "cluster"
    verbose = verbose
    results = {}
    all_mods = []
    for batch_id in pred.keys():
        all_mods.append(list(pred[batch_id]["s"].keys()))
    all_mods = utils.ref_sort(np.unique(np.array(all_mods).flatten()).tolist(), ['atac', 'rna','adt'])
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # SCIB
    all_s = []
    all_z = []
    for batch_id in pred.keys():
        all_s.append(pred[batch_id]["s"]["joint"])
        all_z.append(pred[batch_id]["z"]["joint"][:, :dim_c])
    all_s = np.concatenate(all_s)
    all_z = np.concatenate(all_z)
    adata_joint = sc.AnnData(all_z*0)
    adata_joint.obsm[embed] = all_z
    adata_joint.obs[batch_key] = all_s
    adata_joint.obs[batch_key] = adata_joint.obs[batch_key].astype("str").astype("category")
    adata_joint.obs[label_key] = np.concatenate(label)
    adata_joint.obs[label_key] = adata_joint.obs[label_key].astype("str").astype("category")
    adata = adata_joint
    # print('clustering...')
    print("step 1/8: clustering")
    res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata, label_key=label_key,
        cluster_key=cluster_key, function=me.nmi, use_rep=embed, verbose=verbose, inplace=True)
    
    print("step 2/8: calculating NMI")
    results['NMI'] = me.nmi(adata, group1=cluster_key, group2=label_key, method='arithmetic')
    # print("NMI: " + str(results['NMI']))

    print("step 3/8: calculating ARI")
    results['ARI'] = me.ari(adata, group1=cluster_key, group2=label_key)

    print("step 4/8: calculating kBET")
    type_ =  None
    results['kBET'] = me.kBET(adata, batch_key=batch_key, label_key=label_key, embed=embed, 
        type_=type_, verbose=verbose)

    print("step 5/8: calculating il_score_f1")
    results['il_score_f1'] = me.isolated_labels(adata, label_key=label_key, batch_key=batch_key,
        embed=embed, cluster=True, verbose=verbose)

    print("step 6/8: calculating graph_conn")
    results['graph_conn'] = me.graph_connectivity(adata, label_key=label_key)

    print("step 7/8: calculating cLISI")
    results['cLISI'] = me.clisi_graph(adata, batch_key=batch_key, label_key=label_key, type_="knn",
        subsample=subsample*100, n_cores=1, verbose=verbose)

    print("step 8/8: calculating iLISI")
    results['iLISI'] = me.ilisi_graph(adata, batch_key=batch_key, type_="knn",
        subsample=subsample*100, n_cores=1, verbose=verbose)

    return results

def eval_scmib(pred, label, masks, dim_c=32, si_metric="euclidean", verbose=True):
    """ Evaluation the results for MIDAS.

    Args:
        pred (dict): A dictionary generated by MIDAS.read_embeddings(group_by="subset"). 
        label (list): A list like [label_for_batch1, label_for_batch2, ...].
        masks (list): A list of mask like [mask_for_batch1, mask_for_batch2, ...]. This can be obtained from MIDAS.masks.
        dim_c (int): Dimension of biological variables.
        si_metric (str): See sklearn silhouette score.
        verbose (bool): Wether to print intermediate results.
    Return:
    
    Example:
        >>> MIDAS.predict(mod_latent=True, impute=True, batch_correct=True, translate=True)
        >>> pred = MIDAS.read_preds(mod_latent=True, translate=True, input=True, group_by="subset")
        >>> evaluation = eval(pred, label, MIDAS.masks)

    """
    embed = "X_emb"
    label_key = "label"
    mod_key = "modality"
    verbose = verbose
    results = {
        "asw_mod": {},
        "foscttm": {},
        "f1": {},
        "auroc": {},
        "pearson_rna": {},
        "pearson_adt": {},
    }
    all_mods = []
    for batch_id in pred.keys():
        all_mods.append(list(pred[batch_id]["s"].keys()))
    all_mods = utils.ref_sort(np.unique(np.array(all_mods).flatten()).tolist(), ['atac', 'rna','adt'])
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

    # SCMIB
    for batch_id in pred.keys():
        print(f"calculating batch {batch_id}/{len(pred.keys())}")
        z = pred[batch_id]["z"]
        x = pred[batch_id]["x"]
        x_trans = pred[batch_id]["x_trans"]

        c = {m: v[:, :dim_c] for m, v in z.items()}
        c_cat = np.concatenate([c[k] for k in set(c.keys()) - {"joint"}], axis=0)
        mods_cat = np.concatenate([[k]*len(c[k])  for k in set(c.keys()) - {"joint"}]).tolist()

        label_cat = np.tile(label[batch_id], len(c.keys())-1)
        
        assert len(c_cat) == len(mods_cat) == len(label_cat), "Inconsistent lengths!"
        
        batch = str(batch_id) # toml dict key must be str
        adata = sc.AnnData(c_cat)
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
                results["foscttm"][batch][k] = 1 - utils.calc_foscttm(torch.from_numpy(c[m]), torch.from_numpy(c[m_]))
                
                knn.fit(c[m], label[batch_id])
                label_pred = knn.predict(c[m_])
                results["f1"][batch][k] = f1_score(label[batch_id], label_pred, average='micro')
                
                if m_ in ["atac"] and m_ in c.keys():
                    x_gt = x[m_].reshape(-1)
                    x_pred = x_trans[k].reshape(-1)
                    results["auroc"][batch][k] = roc_auc_score(x_gt, x_pred)
                elif m_ in ["rna", "adt"] and m_ in c.keys():
                    mask = masks[batch_id][m_].astype(bool)
                    x_gt = x[m_][:, mask].reshape(-1)
                    x_pred = x_trans[k][:, mask].reshape(-1)
                    results["pearson_"+m_][batch][k] = pearsonr(x_gt, x_pred)[0]
                    
        for mods in itertools.combinations(c.keys() - {"joint"}, 2): # double to single
            m1, m2 = utils.ref_sort(mods, ref=all_mods)
            m_ = list(set(all_mods) - set(mods))[0]
            k = m1+"_"+m2+"_to_"+m_
            
            if m_ in ["atac"] and m_ in x.keys():
                x_gt = x[m_].reshape(-1)
                x_pred = x_trans[k].reshape(-1)
                results["auroc"][batch][k] = roc_auc_score(x_gt, x_pred)
            elif m_ in ["rna", "adt"] and m_ in x.keys():
                mask = masks[batch_id][m_].astype(bool)
                x_gt = x[m_][:, mask].reshape(-1)
                x_pred = x_trans[k][:, mask].reshape(-1)
                results["pearson_"+m_][batch][k] = pearsonr(x_gt, x_pred)[0]
    return results