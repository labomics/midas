import os
import math
import random
from typing import Union

import scmidas.utils as utils

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MultiDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.Sampler = torch.utils.data.sampler.RandomSampler
        else:
            self.Sampler = torch.utils.data.sampler.SequentialSampler
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([cur_dataset.size for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.Sampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        idx = list(range(self.number_of_datasets))
        for _ in range(0, epoch_samples, step):
            random.shuffle(idx)
            for i in idx:
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class MultimodalDataset(Dataset):

    def __init__(self, data, subset, s_subset, reference_features=None):
        super(MultimodalDataset, self).__init__()

        self.mods = data.mods['subset_%d'%subset]
        self.s_subset = s_subset
        self.transform = None
        self.dims_x = data.feat_dims
        self.dims_chr = data.dims_chr
        if reference_features is not None:
            self.transform = {}
            for k in self.mods:
                if (k != 'atac') and (reference_features[k] != data.features[k]):
                    f, self.transform[k] = utils.merge_features(reference_features[k], data.features[k].copy())
                    self.dims_x[k] = len(f)
        else:
            self.reference_features = data.features

        base_dir = os.path.join(data.data_path, 'subset_%d'%subset)

        self.in_dirs = {}
        self.masks = {}
        for m in self.mods:
            self.in_dirs[m] = os.path.join(base_dir, "vec", m)
            if m in ["rna", "adt"]:
                mask = utils.load_csv(os.path.join(base_dir, "mask", m+".csv"))[1][1:]
                self.masks[m] = np.array(mask, dtype=np.float32)

        filenames_list = []
        for in_dir in self.in_dirs.values():
            filenames_list.append(utils.get_filenames(in_dir, "csv"))
        cell_nums = [len(filenames) for filenames in filenames_list]
        assert cell_nums[0] > 0 and len(set(cell_nums)) == 1, \
            "Inconsistent cell numbers!"
        self.filenames = filenames_list[0]

        self.size = len(self.filenames)


    def __getitem__(self, index):
        items = {"x": {}, "s": {}, "e": {}}
        for m, v in self.s_subset.items():
            items["s"][m] = np.array([v], dtype=np.int64)
        for m in self.mods:
            file_path = os.path.join(self.in_dirs[m], self.filenames[index])
            v = np.array(utils.load_csv(file_path)[0])
            if m == "label":
                items["x"][m] = v.astype(np.int64)
            elif m == "atac":
                items["x"][m] = np.where(v.astype(np.float32) > 0.5, 1, 0).astype(np.float32)
            else:
                items["x"][m] = v.astype(np.float32)

                if self.transform is not None and m in self.transform.keys():
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[self.transform[m][0]] = items["x"][m][self.transform[m][1]]
                    items["x"][m] = temp
                elif items["x"][m].shape[0] < self.dims_x[m]:
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[:items["x"][m].shape[0]] = items["x"][m]
                    items["x"][m] = temp
            if m in self.masks.keys():
                items["e"][m] = self.masks[m]
                if self.transform is not None and m in self.transform.keys():
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[self.transform[m][0]] = items["e"][m][self.transform[m][1]]
                    items["e"][m] = temp
                elif items["e"][m].shape[0] < self.dims_x[m]:
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[:items["e"][m].shape[0]] = items["e"][m]
                    items["e"][m] = temp

        return items


    def __len__(self):
        return self.size

class GetDataInfo():
    """Sumarize data information from path.

    Args:
        data_path (str): Path of directory containing data to be summarized.
        mods (dict, optional): Predefined modalities. If not given, all modalities will be read.
        print_info (bool, optional): If True, prints the summarized information of the data.
        
    Note:
        The directory of data should be organized as::

            data_path
            ├── feat
            │   ├── feat_dims.csv
            │   ├── feat_names_mod1.csv
            │   ├── feat_names_mod2.csv
            │   └── feat_names_mod3.csv
            ├── subset_0  # batch 0
            │   ├── cell_names.csv
            │   ├── mask # feature mask
            │   │   ├── mod1.csv
            │   │   ├── mod2.csv
            │   │   └── mod3.csv
            │   └── vec # counts for each cell
            │       ├── mod1
            │       │   ├── 0000.csv
            │       │   ├── ...
            │       │   └── xxx.csv
            │       ├── mod2
            │       │   ├── 0000.csv
            │       │   ├── ...
            │       │   └── xxx.csv
            │       └── mod3
            │           ├── 0000.csv
            │           ├── ...
            │           └── xxx.csv
            ├── subset_1
            ├── ...
            └── subset_n
    """
    def __init__(
            self, 
            data_path:str, 
            mods:Union[dict, None] = None, 
            print_info:bool = True
            ):

        if mods is not None:
            self.mods = {k:utils.ref_sort(v, ref=['atac', 'rna', 'adt']) for k, v in mods.items()}
            self.predefine_mod = True
        else:
            self.predefine_mod = False
            self.mods = {}
        self.data_path = data_path
        self.__read_dir__()
        if print_info:
            self.info()
    
    def __read_dir__(self):
        assert os.path.exists(self.data_path), "This path does not exist."
        assert os.path.exists(os.path.join(self.data_path, 'feat')), "Feat dir does not exist."
        assert os.path.exists(os.path.join(self.data_path, 'feat', 'feat_dims.csv')), "Feat dimension 'feat_dims.csv' does not exist."
        
        self.subset = []
        self.cell_names = {}
        self.cell_names_orig = {}
        self.subset_cell_num = {}
        self.num_subset = 0
        self.features = {}
        self.masks = []
            
        for i in os.listdir(self.data_path):
            if 'subset_' in i:
                self.num_subset += 1
        for n in range(self.num_subset):
                i = 'subset_%d'%n
                self.subset.append(i)
                assert os.path.exists(os.path.join(self.data_path, i, 'cell_names.csv')), "'cell_names.csv' does not exist in %s."%i
                try:
                    self.cell_names[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names_sampled.csv')).values[:, 1].flatten()
                except:
                    self.cell_names[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names.csv')).values[:, 1].flatten()
                self.cell_names_orig[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names.csv')).values[:, 1].flatten()
                self.subset_cell_num[i] = len(self.cell_names[i])
                
                if not self.predefine_mod:
                    m = []
                    for j in os.listdir(os.path.join(self.data_path, i, 'vec')):
                        if j in ['atac', 'rna', 'adt'] and os.path.exists(os.path.join(self.data_path, 'feat', 'feat_names_%s.csv'%j)):
                            m.append(j)
                    self.mods[i] = utils.ref_sort(m, ref=['atac', 'rna', 'adt'])

        self.mod_combination = utils.combine_mod(self.mods)

        for j in self.mod_combination:
            self.features[j] = pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_names_%s.csv'%j), index_col=0).values.flatten().tolist()
        for n in range(self.num_subset):
            masks = {}
            for m in ["rna", "adt"]:   
                if m in self.mods[f"subset_{n}"]:
                    masks[m] = pd.read_csv(os.path.join(self.data_path, f"subset_{n}", "mask", m+".csv")).values.flatten()
            self.masks.append(masks)
        self.feat_dims = self.__cal_feat_dims__(pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_dims.csv'), index_col=0), self.mod_combination)
        self.subset.sort()
        if 'atac' in self.mod_combination:
            self.dims_chr = pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_dims.csv'), index_col=0)['atac'].values.tolist()
            self.n_chr = len(self.dims_chr)
        else:
            self.dims_chr = []
            self.n_chr = None
    
    def __cal_feat_dims__(self, df, mods):
        feat_dims = {}
        # print(df)
        for c in df.columns:
            if c == 'atac' and c in mods:
                feat_dims['atac'] = df[c].values.sum().tolist()
            elif c in mods:
                feat_dims[c] = df[c].values[0].tolist()
        return feat_dims

    def info(self):
        """Print information.
        """
        print('%d subset(s) in this path' % self.num_subset, self.feat_dims)
        for key,value in self.subset_cell_num.items():
            print('%10s : %5d cells' % (key, value), ';', self.mods[key])

def GenDataFromPath(data_path_list:list, save_dir:str, remove_old:bool = True, feature:str="union"):
    """ Convert csv files to MIDAS input format.

    Args:
        data_path_list (list): A list of dictionaries where each item represents a batch with CSV paths for each modality. Ensure each CSV file (cell * features) has correct column names and cell names. For example: [{"rna": "rna.csv", "adt": "adt.csv"}, {"adt": "adt2.csv", "atac": "atac.csv"}].
        save_dir (str): Target path to save the data.
        remove_old (bool): Whether to remove old directories.
        feature (str): Strategy for features selection. Support "union" and "intersect".
    
    Note:
        For ATAC-seq data, we determine the chromosome number based on the prefix of the feature names, such as "chr1-xxxx", "chr2-xxx", ..., "chr22-xxx" 
        (see scmidas.utils.split_list_by_prefix() and the Supplementary Figure 1 in our paper for better understanding). 
    """
    data = data_path_list
    batch_num = len(data)
    mods = list(set([item for sublist in [list(d.keys()) for d in data] for item in sublist]))
    if remove_old:
        utils.mkdirs(f"{save_dir}",remove_old=remove_old)
    utils.mkdirs(f"{save_dir}/feat", remove_old=remove_old)

    for i in range(batch_num):
        utils.mkdirs(f"{save_dir}/subset_{i}/mask", remove_old=remove_old)
        utils.mkdirs(f"{save_dir}/subset_{i}/vec", remove_old=remove_old)
    
    for i, b in enumerate(data):
        for m in b:
            utils.mkdirs(f"{save_dir}/subset_{i}/vec/{m}", remove_old=remove_old)
    if feature == "union":
        feat_names = {m:[] for m in mods}
        for i, b in enumerate(data):
            cn = []
            for m in b.keys():
                feat_name = list(pd.read_csv(b[m], index_col=0, nrows=1).columns)
                if feat_names[m] != []:
                    feat_names[m], _ = utils.merge_features(feat_names[m], feat_name)
                else:
                    feat_names[m] = feat_name
                cn.append(list(pd.read_csv(b[m], usecols=[0], index_col=0).index))
            assert utils.lists_are_identical(cn), f"inconsistent cell names in batch {i}"
            pd.DataFrame(cn[0]).to_csv(f"{save_dir}/subset_{i}/cell_names.csv")
    elif feature == "intersect":
        feat_names = {m:[] for m in mods}
        for i, b in enumerate(data):
            cn = []
            for m in b.keys():
                feat_name = list(pd.read_csv(b[m], index_col=0, nrows=1).columns)
                if len(feat_names[m]) > 0:
                    feat_names[m] = utils.ref_sort(np.intersect1d(feat_names[m], feat_name), feat_names[m])
                else:
                    feat_names[m] = feat_name
                cn.append(list(pd.read_csv(b[m], usecols=[0], index_col=0).index))
            assert utils.lists_are_identical(cn), f"inconsistent cell names in batch {i}"
            pd.DataFrame(cn[0]).to_csv(f"{save_dir}/subset_{i}/cell_names.csv")
    # Calculate the feature dimensions.
    feat_dims = {}
    for m in utils.ref_sort(feat_names.keys(), ['atac', 'rna', 'adt']):
        if m=="atac":
            chr_dims = []
            split_chr = utils.split_list_by_prefix(feat_names["atac"])
            chr_keys = utils.sort_chromosomes(list(split_chr.keys()))
            for c in chr_keys:
                chr_dims.append(len(split_chr[c]))
            feat_dims[m] = chr_dims
        else:
            if "atac" in feat_names:
                feat_dims[m] = [len(feat_names[m])] * len(chr_keys) # All columns in the pandas DataFrame must have uniform dimensions.

            else:
                feat_dims[m] = [len(feat_names[m])]
    pd.DataFrame(feat_dims).to_csv(f"{save_dir}/feat/feat_dims.csv")
    for m in feat_names.keys():
        pd.DataFrame(feat_names[m]).to_csv(f"{save_dir}/feat/feat_names_{m}.csv")
    
    # After aligning all batches, we need to generate the new matrix and mask for each batch.
    transforms = []
    for b in data:
        transform = {}
        for m in b.keys():
            feat_name = list(pd.read_csv(b[m], index_col=0, nrows=1).columns)
            _, transform[m] = utils.merge_features(feat_names[m], feat_name, only_f1=False if feature == "union" else True)
        transforms.append(transform)

    # save data in MIDAS format
    for i, b in enumerate(data):
        for m in b.keys():
            d = pd.read_csv(b[m], index_col=0)
            new_mat = pd.DataFrame(np.zeros([d.shape[0], len(feat_names[m])]))
            new_mat.index = d.index
            new_mat.columns = feat_names[m]
            new_mat.iloc[:, transforms[i][m][0]] = d.iloc[:, transforms[i][m][1]]
            if m in ["rna", "adt"]:
                mask = np.array([1 if f in feat_names[m] else 0 for f in d.columns])
                new_mask = pd.DataFrame(np.zeros(len(feat_names[m]))).T
                new_mask.iloc[:, transforms[i][m][0]] = mask[transforms[i][m][1]]
                new_mask = new_mask.astype(int)
                new_mask.to_csv(f"{save_dir}/subset_{i}/mask/{m}.csv")
            cell_num = d.shape[0]
            feat_num = d.shape[1]
            vec_name_fmt = os.path.join(f"{save_dir}/subset_{i}/vec/{m}", utils.get_name_fmt(cell_num) + ".csv")
            print("Spliting %s matrix: %d cells, %d features" % (m, cell_num, feat_num))
            new_mat = new_mat.values
            for c in tqdm(range(len(new_mat))):
                utils.save_list_to_csv([new_mat[c]], vec_name_fmt % c)
                