from os import path
from os.path import join as pj
import csv
import math
import numpy as np
import torch as th
from torch.utils.data import Dataset
import modules.utils as utils
import re



class MultimodalDataset(Dataset):

    def __init__(self, task, reference, data_dir, subset=0, split="train", comb=None, train_ratio=None):
        super(MultimodalDataset, self).__init__()
        
        config = utils.gen_data_config(task, reference)
        for kw, arg in config.items():
            setattr(self, kw, arg)

        _, self.combs, self.s, _ = utils.gen_all_batch_ids(self.s_joint, self.combs)
        
        assert subset < len(self.combs) == len(self.s), "Inconsistent subset specifications!"
        self.subset = subset
        self.comb = self.combs[subset] if comb is None else comb
        if train_ratio is not None: self.train_ratio = train_ratio
        self.s_subset = self.s[subset]
        # self.s_drop_rate = s_drop_rate if split == "train" else 0
        
        base_dir = pj(data_dir, "subset_"+str(subset))
        self.in_dirs = {}
        self.masks = {}
        for m in self.comb:
            self.in_dirs[m] = pj(base_dir, "vec", m)
            if m in ["rna", "adt"]:
                mask = utils.load_csv(pj(base_dir, "mask", m+".csv"))[1][1:]
                self.masks[m] = np.array(mask, dtype=np.float32)

        filenames_list = []
        for in_dir in self.in_dirs.values():
            filenames_list.append(utils.get_filenames(in_dir, "csv"))
        cell_nums = [len(filenames) for filenames in filenames_list]
        assert cell_nums[0] > 0 and len(set(cell_nums)) == 1, \
            "Inconsistent cell numbers!"
        filenames = filenames_list[0]

        train_num = int(round(len(filenames) * self.train_ratio))
        if split == "train":
            self.filenames = filenames[:train_num]
        else:
            self.filenames = filenames[train_num:]
        self.size = len(self.filenames)


    def __getitem__(self, index):
        items = {"x": {}, "s": {}, "e": {}}
        
        for m, v in self.s_subset.items():
            items["s"][m] = np.array([v], dtype=np.int64)
        
        for m in self.comb:
            file_path = pj(self.in_dirs[m], self.filenames[index])
            v = np.array(utils.load_csv(file_path)[0])
            if m == "label":
                items["x"][m] = v.astype(np.int64)
            elif m == "atac":
                items["x"][m] = np.where(v.astype(np.float32) > 0.5, 1, 0).astype(np.float32)
            else:
                items["x"][m] = v.astype(np.float32)
            if m in self.masks.keys():
                items["e"][m] = self.masks[m]
        return items


    def __len__(self):
        return self.size


class MultiDatasetSampler(th.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size=1, shuffle=True, max_dataset_size=10000):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.Sampler = th.utils.data.sampler.RandomSampler
        else:
            self.Sampler = th.utils.data.sampler.SequentialSampler
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([cur_dataset.size for cur_dataset in dataset.datasets])
        self.largest_dataset_size = min(self.largest_dataset_size, max_dataset_size)

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
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
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





