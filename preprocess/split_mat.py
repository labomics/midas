# Split csv matrix into csv vectors for pytorch traning
import os
os.chdir("/root/workspace/code/sc-transformer/")
import os.path as path
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import csv
import math
from glob import glob
from tqdm import tqdm

 
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="ct1_tp1")
o = parser.parse_args()


base_dirs = glob(pj("data", "processed", o.task, "subset_*"))
for base_dir in base_dirs:
    # Specify directories
    in_dir = pj(base_dir, "mat")
    out_dir = pj(base_dir, "vec")
    utils.mkdirs(out_dir, remove_old=True)
    print("\nDirectory: %s" % (in_dir))

    # Load and save data
    mat_names = glob(pj(in_dir, '*.csv'))  # get filenames
    for i, mat_name in enumerate(mat_names):
        # load
        mat = utils.load_csv(mat_name)
        # mat = utils.transpose_list(mat)
        mod = path.splitext(path.basename(mat_name))[0]
        cell_num = len(mat) - 1
        feat_num = len(mat[0]) - 1
        print("Spliting %s matrix: %d cells, %d features" % (mod, cell_num, feat_num))
        
        # save
        out_mod_dir = pj(out_dir, mod)
        utils.mkdirs(out_mod_dir, remove_old=True)
        vec_name_fmt = utils.get_name_fmt(cell_num) + ".csv"
        vec_name_fmt = pj(out_mod_dir, vec_name_fmt)
        for cell_id in tqdm(range(cell_num)):
            vec_name = vec_name_fmt % cell_id
            utils.save_list_to_csv([mat[cell_id+1][1:]], vec_name)
