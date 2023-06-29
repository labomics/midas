from os import path
from os.path import join as pj
import time
import random
import argparse

from tqdm import tqdm
import math
import numpy as np
import torch as th
from torch import nn, autograd
import matplotlib.pyplot as plt
import umap
import re
import itertools

from modules import models, utils
from modules.datasets import MultimodalDataset
from modules.datasets import MultiDatasetSampler


parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='atlas',
    help="Choose a task")
parser.add_argument('--reference', type=str, default='',
    help="Choose a reference task")
parser.add_argument('--experiment', type=str, default='e0',
    help="Choose an experiment")
parser.add_argument('--rf_experiment', type=str, default='',
    help="Choose a reference experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
# parser.add_argument('--data', type=str, default='sup',
#     help="Choose a data configuration")
parser.add_argument('--actions', type=str, nargs='+', default=['train'],
    help="Choose actions to run")
parser.add_argument('--method', type=str, default='midas',
    help="Choose an method to benchmark")
parser.add_argument('--init_model', type=str, default='',
    help="Load a trained model")
parser.add_argument('--init_from_ref', type=int, default=0,
    help="Load a model trained on the reference task")
parser.add_argument('--mods_conditioned', type=str, nargs='+', default=[],
    help="Modalities conditioned for sampling")
parser.add_argument('--data_conditioned', type=str, default='prior.csv',
    help="Data conditioned for sampling")
parser.add_argument('--sample_num', type=int, default=0,
    help='Number of samples to be generated')
parser.add_argument('--input_mods', type=str, nargs='+', default=[],
    help="Input modalities for transformation")
## Training
parser.add_argument('--epoch_num', type=int, default=2000,
    help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=-1,
    help='Number of samples in a mini-batch')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s_drop_rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--drop_s', type=int, default=0,
    help="Force to drop s")
parser.add_argument('--map_ref', type=int, default=0,
    help="Map query onto reference for transfer learning")
parser.add_argument('--sample_ref', type=int, default=1,
    help="Sample from reference for reference mapping")
parser.add_argument('--seed', type=int, default=0,
    help="Set the random seed to reproduce the results")
parser.add_argument('--use_shm', type=int, default=0,
    help="Use shared memory to accelerate training")
## Debugging
parser.add_argument('--print_iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log_epochs', type=int, default=100,
    help='Epochs to log the training states')
parser.add_argument('--save_epochs', type=int, default=10,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
    help='Print intermediate variables')
# o, _ = parser.parse_known_args()  # for python interactive
o = parser.parse_args()


# Initialize global varibles
data_config = None
net = None
discriminator = None
optimizer_net = None
optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


def main():
    initialize()
    if o.actions == "print_model":
        print_model()
    if "train" in o.actions:
        train()
    if "test" in o.actions:
        test()
    if "save_input" in o.actions:
        predict(joint_latent=False, input=True)
    if "predict_all" in o.actions:
        predict(mod_latent=True, impute=True, batch_correct=True, translate=True, input=True)
    if "predict_joint" in o.actions:
        predict()
    if "predict_all_latent" in o.actions:
        predict(mod_latent=True)
    if "impute" in o.actions:
        predict(impute=True, input=True)
    if "impute2" in o.actions:
        predict(impute=True)
    if "translate" in o.actions:
        predict(translate=True, input=True)
    if "batch_correct" in o.actions:
        predict(batch_correct=True, input=True)
    if "predict_all_latent_bc" in o.actions:
        predict(mod_latent=True, batch_correct=True, input=True)

    if "visualize" in o.actions:
        visualize()




def initialize():
    init_seed()
    init_dirs()
    load_data_config()
    load_model_config()
    get_gpu_config()
    init_model()


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed)
        th.manual_seed(o.seed)
        th.cuda.manual_seed_all(o.seed)


def init_dirs():
    data_folder = re.sub("_generalize", "_transfer", o.task)
    if o.use_shm == 1:
        o.data_dir = pj("/dev/shm", "processed", data_folder)
    else:
        o.data_dir = pj("data", "processed", data_folder)
    o.result_dir = pj("result", o.task, o.experiment, o.model)
    o.pred_dir = pj(o.result_dir, "predict", o.init_model)
    o.train_dir = pj(o.result_dir, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.experiment, o.model))


def load_data_config():
    
    if o.reference == '':
        o.dims_x, o.dims_chr, o.mods = get_dims_x(ref=0)
        o.ref_mods = o.mods
    else:
        _, _, o.mods = get_dims_x(ref=0)
        o.dims_x, o.dims_chr, o.ref_mods = get_dims_x(ref=1)
    o.mod_num = len(o.mods)

    if o.rf_experiment == '':
        o.rf_experiment = o.experiment
    
    global data_config
    data_config = utils.gen_data_config(o.task, o.reference)
    for k, v in data_config.items():
        vars(o)[k] = v
    if o.batch_size > 0:
        o.N = o.batch_size

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)

    if "continual" in o.task:
        o.continual = True
        o.dim_s_query = len(utils.load_toml("configs/data.toml")[re.sub("_continual", "", o.task)]["s_joint"])
        o.dim_s_ref = len(utils.load_toml("configs/data.toml")[o.reference]["s_joint"])
    else:
        o.continual = False

    if o.reference != '' and o.continual == False and o.map_ref == 1:  # map query onto reference for transfer learning

        o.dims_s = {k: v + 1 for k, v in o.dims_s.items()}
        
        cfg_task_ref = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", o.reference)
        data_config_ref = utils.load_toml("configs/data.toml")[cfg_task_ref]
        _, _, s_ref, dims_s_ref = utils.gen_all_batch_ids(data_config_ref["s_joint"], 
                                                    data_config_ref["combs"])
        o.subset_ids_ref = {m: [] for m in dims_s_ref}
        for subset_id, id_dict in enumerate(s_ref):
            for m in id_dict.keys():
                o.subset_ids_ref[m].append(subset_id)

    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2



def load_model_config():
    model_config = utils.load_toml("configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    if "dims_enc_chr" in vars(o).keys():
        o.dims_dec_chr = o.dims_enc_chr[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim if m != "atac" else o.dims_enc_chr[-1] * 22


def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    global net, discriminator, optimizer_net, optimizer_disc
    
    # Initialize models
    net = models.Net(o).cuda()
    discriminator = models.Discriminator(o).cuda()
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    disc_param_num = sum([param.data.numel() for param in discriminator.parameters()])
    print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))
    
    # Load benchmark
    if o.init_model != '':
        if o.init_from_ref == 0:
            fpath = pj(o.train_dir, o.init_model)
            savepoint_toml = utils.load_toml(fpath+".toml")
            benchmark.update(savepoint_toml['benchmark'])
            o.ref_epoch_num = savepoint_toml["o"]["ref_epoch_num"]
        else:
            fpath = pj("result", o.reference, o.rf_experiment, o.model, "train", o.init_model)
            benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
            o.ref_epoch_num = benchmark["epoch_id_start"]
    else:
        o.ref_epoch_num = 0

    # Initialize optimizers
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    
    # Load models and optimizers
    if o.init_model != '':
        savepoint = th.load(fpath+".pt")
        if o.init_from_ref == 0:
            net.load_state_dict(savepoint['net_states'])
            discriminator.load_state_dict(savepoint['disc_states'])
            optimizer_net.load_state_dict(savepoint['optim_net_states'])
            optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
        else:
            exclude_modules = ["s_enc", "s_dec"]
            pretrained_dict = {}
            for k, v in savepoint['net_states'].items():
                exclude = False
                for exclude_module in exclude_modules:
                    if exclude_module in k:
                        exclude = True
                        break
                if not exclude:
                    pretrained_dict[k] = v
            net_dict = net.state_dict()
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        print('Model is initialized from ' + fpath + ".pt")


def print_model():
    global net, discriminator
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)
        print(discriminator, file=f)


def get_dims_x(ref):
    if ref == 0:
        feat_dims = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    else:
        feat_dims = utils.load_csv(pj("data", "processed", o.reference, "feat", "feat_dims.csv"))
    feat_dims = utils.transpose_list(feat_dims)
    
    dims_x = {}
    dims_chr = []
    for i in range(1, len(feat_dims)):
        m = feat_dims[i][0]
        if m == "atac":
            dims_chr = list(map(int, feat_dims[i][1:]))
            dims_x[m] = sum(dims_chr)
        else:   
            dims_x[m] = int(feat_dims[i][1])
    print("Input feature numbers: ", dims_x)

    mods = list(dims_x.keys())
    
    return dims_x, dims_chr, mods
    
    


def train():
    # train_data_loaders = get_dataloaders("train")
    # test_data_loaders = get_dataloaders("test")
    train_data_loader_cat = get_dataloader_cat("train", train_ratio=None)
    # test_data_loader_cat = get_dataloader_cat("test", train_ratio=None)
    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(train_data_loader_cat, "train", epoch_id)
        # run_epoch(train_data_loaders, "train", epoch_id)
        # run_epoch(test_data_loader_cat, "test", epoch_id)
        # run_epoch(test_data_loaders, "test", epoch_id)
        # utils.calc_subsets_foscttm(net.sct, test_data_loaders, benchmark["foscttm"], "test", 
        #                            o.epoch_num, epoch_id)
        check_to_save(epoch_id)


def get_dataloaders(split, train_ratio=None):
    data_loaders = {}
    for subset in range(len(o.s)):
        data_loaders[subset] = get_dataloader(subset, split, train_ratio=train_ratio)
    return data_loaders


def get_dataloader(subset, split, train_ratio=None):
    dataset = MultimodalDataset(o.task, o.reference, o.data_dir, subset, split, train_ratio=train_ratio)
    shuffle = True if split == "train" else False
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N, shuffle=shuffle,
                                           num_workers=64, pin_memory=True)
    print("Subset: %d, modalities %s: %s size: %d" %
          (subset, str(o.combs[subset]), split, dataset.size))
    return data_loader


def get_dataloader_cat(split, train_ratio=None):
    datasets = []
    for subset in range(len(o.s)):
        datasets.append(MultimodalDataset(o.task, o.reference, o.data_dir, subset, split, train_ratio=train_ratio))
        print("Subset: %d, modalities %s: %s size: %d" %  (subset, str(o.combs[subset]), split,
            datasets[subset].size))
    dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    shuffle = True if split == "train" else False
    sampler = MultiDatasetSampler(dataset_cat, batch_size=o.N, shuffle=shuffle)
    data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
        num_workers=64, pin_memory=True)
    return data_loader


def test():
    data_loaders = get_dataloaders()
    run_epoch(data_loaders, "test")


def run_epoch(data_loader, split, epoch_id=0):
    start_time = time.time()
    if split == "train":
        net.train()
        discriminator.train()
    elif split == "test":
        net.eval()
        discriminator.eval()
    else:
        assert False, "Invalid split: %s" % split

    losses = []
    for i, data in enumerate(data_loader):
        loss = run_iter(split, epoch_id, i, data)
        losses.append(loss)
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('%s\tepoch: %d/%d\tBatch: %d/%d\t%s_loss: %.2f'.expandtabs(3) % 
                  (o.task, epoch_id+1, o.epoch_num, i+1, len(data_loader), split, loss))
    loss_avg = np.nanmean(losses)
    epoch_time = (time.time() - start_time) / 3600 / 24
    elapsed_time = epoch_time * (epoch_id+1)
    total_time = epoch_time * o.epoch_num
    print('%s\t%s\tepoch: %d/%d\t%s_loss: %.2f\ttime: %.2f/%.2f\n'.expandtabs(3) % 
          (o.task, o.experiment, epoch_id+1, o.epoch_num, split, loss_avg, elapsed_time, total_time))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg


# # Train by iterating over subsets
# def run_epoch(data_loaders, split, epoch_id=0):
#     if split == "train":
#         net.train()
#         discriminator.train()
#     elif split == "test":
#         net.eval()
#         discriminator.eval()
#     else:
#         assert False, "Invalid split: %s" % split
#     losses = {}
#     for subset, data_loader in data_loaders.items():
#         losses[subset] = 0
#         for i, data in enumerate(data_loader):
#             loss = run_iter(split, epoch_id, i, data)
#             losses[subset] += loss
#             if o.print_iters > 0 and (i+1) % o.print_iters == 0:
#                 print('Epoch: %d/%d, subset: %s, Batch: %d/%d, %s loss: %.2f' % (epoch_id+1,
#                 o.epoch_num, str(subset), i+1, len(data_loader), split, loss))
#         losses[subset] /= len(data_loader)
#         print('Epoch: %d/%d, subset: %d, %s loss: %.2f' % (epoch_id+1, o.epoch_num, subset, 
#             split, losses[subset]))
#     loss_avg = sum(losses.values()) / len(losses.keys())
#     print('Epoch: %d/%d, %s loss: %.2f\n' % (epoch_id+1, o.epoch_num, split, loss_avg))
#     benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
#     return loss_avg


def run_iter(split, epoch_id, iter_id, inputs):

    if split == "train":
        skip = False
        if o.continual and o.sample_ref == 1:
            subset_id_ref = inputs["s"]["joint"][0, 0].item() - o.dim_s_query
            cycle_id = iter_id // o.dim_s
            if subset_id_ref >= 0 and subset_id_ref != (cycle_id % o.dim_s_ref):
                skip = True
        
        if skip:
            return np.nan
        else:
            inputs = utils.convert_tensors_to_cuda(inputs)
            with autograd.set_detect_anomaly(o.debug == 1):
                loss_net, c_all = forward_net(inputs)
                discriminator.epoch = epoch_id - o.ref_epoch_num
                K = 3
                if o.experiment == "k_1":
                    K = 1
                elif o.experiment == "k_2":
                    K = 2
                elif o.experiment == "k_4":
                    K = 4
                elif o.experiment == "k_5":
                    K = 5

                for _ in range(K):
                    loss_disc = forward_disc(utils.detach_tensors(c_all), inputs["s"])
                    update_disc(loss_disc)
                # c = models.CheckBP('c')(c)
                loss_adv = forward_disc(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
                update_net(loss)
                
                # print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
                #       (loss_net.item(), loss_adv.item(), discriminator.prob))

            # If we want to map query onto reference for transfer learning, then we take the reference as an additional batch
            # and train the discriminator after the last training subset
            if o.reference != '' and o.continual == False and o.map_ref == 1 and inputs["s"]["joint"][0, 0].item() == o.dim_s - 2:
                
                # Randomly load c inferred from the reference dataset
                c_all_ref = {}
                subset_ids_sampled = {m: random.choice(ids) for m, ids in o.subset_ids_ref.items()}
                for m, subset_id in subset_ids_sampled.items():
                    z_dir = pj("result", o.reference, o.rf_experiment, o.model, "predict", o.init_model,
                            "subset_"+str(subset_id), "z", m)
                    filename = random.choice(utils.get_filenames(z_dir, "csv"))
                    z = th.from_numpy(np.array(utils.load_csv(pj(z_dir, filename)), dtype=np.float32))
                    # z = th.tensor(utils.load_csv(pj(z_dir, filename)), dtype=th.float32)
                    c_all_ref[m] = z[:, :o.dim_c]
                c_all_ref = utils.convert_tensors_to_cuda(c_all_ref)

                # Generate s for the reference dataset, which is treated as the last subset
                s_ref = {}
                tmp = inputs["s"]["joint"]
                for m, d in o.dims_s.items():
                    s_ref[m] = th.full((c_all_ref[m].size(0), 1), d-1, dtype=tmp.dtype, device=tmp.device)

                with autograd.set_detect_anomaly(o.debug == 1):
                    for _ in range(K):
                        loss_disc = forward_disc(c_all_ref, s_ref)
                        update_disc(loss_disc)
            
    else:
        with th.no_grad():
            inputs = utils.convert_tensors_to_cuda(inputs)
            loss_net, c_all = forward_net(inputs)
            loss_adv = forward_disc(c_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv
            
            # print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
            #       (loss_net.item(), loss_adv.item(), discriminator.prob))
            
    # if o.time == 1:
    #     print('Runtime: %.3fs' % (forward_time + backward_time))
    return loss.item()


def forward_net(inputs):
    return net(inputs)


def forward_disc(c, s):
    return discriminator(c, s)


def update_net(loss):
    update(loss, net, optimizer_net)


def update_disc(loss):
    update(loss, discriminator, optimizer_disc)
    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()
    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id + 1
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "disc_states": discriminator.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
             "optim_disc_states": optimizer_disc.state_dict()
            }, pj(o.train_dir, filename+".pt"))


def predict(joint_latent=True, mod_latent=False, impute=False, batch_correct=False, translate=False, 
            input=False):
    if translate:
        mod_latent = True
    print("Predicting ...")
    dirs = utils.get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input)
    parent_dirs = list(set(map(path.dirname, utils.extract_values(dirs))))
    utils.mkdirs(parent_dirs, remove_old=True)
    utils.mkdirs(dirs, remove_old=True)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data)
                
                # conditioned on all observed modalities
                if joint_latent:
                    x_r_pre, _, _, _, z, _, _, *_ = net.sct(data)  # N * K
                    utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                if impute:
                    x_r = models.gen_real_data(x_r_pre, sampling=False)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                if input:  # save the input
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["x"][m].int(), pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if mod_latent:
                    for m in data["x"].keys():
                        input_data = {
                            "x": {m: data["x"][m]},
                            "s": data["s"], 
                            "e": {}
                        }
                        if m in data["e"].keys():
                            input_data["e"][m] = data["e"][m]
                        x_r_pre, _, _, _, z, c, b, *_ = net.sct(input_data)  # N * K
                        utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"][m], fname_fmt) % i)
                        if translate: # single to double
                            x_r = models.gen_real_data(x_r_pre, sampling=False)
                            for m_ in set(o.mods) - {m}:
                                utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)
                
                if translate: # double to single
                    for mods in itertools.combinations(data["x"].keys(), 2):
                        m1, m2 = utils.ref_sort(mods, ref=o.mods)
                        input_data = {
                            "x": {m1: data["x"][m1], m2: data["x"][m2]},
                            "s": data["s"], 
                            "e": {}
                        }
                        for m in mods:
                            if m in data["e"].keys():
                                input_data["e"][m] = data["e"][m]
                        x_r_pre, *_ = net.sct(input_data)  # N * K
                        x_r = models.gen_real_data(x_r_pre, sampling=False)
                        m_ = list(set(o.mods) - set(mods))[0]
                        utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_], fname_fmt) % i)

        if batch_correct:
            print("Calculating b_centroid ...")
            # z, c, b, subset_ids, batch_ids = utils.load_predicted(o)
            # b = th.from_numpy(b["joint"])
            # subset_ids = th.from_numpy(subset_ids["joint"])
            
            pred = utils.load_predicted(o)
            b = th.from_numpy(pred["z"]["joint"][:, o.dim_c:])
            s = th.from_numpy(pred["s"]["joint"])

            b_mean = b.mean(dim=0, keepdim=True)
            b_subset_mean_list = []
            for subset_id in s.unique():
                b_subset = b[s == subset_id, :]
                b_subset_mean_list.append(b_subset.mean(dim=0))
            b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
            dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
            net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
            net.sct.batch_correction = True
            
            print("Batch correction ...")
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    x_r_pre, *_ = net.sct(data)
                    x_r = models.gen_real_data(x_r_pre, sampling=True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)
        

def visualize():
    pred = utils.load_predicted(o, mod_latent=True)
    z = pred["z"]
    s = pred["s"]
            
    print("Computing UMAP ...")
    umaps = {}
    mods = o.mods + ["joint"]
    for m in mods:
        print("Computing UMAP: " + m)
        umaps[m] = {
            "z": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m]),
            "c": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m][:, :o.dim_c]),
            "b": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m][:, o.dim_c:])
        }

    print("Plotting ...")
    fig_rows = len(o.s_joint) + 1
    fig_fcols = len(mods) * 3
    fsize = 4
    pt_size = [0.03]
    fg_color = [(0.7, 0, 0)]
    bg_color = [(0.3, 1, 1)]
    fig, ax = plt.subplots(fig_rows, fig_fcols, figsize=(fig_fcols*fsize, fig_rows*fsize))

    for i, m in enumerate(umaps.keys()):
        for j, d in enumerate(umaps[m].keys()):
            col = i * 3 + j
            v = umaps[m][d]

            # for each subset
            for subset_id, batch_id in enumerate(o.s_joint):
                fg = (s[m]==subset_id)
                if sum(fg) > 0:
                    bg = ~fg
                    ax[subset_id, col].scatter(v[bg, 0], v[bg, 1], label=m, s=pt_size, 
                        c=bg_color, marker='.')
                    ax[subset_id, col].scatter(v[fg, 0], v[fg, 1], label=m, s=pt_size, 
                        c=fg_color, marker='.')
                    ax[subset_id, col].set_title("subset_%s, batch_%s, %s, %s"
                                                 % (subset_id, batch_id, m, d))

            # for all subsets
            ax[fig_rows-1, col].scatter(v[:, 0], v[:, 1], label=m, s=pt_size, 
                c=fg_color, marker='.')
            ax[fig_rows-1, col].set_title("subset_all, batch_all, %s, %s" % (m, d))

    plt.tight_layout()
    fig_dir = pj(o.result_dir, "predict", o.init_model, "fig")
    utils.mkdirs(fig_dir, remove_old=True)
    plt.savefig(pj(fig_dir, "predict_all.png"))
    # plt.show()




main()
