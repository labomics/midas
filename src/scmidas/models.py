import numpy as np
import torch
from torch import nn, autograd

import functions.models as F
import scmidas.utils as utils

import os
import time
import itertools
from tqdm import tqdm
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt

from scmidas.datasets import MultiDatasetSampler, MultimodalDataset, GetDataInfo
from scmidas.sample import BallTreeSubsample

class Net(nn.Module):
    def __init__(self, o):
        super(Net, self).__init__()
        self.o = o
        self.sct = SCT(o)
        self.loss_calculator = LossCalculator(o)

    def forward(self, inputs):
        # *outputs, = self.sct(inputs)
        # loss = self.loss_calculator(inputs, *outputs)
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, c_all = self.sct(inputs)
        loss = self.loss_calculator(inputs, x_r_pre, s_r_pre, z_mu, z_logvar, b, z_uni)
        return loss, c_all


class SCT(nn.Module):
    def __init__(self, o):
        super(SCT, self).__init__()
        self.o = o
        self.sampling = False
        self.batch_correction = False
        self.b_centroid = None

        # Modality encoders q(z|x^m)
        x_encs = {}
        x_shared_enc = MLP(o.dims_enc_x+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        for m in o.ref_mods:
            x_indiv_enc = MLP([o.dims_h[m], o.dims_enc_x[0]], out_trans='mish', norm=o.norm,
                              drop=o.drop)
            x_encs[m] = nn.Sequential(x_indiv_enc, x_shared_enc)
        self.x_encs = nn.ModuleDict(x_encs)
        # Modality decoder p(x^m|c, b)
        self.x_dec = MLP([o.dim_z]+o.dims_dec_x+[sum(o.dims_h.values())], hid_norm=o.norm,
                         hid_drop=o.drop)

        # Subject encoder q(z|s)
        self.s_enc = MLP([o.dim_s]+o.dims_enc_s+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        # Subject decoder p(s|b)
        self.s_dec = MLP([o.dim_b]+o.dims_dec_s+[o.dim_s], hid_norm=o.norm, hid_drop=o.drop)

        # Chromosome encoders and decoders
        if "atac" in o.ref_mods:
            chr_encs, chr_decs = [], []
            for dim_chr in o.dims_chr:
                chr_encs.append(MLP([dim_chr]+o.dims_enc_chr, hid_norm=o.norm, hid_drop=o.drop))
                chr_decs.append(MLP(o.dims_dec_chr+[dim_chr], hid_norm=o.norm, hid_drop=o.drop))
            self.chr_encs = nn.ModuleList(chr_encs)
            self.chr_decs = nn.ModuleList(chr_decs)
            self.chr_enc_cat_layer = Layer1D(o.dims_h["atac"], o.norm, "mish", o.drop)
            self.chr_dec_split_layer = Layer1D(o.dims_h["atac"], o.norm, "mish", o.drop)


    def forward(self, inputs):
        o = self.o
        x = inputs["x"]
        e = inputs["e"]
        s = None
        if o.drop_s == 0 and "s" in inputs.keys():
            s_drop_rate = o.s_drop_rate if self.training else 0
            if torch.rand([]).item() < 1 - s_drop_rate:
                s = inputs["s"]

        # Encode x_m
        z_x_mu, z_x_logvar = {}, {}
        x_pp = {}
        for m in x.keys():
            x_pp[m] = preprocess(x[m], m, o.dims_x[m])
            
            if m in ["rna", "adt"]:  # use mask
                h = x_pp[m] * e[m]
            elif m == "atac":        # encode each chromosome
                x_chrs = x_pp[m].split(o.dims_chr, dim=1)
                h_chrs = [self.chr_encs[i](x_chr) for i, x_chr in enumerate(x_chrs)]
                h = self.chr_enc_cat_layer(torch.cat(h_chrs, dim=1))
            else:
                h = x_pp[m]
            # encoding
            # print(m, h.shape)
            z_x_mu[m], z_x_logvar[m] = self.x_encs[m](h).split(o.dim_z, dim=1)

        # Encode s
        if s is not None:
            s_pp = nn.functional.one_hot(s["joint"].squeeze(1), num_classes=o.dim_s).float()  # N * B
            z_s_mu, z_s_logvar = self.s_enc(s_pp).split(o.dim_z, dim=1)
            z_s_mu, z_s_logvar = [z_s_mu], [z_s_logvar]
        else:
            z_s_mu, z_s_logvar = [], []

        # Use product-of-experts
        z_x_mu_list = list(z_x_mu.values())
        z_x_logvar_list = list(z_x_logvar.values())
        z_mu, z_logvar = poe(z_x_mu_list+z_s_mu, z_x_logvar_list+z_s_logvar)  # N * K
        
        # Sample z
        if self.training:
            z = utils.sample_gaussian(z_mu, z_logvar)
        elif self.sampling and o.sample_num > 0:
            z_mu_expand = z_mu.unsqueeze(1)  # N * 1 * K
            z_logvar_expand = z_logvar.unsqueeze(1).expand(-1, o.sample_num, o.dim_z)  # N * I * K
            z = utils.sample_gaussian(z_mu_expand, z_logvar_expand).reshape(-1, o.dim_z)  # NI * K
        else:  # validation
            z = z_mu

        c, b = z.split([o.dim_c, o.dim_b], dim=1)
        
        # Generate x_m activation/probability
        if self.batch_correction:
            z_bc = z.clone()
            z_bc[:, o.dim_c:] = self.b_centroid.type_as(z).unsqueeze(0)
            x_r_pre = self.x_dec(z_bc).split(list(o.dims_h.values()), dim=1)
        else:
            x_r_pre = self.x_dec(z).split(list(o.dims_h.values()), dim=1)
        x_r_pre = utils.get_dict(o.ref_mods, x_r_pre)
        if "atac" in x_r_pre.keys():
            h_chrs = self.chr_dec_split_layer(x_r_pre["atac"]).split(o.dims_dec_chr[0], dim=1)
            x_chrs = [self.chr_decs[i](h_chr) for i, h_chr in enumerate(h_chrs)]
            x_r_pre["atac"] = torch.cat(x_chrs, dim=1).sigmoid()
        
        # Generate s activation
        if s is not None:
            s_r_pre = self.s_dec(b)
        else:
            s_r_pre = None
            
        #
        z_uni, c_all = {}, {}
        for m in z_x_mu.keys():
            # Calculate q(z|x^m, s)
            z_uni_mu, z_uni_logvar = poe([z_x_mu[m]]+z_s_mu, [z_x_logvar[m]]+z_s_logvar)  # N * K
            z_uni[m] = utils.sample_gaussian(z_uni_mu, z_uni_logvar)  # N * K
            c_all[m] = z_uni[m][:, :o.dim_c]  # N * C
        c_all["joint"] = c
  
        return x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, c_all


def poe(mus, logvars):
    """
    Product of Experts
    :param list mus: The Mean. [mu_1, ..., mu_M], where mu_m is N * K
    :param list logvars: The log-scaled variance. [logvar_1, ..., logvar_M], where logvar_m is N * K
    """
    
    mus = [torch.full_like(mus[0], 0)] + mus
    logvars = [torch.full_like(logvars[0], 0)] + logvars
    
    mus_stack = torch.stack(mus, dim=1)  # N * M * K
    logvars_stack = torch.stack(logvars, dim=1)
    
    T = exp(-logvars_stack)  # precision of i-th Gaussian expert at point x
    T_sum = T.sum(1)  # N * K
    pd_mu = (mus_stack * T).sum(1) / T_sum
    pd_var = 1 / T_sum
    pd_logvar = log(pd_var)
    return pd_mu, pd_logvar  # N * K


def gen_real_data(x_r_pre, sampling=True):
    """
    Generate real data using x_r_pre.

    :param bool sampling: whether to generate discrete samples.
    """
    x_r = {}
    for m, v in x_r_pre.items():
        if m in ["rna", "adt"]:
            x_r[m] = v.exp()
            if sampling:
                x_r[m] = torch.poisson(x_r[m]).int()
        else:  # for atac
            x_r[m] = v
            if sampling:
                x_r[m] = torch.bernoulli(x_r[m]).int()
    return x_r


class Discriminator(nn.Module):

    def __init__(self, o):
        super(Discriminator, self).__init__()
        self.o = o

        predictors = {}
        mods = o.mods + ["joint"]
        for m in mods:
            predictors[m] = MLP([o.dim_c]+o.dims_discriminator+[o.dims_s[m]],
                                    hid_norm=o.norm, hid_drop=o.drop)
        self.predictors = nn.ModuleDict(predictors)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  # log_softmax + nll
        
        self.epoch = None
        
        
    def forward(self, c_all, s_all):
        o = self.o
        loss_dict = {}
        
        for m in s_all.keys():
            c, s = c_all[m], s_all[m]
            s_r_pre = self.predictors[m](c)
            loss_dict[m] = self.cross_entropy_loss(s_r_pre, s.squeeze(1))

            if m == "joint":
                prob = s_r_pre.softmax(dim=1)
                mask = nn.functional.one_hot(s.squeeze(1), num_classes=o.dims_s[m])
                self.prob = (prob * mask).sum(1).mean().item()
        loss = sum(loss_dict.values()) / c_all["joint"].size(0) * self.o.loss_disc

        return loss



class LossCalculator(nn.Module):

    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        # self.log_softmax = func("log_softmax")
        # self.nll_loss = nn.NLLLoss(reduction='sum')
        self.pois_loss = nn.PoissonNLLLoss(full=True, reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # log_softmax + nll
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.gaussian_loss = nn.GaussianNLLLoss(full=True, reduction='sum')
        # self.enc_s = MLP([o.dim_s]+o.dims_enc_s+[o.dim_b*2], hid_norm=o.norm, hid_drop=o.drop)

        # self.i = 0


    def forward(self, inputs, x_r_pre, s_r_pre, z_mu, z_logvar, b, z_uni):
        o = self.o
        s = inputs["s"]["joint"]
        x = inputs["x"]
        e = inputs["e"]

        loss_recon = self.calc_recon_loss(x, s, e, x_r_pre, s_r_pre)
        # loss_jsd_s = self.calc_jsd_s_loss(s_r_pre)
        loss_kld_z = self.calc_kld_z_loss(z_mu, z_logvar)
        # if o.experiment in ["no_kl", "no_kl_ad"]:
        #     loss_kld_z = loss_kld_z * 0
        # loss_topo = self.calc_topology_loss(x_pp, z_x_mu, z_x_logvar, z_s_mu, z_s_logvar) * 500
        loss_topo = self.calc_consistency_loss(z_uni) * self.o.loss_mod_alignment
        # loss_mi = self.calc_mi_loss(b, s) * 0

        if o.debug == 1:
            print("recon: %.3f\tkld_z: %.3f\ttopo: %.3f" % (loss_recon.item(),
                loss_kld_z.item(), loss_topo.item()))
        return loss_recon + loss_kld_z + loss_topo


    def calc_recon_loss(self, x, s, e, x_r_pre, s_r_pre):
        losses = {}
        # Reconstruciton losses of x^m
        for m in x.keys():
            if m == "label":
                losses[m] = self.cross_entropy_loss(x_r_pre[m], x[m].squeeze(1)).sum()
            elif m == "atac":
                losses[m] = self.bce_loss(x_r_pre[m], x[m]).sum()
                # losses[m] = self.pois_loss(x_r_pre[m], x[m]).sum()
            else:
                losses[m] = (self.pois_loss(x_r_pre[m], x[m]) * e[m]).sum()
        if s_r_pre is not None:
            losses["s"] = self.cross_entropy_loss(s_r_pre, s.squeeze(1)).sum() * self.o.loss_s_recon
        # print(losses)
        return sum(losses.values()) / s.size(0)


    def calc_kld_z_loss(self, mu, logvar):
        o = self.o
        mu_c, mu_b = mu.split([o.dim_c, o.dim_b], dim=1)
        logvar_c, logvar_b = logvar.split([o.dim_c, o.dim_b], dim=1)
        kld_c_loss = self.calc_kld_loss(mu_c, logvar_c)
        kld_b_loss = self.calc_kld_loss(mu_b, logvar_b)
        beta = 5
        kld_z_loss = kld_c_loss + beta * kld_b_loss
        return kld_z_loss


    def calc_kld_loss(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / mu.size(0)


    def calc_consistency_loss(self, z_uni):
        z_uni_stack = torch.stack(list(z_uni.values()), dim=0)  # M * N * K
        z_uni_mean = z_uni_stack.mean(0, keepdim=True)  # 1 * N * K
        return ((z_uni_stack - z_uni_mean)**2).sum() / z_uni_stack.size(1)


class MLP(nn.Module):
    def __init__(self, features=[], hid_trans='mish', out_trans=False,
                 norm=False, hid_norm=False, drop=False, hid_drop=False):
        super(MLP, self).__init__()
        layer_num = len(features)
        assert layer_num > 1, "MLP should have at least 2 layers!"
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False
        
        layers = []
        for i in range(1, layer_num):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < layer_num - 1:  # hidden layers (if layer number > 2)
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:                  # output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class Layer1D(nn.Module):
    def __init__(self, dim=False, norm=False, trans=False, drop=False):
        super(Layer1D, self).__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if trans:
            layers.append(func(trans))
        if drop:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def preprocess(x, name, dim):
    if name == "label":
        x = nn.functional.one_hot(x.squeeze(1), num_classes=dim).float()
    # elif name == "atac":
    #     x = x.log1p()
    elif name == "rna":
        x = x.log1p()
    elif name == "adt":
        x = x.log1p()
    
    # if task == "snare":
    #     if name == "rna":
    #         x = x.log1p()/2 - 1
    #     elif name == "atac":
    #         x = x.log1p() - 1
    # elif task == "hbm":
    #     if name == "rna":
    #         pass
    #         # x = x.log1p()/4 - 1
    #     elif name == "adt":
    #         pass
    #         # x = x.log1p()/4 - 1
    #     elif name == "label":
    #         x = nn.functional.one_hot(x.squeeze(1), num_classes=dim).float()
    # elif task == "simu":
    #     x = x.log1p()/4 - 1
    # else:
    #     assert False, task+": invalid task!"
    return x


def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0)  # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)


def exp(x, eps=1e-12):
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x, eps=1e-12):
    return (x + eps).log()


def func(func_name):
    if func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'silu':
        return nn.SiLU()
    elif func_name == 'mish':
        return nn.Mish()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    elif func_name == 'log_softmax':
        return nn.LogSoftmax(dim=1)
    else:
        assert False, "Invalid func_name."


class CheckBP(nn.Module):
    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):
    def forward(self, input):
        return F.Identity.apply(input)


class MIDAS():
    """
    Args:
        data (list): A list of 'GetDataInfo' objects or a single 'GetDataInfo' object. For a list, the order of items determines the sequence of batch IDs assigned to them.
        status (list): A list indicating the role of each item in data. 'query' is always assigned after 'reference'.
    Example:
        for offline integration::
        
            >>> model = MIDAS([GetDataInfo1, GetDataInfo2])

        for reciprocal reference mapping::
        
            >>> model = MIDAS([GetDataInfo1, GetDataInfo2], ["reference", "query"])
    """
    def __init__(self, data:Union[list, GetDataInfo], status:Union[list, None] = None):

        if type(data) is not list:
            data = [data]
        self.data = data
        self.batch_num_rep = 0
        self.batch_num_curr = 0
        self.mods = []
        if not status:
            status = ['query' for i in range(len(self.data))] 
        for i, d in enumerate(data):
            if status[i] == 'reference':
                self.batch_num_rep += d.num_subset
            else:
                self.batch_num_curr += d.num_subset
            self.mods += list(d.mods.values())
        self.total_num = self.batch_num_rep + self.batch_num_curr
        self.s_joint = [[i] for i in range(self.total_num)]
        self.s_joint, self.combs, self.s, self.dims_s = utils.gen_all_batch_ids(self.s_joint, [self.mods])
        self.reference_features = {}
        self.dims_x = {}
        self.dims_chr = []
        self.dims_rep = {}
        self.masks = []
        for i, d in enumerate(data):
            # d.info()
            for k in d.mod_combination:
                if k == 'atac':
                    self.dims_x['atac'] = d.feat_dims['atac']
                    self.reference_features[k] = d.features['atac']
                    self.dims_chr = d.dims_chr
                    self.n_chr = d.n_chr
                    if (status[i] == 'reference') and (k not in self.dims_rep):
                        self.dims_rep[k] = d.feat_dims[k]
                
                else:
                    if k not in self.reference_features:
                        self.reference_features[k] = d.features[k]
                    else:
                        self.reference_features[k], _ = utils.merge_features(self.reference_features[k].copy(), d.features[k].copy())
                    if status[i] == 'reference':
                        if k not in self.dims_rep:
                            self.dims_rep[k] = d.feat_dims[k]
                        else:
                            self.dims_rep[k], _  = utils.merge_features(self.dims_rep[k], d.feat_dims[k])
        for i, d in enumerate(data):
            for mask in d.masks:
                mask_ = {}
                for k in mask.keys():
                    _, transform = utils.merge_features(self.reference_features[k], d.features[k].copy())
                    temp = np.zeros(len(self.reference_features[k]), dtype=np.float32)
                    temp[transform[0]] = mask[k][transform[1]]
                    mask_[k] = temp
                self.masks.append(mask_)
                    
        for k in ['atac', 'rna', 'adt']:
            if k in self.reference_features:
                self.dims_x[k] = len(self.reference_features[k])
        
        self.mods = utils.ref_sort(np.unique(np.concatenate(self.mods).flatten()).tolist(), ['atac', 'rna','adt'])
    
    def init_model(
            self, 
            # training related
            train_mod:str = 'offline', 
            lr:float = 1e-4, 
            drop_s:int = 0, 
            s_drop_rate:float = 0.1, 
            grad_clip:int = -1, 
            # structure related
            dim_c:int = 32, 
            dim_b:int = 2, 
            dims_enc_s:list = [16,16], 
            dims_enc_chr:list = [128,32], 
            dims_enc_x:list = [1024,128], 
            dims_discriminator:list = [128,64],
            norm:str = "ln", 
            drop:float = 0.2, 
            disc_train:int = 3,
            # loss related
            loss_s_recon:float = 1000.0,
            loss_mod_alignment:float = 50.0,
            loss_disc:float = 30.0, 
            # checkpoint related
            model_path:Union[str, None] = None,
            log_path:Union[str, None] = None,
            # reciprocal integration related
            reciprocal_from:Union[str, None] = None
            ):
        """Initialize the model structure.

        Args:
            train_mod (str): 'offline': Classic training method (See MIDAS). 'reciprocal': Reciprocal integration method.
            lr (float): Learning rate for training.
            drop_s (float): Whether to force dropping s (batch ID) during training.
            s_drop_rate (float): Dropout rate for s (batch ID).
            grad_clip (int): Whether to clip gradients during training.
            dim_c (list): Dimension of the variable c (biological information).
            dim_b (list): Dimension of the variable b (batch information).
            dims_enc_s (list): List of dimensions for the encoder hidden layers (MLP) for s (batch ID).
            dims_enc_chr (list): List of dimensions for the encoder hidden layers (MLP) for chromosomes (used when there is ATAC data).
            dims_enc_x (list): List of dimensions for the encoder hidden layers (MLP) for data (except ATAC).
            dims_discriminator (list): List of dimensions for the discriminator hidden layers (MLP).
            norm (str): Type of normalization. 'ln' or 'bn'.
            drop (float): Dropout rate for the hidden layers.
            disc_train (int): Number of training iterations for the discriminator.
            loss_s_recon (float): Scaling factor for s (batch ID) reconstruction loss.
            loss_mod_alignment (float): Scaling factor for modality alignment loss.
            loss_disc  (float): Scaling factor for the loss used to train the discriminator.
            model_path (str, optional): Path to save the model weights (a ".pt" file).
            log_path (str, optional): Path to save the training status (a ".toml" file).
            reciprocal_from (str, optional): Path to the model weights when using 'reciprocal' training mode (a ".pt" file). This is used only when train_mod == 'reciprocal'.
        """

        assert not (train_mod == 'reciprocal' and reciprocal_from==None), 'Missing weight path to initialize the model when trying to implement reciprocal integration'
        dims_h = {}
        self.train_mod = train_mod
        for m, dim in self.dims_x.items():
            dims_h[m] = dim if m != "atac" else dims_enc_chr[-1] * self.n_chr

        self.log = {
            "train_loss": [],
            "test_loss": [],
            "foscttm": [],
            "epoch_id_start": 0,
            }

        self.o = utils.simple_obj({
            # data related
            'mods' : self.mods,
            'dims_x' : self.dims_x,
            'ref_mods': self.mods, # no meanings here
            's_joint' : self.s_joint,
            'combs' : self.combs, 
            's' : self.s, 
            'dims_s' : self.dims_s,
            'dims_chr' : self.dims_chr,
            # model hyper-parameters
            'drop' : drop,
            'drop_s' : drop_s,
            's_drop_rate' : s_drop_rate,
            'grad_clip' : grad_clip,
            'norm' : norm,
            'lr' : lr,
            # model structure
            'dim_c' : dim_c, 
            'dim_b' : dim_b, 
            'dim_s' : self.dims_s["joint"], 
            'dim_z' : dim_c + dim_b, 
            'dims_enc_s' : dims_enc_s, 
            'dims_enc_chr' : dims_enc_chr, 
            'dims_enc_x' : dims_enc_x, 
            'dims_dec_x' : dims_enc_x[::-1],
            'dims_dec_s' : dims_enc_s[::-1],
            'dims_dec_chr' : dims_enc_chr[::-1],
            "dims_h" : dims_h,
            'dims_discriminator' : dims_discriminator,
            "disc_train" : disc_train,
            # loss related
            "loss_s_recon" : loss_s_recon,
            "loss_mod_alignment" : loss_mod_alignment,
            "loss_disc": loss_disc, 
            })

        self.net = Net(self.o).cuda()
        self.discriminator = Discriminator(self.o).cuda()
        self.optimizer_net = torch.optim.AdamW(self.net.parameters(), lr=self.o.lr)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=self.o.lr)
        
        # initialization for reciprocal learning
        # model structure adaptation
        if self.train_mod == 'reciprocal':
            print('load an old model from', reciprocal_from)
            savepoint = torch.load(reciprocal_from)
            dims_h_rep = {}
            for m, dim in self.dims_rep.items():
                dims_h_rep[m] = dim if m != "atac" else dims_enc_chr[-1] * self.n_chr
            self.net = utils.update_model(savepoint, dims_h_rep, self.o.dims_h, self.net)
            self.discriminator = Discriminator(self.o).cuda()
            self.optimizer_net = torch.optim.AdamW(self.net.parameters(), lr=self.o.lr)
            self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=self.o.lr)
        # start training from a breakpoint
        if model_path is not None:
            print('load a pretrained model from', model_path)
            savepoint = torch.load(model_path)
            self.net.load_state_dict(savepoint['net_states'])
            self.discriminator.load_state_dict(savepoint['disc_states'])
            self.optimizer_net.load_state_dict(savepoint['optim_net_states'])
            self.optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
        if log_path is not None:
            savepoint_toml = utils.load_toml(log_path)
            self.log.update(savepoint_toml['log'])

        net_param_num = sum([param.data.numel() for param in self.net.parameters()])
        disc_param_num = sum([param.data.numel() for param in self.discriminator.parameters()])
        print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))

    
    def __forward_net__(self, inputs):
        return self.net(inputs)


    def __forward_disc__(self, c, s):
        return self.discriminator(c, s)
    
    def __update_disc__(self, loss):
        self.__update__(loss, self.discriminator, self.optimizer_disc)


    def __update_net__(self,loss):
        self.__update__(loss, self.net, self.optimizer_net)


    def __update_disc__(self,loss):
        self.__update__(loss, self.discriminator, self.optimizer_disc)
        

    def __update__(self,loss, model, optimizer):
        optimizer.zero_grad()
        loss.backward()
        if self.o.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.o.grad_clip)
        optimizer.step()
        
    def __run_iter__(self, split, epoch_id, inputs, rnt=1):
        inputs = utils.convert_tensors_to_cuda(inputs)
        if split == "train":
            with autograd.set_detect_anomaly(self.debug == 1):
                loss_net, c_all = self.__forward_net__(inputs)
                self.discriminator.epoch = epoch_id
                for _ in range(self.o.disc_train):
                    loss_disc = self.__forward_disc__(utils.detach_tensors(c_all), inputs["s"])
                    loss_disc = loss_disc * rnt
                    self.__update_disc__(loss_disc)
                loss_adv = self.__forward_disc__(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
                loss = rnt * loss
                self.__update_net__(loss)
            
        else:
            with torch.no_grad():
                loss_net, c_all = self.__forward_net__(inputs)
                loss_adv = self.__forward_disc__(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
        return loss.item()

    def __run_epoch__(self, data_loader, split, epoch_id=0):
        start_time = time.time()
        if split == "train":
            self.net.train()
            self.discriminator.train()
        elif split == "test":
            self.net.eval()
            self.discriminator.eval()
        else:
            assert False, "Invalid split: %s" % split
        loss_total = 0
        if self.train_mod == 'reciprocal':
            assert type(data_loader) is list, "Wrong type of dataloader for reciprocal learning."
            query_dataloader = data_loader[1]
            reference_dataloader = data_loader[0]
            for _, data in enumerate(zip(query_dataloader, reference_dataloader)):
                data1, data2 = data[0], data[1]
                rnt_ = self.batch_num_curr / (self.batch_num_rep + self.batch_num_curr)
                loss = self.__run_iter__(split, epoch_id, data1, rnt_)
                loss_total += loss
                rnt_ = self.batch_num_rep / (self.batch_num_rep  + self.batch_num_curr)
                loss = self.__run_iter__(split, epoch_id, data2, rnt_)
                loss_total += loss
            loss_avg = loss_total / len(query_dataloader) / 2
        elif self.train_mod == 'offline':
            rnt_ = 1
            for _, data in enumerate(data_loader):
                loss = self.__run_iter__(split, epoch_id, data, rnt_)
                loss_total += loss
            loss_avg = loss_total / len(data_loader)

        epoch_time = (time.time() - start_time) / 3600 / 24
        self.log[split+'_loss'].append((float(epoch_id), float(loss_avg)))
        return loss_avg, epoch_time
    
    def train(
            self, 
            n_epoch:int = 2000, 
            mini_batch_size:int = 256, 
            shuffle:bool = True, 
            save_epochs:int = 50, 
            debug:int = 0, 
            save_path:str = './result/experiment/'
            ):
        """Train the model.
        
        Args:
            n_epoch (int): Number of training epochs.
            mini_batch_size (int): Size of mini-batches for training.
            shuffle (bool): Whether to shuffle the training data during each epoch.
            save_epochs (int): Frequency to save the latest weights and logs during training.
            debug (int): If True, print intermediate variables for debugging purposes.
            save_path (str): Path to save the trained model and related files.

        """
        print("Training ...")
        self.save_epochs = save_epochs
        self.debug = debug
        self.o.debug = debug
        if self.train_mod == 'reciprocal':
            datasets = self.gen_datasets(self.data)
            reference_datasets = list(datasets[:self.batch_num_rep])
            query_datasets = list(datasets[self.batch_num_rep:])
            reference_datasets = torch.utils.data.dataset.ConcatDataset(reference_datasets)
            query_datasets = torch.utils.data.dataset.ConcatDataset(query_datasets)
            reference_sampler = MultiDatasetSampler(reference_datasets, batch_size=mini_batch_size, shuffle=shuffle)
            query_sampler = MultiDatasetSampler(query_datasets, batch_size=mini_batch_size, shuffle=shuffle)
            self.data_loader = [
                torch.utils.data.DataLoader(reference_datasets, batch_size=mini_batch_size, sampler=reference_sampler, num_workers=64, pin_memory=True),
                torch.utils.data.DataLoader(query_datasets, batch_size=mini_batch_size, sampler=query_sampler, num_workers=64, pin_memory=True)
            ]
        elif self.train_mod == 'offline':
            datasets = self.gen_datasets(self.data)
            sampler = MultiDatasetSampler(torch.utils.data.dataset.ConcatDataset(datasets), batch_size=mini_batch_size, shuffle=shuffle)
            self.data_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.ConcatDataset(datasets), batch_size=mini_batch_size, sampler=sampler, num_workers=64, pin_memory=True)
        with tqdm(total=n_epoch) as pbar:
            pbar.update(self.log['epoch_id_start'])
            for epoch_id in range(self.log['epoch_id_start'], n_epoch):
                loss_avg, epoch_time = self.__run_epoch__(self.data_loader, "train", epoch_id)
                self.__check_to_save__(epoch_id, n_epoch, save_path)
                pbar.update(1)
                pbar.set_description("Loss: %.4f" % loss_avg)
        
    def predict(self, 
                save_dir:str = './result/experiment/predict/', 
                joint_latent:bool = True, 
                mod_latent:bool = False, 
                impute:bool = False, 
                batch_correct:bool = False, 
                translate:bool = False, 
                input:bool = False, 
                mini_batch_size:int = 256,
                remove_old=True
                ):
        """Predict the embeddings or their imputed expression.

        Args:
            save_dir (str): The path to save the predicted files.
            joint_latent (bool): Whether to generate the joint embeddings.
            impute (bool): Whether to generate the imputed counts data.
            batch_correct (bool): Whether to generate the batch-corrected counts data.
            translate (bool): Whether to generate the translated counts.
            input (bool): Whether to generate the input data.
            mini_batch_size (int): The mini-batch size for saving. Influence the cell number in the csv file.
        """
        if translate:
            mod_latent = True
        print("Predicting ...")
        self.o.pred_dir = save_dir
        if not os.path.exists(self.o.pred_dir):
            os.makedirs(self.o.pred_dir)
        dirs = utils.get_pred_dirs(self.o, joint_latent, mod_latent, impute, batch_correct, translate, input)
        parent_dirs = list(set(map(os.path.dirname, utils.extract_values(dirs))))
        utils.mkdirs(parent_dirs, remove_old=remove_old)
        utils.mkdirs(dirs, remove_old=remove_old)
        datasets = self.gen_datasets(self.data)
        data_loaders = {k:torch.utils.data.DataLoader(datasets[k], batch_size=mini_batch_size, \
            num_workers=64, pin_memory=True, shuffle=False) for k in range(self.batch_num_curr+self.batch_num_rep)}
        # data_loaders = get_dataloaders("test", train_ratio=0)
        self.net.eval()
        with torch.no_grad():
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(self.o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    
                    # conditioned on all observed modalities
                    if joint_latent:
                        x_r_pre, _, _, _, z, _, _, *_ = self.net.sct(data)  # N * K
                        utils.save_tensor_to_csv(z, os.path.join(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                    if impute:
                        x_r = gen_real_data(x_r_pre, sampling=False)
                        for m in self.o.mods:
                            utils.save_tensor_to_csv(x_r[m], os.path.join(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                    if input:  # save the input
                        for m in self.o.combs[subset_id]:
                            utils.save_tensor_to_csv(data["x"][m].int(), os.path.join(dirs[subset_id]["x"][m], fname_fmt) % i)

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
                            x_r_pre, _, _, _, z, c, b, *_ = self.net.sct(input_data)  # N * K
                            utils.save_tensor_to_csv(z, os.path.join(dirs[subset_id]["z"][m], fname_fmt) % i)
                            if translate: # single to double
                                x_r = gen_real_data(x_r_pre, sampling=False)
                                for m_ in set(self.o.mods) - {m}:
                                    utils.save_tensor_to_csv(x_r[m_], os.path.join(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)
                    
                    if translate: # double to single
                        for mods in itertools.combinations(data["x"].keys(), 2):
                            m1, m2 = utils.ref_sort(mods, ref=self.o.mods)
                            input_data = {
                                "x": {m1: data["x"][m1], m2: data["x"][m2]},
                                "s": data["s"], 
                                "e": {}
                            }
                            for m in mods:
                                if m in data["e"].keys():
                                    input_data["e"][m] = data["e"][m]
                            x_r_pre, *_ = self.net.sct(input_data)  # N * K
                            x_r = gen_real_data(x_r_pre, sampling=False)
                            m_ = list(set(self.o.mods) - set(mods))[0]
                            utils.save_tensor_to_csv(x_r[m_], os.path.join(dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_], fname_fmt) % i)

            if batch_correct:
                print("Calculating b_centroid ...")
                pred = utils.load_predicted(self.o)
                b = torch.from_numpy(pred["z"]["joint"][:, self.o.dim_c:])
                s = torch.from_numpy(pred["s"]["joint"])

                b_mean = b.mean(dim=0, keepdim=True)
                b_subset_mean_list = []
                for subset_id in s.unique():
                    b_subset = b[s == subset_id, :]
                    b_subset_mean_list.append(b_subset.mean(dim=0))
                b_subset_mean_stack = torch.stack(b_subset_mean_list, dim=0)
                dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
                self.net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
                self.net.sct.batch_correction = True
                
                print("Batch correction ...")
                for subset_id, data_loader in data_loaders.items():
                    print("Processing subset %d: %s" % (subset_id, str(self.o.combs[subset_id])))
                    fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                    
                    for i, data in enumerate(tqdm(data_loader)):
                        data = utils.convert_tensors_to_cuda(data)
                        x_r_pre, *_ = self.net.sct(data)
                        x_r = gen_real_data(x_r_pre, sampling=True)
                        for m in self.o.mods:
                            utils.save_tensor_to_csv(x_r[m], os.path.join(dirs[subset_id]["x_bc"][m], fname_fmt) % i)
    
    def read_preds(
            self, 
            pred_path:str = None, 
            joint_latent:bool = True, 
            mod_latent:bool = False, 
            impute:bool = False, 
            batch_correct:bool = False, 
            translate:bool = False, 
            input:bool = False, 
            group_by:str = "modality"):
        """Get embeddings or other outputs from a specified path.

        Args:
            pred_path (str): The path from which to retrieve the embeddings. If not provided, it uses the path from the previous `predict()` function call, if available.
            joint_latent (bool): Whether to retrieve the joint embeddings.
            impute (bool): Whether to retrieve the imputed counts data.
            batch_correct (bool): Whether to retrieve the batch-corrected counts data.
            translate (bool): Whether to retrieve the translated counts.
            input (bool): Whether to retrieve the input data.
            group_by (str): Specify how to group the data: "modality" or "batch".

        Returns:
            Embeddings or other outputs obtained from the specified path.
        """

        if pred_path is not None:
            self.o.pred_dir = pred_path
        pred = utils.load_predicted(self.o, joint_latent=joint_latent, mod_latent=mod_latent, impute=impute, batch_correct=batch_correct, 
                   translate=translate, input=input, group_by=group_by)
        return pred
    
    def gen_datasets(self, data:list):
        """ Generate dataset object.

        Args:
            data (list): A list of GetDataInfo containing information about the dataset.

        Returns:
            A list containing torch.utils.data.Dataset objects.
        """
        datasets = []
        n = 0
        for d in data:
            for i in range(d.num_subset):
                datasets.append(MultimodalDataset(d, subset=i, s_subset=self.s[n], reference_features=self.reference_features))
                n += 1
        return datasets

    def viz_loss(self):
        """ 
        Visualize the loss.
        """
        plt.figure(figsize=(4,2))
        plt.plot(np.array(self.log['train_loss'])[:, 0]+1, np.array(self.log['train_loss'])[:, 1])
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.title('Loss curve')

    def reduce_data(self, 
             output_task_name:str = 'pack', 
             des_dir:str = './data/processed/', 
             n_sample:int = 100000, 
             pred_dir:Union[str, None] = None):
        """ Reduces data by proportionally sampling from each batch and eventually merging them into a unified dataset with consistent features for storage.
    
        Args:
            output_task_name (str): The name of the output. It will be concatenated with 'des_dir' to form the output path.
            des_dir (str): The directory path to save the data.
            n_sample (int): The desired number of samples.
            pred_dir (str): The directory path where the embeddings are stored. The embeddings are used for sampling. 
        """

        if pred_dir is None:
            pred_dir = self.o.pred_dir
        else:
            self.o.pred_dir = pred_dir
        print("reducing data ...")

        if not os.path.exists(os.path.join(des_dir, output_task_name)):
            os.makedirs(os.path.join(des_dir, output_task_name, 'feat'))

        # load info
        datasets = self.gen_datasets(self.data)
        data_loaders = {k:torch.utils.data.DataLoader(datasets[k], batch_size=1, \
            num_workers=64, pin_memory=True, shuffle=False) for k in range(self.batch_num_curr+self.batch_num_rep)}
        
        emb = self.read_embeddings()
        # print(emb)
        cell_names = {}
        cell_names_sampled = {}
        cell_nums = []
        n = 0
        for d in self.data:
            for k in range(d.num_subset):
                cell_names_sampled[n] = d.cell_names['subset_%d'%k]
                cell_names[n] = d.cell_names_orig['subset_%d'%k]
                cell_nums.append(len(cell_names[n]))
                n += 1
        if sum(cell_nums) > n_sample:
            rate = (np.array(cell_nums) / sum(cell_nums)  * n_sample).astype(int)
        else:
            rate = [len(i) for i in datasets]
        sample_preserve = {}
        if not os.path.exists(os.path.join(des_dir, output_task_name, 'feat')):
            os.makedirs(os.path.join(des_dir, output_task_name, 'feat'))
        for i in range(self.batch_num_rep+self.batch_num_curr):
            emb_subset = emb['z']['joint'][emb['s']['joint']==i]
            sample_preserve['subset_%d'%i] = BallTreeSubsample(emb_subset, rate[i])
            sample_preserve['subset_%d'%i].sort()
            # print(sample_preserve['subset_%d'%i])

            if not os.path.exists(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'mask')):
                os.makedirs(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'mask'))
            if i in cell_names_sampled:
                # print(i)
                pd.DataFrame(cell_names_sampled[i][sample_preserve['subset_%d'%i]]).to_csv(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'cell_names_sampled.csv'))
                pd.DataFrame(cell_names[i]).to_csv(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'cell_names.csv'))
            else:
                pd.DataFrame(cell_names[i][sample_preserve['subset_%d'%i]]).to_csv(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'cell_names_sampled.csv'))
                pd.DataFrame(cell_names[i]).to_csv(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'cell_names.csv'))
            fname_fmt = utils.get_name_fmt(rate[i])+".csv"
            n = 0
            for k, data in enumerate(data_loaders[i]):
                if k in sample_preserve['subset_%d'%i]:
                    for m in self.o.combs[i]:
                        if not os.path.exists(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'vec', m)):
                            os.makedirs(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'vec', m))
                        utils.save_tensor_to_csv(data["x"][m].int(), os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'vec', m, fname_fmt) % n)
                    n += 1
            for k, data in enumerate(data_loaders[i]):
                for m in self.o.combs[i]:
                    if m != 'atac':
                        pd.DataFrame(utils.convert_tensor_to_list(data["e"][m].int())).to_csv(os.path.join(des_dir, output_task_name, 'subset_%d'%i, 'mask', '%s.csv'%m))
                break

        features_dims = {}
        if self.dims_chr != []:
            features_dims['atac'] = self.dims_chr
        for m in self.reference_features:
            if m != 'atac':
                features_dims[m] = [self.dims_x[m] for i in range(self.n_chr)]
            pd.DataFrame(self.reference_features[m]).to_csv(os.path.join(des_dir, output_task_name, 'feat','feat_names_%s.csv'%m))
        pd.DataFrame(features_dims).to_csv(os.path.join(des_dir, output_task_name, 'feat','feat_dims.csv'))


    def __check_to_save__(self, epoch_id, epoch_num, save_path):
        if (epoch_id+1) % self.save_epochs == 0 or epoch_id+1 == epoch_num:
            self.__save_training_states__(epoch_id, "sp_%08d"%(epoch_id+1), save_path)
            self.__save_training_states__(epoch_id, "sp_latest", save_path)
    
    def __save_training_states__(self, epoch_id, filename,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log['epoch_id_start'] = epoch_id + 1
        utils.save_toml({"o": vars(self.o), "log": self.log}, os.path.join(save_path, filename+".toml"))
        torch.save({"net_states": self.net.state_dict(),
                "disc_states": self.discriminator.state_dict(),
                "optim_net_states": self.optimizer_net.state_dict(),
                "optim_disc_states": self.optimizer_disc.state_dict()
                }, os.path.join(save_path, filename+".pt"))
        
    