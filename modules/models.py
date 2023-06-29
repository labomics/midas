from os import path
from os.path import join as pj
import math
import numpy as np
import torch as th
import torch.nn as nn

import functions.models as F
import modules.utils as utils


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
        if len(o.dims_enc_x) > 0:
            x_shared_enc = MLP(o.dims_enc_x+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
            for m in o.ref_mods:
                x_indiv_enc = MLP([o.dims_h[m], o.dims_enc_x[0]], out_trans='mish', norm=o.norm,
                                drop=o.drop)
                x_encs[m] = nn.Sequential(x_indiv_enc, x_shared_enc)
        else:
            for m in o.ref_mods:
                x_encs[m] = MLP([o.dims_h[m], o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        self.x_encs = nn.ModuleDict(x_encs)
        # Modality decoder p(x^m|c, b)
        self.x_dec = MLP([o.dim_z]+o.dims_dec_x+[sum(o.dims_h.values())], hid_norm=o.norm,
                         hid_drop=o.drop)

        # Batch encoder q(z|s)
        self.s_enc = MLP([o.dim_s]+o.dims_enc_s+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        # Batch decoder p(s|b)
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
            if th.rand([]).item() < 1 - s_drop_rate:
                s = inputs["s"]

        # Encode x_m
        z_x_mu, z_x_logvar = {}, {}
        x_pp = {}
        for m in x.keys():
            x_pp[m] = preprocess(x[m], m, o.dims_x[m], o.task)
            
            if m in ["rna", "adt"]:  # use mask
                h = x_pp[m] * e[m]
            elif m == "atac":        # encode each chromosome
                x_chrs = x_pp[m].split(o.dims_chr, dim=1)
                h_chrs = [self.chr_encs[i](x_chr) for i, x_chr in enumerate(x_chrs)]
                h = self.chr_enc_cat_layer(th.cat(h_chrs, dim=1))
            else:
                h = x_pp[m]
            # encoding
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
            x_r_pre["atac"] = th.cat(x_chrs, dim=1).sigmoid()
        
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
    - mus: [mu_1, ..., mu_M], where mu_m is N * K
    - logvars: [logvar_1, ..., logvar_M], where logvar_m is N * K
    """
    
    mus = [th.full_like(mus[0], 0)] + mus
    logvars = [th.full_like(logvars[0], 0)] + logvars
    
    mus_stack = th.stack(mus, dim=1)  # N * M * K
    logvars_stack = th.stack(logvars, dim=1)
    
    T = exp(-logvars_stack)  # precision of i-th Gaussian expert at point x
    T_sum = T.sum(1)  # N * K
    pd_mu = (mus_stack * T).sum(1) / T_sum
    pd_var = 1 / T_sum
    pd_logvar = log(pd_var)
    return pd_mu, pd_logvar  # N * K


def gen_real_data(x_r_pre, sampling=True):
    """
    Generate real data using x_r_pre
    - sampling: whether to generate discrete samples
    """
    x_r = {}
    for m, v in x_r_pre.items():
        if m in ["rna", "adt"]:
            x_r[m] = v.exp()
            if sampling:
                x_r[m] = th.poisson(x_r[m]).int()
        else:  # for atac
            x_r[m] = v
            if sampling:
                x_r[m] = th.bernoulli(x_r[m]).int()
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
        loss = sum(loss_dict.values()) / c_all["joint"].size(0)

        if o.experiment == "bio_ib_0":
            loss = loss * 0
        elif o.experiment == "bio_ib_10":
            loss = loss * 10
        elif o.experiment == "bio_ib_15":
            loss = loss * 15
        elif o.experiment == "bio_ib_20":
            loss = loss * 20
        elif o.experiment == "bio_ib_25":
            loss = loss * 25
        elif o.experiment == "bio_ib_40":
            loss = loss * 40
        elif o.experiment == "bio_ib_50":
            loss = loss * 50
        elif o.experiment == "bio_ib_100":
            loss = loss * 100
        else:
            loss = loss * 30


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

        self.tech_ib_coef = 4
        if o.experiment == "tech_ib_0":
            self.tech_ib_coef = 0
        elif o.experiment == "tech_ib_1":
            self.tech_ib_coef = 1
        elif o.experiment == "tech_ib_2":
            self.tech_ib_coef = 2
        elif o.experiment == "tech_ib_8":
            self.tech_ib_coef = 8
        elif o.experiment == "tech_ib_16":
            self.tech_ib_coef = 16

        # self.i = 0


    def forward(self, inputs, x_r_pre, s_r_pre, z_mu, z_logvar, b, z_uni):
        o = self.o
        s = inputs["s"]["joint"]
        x = inputs["x"]
        e = inputs["e"]

        loss_recon = self.calc_recon_loss(x, s, e, x_r_pre, s_r_pre)
        loss_kld_z = self.calc_kld_z_loss(z_mu, z_logvar)

        loss_mod = self.calc_mod_align_loss(z_uni)
        if o.experiment == "mod_0":
            loss_mod = loss_mod * 0
        elif o.experiment == "mod_10":
            loss_mod = loss_mod * 10
        elif o.experiment == "mod_20":
            loss_mod = loss_mod * 20
        elif o.experiment == "mod_100":
            loss_mod = loss_mod * 100
        elif o.experiment == "mod_200":
            loss_mod = loss_mod * 200
        else:
            loss_mod = loss_mod * 50

        if o.debug == 1:
            print("recon: %.3f\tkld_z: %.3f\ttopo: %.3f" % (loss_recon.item(),
                loss_kld_z.item(), loss_mod.item()))
        return loss_recon + loss_kld_z + loss_mod


    def calc_recon_loss(self, x, s, e, x_r_pre, s_r_pre):
        o = self.o
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
 
            s_coef = 1000
            if o.experiment == "s_coef_1":
                s_coef = 1
            elif o.experiment == "s_coef_200":
                s_coef = 200
            elif o.experiment == "s_coef_500":
                s_coef = 500
            elif o.experiment == "s_coef_2000":
                s_coef = 2000
            elif o.experiment == "s_coef_5000":
                s_coef = 5000

            losses["s"] = self.cross_entropy_loss(s_r_pre, s.squeeze(1)).sum() * (self.tech_ib_coef + s_coef)
        # print(losses)
        return sum(losses.values()) / s.size(0)


    def calc_kld_z_loss(self, mu, logvar):
        o = self.o
        mu_c, mu_b = mu.split([o.dim_c, o.dim_b], dim=1)
        logvar_c, logvar_b = logvar.split([o.dim_c, o.dim_b], dim=1)
        kld_c_loss = self.calc_kld_loss(mu_c, logvar_c)
        kld_b_loss = self.calc_kld_loss(mu_b, logvar_b)
        kld_z_loss = kld_c_loss + (1 + self.tech_ib_coef) * kld_b_loss
        return kld_z_loss


    def calc_kld_loss(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / mu.size(0)


    def calc_mod_align_loss(self, z_uni):
        z_uni_stack = th.stack(list(z_uni.values()), dim=0)  # M * N * K
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


def preprocess(x, name, dim, task):
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
