import os
import datetime

from typing import Dict, List, Optional, Tuple
from anndata import AnnData
from mudata import MuData

import natsort

import toml
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as L
from pytorch_lightning.utilities import rank_zero_only

import logging
logging.basicConfig(level=logging.INFO)
import threading

# Project-Specific Imports
from .data import MyDistributedSampler, MultiBatchSampler, MultiModalDataset
from .utils import *
from .nn import MLP, Layer1D, distribution_registry, transform_registry

class Encoder(nn.Module):
    """
    Encoder class for multi-modal data with modality-specific pre-processing, encoding, 
    and shared encoding layers.

    Parameters:
        dims_x : Dict[str, list]
            Input dimensions for each modality (e.g, {'rna':[1000], 'adt':[100]}).
        dims_h : Dict[str, list]
            Hidden dimensions for each modality after pre-encoding (e.g, {'rna':256, 'adt':256}).
        dim_z : int
            Latent dimension size (e.g, 32).
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        out_trans : str
            Output activation function (e.g., 'mish').
        drop : float
            Dropout rate.
        kwargs : Dict[str, Any]
            Additional modality-specific configurations.

    Notes:
        By default, RNA and ADT data are log1p-transformed in the encoder and will be exponentiated after decoding. 
        To skip this step, modify the configuration file. See parameter 'trsf_before_enc_'. 
    """
    def __init__(
        self,
        dims_x: Dict[str, list],
        dims_h: Dict[str, list],
        dim_z: int,
        norm: str,
        out_trans: str,
        drop: float,
        **kwargs,
    ):
        super(Encoder, self).__init__()
        self.dims_x = dims_x
        self.dims_h = dims_h
        self.dim_z = dim_z
        self.norm = norm
        self.out_trans = out_trans
        self.drop = drop

        # Dynamically set additional arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Extract transformations to apply before encoding
        self.trsf_before_enc = filter_keys(kwargs, 'trsf_before_enc')

        # Shared encoder across all modalities
        shared_encoder = MLP(
            self.dims_shared_enc + [self.dim_z * 2],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

        # Initialize modality-specific encoders
        # mod1 -> (opt) transform[mod1] -> (opt) pre_encoder[mod1] ->
        #      (opt) transform_concat[mod1] -> indiv_enc[mod1] -> share_encoder -> z_mod1
        self.pre_encoders = nn.ModuleDict()  # Modality-specific pre-encoding layers
        self.transform_concat = nn.ModuleDict()  # Post-concatenation layers
        encoders = {}  # Final encoders for each modality

        for modality, input_dims in dims_x.items():
            # For truncated input, such as ATAC
            if len(input_dims) > 1:
                self.pre_encoders[modality] = nn.ModuleList([
                    MLP([dim] + kwargs[f'dims_before_enc_{modality}'], hid_norm=self.norm, hid_drop=self.drop)
                    for dim in input_dims
                ])
                self.transform_concat[modality] = Layer1D(self.dims_h[modality], 
                                                      self.norm, 
                                                      self.out_trans, 
                                                      self.drop)

            # Create individual encoder for the modality
            indiv_enc = MLP(
                [self.dims_h[modality][0], self.dims_shared_enc[0]],
                out_trans=self.out_trans,
                norm=self.norm,
                drop=self.drop,
            )
            encoders[modality] = nn.Sequential(indiv_enc, shared_encoder)

        self.encoders = nn.ModuleDict(encoders)

    def forward(
            self, 
            data: Dict[str, torch.Tensor], 
            mask: Dict[str, torch.Tensor]
            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for the encoder.

        Parameters:
            data : Dict[str, torch.Tensor]
                Input data for each modality.
            mask : Dict[str, torch.Tensor]
                Masks for each modality.

        Returns:
            Tuple:
                - z_x_mu : Dict[str, torch.Tensor]
                    Mean values for latent space for each modality.
                - z_x_logvar : Dict[str, torch.Tensor]
                    Log-variance values for latent space for each modality.
        """
        data = data.copy()
        mask = mask.copy()
        # Apply transformations before encoding
        for modality in data.keys():
            if f'trsf_before_enc_{modality}' in self.trsf_before_enc:
                transformation = self.trsf_before_enc[f'trsf_before_enc_{modality}']
                data[modality] = transform_registry.get(transformation)(data[modality])

        # Apply masks to data
        for modality, mask_value in mask.items():
            data[modality] *= mask_value

        # Pre-encode and concatenate if necessary, for truncated inputs
        for modality in data.keys():
            if modality in self.pre_encoders:
                # Split and process individual dimensions
                batches = data[modality].split(self.dims_x[modality], dim=1)
                processed_batches = [
                    self.pre_encoders[modality][i](batch) for i, batch in enumerate(batches)
                ]
                # Concatenate processed batches and transform
                data[modality] = self.transform_concat[modality](torch.cat(processed_batches, dim=1))

        # Encode data and split into mean and log-variance
        z_x_mu, z_x_logvar = {}, {}
        for modality, modality_data in data.items():
            encoded = self.encoders[modality](modality_data)
            z_x_mu[modality], z_x_logvar[modality] = encoded.split(self.dim_z, dim=1)

        return z_x_mu, z_x_logvar


class Decoder(nn.Module):
    """
    Decoder class for multi-modal data with shared and modality-specific decoding layers.

    Parameters:
        dims_x : Dict[str, list]
            Output dimensions for each modality.
        dims_h : Dict[str, list]
            Hidden dimensions for each modality.
        dim_z : int
            Latent dimension size.
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        out_trans : str
            Output activation function (e.g., 'relu').
        drop : float
            Dropout rate.
        kwargs : Dict[str, Any]
            Additional modality-specific configurations.
    """

    def __init__(
        self,
        dims_x: Dict[str, list],
        dims_h: Dict[str, list],
        dim_z: int,
        norm: str,
        out_trans: str,
        drop: float,
        **kwargs,
    ):
        super(Decoder, self).__init__()
        self.dims_x = dims_x
        self.dims_h = dims_h
        self.dim_z = dim_z
        self.norm = norm
        self.out_trans = out_trans
        self.drop = drop

        # Dynamically set additional arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # z -> shared_decoder -> (opt) post_decoders[mod1] -> (opt) transform_concat[mod1] -> mod1

        # Shared decoder layer
        total_hidden_dims = sum(dim[0] for dim in dims_h.values())
        self.shared_decoder = MLP(
            [self.dim_z] + self.dims_shared_dec + [total_hidden_dims],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

        # Modality-specific decoders
        self.post_decoders = nn.ModuleDict()
        self.transform_concat = nn.ModuleDict()

        for modality, output_dims in dims_x.items():
            # Modality-specific post-decoding layers
            if len(output_dims) > 1:
                self.post_decoders[modality] = nn.ModuleList([
                    MLP(kwargs[f'dims_after_dec_{modality}'] + [dim], 
                        hid_norm=self.norm, hid_drop=self.drop)
                    for dim in output_dims
                ])

                # Layer to process concatenated outputs
                self.transform_concat[modality] = Layer1D(self.dims_h[modality], 
                                                    self.norm, self.out_trans, 
                                                    self.drop)

    def forward(self, latent_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the decoder.

        Parameters:
            latent_data : torch.Tensor
                Latent variable input tensor of shape (batch_size, dim_z).

        Returns:
            Dict[str, torch.Tensor] : 
                Decoded outputs for each modality.
        """
        # Pass through the shared decoder
        shared_output = self.shared_decoder(latent_data)

        # Split shared decoder output into modality-specific chunks
        modality_outputs = shared_output.split(
            [dim[0] for dim in self.dims_h.values()],
            dim=1,
        )

        # Create a dictionary to hold the modality-specific outputs
        data_dict = {modality: output 
                     for modality, output in zip(self.dims_x.keys(), modality_outputs)}

        # Process each modality-specific output
        for modality, post_decoders in self.post_decoders.items():
            # Apply transformation layer
            processed_output = self.transform_concat[modality](data_dict[modality])
            batches = processed_output.split(self.__dict__[f'dims_after_dec_{modality}'][0], dim=1)

            # Apply modality-specific post-decoders
            data_dict[modality] = torch.cat(
                [post_decoders[i](batch) for i, batch in enumerate(batches)],
                dim=1,
            )

        # Apply activation functions based on distribution
        for modality, output in data_dict.items():
            distribution = self.__dict__[f'distribution_dec_{modality}']
            activation_fn = distribution_registry.get_activate(distribution)
            data_dict[modality] = activation_fn(output)

        return data_dict


class S_Encoder(nn.Module):
    """
    Encoder for batch ID latent variables.

    Parameters:
        n_batches : int
            Number of distinct batches.
        dims_enc_s : List[int]
            List of dimensions for hidden layers in the encoder.
        dim_z : int
            Latent dimension size for the latent.
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        drop : float
            Dropout rate.
    """

    def __init__(
            self, 
            n_batches: int, 
            dims_enc_s: List[int], 
            dim_z: int, 
            norm: str, 
            drop: float
            ):
        super(S_Encoder, self).__init__()
        self.n_batches = n_batches
        self.dims_enc_s = dims_enc_s
        self.dim_z = dim_z
        self.norm = norm
        self.drop = drop

        # Define the encoder MLP
        self.s_encoder = MLP(
            [self.n_batches] + self.dims_enc_s + [self.dim_z * 2],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for S_Encoder.

        Parameters:
            data : torch.Tensor
                Input tensor of shape (batch_size, 1), containing batch indices.

        Returns:
            torch.Tensor :
                Encoded tensor of shape (batch_size, dim_z * 2).
        """
        # One-hot encode the batch indices
        one_hot_data = nn.functional.one_hot(data.squeeze(1), num_classes=self.n_batches).float()

        # Pass through the encoder
        return self.s_encoder(one_hot_data)


class S_Decoder(nn.Module):
    """
    Decoder for reconstructing batch ID.

    Parameters:
        n_batches : int
            Number of distinct batches.
        dims_dec_s : List[int]
            List of dimensions for hidden layers in the decoder.
        dim_u : int
            Latent dimension size for the input (e.g, 2).
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        drop : float
            Dropout rate.
    """

    def __init__(
            self, 
            n_batches: int, 
            dims_dec_s: List[int], 
            dim_u: int, 
            norm: str, 
            drop: float):
        super(S_Decoder, self).__init__()
        self.n_batches = n_batches
        self.dims_dec_s = dims_dec_s
        self.dim_u = dim_u
        self.norm = norm
        self.drop = drop

        # Define the decoder MLP
        self.s_decoder = MLP(
            [self.dim_u] + self.dims_dec_s + [self.n_batches],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for S_Decoder.

        Parameters:
            data : torch.Tensor
                Latent input tensor of shape (batch_size, dim_u).

        Returns:
            torch.Tensor :
                Reconstructed tensor of shape (batch_size, n_batches).
        """
        return self.s_decoder(data)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for multi-modal data, supporting batch correction and 
    sampling from distributions.

    Parameters:
        dims_x : Dict[str, list]
            Input dimensions for each modality, e.g {'rna'=[1000], 'adt'=[100], 'atac'=[10,10,10]}.
        dims_s : Dict[str, int]
            Dimensions of the classes for each modality.
        kwargs : Dict[str, Any]
            Additional configurations for encoders, decoders, and other modules.
    """
    def __init__(self, dims_x: Dict[str, list], dims_s: Dict[str, int], **kwargs):
        super(VAE, self).__init__()
        self.dims_x = dims_x
        self.dims_s = dims_s
        self.mods = set(dims_x.keys())
        logging.debug(f'Initializing VAE with modalities: {self.mods}')
        logging.debug(f'Initializing VAE with dims_s: {self.dims_s}')
        logging.debug(f'Initializing VAE with dims_x: {self.dims_x}')


        # Dynamically set additional arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.available_mods = set(self.dims_x.keys())
        self.dim_z = self.dim_c + self.dim_u
        self.dims_h = self.get_dim_h()
        self.n_batches = dims_s['joint']

        # Initialize modules
        self.encoder = Encoder(self.dims_x, self.dims_h, self.dim_z, self.norm, self.out_trans, self.drop,
                               **filter_keys(self.__dict__, '_enc'))
        self.decoder = Decoder(self.dims_x, self.dims_h, self.dim_z, self.norm, self.out_trans, self.drop,
                               **filter_keys(self.__dict__, '_dec'))
        self.s_encoder = S_Encoder(self.n_batches, self.dims_enc_s, self.dim_z, self.norm, self.drop)
        self.s_decoder = S_Decoder(self.n_batches, self.dims_dec_s, self.dim_u, self.norm, self.drop)

        # Batch correction and sampling configurations
        self.batch_correction = False
        self.u_centroid = None
        self.drop_s = False
        self.sampling = False
        self.sample_num = 0

    def forward(self, data: Dict[str, torch.Tensor]
                ) -> Tuple[Dict[str, torch.Tensor], 
                           Optional[torch.Tensor], 
                           torch.Tensor, 
                           torch.Tensor, 
                           torch.Tensor, 
                           torch.Tensor, 
                           torch.Tensor, 
                           Dict[str, torch.Tensor], 
                           Dict[str, torch.Tensor]]:
        """
        Forward pass for the VAE.

        Parameters:
            data : Dict[str, torch.Tensor]
                Input data dictionary containing:
                - 'x': Dict[str, torch.Tensor], modality-specific input data.
                - 'e': Dict[str, torch.Tensor], modality-specific masks.
                - 's' (optional): torch.Tensor, dimensions of the output classes for each modality.
        Returns:
            Tuple:
                - x_r_pre : Dict[str, torch.Tensor]
                    Reconstructed modality-specific data.
                - s_r_pre : Optional[torch.Tensor]
                    If 's' is provided, return reconstructed batch indices.
                    If 's' is not provided, return None.
                - z_mu : torch.Tensor
                    Mean of the combined latent variables.
                - z_logvar : torch.Tensor
                    Log-variance of the combined latent variables.
                - z : torch.Tensor
                    Sampled latent variables.
                - c : torch.Tensor
                    Biological information variables.
                - u : torch.Tensor
                    Technical noise variables.
                - z_uni : Dict[str, torch.Tensor]
                    Unified latent variables for each modality.
                - c_all : Dict[str, torch.Tensor]
                    Modality-specific Biological information variables.
        """
        x = data['x']
        e = data['e']
        s = None

        # Handle batch-specific information. See https://github.com/labomics/midas/issues/12.
        if not self.drop_s and 's' in data:
            s_drop_rate = self.s_drop_rate if self.training else 0
            if torch.rand([]).item() < 1 - s_drop_rate:
                s = data['s']

        # Encode data
        # check device:
        logging.debug(f"x device: {next(iter(x.values())).device}")
        logging.debug(f"model device: {next(self.parameters()).device}")
        z_x_mu, z_x_logvar = self.encoder(x, e)
        z_s_mu, z_s_logvar = self.encode_batch(s)

        # Combine latent variables using Product of Experts
        try:
            z_mu, z_logvar = self.poe(list(z_x_mu.values()) + z_s_mu, list(z_x_logvar.values()) + z_s_logvar)
        except:
            logging.debug(z_x_mu, z_s_mu, x, e, s)

        # Sample latent variables
        z = self.sample_latent(z_mu, z_logvar)

        # Split latent variables into c and u
        c, u = z.split([self.dim_c, self.dim_u], dim=1)

        # Perform batch correction if enabled
        if self.batch_correction:
            z[:, self.dim_c:] = self.u_centroid.type_as(z).unsqueeze(0)

        # Decode data
        x_r_pre = self.decoder(z)

        # Decode batch-specific information
        s_r_pre = self.s_decoder(u) if s is not None else None

        # Generate unified latent variables and modality-specific c
        z_uni, c_all = self.generate_unified_latent(z_x_mu, z_x_logvar, z_s_mu, z_s_logvar, c)

        return x_r_pre, s_r_pre, z_mu, z_logvar, z, c, u, z_uni, c_all

    def encode_batch(self, s: torch.Tensor) -> Optional[Tuple[list, list]]:
        """
        Encode batch IDs latent variables.

        Parameters:
            s : torch.Tensor
                Batch IDs.

        Returns:
            Optional[Tuple[list, list]]:
                - z_s_mu : List[torch.Tensor]
                    Mean of batch IDs latent variables.
                - z_s_logvar : List[torch.Tensor]
                    Log-variance of batch IDs latent variables.
        """
        if s is not None:
            z_s_mu, z_s_logvar = self.s_encoder(s['joint']).split(self.dim_z, dim=1)
            return [z_s_mu], [z_s_logvar]
        return [], []

    def sample_latent(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample latent variables from a Gaussian distribution.

        Parameters:
            z_mu : torch.Tensor
                Mean of the latent variables of shape (batch_size, latent_dim).
            z_logvar : torch.Tensor
                Log-variance of the latent variables of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor:
                Sampled latent variables of shape (batch_size, latent_dim).
        """
        if self.training:
            return self.sample_gaussian(z_mu, z_logvar)
        elif self.sampling and self.sample_num > 0:
            z_mu_expand = z_mu.unsqueeze(1)
            z_logvar_expand = z_logvar.unsqueeze(1).expand(-1, self.sample_num, self.dim_z)
            return self.sample_gaussian(z_mu_expand, z_logvar_expand).reshape(-1, self.dim_z)
        return z_mu

    def generate_unified_latent(
        self,
        z_x_mu: Dict[str, torch.Tensor],
        z_x_logvar: Dict[str, torch.Tensor],
        z_s_mu: List[torch.Tensor],
        z_s_logvar: List[torch.Tensor],
        c: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Generate unified latent variables and modality-specific representations.

        Parameters:
            z_x_mu : Dict[str, torch.Tensor]
                Means of modality-specific latent variables.
            z_x_logvar : Dict[str, torch.Tensor]
                Log-variances of modality-specific latent variables.
            z_s_mu : List[torch.Tensor]
                Mean of the batch-ID latent variables.
            z_s_logvar : List[torch.Tensor]
                Log-variance of the batch-ID latent variables.
            c : torch.Tensor
                Biological information.

        Returns:
            Tuple:
                - z_uni : Dict[str, torch.Tensor]:
                    Collection of latent variables for the unimodal inputs.
                - c_all : Dict[str, torch.Tensor]:
                    Collection of biological information for the unimodal and joint inputs.
        """
        z_uni = {}
        c_all = {}

        for modality, z_x_mu_mod in z_x_mu.items():
            # Combine modality-specific and batch-specific latent variables
            z_uni_mu, z_uni_logvar = self.poe([z_x_mu_mod] + z_s_mu, [z_x_logvar[modality]] + z_s_logvar)
            # fix here
            z_uni[modality] = self.sample_latent(z_uni_mu, z_uni_logvar)
            
            # Extract shared latent representation (biological information)
            c_all[modality] = z_uni[modality][:, :self.dim_c]

        # Add joint representation
        c_all['joint'] = c
        return z_uni, c_all

    def get_dim_h(self) -> Dict[str, List[int]]:
        """
        Compute hidden dimensions for each modality.

        Returns:
            Dict[str, List[int]]:
                A dictionary containing the hidden dimensions for each modality.
        """
        dims_h = self.dims_x.copy()

        # Adjust dimensions based on pre-encoding layers
        for key in filter_keys(self.__dict__, 'dims_before_enc_'):
            modality = key.split('_')[-1]
            if  (modality in self.dims_x) and (len(self.dims_x[modality]) > 1):
                dims_h[modality] = [sum([self.__dict__[key][-1]] * len(self.dims_x[modality]))]
        return dims_h

    def gen_real_data(self, 
                      x_r_pre: Dict[str, torch.Tensor], 
                      sampling: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate real data from reconstructed data.

        Parameters:
            x_r_pre : Dict[str, torch.Tensor]
                Dictionary of reconstructed data tensors for each modality.
            sampling : bool, optional
                Whether to sample the output (default: True).

        Returns:
            Dict[str, torch.Tensor]:
                Generated real data for each modality.
        """
        x_r = {}
        for modality, tensor in x_r_pre.items():
            # Apply inverse transformations if needed
            if f'trsf_before_enc_{modality}' in self.__dict__:
                tensor = reverse_trsf(self.__dict__[f'trsf_before_enc_{modality}'].split('_')[-1], tensor)

            # Apply sampling or directly return the data
            x_r[modality] = self.sample(
                self.__dict__[f'distribution_dec_{modality}'].split('_')[-1], tensor, sampling)

        return x_r

    @staticmethod
    def sample(name: str, data: torch.Tensor, sampling: bool) -> torch.Tensor:
        """
        Map a sampling function based on the distribution name.

        Parameters:
            name : str
                Name of the distribution.
            data : torch.Tensor
                Input data tensor.
            sampling : bool
                Whether to apply sampling.

        Returns:
            torch.Tensor
                Sampled or original data tensor.
        """
        if sampling:
            return distribution_registry.get_sampling(name)(data)
        return data

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from a Gaussian distribution using the reparameterization trick.

        Parameters:
            mu : torch.Tensor
                Mean of the Gaussian distribution.
            logvar : torch.Tensor
                Log-variance of the Gaussian distribution.

        Returns:
            torch.Tensor
                Sampled tensor.
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def poe(mus: List[torch.Tensor], logvars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Product of Experts (PoE) for combining Gaussian distributions.

        Parameters:
            mus : list of torch.Tensor
                List of mean tensors for each Gaussian.
            logvars : list of torch.Tensor
                List of log-variance tensors for each Gaussian.

        Returns:
            Tuple : 
                - combined_mean: torch.Tensor
                    Mean of the combined Gaussian distribution.
                - combined_logvar: torch.Tensor
                    Log-variance of the combined Gaussian distribution.
        """
        # Add prior distributions with zero mean and unit variance

        try:
            mus = [torch.zeros_like(mus[0])] + mus
        except:
            logging.debug(mus)
        logvars = [torch.zeros_like(logvars[0])] + logvars

        # Calculate precision and combined precision
        precisions = torch.exp(-torch.stack(logvars, dim=1))  # Shape: (batch_size, num_experts, latent_dim)
        precision_sum = precisions.sum(dim=1)

        # Calculate combined mean and variance
        weighted_means = (torch.stack(mus, dim=1) * precisions).sum(dim=1)
        combined_mean = weighted_means / precision_sum
        combined_logvar = torch.log(1 / precision_sum)

        return combined_mean, combined_logvar
    

class Discriminator(nn.Module):
    """
    Discriminator class for multi-modal latent variables.

    Parameters:
        dims_x : Dict[str, list]
            Input dimensions for each modality.
        dims_s : Dict[str, int]
            Dimensions of the classes for each modality.
        kwargs : Dict[str, Any]
            Additional configurations, such as hidden layer sizes, dropout rate, and normalization type.
    """

    def __init__(self, dims_x: Dict[str, list], dims_s: Dict[str, int], **kwargs):
        super(Discriminator, self).__init__()
        self.dims_x = dims_x
        self.dims_s = dims_s

        # Dynamically set additional arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Combine modality keys with 'joint' modality
        self.modalities = list(self.dims_x.keys()) + ['joint']

        # Create predictors for each modality
        self.predictors = nn.ModuleDict({
            modality: MLP(
                [self.dim_c] + self.dims_dsc + [self.dims_s[modality]],
                hid_norm=self.norm,
                hid_drop=self.drop
            )
            for modality in self.modalities
        })

        # Cross-entropy loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  # log_softmax + nll

    def forward(self, latent_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the discriminator.

        Parameters:
            latent_inputs : Dict[str, torch.Tensor]
                Dictionary of latent inputs for each modality, where keys are modality names
                and values are tensors of shape (batch_size, dim_c).

        Returns:
            Dict[str, torch.Tensor] : 
                Dictionary of logits for each modality, where keys are modality names
                and values are tensors of shape (batch_size, dims_s[modality]).
        """
        return {modality: self.predictors[modality](latent_input)
                for modality, latent_input in latent_inputs.items()}

    def calculate_loss(self, 
                       predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate cross-entropy loss for all modalities.

        Parameters:
            predictions : Dict[str, torch.Tensor]
                Dictionary of predicted logits for each modality.
            targets : Dict[str, torch.Tensor]
                Dictionary of ground truth labels for each modality.

        Returns:
            torch.Tensor : 
                Total normalized loss.
        """
        total_loss = sum(
            self.cross_entropy_loss(pred, targets[modality].squeeze(1))
            for modality, pred in predictions.items()
        )

        # Normalize the total loss by the batch size of the joint modality
        batch_size = predictions['joint'].size(0)
        return total_loss / batch_size
    

class MIDAS(L.LightningModule):
    """
    MIDAS processes mosaic single-cell data into imputed and batch-corrected data for multimodal analysis.

    Attributes:
        net : VAE
            Variational Autoencoder for multi-modal data encoding and decoding.
        dsc : Discriminator
            Discriminator for distinguishing latent variables across batches.
        configs : Dict[str, Any]
            Model and training configurations dynamically set as attributes.
        automatic_optimization : bool
            Controls whether optimization is automatic or manually defined. Always True.
    """

    def __init__(self):
        super(MIDAS, self).__init__()

        # Initialize VAE and Discriminator
        self.net = VAE(self.dims_x, self.dims_s, **self.configs)
        self.dsc = Discriminator(self.dims_x, self.dims_s, **self.configs)

        # Dynamically set configurations as attributes
        for key, value in self.configs.items():
            setattr(self, key, value)

        # Disable automatic optimization to manually control training steps. Always True.
        self.automatic_optimization = False

        self.thread_lock = threading.Lock()

    @classmethod
    def configure_data(
        cls, 
        configs: dict,
        datalist: List[Dataset], 
        dims_x: Dict[str, list], 
        dims_s: Dict[str, int] , 
        s_joint: List[Dict[str, int]], 
        combs: List[List[str]], 
        batch_size: int = 256, 
        n_save:int = 500, 
        save_model_path: str = './saved_models/', 
        sampler_type:str = 'auto', 
        viz_umap_tb=False,
        batch_names=None,
    ) -> 'MIDAS':
        """
        Configure the data and model parameters for training.
        
        Parameters:
            configs : dict,
                Configurations of the model.
            datalist : List[Dataset]
                List of datasets to be used for training.
            dims_x : Dict[str, list]
                Dictionary specifying the dimensions of input features for each modality.
            dims_s : Dict[str, int]
                Dimensions of the classes for each modality.
            s_joint : List[Dict[str, int]]
                Modality ID for each batch.
            combs : List[List[str]]
                Combinations of modalities.
            batch_size : int, optional
                Size of each training batch, by default 256.
            n_save : int, optional
                Interval (in epochs) for saving model checkpoints, by default 500.
            save_model_path : str, optional
                Directory path for saving model checkpoints, by default './saved_models/'.
            sampler_type : str, optional
                Type of sampler to use, by default 'auto'. For 'ddp', use distributed sampler.
            viz_umap_tb: bool, optional
                Whether to visualize UMAP embeddings in TensorBoard, by default False.
            batch_names: list, optional
                List of batch names, by default None.
        Returns:
            class 'MIDAS': 
                Returns MIDAS instance.
        """

        # Set class-level attributes
        cls.configs = configs
        cls.dims_x = dims_x

        # check config
        atac_dims = dims_x.get('atac', None)
        if atac_dims is not None and len(atac_dims) == 1:
            logging.warning(
                f"Detected ATAC with only one dimension [{atac_dims[0]}]. "
                "This will cause the data to be encoded directly instead of by chromosome, as described in our paper. "
                "We recommend splitting the ATAC data by chromosome."
            )
            if 'dims_before_enc_atac' in configs and 'dims_after_dec_atac' in configs:
                logging.error(
                    'Invalid ATAC configuration: both "dims_before_enc_atac" and "dims_after_dec_atac" exist in the configs, '
                    'but len(dims_x["atac"]) == 1. To forcibly encode ATAC data directly, please remove these settings from configs.'
                )
                exit()
        if batch_names is None:
            batch_names = ['batch_%d' for i in range(len(datalist))]
        cls.batch_names = batch_names
        cls.sampler_type = sampler_type
        cls.datalist = datalist
        cls.dims_s = dims_s
        cls.s_joint = s_joint
        cls.combs = combs
        cls.mods = list(dims_x.keys())  # Extract modality names from dims_x keys
        cls.save_model_path = save_model_path
        cls.batch_size = batch_size
        cls.n_save = n_save
        cls.viz_umap_tb = viz_umap_tb
        return cls()
    
    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for training, using the appropriate sampler.

        Returns:
            DataLoader : 
                Configured DataLoader instance for training.
        """
        # Concatenate all datasets
        try:
            dataset = ConcatDataset(self.datalist)
            logging.info(f'Total number of samples: {len(dataset)} from {len(self.datalist)} datasets.')
        except Exception as e:
            raise ValueError('Failed to concatenate datasets. Please check the input datalist.') from e

        # Select the appropriate sampler
        if self.sampler_type == 'ddp':
            logging.info('Using Distributed Data Parallel (DDP) sampler.')
            sampler = MyDistributedSampler(dataset, batch_size=self.batch_size, n_max=self.n_max)
        else:
            logging.info('Using MultiBatchSampler for data loading.')
            sampler = MultiBatchSampler(dataset, batch_size=self.batch_size, n_max=self.n_max)

        # Create the DataLoader
        try:
            train_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers
            )
            logging.info(f'DataLoader created with batch size {self.batch_size} and {self.num_workers} workers.')
        except Exception as e:
            raise RuntimeError('Failed to create DataLoader. Check DataLoader configuration.') from e
        logging.debug(f'DataLoader: {len(train_loader)}')
        return train_loader
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configure optimizers for the MIDAS model.

        Returns:
            List[torch.optim.Optimizer] : 
                List of optimizers for the network and discriminator.
        """
        logging.debug(f'net:{self.net}')
        logging.debug(f'dsc:{self.dsc}')
        self.net_optim = getattr(torch.optim, self.optim_net)(self.net.parameters(), lr=self.lr_net)
        self.dsc_optim = getattr(torch.optim, self.optim_dsc)(self.dsc.parameters(), lr=self.lr_dsc)

        if self.load_optimizer_state:
            self.net_optim.load_state_dict(self.loaded_net_optim_state)
            self.dsc_optim.load_state_dict(self.loaded_dsc_optim_state)

        return [self.net_optim, self.dsc_optim]
    
    def training_step(self, 
                      batch: Dict[str, Dict[str, torch.Tensor]], 
                      batch_idx: int) -> torch.Tensor:
        """
        Executes a single training step for MIDAS.

        Parameters:
            batch : Dict[str, Dict[str, torch.Tensor]]
                Input batch containing modality data, batch indices, and masks.
            batch_idx : int
                Index of the current training batch.

        Returns:
            torch.Tensor : 
                Total VAE loss for the current batch.
        """
        # Forward pass through the VAE
        logging.debug(f"Training step - batch index: {batch_idx}")
        logging.debug(f"Input: {batch}")
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, u, z_uni, c_all = self.net(batch)
        logging.debug(f"Current batch: {batch['s']['joint'][0]}")
        c_all['joint'] = c

        # Compute reconstruction loss
        recon_loss, recon_dict = self.calc_recon_loss(
            batch['x'], batch['s']['joint'], batch['e'], 
            x_r_pre, s_r_pre,
            filter_keys(self.__dict__, 'distribution_dec_'), 
            filter_keys(self.__dict__, 'lam_recon_')
        )
        recon_loss *= self.lam_recon

        # Compute KLD loss
        kld_loss = self.calc_kld_z_loss(
            self.dim_c, self.dim_u, self.lam_kld_c, self.lam_kld_u, z_mu, z_logvar
        ) * self.lam_kld

        # Compute consistency loss
        consistency_loss = self.calc_consistency_loss(z_uni) * self.lam_alignment

        # Compute total VAE loss
        loss_net = recon_loss + kld_loss + consistency_loss

        # Train discriminator for n_iter_disc iterations
        for _ in range(self.n_iter_disc):
            self.train_discriminator(c_all, batch['s'])

        # Compute adversarial loss for the VAE
        s_pred = self.dsc(c_all)
        loss_dsc = self.calc_dsc_loss(s_pred, batch['s']) * self.lam_dsc
        loss_net = loss_net - loss_dsc * self.lam_adv
    
        # Update VAE model
        self.update_model(loss_net, self.net, self.net_optim, self.grad_clip)

        # Log training losses
        self.log_losses(recon_loss, kld_loss, consistency_loss, loss_net, loss_dsc, recon_dict)

        return loss_net

    def train_discriminator(self, 
                            c_all: Dict[str, torch.Tensor], 
                            targets: Dict[str, torch.Tensor]):
        """
        Train the discriminator with modality-specific latent representations.

        Parameters:
            c_all : Dict[str, torch.Tensor]
                Dictionary of latent representations for each modality.
            targets : Dict[str, torch.Tensor]
                Ground truth batch labels for each modality.
        """
        s_pred = self.dsc(detach_tensors(c_all))
        loss_dsc = self.calc_dsc_loss(s_pred, targets) * self.lam_dsc
        self.update_model(loss_dsc, self.dsc, self.dsc_optim, self.grad_clip)

    @rank_zero_only
    def predict(
        self,
        return_in_memory: bool = True,
        save_dir: Optional[str] = None,
        save_format: str = "npy",  # "npy" or "csv"
        joint_latent: bool = True,
        mod_latent: bool = False,
        impute: bool = False,
        batch_correct: bool = False,
        translate: bool = False,
        input: bool = False,
        verbose: bool = True
    ):
        """
        Run model inference in a streaming manner.

        This method supports three prediction modes:

            1. Return predictions in memory (recommended for small or medium datasets).
            2. Stream predictions to disk per mini-batch (recommended for large datasets).
            3. Perform both simultaneously.

        Notes:
            - If `return_in_memory=False`, prediction tensors will not accumulate in RAM,
            making this method suitable for very large datasets.
            - If `save_dir` is provided, predictions are written incrementally to disk
            per mini-batch.
            - If `batch_correct=True`, a second pass over the dataset is performed:

                Pass 1:
                    Compute joint latent representations and estimate the technical
                    centroid using online statistics.

                Pass 2:
                    Reconstruct data with batch correction and stream the corrected
                    outputs.

            - If `translate=True`, `mod_latent` will be automatically set to True.

        Parameters:
            return_in_memory : bool, default=True
                Whether to keep predictions in memory and return them as a nested
                dictionary. Set to False for large datasets to avoid OOM.

            save_dir : str or None, default=None
                Output directory for streaming prediction results. If None,
                predictions are not saved to disk.

            save_format : {"npy", "csv"}, default="npy"
                File format used when `save_dir` is provided.

                - `"npy"` : NumPy binary format (recommended; fast and compact).
                - `"csv"` : CSV text format (not recommended for large arrays).

            joint_latent : bool, default=True
                Whether to compute the joint latent representation conditioned on all
                observed modalities.

                Stored as:
                    - `z["joint"]` (raw latent)
                    - or postprocessed into `z_c["joint"]` and `z_u["joint"]`.

            mod_latent : bool, default=False
                Whether to compute latent representations conditioned on each
                individual modality.

                For each modality `m`, a single-modality forward pass is performed
                and stored as:

                    `z[m]`

            impute : bool, default=False
                Whether to generate imputed data (`x_impt`) from the joint latent
                representation.

                Stored as:

                    `x_impt[modality]`

            batch_correct : bool, default=False
                Whether to estimate a technical centroid and perform batch-effect
                correction on reconstructed data.

                Stored as:

                    `x_bc[modality]`

            translate : bool, default=False
                Whether to perform cross-modality translation.

                For each available input modality subset, missing modalities are
                generated and stored as:

                    `x_trans["<input_mods>_to_<target_mod>"]`

            input : bool, default=False
                Whether to include the original input data and masks in the output.

                Stored as:

                    `x[modality]`
                    `mask[modality]` (if available)

            verbose : bool, default=True
                Whether to display progress bars (`tqdm`) and logging messages.

        Returns:
            output : dict or None

        Raises:
            ValueError
                If both `return_in_memory=False` and `save_dir=None`, or if an
                unsupported `save_format` is specified.

            KeyError
                If required prediction fields are missing.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            logging.info(f"Predicting using device: {device}")
        model = self.net.to(device).eval()

        _old_bc = getattr(model, "batch_correction", None)
        _old_uc = getattr(model, "u_centroid", None)
        if hasattr(model, "batch_correction"):
            model.batch_correction = False

        if translate:
            mod_latent = True

        # Choose sink(s)
        sinks: List[BaseSink] = []
        mem_sink = MemorySink() if return_in_memory else None
        if mem_sink is not None:
            sinks.append(mem_sink)

        disk_sink = None
        if save_dir is not None:
            disk_sink = DiskSink(DiskSinkConfig(save_dir=save_dir, save_format=save_format))
            sinks.append(disk_sink)

        if not sinks:
            raise ValueError("You must enable at least one of return_in_memory=True or save_dir!=None.")

        # For batch_correct centroid computation (online; no full z storage)
        online_stats: Optional[OnlineMeanByGroup] = None
        if batch_correct:
            online_stats = None

        all_combinations = generate_all_combinations(self.mods) if translate else None

        with torch.no_grad():
            # -----------------------
            # Pass 1: standard outputs (+ collect stats for batch_correct)
            # -----------------------
            for batch_id, dataset in enumerate(self.datalist):
                batch_name = self.batch_names[batch_id]
                loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
                if verbose:
                    logging.info("Processing batch %s: %s", batch_name, str(self.combs[batch_id]))

                for i, batch in enumerate(tqdm(loader, desc=f"predict:{batch_name}", disable=not verbose)):
                    batch = convert_tensors_to_cuda(batch, device)

                    # Save s (labels / subset ids) if present
                    if "s" in batch and isinstance(batch["s"], dict):
                        for k, v in batch["s"].items():
                            for s in sinks:
                                s.write(batch_name, ["s", k], v)

                    # Always compute forward once (cheap vs branching), then selectively write
                    # Expected forward signature: x_r_pre, ..., z, c, u, ...
                    out = model(batch)
                    x_r_pre = out[0]
                    z = out[4]

                    # init online_stats after z known
                    if batch_correct and online_stats is None:
                        z_dim = z.shape[1]
                        u_dim = z_dim - self.dim_c
                        if u_dim <= 0:
                            raise ValueError(f"dim_c={self.dim_c} is invalid for z_dim={z_dim}")
                        online_stats = OnlineMeanByGroup(dim=u_dim)

                    # joint latent (z)
                    if joint_latent:
                        for s in sinks:
                            s.write(batch_name, ["z", "joint"], z)

                    # online stats for batch correction: use u=z[:, dim_c:], group id from s['joint'] if exists
                    if batch_correct:
                        # try common field names
                        if "s" in batch and isinstance(batch["s"], dict):
                            if "joint" in batch["s"]:
                                g = batch["s"]["joint"]
                            else:
                                # fallback: first key
                                g = next(iter(batch["s"].values()))
                        else:
                            raise ValueError("batch_correct=True requires batch['s'][...] for grouping.")
                        u = z[:, self.dim_c:]
                        online_stats.update(u, g)

                    # impute
                    if impute:
                        x_r = model.gen_real_data(x_r_pre, sampling=False)
                        for m, xm in x_r.items():
                            for s in sinks:
                                s.write(batch_name, ["x_impt", m], xm)

                    # save input + masks
                    if input:
                        for m in self.combs[batch_id]:
                            if "x" in batch and m in batch["x"]:
                                for s in sinks:
                                    s.write(batch_name, ["x", m], batch["x"][m])
                            if "e" in batch and isinstance(batch["e"], dict) and m in batch["e"]:
                                # mask typically small; store as meta or normal tensor
                                # if you prefer single-file meta, use write_meta
                                mask_np = to_numpy(batch["e"][m])[0] if batch["e"][m].ndim >= 2 else to_numpy(batch["e"][m])
                                for s in sinks:
                                    s.write_meta(batch_name, ["mask", m], mask_np)

                    # per-modality latent
                    if mod_latent:
                        for m in batch.get("x", {}).keys():
                            input_data = {"x": {m: batch["x"][m]}, "s": batch.get("s", {}), "e": {}}
                            if "e" in batch and m in batch["e"]:
                                input_data["e"][m] = batch["e"][m]
                            out_m = model(input_data)
                            z_m = out_m[4]
                            for s in sinks:
                                s.write(batch_name, ["z", m], z_m)

                    # translate (general: any input subset -> remaining outputs)
                    if translate and all_combinations is not None:
                        for input_mods, output_mods in all_combinations:
                            input_mods_sorted = sorted(input_mods)
                            # check availability in this minibatch
                            if not all(m in batch.get("x", {}) for m in input_mods_sorted):
                                continue
                            input_data = {
                                "x": {m: batch["x"][m] for m in input_mods_sorted},
                                "s": batch.get("s", {}),
                                "e": {}
                            }
                            if "e" in batch:
                                for m in input_mods_sorted:
                                    if m in batch["e"]:
                                        input_data["e"][m] = batch["e"][m]

                            out_t = model(input_data)
                            x_r_pre_t = out_t[0]
                            x_r_t = model.gen_real_data(x_r_pre_t, sampling=False)
                            for mod in output_mods:
                                key = "_".join(input_mods_sorted) + "_to_" + mod
                                for s in sinks:
                                    s.write(batch_name, ["x_trans", key], x_r_t[mod])

            # -----------------------
            # Pass 2: batch correction reconstruction (streaming)
            # -----------------------
            if batch_correct:
                if online_stats is None:
                    raise RuntimeError("Internal error: online_stats not initialized.")
                    
                u_centroid = online_stats.finalize_centroid().to(device)

                # expected: model has fields for correction; adapt to your implementation
                # (match your new code: model.u_centroid / model.batch_correction)
                _bc_prev = getattr(model, "batch_correction", None)
                _uc_prev = getattr(model, "u_centroid", None)

                try:
                    if hasattr(model, "u_centroid"):
                        model.u_centroid = u_centroid
                    if hasattr(model, "batch_correction"):
                        model.batch_correction = True
                    if verbose:
                        logging.info("Batch correction (second pass) ...")
                    for batch_id, dataset in enumerate(self.datalist):
                        batch_name = self.batch_names[batch_id]
                        loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
                        if verbose:
                            logging.info("Processing batch %s: %s", batch_name, str(self.combs[batch_id]))

                        for i, batch in enumerate(tqdm(loader, desc=f"batch_correct:{batch_name}", disable=not verbose)):
                            batch = convert_tensors_to_cuda(batch, device)
                            out = model(batch)
                            x_r_pre = out[0]
                            x_r = model.gen_real_data(x_r_pre, sampling=True)
                            for m in self.mods:
                                if m in x_r:
                                    for s in sinks:
                                        s.write(batch_name, ["x_bc", m], x_r[m])
                finally:
                    if hasattr(model, "batch_correction"):
                        model.batch_correction = (_bc_prev if _bc_prev is not None else False)
                    if hasattr(model, "u_centroid"):
                        model.u_centroid = _uc_prev

        # finalize sinks
        disk_out = disk_sink.finalize() if disk_sink is not None else None
        if mem_sink is None:
            # pure disk mode
            return disk_out

        # Post-process memory output into pred_b like you had (z -> z_c / z_u)
        raw = mem_sink.finalize()
        pred_b: Dict[str, Any] = {}
        for batch_name, d in raw.items():
            pred_b[batch_name] = {}
            # z split
            if "z" in d:
                pred_b[batch_name]["z_c"] = {}
                pred_b[batch_name]["z_u"] = {}
                for k, zt in d["z"].items():
                    znp = to_numpy(zt)
                    pred_b[batch_name]["z_c"][k] = znp[:, :self.dim_c]
                    pred_b[batch_name]["z_u"][k] = znp[:, self.dim_c:]
            # others
            for var in ["x_impt", "x_trans", "x_bc", "x", "s"]:
                if var in d:
                    pred_b[batch_name][var] = {}
                    for k, vt in d[var].items():
                        pred_b[batch_name][var][k] = to_numpy(vt)
            # masks (meta)
            if "mask" in d:
                pred_b[batch_name]["mask"] = d["mask"]

            if not joint_latent:
                pred_b[batch_name]["z_c"].pop("joint", None)
                pred_b[batch_name]["z_u"].pop("joint", None)

        if disk_out is not None:
            # if both: return both memory + disk manifest
            return {"memory": pred_b, "disk": disk_out}

        if hasattr(model, "batch_correction"):
            model.batch_correction = (_old_bc if _old_bc is not None else False)
        if hasattr(model, "u_centroid"):
            model.u_centroid = _old_uc

        return pred_b
        
    def on_train_epoch_end(self):
        """
        Save a model checkpoint at the end of each training epoch with a meaningful filename.
        """
        # Save the checkpoint periodically based on n_save
        total_epoch = self.current_epoch + self.start_epoch
        if (total_epoch + 1) % self.n_save == 0:
            os.makedirs(self.save_model_path, exist_ok=True)
            
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            # Generate a descriptive checkpoint filename
            checkpoint_filename = f'model_epoch{total_epoch+1}_{timestamp}.pt'
            checkpoint_path = os.path.join(self.save_model_path, checkpoint_filename)
            
            # Save the checkpoint
            self.save_checkpoint(checkpoint_path)
            if self.viz_umap_tb:
                logging.info('Plotting UMAP...')
                umap_thread = threading.Thread(
                target=self.get_emb_umap,
                kwargs={"save_dir":self.save_model_path, "n_obs": 20000, "verbose": False},  # 关键字参数
                daemon=True  # 守护线程：主线程退出时子线程自动结束，避免内存泄漏
                )
                # 启动线程
                umap_thread.start()
                # self.get_emb_umap(save_dir=self.save_model_path, n_obs=20000, verbose=False)
                # shutil.rmtree(self.save_model_path+'/predict'+timestamp)

    def on_train_end(self):
        """
        Save the final model checkpoint at the end of training.
        """
        os.makedirs(self.save_model_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        checkpoint_filename = f'model_epoch{self.current_epoch+self.start_epoch}_{timestamp}.pt'
        checkpoint_path = os.path.join(self.save_model_path, checkpoint_filename)
        self.save_checkpoint(checkpoint_path)
        
    @rank_zero_only
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save the current model and optimizer states to a checkpoint file.
        
        Parameters:
            checkpoint_path : str
                Path to save the checkpoint file.
        
        Raises:
            ValueError:
                If `checkpoint_path` is an invalid or empty string.
        """
        # Validate the output path
        if not checkpoint_path or not isinstance(checkpoint_path, str):
            raise ValueError('Invalid checkpoint path. Please provide a valid string.')

        # Create a state dictionary with model and optimizer states
        checkpoint_data = {
            'net': self.net.state_dict(),          # State dictionary of the main model
            'dsc': self.dsc.state_dict(),          # State dictionary of the discriminator
            'optim_net': self.net_optim.state_dict(),  # State dictionary of the main optimizer
            'optim_dsc': self.dsc_optim.state_dict()   # State dictionary of the discriminator optimizer
        }

        # Save the state dictionary to the specified path
        torch.save(checkpoint_data, checkpoint_path)

        # Inform the user of successful save
        logging.info(f'Checkpoint successfully saved to "{checkpoint_path}".')

    def load_checkpoint(self, checkpoint_path: str, start_epoch: int = 0, **kwargs):
        """
        Load model and optimizer states from a checkpoint file.
        
        Parameters:
            checkpoint_path : str
                Path to the checkpoint file containing saved model and optimizer states.
            start_epoch: int
                Indicate how many epoch the model has been trained.
            kwargs : Dict[str, Any]
                Additional configurations for torch.load().
        Raises:
            AssertionError:
                If the provided checkpoint path does not exist.
        """
        # Verify the checkpoint path exists
        assert os.path.exists(checkpoint_path), f'Checkpoint path "{checkpoint_path}" does not exist.'

        # Load the checkpoint file
        checkpoint_data = torch.load(checkpoint_path, weights_only=True, **kwargs)

        # Load the model state dictionaries
        self.net.load_state_dict(checkpoint_data['net'])
        self.dsc.load_state_dict(checkpoint_data['dsc'])

        # Load the optimizer state dictionaries
        self.load_optimizer_state = True
        self.loaded_net_optim_state = checkpoint_data['optim_net']
        self.loaded_dsc_optim_state = checkpoint_data['optim_dsc']
    
        self.start_epoch = start_epoch # influence saving name of checkpoints
                                       
    @rank_zero_only
    def get_emb_umap(
        self,
        pred_dir: str = None,
        pred_format: str = None, #'npy' or 'csv'
        save_dir: str = None,
        drop_c_umap: bool = False,
        drop_u_umap: bool = False,
        color_by: str = "batch",          # NEW: "batch" (default) or "s_joint" or any obs column you add
        n_obs: int = None, 
        verbose=True,
        **kwargs
    ) -> Tuple[List[sc.AnnData], List[plt.Figure]]:
        """
        Generate UMAP visualizations for biological and technical latent embeddings.

        This function loads predicted latent representations and computes UMAP
        embeddings for visualization. Two types of embeddings are supported:

            1. Biological embedding (`z_c`)
            2. Technical embedding (`z_u`)

        For large datasets, the function can optionally subsample observations to
        accelerate UMAP computation.

        Parameters:
            pred_dir : str, optional
                Directory containing predicted results generated by `predict()` or
                `predict_streaming()`. If None, predictions will be generated on-the-fly
                using `self.predict()`.

            pred_format : {"npy", "csv"}, optional
                File format of saved prediction files when loading from disk.
                Only used when `pred_dir` is provided.

            save_dir : str, optional
                Directory to save the generated UMAP figures. If None, figures will not
                be written to disk.

            drop_c_umap : bool, default=False
                Whether to skip UMAP visualization for the biological embedding (`z_c`).

            drop_u_umap : bool, default=False
                Whether to skip UMAP visualization for the technical embedding (`z_u`).

            color_by : str, default="batch"
                Column name in `adata.obs` used to color cells in UMAP plots.
                Common options include:

                - `"batch"` : batch label
                - `"s_joint"` : subset or dataset identifier
                - any other metadata column stored in `adata.obs`

            n_obs : int, optional
                Number of observations to randomly subsample before computing UMAP.
                Useful for large datasets to speed up visualization. If None, all
                observations will be used.

            verbose : bool, default=True
                Whether to display progress bars and logging information.

            **kwargs : Dict[str, Any]
                Additional keyword arguments passed to `scanpy.pl.umap()`.

        Returns:
            all_adata : List[AnnData]
                List of AnnData objects containing the computed UMAP embeddings.

            all_figures : List[matplotlib.figure.Figure]
                List of generated UMAP figure objects.

        Notes:
            - UMAP is computed using `scanpy.pp.neighbors()` followed by
            `scanpy.tl.umap()`.
            - The biological embedding (`z_c`) captures biological variation,
            while the technical embedding (`z_u`) reflects batch or technical effects.
            - For very large datasets (e.g., >1M cells), it is recommended to set
            `n_obs` (e.g., 20,000) to reduce computation time.

        Examples
        --------
        Generate UMAP from saved predictions:

        >>> model.get_emb_umap(pred_dir="./predictions")

        Generate UMAP with subsampling and custom coloring:

        >>> model.get_emb_umap(
        ...     pred_dir="./predictions",
        ...     n_obs=20000,
        ...     color_by="batch"
        ... )

        Generate UMAP and save figures:

        >>> model.get_emb_umap(
        ...     pred_dir="./predictions",
        ...     save_dir="./figs"
        ... )
        """
        def _unwrap_pred(p: Any) -> Dict[str, Any]:
            # If predict returned {"memory": ..., "disk": ...}
            if isinstance(p, dict) and "memory" in p and isinstance(p["memory"], dict):
                return p["memory"]
            return p
        if verbose:
            logging.info(f"Loading predicted data from: {pred_dir}")
        if pred_dir is not None:
            # IMPORTANT: adapt this call to your actual loader signature.
            # If you're using the loader we discussed earlier, it would be something like:
            pred = load_predicted(pred_dir, save_format=pred_format, dim_c=self.dim_c, split_z=True, var_names=['z'])
            #
            # If your project already has load_predicted(pred_dir, self.combs, mtx=use_mtx), keep it.
            # pred = load_predicted(pred_dir, dim_c=self.dim_c, split_z=True)  # <- adjust if needed
        else:
            # Use the new streaming predict (in-memory) by default
            pred = self.predict(
                return_in_memory=True,
                save_dir=None,
                joint_latent=True,
                mod_latent=False,
                impute=False,
                batch_correct=False,
                translate=False,
                input=False,
                verbose=verbose
            )

        pred = _unwrap_pred(pred)

        # pred is expected to be: {batch_name: {...}, batch_name2: {...}}
        if not isinstance(pred, dict) or len(pred) == 0:
            raise ValueError("Empty prediction results.")

        # ----------------------------
        # Concatenate z_c/z_u across batches
        # ----------------------------
        zc_list, zu_list = [], []
        batch_labels = []
        s_joint_labels = []  # optional

        for batch_name, data in pred.items():
            if "z_c" not in data or "joint" not in data["z_c"]:
                raise KeyError(f"Missing z_c/joint in batch '{batch_name}'")
            if "z_u" not in data or "joint" not in data["z_u"]:
                raise KeyError(f"Missing z_u/joint in batch '{batch_name}'")

            zc = data["z_c"]["joint"]
            zu = data["z_u"]["joint"]

            # allow torch or numpy
            if hasattr(zc, "detach"):
                zc = zc.detach().cpu().numpy()
            if hasattr(zu, "detach"):
                zu = zu.detach().cpu().numpy()

            n = zc.shape[0]
            zc_list.append(zc)
            zu_list.append(zu)
            batch_labels.append(np.array([batch_name] * n, dtype=object))

            # optional: keep s['joint'] if present (useful for coloring)
            sj = None
            if "s" in data and isinstance(data["s"], dict):
                # common key names: 'joint' or first key
                if "joint" in data["s"]:
                    sj = data["s"]["joint"]
                else:
                    sj = next(iter(data["s"].values()))
            if sj is not None:
                if hasattr(sj, "detach"):
                    sj = sj.detach().cpu().numpy()
                sj = np.asarray(sj).reshape(-1)
                if sj.shape[0] == n:
                    s_joint_labels.append(sj.astype(int).astype(str))

        bio_embedding = np.concatenate(zc_list, axis=0)
        tech_embedding = np.concatenate(zu_list, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)

        s_joint_labels = np.concatenate(s_joint_labels, axis=0) if len(s_joint_labels) else None

        # ----------------------------
        # Build UMAPs
        # ----------------------------
        all_adata: List[sc.AnnData] = []
        all_figures: List[plt.Figure] = []

        file_names = ["biological_information.png", "technical_noise.png"]
        embeddings = [bio_embedding, tech_embedding]

        for index, (embedding, file_name) in enumerate(zip(embeddings, file_names)):
            if file_name == "biological_information.png" and drop_c_umap:
                logging.info("Skipping biological embedding UMAP generation (drop_c_umap=True).")
                continue
            if file_name == "technical_noise.png" and drop_u_umap:
                logging.info("Skipping technical embedding UMAP generation (drop_u_umap=True).")
                continue
            
            if verbose:
                logging.info(f"Processing {'biological' if index == 0 else 'technical'} embedding...")

            adata = sc.AnnData(embedding)
            adata.obs["batch"] = batch_labels
            if s_joint_labels is not None:
                adata.obs["s_joint"] = s_joint_labels

            # neighbors + umap (use the embedding directly as X)
            if verbose:
                logging.info(" - Computing neighbors...")
            if n_obs:
                sc.pp.subsample(adata, n_obs=n_obs)
            sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")  # X is already embedding
            if verbose:
                logging.info(" - Computing UMAP...")
            sc.tl.umap(adata)

            # pick color
            plot_color = color_by
            if plot_color is not None and plot_color not in adata.obs.columns:
                logging.warning(
                    f"color_by='{plot_color}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}. Falling back to 'batch'."
                )
                plot_color = "batch"

            if verbose:
                logging.info(f" - Generating UMAP plot for {file_name}...")
            fig = sc.pl.umap(
                adata,
                title=file_name[:-4],
                color=plot_color,
                show=False,
                return_fig=True,
                **kwargs,
            )
            all_figures.append(fig)

            if save_dir:
                fig_save_path = os.path.join(save_dir, "figs", f"epoch_{self.current_epoch + self.start_epoch + 1}_"+file_name)
                os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
                fig.savefig(fig_save_path, dpi=200, bbox_inches="tight")
                if verbose:
                    logging.info(f" - UMAP plot saved to: {fig_save_path}")

            if getattr(self, "logger", None) is not None and getattr(self, "viz_umap_tb", False):
                self.logger.experiment.add_figure(file_name, fig, self.current_epoch + self.start_epoch)

            all_adata.append(adata)
        if verbose:
            logging.info("UMAP generation completed.")
        return all_adata, all_figures

    def log_losses(self, 
                   recon_loss: torch.Tensor, 
                   kld_loss, consistency_loss: torch.Tensor, 
                   loss_net: torch.Tensor, 
                   loss_dsc: torch.Tensor, 
                   recon_dict: Dict[str, torch.Tensor]):
        """
        Log losses for monitoring and debugging during training.

        Parameters:
            recon_loss : torch.Tensor
                Reconstruction loss.
            kld_loss : torch.Tensor
                KLD loss.
            consistency_loss : torch.Tensor
                Consistency loss.
            recon_dict : Dict[str, torch.Tensor]
                Per-modality reconstruction losses.
            loss_net : torch.Tensor
                Total VAE loss.
            loss_dsc : torch.Tensor
                Discriminator loss.
        """
        self.log_dict(
            {
                'loss_/recon_loss': recon_loss,
                'loss_/kld_loss': kld_loss,
                'loss_/consistency_loss': consistency_loss,
                'loss/net': loss_net,
                'loss/dsc':loss_dsc
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(recon_dict, on_epoch=True, sync_dist=True)

    @staticmethod
    def update_model(
        loss: torch.Tensor, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        grad_clip: int=-1):
        """
        Update model parameters using backpropagation.

        Parameters:
            loss : torch.Tensor
                Computed loss for backpropagation.
            model : torch.nn.Module
                Model to update.
            optimizer : torch.optim.Optimizer
                Optimizer for parameter updates.
            grad_clip : int
                True to allow clipping gradient.
        """
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    @staticmethod
    def calc_dsc_loss(pred: Dict[str, torch.Tensor], true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the discriminator loss using cross-entropy.

        Parameters:
            pred : Dict[str, torch.Tensor]
                Predicted logits for each modality.
            true : Dict[str, torch.Tensor]
                Ground truth labels for each modality.

        Returns:
            torch.Tensor : 
                Computed discriminator loss.
        """
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  # Cross-entropy loss
        loss = {}

        # Compute loss for each modality
        for modality in pred:
            loss[modality] = cross_entropy_loss(pred[modality], true[modality].squeeze(1))

        # Normalize total loss by batch size
        total_loss = sum(loss.values()) / pred['joint'].size(0)
        return total_loss
    
    @staticmethod
    def calc_kld_z_loss(dim_c: int, 
                        dim_u: int, 
                        lam_kld_c: float, 
                        lam_kld_u: float, 
                        mu: torch.Tensor, 
                        logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Kullback-Leibler Divergence (KLD) loss for latent variables z.

        Parameters:
            dim_c : int
                Dimension of the biological latent space.
            dim_u : int
                Dimension of the technical latent space.
            lam_kld_c : float
                Weight for KLD loss of the biological latent space.
            lam_kld_u : float
                Weight for KLD loss of the technical latent space.
            mu : torch.Tensor
                Mean of the latent variable distribution (batch_size x (dim_c + dim_u)).
            logvar : torch.Tensor
                Log-variance of the latent variable distribution (batch_size x (dim_c + dim_u)).

        Returns:
            torch.Tensor:
                Weighted sum of KLD losses for the biological and technical latent spaces.
        """
        # Split the mean and log-variance into biological (c) and technical (u) components
        mu_c, mu_u = mu.split([dim_c, dim_u], dim=1)
        logvar_c, logvar_u = logvar.split([dim_c, dim_u], dim=1)

        # Calculate KLD losses for biological and technical latent spaces
        kld_c_loss = MIDAS.calc_kld_loss(mu_c, logvar_c)
        kld_u_loss = MIDAS.calc_kld_loss(mu_u, logvar_u)

        # Combine the losses with their respective weights
        kld_z_loss = kld_c_loss * lam_kld_c + kld_u_loss * lam_kld_u
        return kld_z_loss

    @staticmethod
    def calc_kld_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate the KLD loss for a single latent space.

        Parameters:
            mu : torch.Tensor
                Mean of the latent variable distribution (batch_size x latent_dim).
            logvar : torch.Tensor
                Log-variance of the latent variable distribution (batch_size x latent_dim).

        Returns:
            torch.Tensor :
                KLD loss for the latent space, normalized by batch size.
        """
        # KLD loss formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        return kld_loss

    @staticmethod
    def calc_consistency_loss(z_uni: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the consistency loss for unified latent variables across modalities.

        Parameters:
            z_uni : Dict[str, torch.Tensor]
                Dictionary of unified latent variables for each modality, where each value is a
                tensor of shape (batch_size x latent_dim).

        Returns:
            torch.Tensor : 
                Consistency loss computed as the variance of the unified latent variables.
        """
        # Stack the unified latent variables along a new dimension (modalities)
        z_uni_stack = torch.stack(list(z_uni.values()), dim=0)  # Shape: M x N x K (M=modalities, N=batch_size, K=latent_dim)

        # Calculate the mean across modalities
        z_uni_mean = z_uni_stack.mean(0, keepdim=True)  # Shape: 1 x N x K

        # Consistency loss is the variance across modalities
        consistency_loss = ((z_uni_stack - z_uni_mean) ** 2).sum() / z_uni_stack.size(1)  # Normalize by batch size
        return consistency_loss
 
    @staticmethod
    def calc_recon_loss(
        x: Dict[str, torch.Tensor],    
        s: torch.Tensor,
        e: Dict[str, torch.Tensor],
        x_r_pre: Dict[str, torch.Tensor],
        s_r_pre: Dict[str, torch.Tensor],
        dist: Dict[str, str],
        lam: Dict[str, float]
    ) -> Tuple[float, Dict[torch.Tensor, torch.Tensor]]:
        """
        Calculate the reconstruction loss for input data and predicted outputs.

        Parameters:
            x : Dict[str, torch.Tensor]
                Original input data for each modality (x^m).
            s : torch.Tensor
                Ground truth batch labels.
            e : Dict[str, torch.Tensor]
                Mask.
            x_r_pre : Dict[str, torch.Tensor]
                Reconstructed predictions for each modality (x_r^m).
            s_r_pre : Dict[str, torch.Tensor]
                Reconstructed predictions for batch labels.
            dist : Dict[str, str]
                Dictionary specifying the distribution type for each modality's decoder.
            lam : Dict[str, float]
                Dictionary containing reconstruction loss weights for each modality and for s.

        Returns:
            Tuple:
                - total_loss : torch.Tensor
                    Total reconstruction loss, normalized by batch size.
                - losses : Dict[str, torch.Tensor]
                    Dictionary containing reconstruction losses for each modality and for batch labels.
        """
        losses = {}

        # Compute reconstruction loss for each modality
        for modality, x_original in x.items():
            # Get the appropriate loss function based on the modality's decoder distribution
            loss_fn = distribution_registry.get_loss(dist[f'distribution_dec_{modality}'])

            # Check if there is an event-specific mask for the modality
            if modality in e:
                # Apply event-specific mask to the reconstruction loss
                losses[f'recon_loss/{modality}'] = (
                    loss_fn(x_r_pre[modality], x_original) * e[modality]
                ).sum() * lam[f'lam_recon_{modality}']
            else:
                # Compute the reconstruction loss without a mask
                losses[f'recon_loss/{modality}'] = (
                    loss_fn(x_r_pre[modality], x_original)
                ).sum() * lam[f'lam_recon_{modality}']

        # Compute reconstruction loss for batch labels, if provided
        if s_r_pre is not None:
            # Use cross-entropy loss for batch label reconstruction
            losses['recon_loss/s'] = (
                distribution_registry.get_loss('CE')(s_r_pre, s.squeeze(1))
            ).sum() * lam['lam_recon_s']

        # Normalize total loss by the batch size
        total_loss = sum(losses.values()) / s.size(0)
        return total_loss, losses

    @staticmethod
    def get_info_from_mdata(mdata, batch_key='batch'):
        batch_names = []
        for k in mdata.mod.keys():
            batch_names.extend(np.unique(mdata[k].obs[batch_key]).tolist())
        batch_names = np.unique(batch_names)
        data = []
        mask = []
        for b in batch_names:
            t = {}
            mt = {}
            for m in mdata.mod.keys():
                if b in mdata[m].obs[batch_key].values:
                    t[m] = mdata[m][mdata[m].obs[batch_key]==b]
                if f'mask_{b}' in mdata[m].uns:
                    mt[m] = mdata[m].uns[f'mask_{b}']
            data.append(t)
            mask.append(mt)
        return data, mask, batch_names

    @staticmethod
    def get_info_from_dir(dir_path: str, format: str):
        """
        Extract data, mask, and feature dimensions from a directory of vectors.

        Parameters:
            dir_path : str
                Path to the directory containing data and mask files.
            format : str
                Support 'mtx', 'csv', and 'vec'.

        Returns:
            Tuple:
                - data : List[Dict[str, str]]
                    List of dictionaries where keys are modalities and values are file paths.
                - mask : List[Dict[str, str]]
                    List of dictionaries where keys are modalities and values are mask file paths.
                - dims_x : Dict[str, list]
                    Dictionary containing feature dimensions for each modality.
        
        Notes:
            The directory should be organized as::
    
                dataset/
                    feat/
                        # Dimensions of each modality: {mod1=[...], mod2=[...]}.
                        # Split the data into chunks if the length of the list is greater than 1. 
                        # For instance, you can split the ATAC data by chromosomes.
                        feat_dims.toml 
                    batch_0/
                        mask/mod1.csv
                        mask/mod2.csv
                        vec/mod1/0000.csv # the first sample
                        vec/mod1/0001.csv # the second sample
                        ....
                        vec/mod2/0000.csv
                        vec/mod2/0001.csv
                        ....
                    batch_1/
                        mask/mod1.csv
                        mask/mod2.csv
                        vec/mod1/0000.csv
                        vec/mod1/0001.csv
                        ....
                        vec/mod2/0000.csv
                        vec/mod2/0001.csv
                    ....
                
            or  like::

                dataset/
                    feat/
                        # Dimensions of each modality: {mod1=[...], mod2=[...]}.
                        # Split the data into chunks if the length of the list is greater than 1. 
                        # For instance, you can split the ATAC data by chromosomes.
                        feat_dims.toml 
                    batch_0/
                        mask/mod1.csv
                        mask/mod2.csv
                        mat/mod1.mtx (.csv)
                        mat/mod2.mtx (.csv)
                        ....
                    batch_1/
                        mask/mod1.csv
                        mask/mod2.csv
                        mat/mod1.mtx (.csv)
                        mat/mod2.mtx (.csv)
                    ....
            
        """
        data = []  # List to store data file paths
        mask = []  # List to store mask file paths
        batch_names = []

        for batch_dir in natsort.natsorted(os.listdir(dir_path)):
            if batch_dir != 'feat':  # Ignore the 'feat' directory
                data_batch = {}
                mask_batch = {}
                batch_path = os.path.join(dir_path, batch_dir)
                batch_names.append(batch_dir)

                # Collect file paths for data and masks
                if format == 'vec':
                    if os.path.exists(batch_path):
                        vec_dir = os.path.join(batch_path, 'vec')
                        mask_dir = os.path.join(batch_path, 'mask')
                        for file in os.listdir(vec_dir):
                            data_batch[file] = os.path.join(vec_dir, file)
                        for file in os.listdir(mask_dir):
                            mask_batch[file[:-4]] = os.path.join(mask_dir, file)

                elif format in ['csv', 'mtx']:
                    if os.path.exists(batch_path):
                        mat_dir = os.path.join(batch_path, 'mat')
                        mask_dir = os.path.join(batch_path, 'mask')
                        for file in os.listdir(mat_dir):
                            data_batch[file[:-4]] = os.path.join(mat_dir, file)
                        for file in os.listdir(mask_dir):
                            mask_batch[file[:-4]] = os.path.join(mask_dir, file)

                data.append(data_batch)
                mask.append(mask_batch)

        # Load feature dimensions from 'feat_dims.toml'
        dims_x = toml.load(os.path.join(dir_path, 'feat', 'feat_dims.toml'))
        return data, mask, dims_x, batch_names

    @classmethod
    def configure_data_from_mdata(
        cls,
        configs: Dict[str, Any], 
        mdata: "MuData", # Assuming MuData is imported or available
        dims_x: Dict[str, list],
        batch_key: str = 'batch',
        transform: Optional[Dict[str, str]] = None, 
        sampler_type: str = 'auto', 
        viz_umap_tb: bool = False, 
        save_model_path: str = './saved_models/', 
        n_save: int = 500,
        **kwargs : Any
    ) -> 'MIDAS':
        """
        Configure the MIDAS model directly from a MuData object.

        This method processes the MuData input to extract data, masks, and batch information,
        initializes the datasets, and sets up the model configuration.

        Parameters:
            configs : Dict[str, Any]
                Configurations of the model.
            mdata : MuData
                The input MuData object containing multi-modal single-cell data.
                It is expected to contain `AnnData` objects for different modalities (e.g., RNA, ATAC).
            dims_x : Dict[str, list]
                A dictionary specifying the input feature dimensions for each modality.
                Keys are modality names, and values are lists of dimensions (e.g., `{'rna': [2000]}`).
            transform : Optional[Dict[str, str]], default=None
                A dictionary specifying specific transformations to apply to each modality.
                Example: `{'atac': 'binarize'}`. If None, default transformations are used.
            sampler_type : str, default='auto'
                Strategy for data sampling. Use 'ddp' for Distributed Data Parallel training, 
                or 'auto' for standard training.
            viz_umap_tb: bool, default=False
                If True, enables UMAP visualization logs in TensorBoard during the training process.
            save_model_path : str, optional
                Directory path for saving model checkpoints, by default './saved_models/'.
            n_save : int, optional
                Interval (in epochs) for saving model checkpoints, by default 500.
            **kwargs : Any
                Additional keyword arguments passed to the underlying `configure_data` method
                (e.g., `batch_size`, `num_workers`).

        Returns:
            MIDAS:
                An initialized instance of the MIDAS class, ready for training or inference.
        """
        # Note: get_info_from_mdata is expected to return:
        # data: List[Dict[str, AnnData]], mask: List[Dict[str, np.ndarray]], batch_names: List[str]
        data, mask, batch_names = cls.get_info_from_mdata(mdata, batch_key)

        # Configure datasets and calculate dimensions for batch correction
        # This calls the updated get_datasets_from_dir which handles in-memory masks (numpy arrays)
        datalist, dims_s, s_joint, combs = cls.get_datasets_from_adata(
            data, mask, batch_names, transform
        )
        
        # Reset training state flags
        cls.start_epoch = 0
        cls.load_optimizer_state = False
        
        # Finalize configuration and return the class instance
        return cls.configure_data(
            configs, 
            datalist, 
            dims_x,
            dims_s,
            s_joint, 
            combs, 
            sampler_type=sampler_type, 
            viz_umap_tb=viz_umap_tb,
            batch_names=batch_names,
            save_model_path = save_model_path,
            n_save=n_save,
            **kwargs
        )

    @classmethod
    def configure_data_from_dir(cls,
                                configs: Dict[str, Any], 
                                dir_path: str,
                                format: str = 'mtx',
                                transform: Dict[str, str] = None, 
                                sampler_type: str = 'auto', 
                                viz_umap_tb: bool = False,
                                save_model_path: str = './saved_models/', 
                                n_save: int = 500,
                                **kwargs : Dict[str, Any]) -> 'MIDAS':
        """
        Configure data from a directory and apply optional transformations.

        Parameters:
            configs : Dict[str, Any]
                Configurations of the model.
            dir_path : str
                Path to the directory containing data files.
            transform : Dict[str, str], optional
                A dictionary specifying transformations to apply to specific modalities.
                Example: {'atac': 'binarize'}
                Default is None, which uses the default transformation settings.
            sampler_type : str, optional
                Type of sampler to use, by default 'auto'. For 'ddp', use distributed sampler.
            viz_umap_tb: bool, optional
                Whether to visualize UMAP embeddings in TensorBoard, by default False.
            save_model_path : str, optional
                Directory path for saving model checkpoints, by default './saved_models/'.
            n_save : int, optional
                Interval (in epochs) for saving model checkpoints, by default 500.
            kwargs : Dict[str, Any]
                Additional parameters passed to configure_data().

        Returns:
            class 'MIDAS':
                Returns the configured class instance.

        Examples:
            >>> from scmidas.model import MIDAS
            >>> from scmidas.config import load_config
            >>> configs = load_config()
            >>> dir_path = 'XXX'
            >>> transform = {'atac': 'binarize'}
            >>> model = MIDAS.configure_data_from_dir(configs, dir_path, transform)

        """
        # Extract data, mask, and feature dimensions from the directory
        data, mask, dims_x, batch_names = cls.get_info_from_dir(dir_path, format)

        # Configure datasets and associated parameters
        datalist, dims_s, s_joint, combs = cls.get_datasets_from_dir(data, mask, batch_names, transform, format)
        
        cls.start_epoch = 0
        cls.load_optimizer_state = False
        # Finalize and return class instance
        return cls.configure_data(
            configs, 
            datalist, 
            dims_x,
            dims_s,
            s_joint, 
            combs, 
            sampler_type=sampler_type, 
            viz_umap_tb=viz_umap_tb,
            batch_names=batch_names,
            n_save=n_save,
            save_model_path = save_model_path,
            **kwargs)
    
    @staticmethod
    def get_datasets_from_dir(
        data: List[Dict[str, str]], 
        mask: List[Dict[str, str]], 
        batch_names: List[str],
        transform: Dict[str, str]=None, 
        format: str='mtx'):
        """
        Configure data from directory.

        Parameters:
            data : List[Dict[str, str]]
                List of data dictionaries, where keys are modalities and values are file paths.
            mask : List[Dict[str, str]]
                List of mask dictionaries, where keys are modalities and values are mask file paths.
            batch_name : List[str]
                List of batch names.
            transform : Optional[Dict[str, str]]
                Transformations to apply to specific modalities.
            format : str
                File type of the input data, default is 'vec'. ['vec', 'mtx', 'csv']

        Returns:
            Tuple:
                - datasets : List[MultiModalDataset]
                    List of initialized `MultiModalDataset` objects.
                - dims_s : Dict[str, int]
                    Dimensions for batch correction for each modality.
                - s_joint : List[Dict[str, int]]
                    Modality indices for each batch.
                - combs : List[List[str]]
                    List of modality combinations for each batch.
        """
        s_joint = []  # Modality indices for each batch
        n_s = {}  # Counter for each modality
        combs = []  # Modality combinations for each batch
        datasets = []  # List of datasets
        dims_s = {}  # Dimensions for batch correction

        for i, batch_data in enumerate(data):
            batch_s = {}  # Store batch-specific indices
            batch_combs = []  # Modality combination for the current batch

            # Assign batch index for each modality
            for modality in batch_data.keys():
                if modality in n_s:
                    batch_s[modality] = n_s[modality] + 1
                    n_s[modality] += 1
                else:
                    batch_s[modality] = 0
                    n_s[modality] = 0
                batch_combs.append(modality)

            # Add joint batch information
            batch_s['joint'] = i
            n_s['joint'] = i
            s_joint.append(batch_s)
            combs.append(batch_combs)

            # Determine file types for each modality
            file_types = {
                modality: format
                for modality in batch_data.keys()
            }

            # Initialize MultiModalDataset
            dataset = MultiModalDataset(batch_data, batch_s, file_types, mask[i], transform)
            datasets.append(dataset)

        # Define dimensions for batch correction
        dims_s = {modality: count + 1 for modality, count in n_s.items()}
        MIDAS.print_info(mask, datasets, batch_names)
        return datasets, dims_s, s_joint, combs

    @staticmethod
    def get_datasets_from_adata(
        data: List[Dict[str, AnnData]], 
        mask: List[Dict[str, str]], 
        batch_names: List[str],
        transform: Dict[str, str]=None):
        """
        Configure data from a CSV input.

        Parameters:
            data : List[Dict[str, str]]
                List of data dictionaries, where keys are modalities and values are adata object.
            mask : List[Dict[str, str]]
                List of mask dictionaries, where keys are modalities and values are mask values.
            batch_name : List[str]
                List of batch names.
            transform : Optional[Dict[str, str]]
                Transformations to apply to specific modalities.
            format : str
                File type of the input data, default is 'vec'. ['vec', 'mtx', 'csv']

        Returns:
            Tuple:
                - datasets : List[MultiModalDataset]
                    List of initialized `MultiModalDataset` objects.
                - dims_s : Dict[str, int]
                    Dimensions for batch correction for each modality.
                - s_joint : List[Dict[str, int]]
                    Modality indices for each batch.
                - combs : List[List[str]]
                    List of modality combinations for each batch.
        """
        s_joint = []  # Modality indices for each batch
        n_s = {}  # Counter for each modality
        combs = []  # Modality combinations for each batch
        datasets = []  # List of datasets
        dims_s = {}  # Dimensions for batch correction

        for i, batch_data in enumerate(data):
            batch_s = {}  # Store batch-specific indices
            batch_combs = []  # Modality combination for the current batch

            # Assign batch index for each modality
            for modality in batch_data.keys():
                if modality in n_s:
                    batch_s[modality] = n_s[modality] + 1
                    n_s[modality] += 1
                else:
                    batch_s[modality] = 0
                    n_s[modality] = 0
                batch_combs.append(modality)

            # Add joint batch information
            batch_s['joint'] = i
            n_s['joint'] = i
            s_joint.append(batch_s)
            combs.append(batch_combs)

            # Determine file types for each modality
            file_types = {
                modality: 'anndata'
                for modality in batch_data.keys()
            }

            # Initialize MultiModalDataset
            dataset = MultiModalDataset(batch_data, batch_s, file_types, mask[i], transform)
            datasets.append(dataset)

        # Define dimensions for batch correction
        dims_s = {modality: count + 1 for modality, count in n_s.items()}
        MIDAS.print_info(mask, datasets, batch_names)
        return datasets, dims_s, s_joint, combs

    @staticmethod
    @rank_zero_only
    def print_info(mask: List[Dict[str, str]], datalist: List[Dataset], batch_names: List[str]):
        """
        Print summary of mask density and dataset information.

        Parameters:
            mask : List[Dict[str, str]]
                List of mask.
            datalist : List[Dataset]
                List of datasets.
            batch_name : List[str]
                List of batch names.
        """
        # Calculate mask density for each batch
        feature = []
        valid_feature = []
        for i, dataset in enumerate(datalist):
            s1 = {}
            s2 = {}
            dataset = dataset[0]
            mask_ = mask[i]
            for m in dataset['x']:
                s1['#%s'%m.upper()] = len(dataset['x'][m])
                if m in mask_:
                    if isinstance(mask_[m], str):
                        t = pd.read_csv(mask_[m], index_col=0).values
                    else:
                        t = mask_[m]
                    s2['#VALID_'+m.upper()] = t.sum()
            feature.append(s1)
            valid_feature.append(s2)
        valid_feature = pd.DataFrame(valid_feature)
        valid_feature.index = batch_names
        cell_number = pd.DataFrame({'#CELL':[len(dataset) for dataset in datalist]})
        cell_number.index = batch_names
        feature = pd.DataFrame(feature)
        feature.index = batch_names
        data = pd.concat([cell_number, feature, valid_feature], axis=1)
        # Print summary
        logging.info('Input data: \n' + data.to_string())