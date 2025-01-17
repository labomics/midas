# Standard Library Imports
import os
import datetime
from typing import Dict, List, Tuple
import toml
import natsort
# Third-Party Library Imports
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as L
import logging
logging.basicConfig(level=logging.INFO)

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
            Input dimensions for each modality.
        dims_h : Dict[str, list]
            Hidden dimensions for each modality after pre-encoding.
        dim_z : int
            Latent dimension size.
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        out_trans : str
            Output activation function (e.g., 'mish').
        drop : float
            Dropout rate.
        kwargs : dict
            Additional modality-specific configurations.

    Notes:
        By default, RNA and ADT data are log1p-transformed and will be exponentiated after decoding. 
        To skip this step, modify the configuration file. See parameter trsf_before_enc_. 
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
        self.pre_encoders = nn.ModuleDict()  # Modality-specific pre-encoding layers
        self.after_concat = nn.ModuleDict()  # Post-concatenation layers
        encoders = {}  # Final encoders for each modality

        for modality, input_dims in dims_x.items():
            # If multiple input dimensions, create pre-encoding layers
            if len(input_dims) > 1:
                pre_encoders = nn.ModuleList([
                    MLP([dim] + kwargs[f'dims_before_enc_{modality}'], hid_norm=self.norm, hid_drop=self.drop)
                    for dim in input_dims
                ])
                self.pre_encoders[modality] = pre_encoders
                self.after_concat[modality] = Layer1D(self.dims_h[modality], 
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

    def forward(self, data: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]):
        """
        Forward pass for the encoder.

        Parameters:
            data : dict
                Input data for each modality.
            mask : dict
                Masks for each modality.

        Returns:
            z_x_mu : dict
                Mean values for latent space for each modality.
            z_x_logvar : dict
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

        # Pre-encode and concatenate if necessary
        for modality in data.keys():
            if modality in self.pre_encoders:
                # Split and process individual dimensions
                batches = data[modality].split(self.dims_x[modality], dim=1)
                processed_batches = [
                    self.pre_encoders[modality][i](batch) for i, batch in enumerate(batches)
                ]
                # Concatenate processed batches
                data[modality] = self.after_concat[modality](torch.cat(processed_batches, dim=1))

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
        kwargs : Dict
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

        # Shared decoder layer
        total_hidden_dims = sum(dim[0] for dim in dims_h.values())
        self.shared_decoder = MLP(
            [self.dim_z] + self.dims_shared_dec + [total_hidden_dims],
            hid_norm=self.norm,
            hid_drop=self.drop,
        )

        # Modality-specific decoders
        self.post_decoders = nn.ModuleDict()
        self.before_concat = nn.ModuleDict()

        for modality, output_dims in dims_x.items():
            # Modality-specific post-decoding layers
            if len(output_dims) > 1:
                post_decoders = nn.ModuleList([
                    MLP(kwargs[f'dims_after_dec_{modality}'] + [dim], 
                        hid_norm=self.norm, hid_drop=self.drop)
                    for dim in output_dims
                ])
                self.post_decoders[modality] = post_decoders

            # Layer to process concatenated outputs
            self.before_concat[modality] = Layer1D(self.dims_h[modality], 
                                                   self.norm, self.out_trans, 
                                                   self.drop)

    def forward(self, latent_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the decoder.

        Parameters:
            latent_data : torch.Tensor
                Latent variable input tensor of shape (batch_size, dim_z).

        Returns:
            Dict[str, torch.Tensor]
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
            # Apply pre-concatenation layer
            processed_output = self.before_concat[modality](data_dict[modality])
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
    Encoder for batch-specific latent variables.

    Parameters:
        n_batches : int
            Number of distinct batches.
        dims_enc_s : List[int]
            List of dimensions for hidden layers in the encoder.
        dim_z : int
            Latent dimension size for the output.
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        drop : float
            Dropout rate.
    """

    def __init__(self, n_batches: int, dims_enc_s: List[int], dim_z: int, norm: str, drop: float):
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
            torch.Tensor
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
            Latent dimension size for the input.
        norm : str
            Normalization type (e.g., 'ln' for LayerNorm).
        drop : float
            Dropout rate.
    """

    def __init__(self, n_batches: int, dims_dec_s: List[int], dim_u: int, norm: str, drop: float):
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
            torch.Tensor
                Reconstructed tensor of shape (batch_size, n_batches).
        """
        return self.s_decoder(data)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for multi-modal data, supporting batch correction and sampling.

    Parameters:
        dims_x : dict
            Input dimensions for each modality.
        dims_s : dict
            Dimensions of the classes for each modality.
        kwargs : dict
            Additional configurations for encoders, decoders, and other modules.
    """
    def __init__(self, 
                 dims_x: Dict[str, list], 
                 dims_s: Dict[str, int], 
                 **kwargs):
        super(VAE, self).__init__()
        self.dims_x = dims_x
        self.dims_s = dims_s
        self.mods = set(dims_x.keys())

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

    def forward(self, data: Dict[str, torch.Tensor]) -> tuple:
        """
        Forward pass for the VAE.

        Parameters:
            data : dict
                Input data dictionary containing:
                - 'x': Dict[str, torch.Tensor], modality-specific input data.
                - 'e': Dict[str, torch.Tensor], modality-specific masks.
                - 's' (optional): torch.Tensor, dimensions of the output classes for each modality.

        Returns:
            Tuple:
                - x_r_pre (dict): Reconstructed modality-specific data.
                - s_r_pre (torch.Tensor or None): Reconstructed batch indices.
                - z_mu (torch.Tensor): Mean of the combined latent variables.
                - z_logvar (torch.Tensor): Log-variance of the combined latent variables.
                - z (torch.Tensor): Sampled latent variables.
                - c (torch.Tensor): Biological information variables.
                - u (torch.Tensor): Technical noise variables.
                - z_uni (dict): Unified latent variables for each modality.
                - c_all (dict): Modality-specific Biological information variables.
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
        z_x_mu, z_x_logvar = self.encoder(x, e)
        z_s_mu, z_s_logvar = self.encode_batch(s)

        # Combine latent variables using Product of Experts
        z_mu, z_logvar = self.poe(list(z_x_mu.values()) + z_s_mu, list(z_x_logvar.values()) + z_s_logvar)

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

    def encode_batch(self, s: torch.Tensor) -> Tuple[list, list]:
        """
        Encode batch indices latent variables.

        Parameters:
            s : torch.Tensor
                Batch indices.

        Returns:
            Tuple:
                - List[torch.Tensor]: Mean of batch indices latent variables.
                - List[torch.Tensor]: Log-variance of batch indices latent variables.
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
            z_x_mu : dict
                Mean of modality-specific latent variables.
            z_x_logvar : dict
                Log-variance of modality-specific latent variables.
            z_s_mu : list
                Mean of modality-specific batch indices latent variables.
            z_s_logvar : list
                Log-variance of modality-specific batch indices latent variables.
            c : torch.Tensor
                Biological information.

        Returns:
            Tuple
                - Unified latent variables (z_uni) for each modality.
                - Modality-specific shared representations (c_all).
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
            dict
                A dictionary containing the hidden dimensions for each modality.
        """
        dims_h = self.dims_x.copy()

        # Adjust dimensions based on pre-encoding layers
        for key in filter_keys(self.__dict__, 'dims_before_enc_'):
            modality = key.split('_')[-1]
            if modality in self.dims_x:
                dims_h[modality] = [sum([self.__dict__[key][-1]] * len(self.dims_x[modality]))]
        return dims_h

    def gen_real_data(self, 
                      x_r_pre: Dict[str, torch.Tensor], 
                      sampling: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate real data from reconstructed data.

        Parameters:
            x_r_pre : dict
                Dictionary of reconstructed data tensors for each modality.
            sampling : bool, optional
                Whether to sample the output (default: True).

        Returns:
            dict
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
            Tuple
                - Mean of the combined Gaussian distribution.
                - Log-variance of the combined Gaussian distribution.
        """
        # Add prior distributions with zero mean and unit variance
        mus = [torch.zeros_like(mus[0])] + mus
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
        dims_x : dict
            Input dimensions for each modality.
        dims_s : dict
            Dimensions of the classes for each modality.
        kwargs : dict
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
            latent_inputs : dict
                Dictionary of latent inputs for each modality, where keys are modality names
                and values are tensors of shape (batch_size, dim_c).

        Returns:
            dict
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
            predictions : dict
                Dictionary of predicted logits for each modality.
            targets : dict
                Dictionary of ground truth labels for each modality.

        Returns:
            torch.Tensor
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
        configs : dict
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
    ):
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
                Modality indices for each batch.
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

        Returns:
            cls
                Returns MIDAS instance.
        """

        # Set class-level attributes
        cls.configs = configs
        cls.sampler_type = sampler_type
        cls.datalist = datalist
        cls.dims_x = dims_x
        cls.dims_s = dims_s
        cls.s_joint = s_joint
        cls.combs = combs
        cls.mods = list(dims_x.keys())  # Extract modality names from dims_x keys
        cls.save_model_path = save_model_path
        cls.batch_size = batch_size
        cls.n_save = n_save

        return cls()
    
    def train_dataloader(self):
        """
        Create a DataLoader for training, using the appropriate sampler.

        Returns:
            DataLoader
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

        return train_loader
    
    def configure_optimizers(self):
        """
        Configure optimizers for the network and discriminator.

        Returns:
            list
                List of configured optimizers.
        """
        # Dynamically create optimizers based on provided optimizer names
        self.net_optim = getattr(torch.optim, self.optim_net)(self.net.parameters(), lr=self.lr_net)
        self.dsc_optim = getattr(torch.optim, self.optim_dsc)(self.dsc.parameters(), lr=self.lr_dsc)
        return [self.net_optim, self.dsc_optim]
    
    def training_step(self, 
                      batch: Dict[str, Dict[str, torch.Tensor]], 
                      batch_idx: int) -> torch.Tensor:
        """
        Executes a single training step for MIDAS.

        Parameters:
            batch : dict
                Input batch containing modality data, batch indices, and masks.
            batch_idx : int
                Index of the current training batch.

        Returns:
            torch.Tensor
                Total VAE loss for the current batch.
        """
        # Forward pass through the VAE
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, u, z_uni, c_all = self.net(batch)
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
            c_all : dict
                Dictionary of latent representations for each modality.
            targets : dict
                Ground truth batch labels for each modality.
        """
        s_pred = self.dsc(detach_tensors(c_all))
        loss_dsc = self.calc_dsc_loss(s_pred, targets) * self.lam_dsc
        self.update_model(loss_dsc, self.dsc, self.dsc_optim, self.grad_clip)

    @torch.no_grad
    def predict(self, 
                pred_dir: str, 
                joint_latent: bool = True, 
                mod_latent: bool = False, 
                impute: bool = False, 
                batch_correct: bool = False, 
                translate: bool = False, 
                input: bool = False):
        """
        Predict and save results for multiple modes, including joint latent, 
        imputation, batch correction, and translation.

        Parameters:
            pred_dir : str
                Directory for saving prediction results.
            joint_latent : bool, optional
                Whether to calculate and save joint latent representations.
            mod_latent : bool, optional
                Whether to calculate and save modality-specific latent representations.
            impute : bool, optional
                Whether to perform data imputation.
            batch_correct : bool, optional
                Whether to apply batch correction.
            translate : bool, optional
                Whether to perform modality translation.
            input : bool, optional
                Whether to save input data.
        
        Notes:
            See https://github.com/labomics/midas/issues/7.
        """
        self.net.eval()
        device = next(self.net.parameters()).device
        if translate:
            mod_latent = True
        logging.info('Predicting ...')
        dirs = get_pred_dirs(pred_dir, 
                             self.combs, 
                             joint_latent, 
                             mod_latent, 
                             impute, 
                             batch_correct, 
                             translate, 
                             input)
        parent_dirs = list(set(map(os.path.dirname, extract_values(dirs))))
        mkdirs(parent_dirs, remove_old=False)
        mkdirs(dirs, remove_old=False)
        for batch_id, data in enumerate(self.datalist):
            data_loader = DataLoader(data, shuffle=False, batch_size=self.batch_size)
            logging.info('Processing batch %d: %s' % (batch_id, str(self.combs[batch_id])))
            fname_fmt = get_name_fmt(len(data_loader))+'.csv'
            for i, data in enumerate(tqdm(data_loader)):
                data = convert_tensors_to_cuda(data, device)
                # conditioned on all observed modalities
                if joint_latent:
                    x_r_pre, _, _, _, z, _, _, *_ = self.net(data)  # N * K
                    save_tensor_to_csv(z, os.path.join(dirs[batch_id]['z']['joint'], fname_fmt) % i)
                if impute:
                    x_r = self.net.gen_real_data(x_r_pre, sampling=False)
                    for m in self.mods:
                        save_tensor_to_csv(x_r[m], os.path.join(dirs[batch_id]['x_impt'][m], fname_fmt) % i)
                if input:  # save the input
                    for m in self.combs[batch_id]:
                        save_tensor_to_csv(data['x'][m], os.path.join(dirs[batch_id]['x'][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if mod_latent:
                    for m in data['x'].keys():
                        input_data = {
                            'x': {m: data['x'][m]},
                            's': data['s'], 
                            'e': {}
                        }
                        if m in data['e'].keys():
                            input_data['e'][m] = data['e'][m]
                        x_r_pre, _, _, _, z, c, u, *_ = self.net(input_data)  # N * K
                        save_tensor_to_csv(z, os.path.join(dirs[batch_id]['z'][m], fname_fmt) % i)
                if translate: # double to single
                    all_combinations = generate_all_combinations(self.mods)
                    for input_mods, output_mods in all_combinations:
                        input_mods_sorted = sorted(input_mods)
                        input_data = {
                            'x': {m: data['x'][m] for m in input_mods_sorted if m in data['x']},
                            's': data['s'], 
                            'e': {}
                        }
                        for m in input_mods_sorted:
                            if m in data['e'].keys():
                                input_data['e'][m] = data['e'][m]
                        x_r_pre, *_ = self.net(input_data)  # N * K
                        x_r = self.net.gen_real_data(x_r_pre, sampling=False)
                        for mod in output_mods:
                            save_tensor_to_csv(x_r[mod], 
                                               os.path.join(dirs[batch_id]['x_trans']['_'.join(input_mods_sorted) + '_to_' + mod], fname_fmt) % i)

        if batch_correct:
            logging.info('Calculating u_centroid ...')
            
            pred = load_predicted(pred_dir, self.s_joint, self.combs, self.mods)
            u = torch.from_numpy(pred['z']['joint'][:, self.dim_c:])
            s = torch.from_numpy(pred['s']['joint'])

            u_mean = u.mean(dim=0, keepdim=True)
            u_batch_mean_list = []
            for batch_id in s.unique():
                u_batch = u[s == batch_id, :]
                u_batch_mean_list.append(u_batch.mean(dim=0))
            u_batch_mean_stack = torch.stack(u_batch_mean_list, dim=0)
            dist = ((u_batch_mean_stack - u_mean) ** 2).sum(dim=1)
            self.net.u_centroid = u_batch_mean_list[dist.argmin()]
            self.net.batch_correction = True
            
            logging.info('Batch correction ...')
            for batch_id, data in enumerate(self.datalist):
                data_loader = DataLoader(data, shuffle=False, batch_size=self.batch_size)
                logging.info('Processing batch %d: %s' % (batch_id, str(self.combs[batch_id])))
                fname_fmt = get_name_fmt(len(data_loader))+'.csv'
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = convert_tensors_to_cuda(data, device)
                    x_r_pre, *_ = self.net(data)
                    x_r = self.net.gen_real_data(x_r_pre, sampling=True)
                    for m in self.mods:
                        save_tensor_to_csv(x_r[m], os.path.join(dirs[batch_id]['x_bc'][m], fname_fmt) % i)

    def on_train_epoch_end(self):
        """
        Save a model checkpoint at the end of each training epoch with a meaningful filename.
        """
        # Save the checkpoint periodically based on n_save
        if self.current_epoch!=0 and self.current_epoch % self.n_save == 0:
            os.makedirs(self.save_model_path, exist_ok=True)
            
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            # Generate a descriptive checkpoint filename
            checkpoint_filename = f'model_epoch{self.current_epoch}_{timestamp}.pt'
            checkpoint_path = os.path.join(self.save_model_path, checkpoint_filename)
            
            # Save the checkpoint
            self.save_checkpoint(checkpoint_path)
            logging.info(f'Checkpoint saved for epoch "{self.current_epoch}" at "{checkpoint_path}".')

    def on_train_end(self):
        """
        Save the final model checkpoint at the end of training.
        """
        os.makedirs(self.save_model_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        checkpoint_filename = f'model_epoch{self.current_epoch}_{timestamp}.pt'
        checkpoint_path = os.path.join(self.save_model_path, checkpoint_filename)
        self.save_checkpoint(checkpoint_path)
        logging.info(f'Checkpoint saved for epoch "{self.current_epoch}" at {checkpoint_path}".')

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

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model and optimizer states from a checkpoint file.
        
        Parameters:
            checkpoint_path : str
                Path to the checkpoint file containing saved model and optimizer states.
        
        Raises:
            AssertionError:
                If the provided checkpoint path does not exist.
        """
        # Verify the checkpoint path exists
        assert os.path.exists(checkpoint_path), f'Checkpoint path "{checkpoint_path}" does not exist.'

        # Load the checkpoint file
        checkpoint_data = torch.load(checkpoint_path)

        # Load the model state dictionaries
        self.net.load_state_dict(checkpoint_data['net'])
        self.dsc.load_state_dict(checkpoint_data['dsc'])

        # Check if optimizers are already initialized
        if not (hasattr(self, 'net_optim') and hasattr(self, 'dsc_optim')):
            logging.warning('Optimizers not initialized. Running `configure_optimizers` to set them up.')
            # Initialize optimizers if they are not already set
            self.net_optim, self.dsc_optim = self.configure_optimizers()

        # Load the optimizer state dictionaries
        self.net_optim.load_state_dict(checkpoint_data['optim_net'])
        self.dsc_optim.load_state_dict(checkpoint_data['optim_dsc'])

    def get_emb_umap(self, pred_dir: str, save_dir='.', save_fig=True, **kwargs):
        """
        Generate UMAP embeddings for biological (c) and technical (u) latent variables.

        Parameters:
            pred_dir : str
                Directory containing predicted data.
            save_dir : str, optional
                Directory to save UMAP plots, by default '.'.
            save_fig : bool, optional
                Whether to save the UMAP figures, by default True.
            kwargs : dict
                Additional configurations for sc.pl.umap().

        Returns:
            tuple
                List of AnnData objects and UMAP figures.
        """
        logging.info(f'Loading predicted data from: {pred_dir}')
        pred = load_predicted(pred_dir, self.combs)

        # Extract biological and technical embeddings and batch labels
        bio_embedding = pred['z']['joint'][:, :self.dim_c]  # Biological embedding
        tech_embedding = pred['z']['joint'][:, self.dim_c:]  # Technical embedding
        batch_labels = pred['s']['joint'].astype('int').astype('str')  # Batch labels

        all_adata = []  # List to store AnnData objects
        all_figures = []  # List to store UMAP figures
        file_names = ['biological_embedding.png', 'technical_embedding.png']  # File names for UMAP plots

        # Generate UMAP for both embeddings
        for index, (embedding, file_name) in enumerate(zip([bio_embedding, tech_embedding], file_names)):
            logging.info(f'Processing {'biological' if index == 0 else 'technical'} embedding...')

            # Create AnnData object for the embedding
            adata = sc.AnnData(embedding)
            adata.obs['batch'] = batch_labels

            # Compute nearest neighbors and UMAP
            logging.info(' - Computing neighbors...')
            sc.pp.neighbors(adata)
            logging.info(' - Computing UMAP...')
            sc.tl.umap(adata)

            # Plot UMAP and optionally save the figure
            logging.info(f' - Generating UMAP plot for {file_name}...')
            fig = sc.pl.umap(adata, title=file_name.strip('.png'), color='batch', show=False, return_fig=True, **kwargs)
            all_figures.append(fig)

            if save_fig:
                fig_save_path = os.path.join(save_dir, 'figs', file_name)
                os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
                fig.savefig(fig_save_path)
                logging.info(f' - UMAP plot saved to: {fig_save_path}')

            all_adata.append(adata)

        logging.info('UMAP generation completed.')
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
            recon_dict : dict
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
        grad_clip=-1):
        """
        Update model parameters using backpropagation.

        Parameters:
            loss : torch.Tensor
                Computed loss for backpropagation.
            model : torch.nn.Module
                Model to update.
            optimizer : torch.optim.Optimizer
                Optimizer for parameter updates.
            grad_clip : bool
                True to allow clipping gradient.
        """
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    @staticmethod
    def calc_dsc_loss(pred: dict, true: dict):
        """
        Calculate the discriminator loss using cross-entropy.

        Parameters:
            pred : dict
                Predicted logits for each modality.
            true : dict
                Ground truth labels for each modality.

        Returns:
            float
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
                        logvar: torch.Tensor):
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
            kld_z_loss : float
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
    def calc_kld_loss(mu: torch.Tensor, logvar: torch.Tensor):
        """
        Calculate the KLD loss for a single latent space.

        Parameters:
            mu : torch.Tensor
                Mean of the latent variable distribution (batch_size x latent_dim).
            logvar : torch.Tensor
                Log-variance of the latent variable distribution (batch_size x latent_dim).

        Returns:
            kld_loss : float
                KLD loss for the latent space, normalized by batch size.
        """
        # KLD loss formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        return kld_loss

    @staticmethod
    def calc_consistency_loss(z_uni: dict):
        """
        Calculate the consistency loss for unified latent variables across modalities.

        Parameters:
            z_uni : dict
                Dictionary of unified latent variables for each modality, where each value is a
                tensor of shape (batch_size x latent_dim).

        Returns:
            consistency_loss : float
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
        x: dict,    
        s: torch.Tensor,
        e: dict,
        x_r_pre: dict,
        s_r_pre: dict,
        dist: dict,
        lam: dict
    ):
        """
        Calculate the reconstruction loss for input data and predicted outputs.

        Parameters:
            x : dict
                Original input data for each modality (x^m).
            s : torch.Tensor
                Ground truth batch labels.
            e : dict
                Mask.
            x_r_pre : dict
                Reconstructed predictions for each modality (x_r^m).
            s_r_pre : dict
                Reconstructed predictions for batch labels.
            dist : dict
                Dictionary specifying the distribution type for each modality's decoder.
            lam : dict
                Dictionary containing reconstruction loss weights for each modality and for s.

        Returns:
            total_loss : float
                Total reconstruction loss, normalized by batch size.
            losses : dict
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
    def get_info_from_dir(dir_path: str):
        """
        Extract data, mask, and feature dimensions from a directory.

        Parameters:
            dir_path : str
                Path to the directory containing data and mask files.

        Returns:
            data : list of dict
                List of dictionaries where keys are modalities and values are file paths.
            mask : list of dict
                List of dictionaries where keys are modalities and values are mask file paths.
            dims_x : dict
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
            
        """
        data = []  # List to store data file paths
        mask = []  # List to store mask file paths

        for batch_dir in natsort.natsorted(os.listdir(dir_path)):
            if batch_dir != 'feat':  # Ignore the 'feat' directory
                data_batch = {}
                mask_batch = {}
                batch_path = os.path.join(dir_path, batch_dir)

                # Collect file paths for data and masks
                if os.path.exists(batch_path):
                    vec_dir = os.path.join(batch_path, 'vec')
                    mask_dir = os.path.join(batch_path, 'mask')
                    for file in os.listdir(vec_dir):
                        data_batch[file] = os.path.join(vec_dir, file)
                    for file in os.listdir(mask_dir):
                        mask_batch[file[:-4]] = os.path.join(mask_dir, file)

                data.append(data_batch)
                mask.append(mask_batch)

        # Load feature dimensions from 'feat_dims.toml'
        dims_x = toml.load(os.path.join(dir_path, 'feat', 'feat_dims.toml'))
        return data, mask, dims_x
    
    @classmethod
    def configure_data_from_dir(cls,
                                configs: dict, 
                                dir_path: str, 
                                transform: dict = None, 
                                sampler_type: str = 'auto', 
                                **kwargs : dict):
        """
        Configure data from a directory and apply optional transformations.

        Parameters:
            configs : dict,
                Configurations of the model.
            dir_path : str
                Path to the directory containing data files.
            transform : dict, optional
                A dictionary specifying transformations to apply to specific modalities.
                Example: {'atac': 'binarize'}
                Default is None, which uses the default transformation settings.
            sampler_type : str, optional
                Type of sampler to use, by default 'auto'. For 'ddp', use distributed sampler.
            kwargs : dict
                Additional parameters passed to configure_data().

        Returns:
            cls
                Returns the configured class instance.

        Raises:
            ValueError:
                If `transform` is not a dictionary.

        Examples:
            >>> from scmidas.model import MIDAS
            >>> from scmidas.config import load_config
            >>> configs = load_config()
            >>> dir_path = './data_processed/xxx'
            >>> transform = {'atac': 'binarize'}
            >>> model = MIDAS.configure_data_from_dir(configs, dir_path, transform)

        """
        # Extract data, mask, and feature dimensions from the directory
        data, mask, dims_x = cls.get_info_from_dir(dir_path)

        # Configure datasets and associated parameters
        datalist, dims_s, s_joint, combs = cls.configure_data_from_csv(data, mask, transform)

        # Finalize and return class instance
        return cls.configure_data(configs, datalist, dims_x, dims_s, s_joint, combs, sampler_type=sampler_type, **kwargs)

    @staticmethod
    def configure_data_from_csv(data: dict, mask: dict, transform: dict=None):
        """
        Configure data from a CSV input.

        Parameters:
            data : list of dict
                List of data dictionaries, where keys are modalities and values are file paths.
            mask : list of dict
                List of mask dictionaries, where keys are modalities and values are mask file paths.
            transform : dict, optional
                Transformations to apply to specific modalities.

        Returns:
            datasets : list
                List of initialized `MultiModalDataset` objects.
            dims_s : dict
                Dimensions for batch correction for each modality.
            s_joint : list
                Modality indices for each batch.
            combs : list
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
                modality: 'mat' if batch_data[modality].endswith(('.csv', '.csv.gz')) else 'vec'
                for modality in batch_data.keys()
            }

            # Initialize MultiModalDataset
            dataset = MultiModalDataset(batch_data, batch_s, file_types, mask[i], transform)
            datasets.append(dataset)

        # Define dimensions for batch correction
        dims_s = {modality: count + 1 for modality, count in n_s.items()}
        MIDAS.print_info(mask, datasets)
        return datasets, dims_s, s_joint, combs

    @staticmethod
    def print_info(mask: List[Dict[str, str]], datalist: List[Dataset]):
        """
        Print summary of mask density and dataset information.

        Parameters:
            mask : list of dict
                List of mask.
            datalist : list
                List of datasets.
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
                    t = pd.read_csv(mask_[m], index_col=0).values
                    s2['#VALID_'+m.upper()] = t.sum()
            feature.append(s1)
            valid_feature.append(s2)
        valid_feature = pd.DataFrame(valid_feature)
        valid_feature.index = [f'BATCH {i}' for i in range(len(valid_feature))]
        cell_number = pd.DataFrame({'#CELL':[len(dataset) for dataset in datalist]})
        cell_number.index = [f'BATCH {i}' for i in range(len(cell_number))]
        feature = pd.DataFrame(feature)
        feature.index = [f'BATCH {i}' for i in range(len(feature))]
        data = pd.concat([cell_number, feature, valid_feature], axis=1)
        # Print summary
        logging.info('Input data: \n' + data.to_string())