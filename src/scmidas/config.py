import logging
logging.basicConfig(level=logging.INFO)

configs_all = {}
configs_all["default"] = {
# available_mods : ["rna", "adt", "atac"]  # Supported modalities

# Latent
"dim_c" : 32,  # Latent dimension for biological information c.
"dim_u" : 2,   # Latent dimension for technical information u (always be small to avoid capturing biological information).

# Loss function weights
"lam_kld_c" : 1,         # Weight for variable c’s KLD loss.
"lam_kld_u" : 5,         # Weight for variable u’s KLD loss.
"lam_kld" : 1,           # Weight for total KLD loss.
"lam_recon" : 1,         # Weight for reconstruction loss.
"lam_dsc" : 30,          # Weight for discriminator loss (for training the discriminator).
"lam_adv" : 1,          # Weight for adversarial loss. loss : VAE_loss - disc_loss * lam_adv.
"lam_alignment" : 50,    # Weight for modality alignment loss.
"lam_recon_rna" : 1,     # Weight for RNA reconstruction loss.
"lam_recon_adt" : 1,     # Weight for ADT reconstruction loss.
"lam_recon_atac" : 1,    # Weight for ATAC reconstruction loss.
"lam_recon_s" : 1000,    # Weight for batch indices reconstruction loss.

# Discriminator iteration
"n_iter_disc" : 3,  # Number of discriminator iterations before training the VAE.

# Basic network structure (MLP)
"norm" : "ln",           # Use layer normalization. ‘bn’, ‘ln’, or False.
"drop" : 0.2 ,           # Dropout rate.
"out_trans" : "mish",    # Activation function for the output. Support: ‘tanh’, ‘relu’, ‘silu’, ‘mish’, ‘sigmoid’, ‘softmax’, ‘log_softmax’.

# Modality configuration
"dims_shared_enc" : [1024, 128],  # Shared encoder structure across all modalities.
"dims_shared_dec" : [128, 1024],  # Shared decoder structure across all modalities.

# RNA modality configuration
"trsf_before_enc_rna" : "log1p",      # Apply log1p transformation before encoding. Exponential transformation will be applied after decoding.
"distribution_dec_rna" : "POISSON",   # Poisson distribution assumption for decoder.

# ADT modality configuration
"trsf_before_enc_adt" : "log1p",      # Apply log1p transformation before encoding. Exponential transformation will be applied after decoding.
"distribution_dec_adt" : "POISSON",   # Poisson distribution assumption for decoder.

# ATAC modality configuration
"dims_before_enc_atac" : [128, 32],  # Independent MLP structure before shared encoder. It is used to compress the data chunks of the ATAC modality.
"dims_after_dec_atac" : [32, 128],   # Independent MLP structure after shared decoder. It expands the embeddings to reconstruct the ATAC modality.
"distribution_dec_atac" : "BERNOULLI",  # Bernoulli distribution assumption for decoder. Use BCE loss.

# Batch-related configuration
"s_drop_rate" : 0.1,              # Rate to drop batch indices during training.
"dims_enc_s" : [16, 16],          # Encoder structure.
"dims_dec_s" : [16, 16],          # Decoder structure.
"dims_dsc" : [128, 64],           # Structure of the discriminator.

# Training configuration
"optim_net" : "AdamW",            # Optimizer for the main network.
"lr_net" : 1e-4 ,                 # Learning rate for the main network.
"optim_dsc" : "AdamW",            # Optimizer for the discriminator.
"lr_dsc" : 1e-4,                  # Learning rate for the discriminator.
"grad_clip" : -1,                 # Gradient clipping (grad_clip>0 means clipping).

# Data loader configuration
"num_workers" : 20,               # Number of worker threads for data loading.
"pin_memory" : True,              # Load data into pinned memory.
"persistent_workers" : True,     # Persistent worker threads.
"n_max" : 10000                  # Maximum number of samples per batch.
}


def load_config(config_name :str = "default"):
    """
    Load configurations to construct the model.

    Parameters:
        config_name : str
            Item name from the configuration.
    """
    logging.info(f'The model is initialized with the {config_name} configurations.')
    return configs_all[config_name]