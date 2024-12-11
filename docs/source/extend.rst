Extending MIDAS to Support More Modalities
==========================================

MIDAS currently supports the integration of RNA, ADT, and ATAC data. To extend the model to support new modalities, follow the guidelines below.

About Our Framework
~~~~~~~~~~~~~~~~~~~

The MIDAS model is configured via the ``scmidas/model_config.toml`` file and is primarily implemented using Multi-Layer Perceptrons (MLP). The framework consists of several key components:

Key Components of MIDAS
-----------------------

1. **Data Encoder**: Encodes each modality (as a dictionary) into the means and log-transformed variances of a Gaussian distribution, representing the modality-specific latent features.
2. **Data Decoder**: Reconstructs the counts for each modality by using the joint latent features as input.
3. **Batch Indices Encoder**: Encodes batch indices for each modality into Gaussian-distributed means and variances.
4. **Batch Indices Decoder**: Reconstructs the batch indices for each modality using the joint latent features.
5. **Discriminator**: A set of classifiers that categorize each modality's latents as well as the joint latents. Only the biological part of the latents is used for this classification.

Each of these components uses MLPs as the base architecture, which means that convolutional neural networks (CNNs) or more complex structures are currently not supported (though they can be incorporated if needed).

Transformation and Distribution Functions
-----------------------------------------

MIDAS employs a variety of transformation and distribution functions that can be modified or extended for new modalities.

Transformation Functions
^^^^^^^^^^^^^^^^^^^^^^^^

- **binarize**

  - **Input Transformation**: Binarize the data

  - **Output Transformation**: None

- **log1p**

  - **Input Transformation**: Apply ``log1p`` (log(x + 1)) transformation

  - **Output Transformation**: Apply the exponential function (``exp``)

.. note::
   The transformation functions specified during the model configuration (e.g., in the ``model = MIDAS.configure_data_from_dir(task, transform)``) are applied only when retrieving items from the dataset (during the `get_item` step). However, when the data is passed through the encoder (after the initial step) and before the decoder output (just before the final step), both the transformation and its inverse transformation will be applied. This ensures that the data is properly transformed during the forward pass and restored during the decoding process.

Distribution Functions
^^^^^^^^^^^^^^^^^^^^^^

1. **POISSON**

   - **Loss Function**: Poisson loss

   - **Sampling**: Poisson sampling

   - **Activation**: None

2. **BERNOULLI**

   - **Loss Function**: Cross-entropy

   - **Sampling**: Bernoulli sampling

   - **Activation**: Sigmoid

The ``loss`` defines the reconstruction loss function, ``sampling`` defines how batch-corrected counts are calculated, and ``activation`` sets the output layer activation for the decoder.


Step 1: Extend the Framework for New Modalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data Encoder
------------

The encoder transforms data through modality-specific and shared layers to obtain latent representations. You can modify the structure in the ``scmidas/model_config.toml`` as follows:

.. tip::

   To customize the configuration, you have two options:

   1. Modify the Default Configuration: Directly update the default configuration to suit your requirements.

   2. Create and Customize a New Configuration:
      Duplicate the default configuration as a new item.
      Make your modifications to the new item.
      Specify the new configuration item when calling the model's configure functions (e.g., ``MIDAS.configure_data_from_dir(task, transform, config_name=new_item)``).

1. **Transformation Before Encoding**: Set the transformation function for the data before encoding.

   Example:

   .. code-block:: python
      
      trsf_before_enc_mod = 'log1p'  # Default transformation
   
2. **Dimensionality Reduction Layer**: If the data is split into chunks, define the modality-specific layers for encoding each chunk individually before merging them.

   Example:

   .. code-block:: python
      
      dims_before_enc_mod = [512, 128]  # First encode to 512 dimensions, then to 128
   
3. **(Optional) Shared Layer Configuration**: Define the shared layer structure (e.g., ``[1024, 128]``).

   Example:

   .. code-block:: python
      
      dims_shared_enc = [1024, 128]
   

Data Decoder
------------

The decoder reconstructs the original data by decoding latents through shared and modality-specific layers. Configure the shared layers and post-decoders in the ``scmidas/model_config.toml`` as follows:

1. **(Optional) Shared Layer Setup**: Define the structure for the shared decoder layers.

   Example:

   .. code-block:: python
      
      dims_shared_dec = [128, 1024]
   
2. **Dimensionality Expansion Layer**: If the data is split into chunks, define the dimensionality expansion layers after the shared layers.

   Example:

   .. code-block:: python
      
      dims_after_dec_mod = [128, 512]
   

After decoding, the output will be transformed according to the registered transformation functions.

Reconstruction Loss Weight
--------------------------

Set the weight for the reconstruction loss function in the ``scmidas/model_config.toml`` as follows:

.. code-block:: python

   lam_recon_mod = 1  # Adjust as needed

(Optional) Batch Indices Encoder and Decoder
--------------------------------------------

Set up the batch indices encoder and decoder layers in the ``scmidas/model_config.toml`` as follows:

1. **Batch Indices Encoder Setup**:

   .. code-block:: python
      
      dims_enc_s = [16, 16]
   
2. **Batch Indices Decoder Setup**:

   .. code-block:: python

      dims_dec_s = [16, 16]

Step 2: Register New Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To extend MIDAS with new functionalities, you need to register new transformation and distribution functions.

Registering New Transformation Functions
----------------------------------------

Use the ``TransformRegistry`` to register new transformation functions in the ``scmidas/nn.py``, or import the instance as needed:

.. code-block:: python

   TransformRegistry.register(name, func, inverse_func)


Register New Distribution Functions
--------------------------------------

Use the ``DistributionRegistry`` to register new distribution-related functions in the ``scmidas/nn.py``, or import the instance as needed:

.. code-block:: python

   DistributionRegistry.register(name, loss_fn, sampling_fn, activate_fn)

Step 3: Register New Modality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can add the name of new modality in the ``scmidas/model_config.toml`` as:

.. code-block:: python

   available_mods = ["rna", "adt", "atac", "mod"]

Call for Contributions
~~~~~~~~~~~~~~~~~~~~~~

If you've implemented new features or improvements and would like to contribute to the MIDAS project, we encourage you to submit a **pull request**. We welcome your enhancements and will review them for inclusion in the main repository.
