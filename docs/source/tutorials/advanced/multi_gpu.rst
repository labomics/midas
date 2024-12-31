Training with Multi-GPU
========================

1. Use the Distributed Sampler for DDP Training
To enable Distributed Data Parallel (DDP) training, configure the data sampler as follows:

.. code-block:: python

    # If using this method:
    MIDAS.configure_data(sampler_type='ddp')

    # Or, if using this method:
    MIDAS.configure_data_from_dir(sampler_type='ddp')



2. Configure the Trainer for Multi-GPU Training
Set up the trainer in your training script with the following settings:

.. code-block:: python

    trainer = L.Trainer(
        devices='auto',                # Automatically use all available GPUs
        strategy='ddp'                 # Enable distributed training with DDP
    )

- ``devices='auto'``: This will automatically detect and use all available GPUs. 
Alternatively, specify a specific number of GPUs by setting devices=n, 
where n is the desired number of GPUs (e.g., devices=2 for two GPUs).

- ``strategy='ddp'``: Use the Distributed Data Parallel (DDP) strategy 
for training across multiple GPUs on a single node. DDP helps to parallelize 
the model training by splitting the data and computing on different GPUs, improving performance.

.. note::
    Run 'ddp' in ``.py`` script: For multi-GPU training, it is recommended to run 
    your code in a ``.py`` file rather than a Jupyter notebook (``.ipynb``).