Training with Multi-GPU
========================

To enable training on multiple GPUs, configure the trainer in the training code snippet with the following settings:

.. code-block:: python

    trainer = L.Trainer(
        devices='auto',                # Automatically use all available GPUs
        strategy='ddp'                 # Enable distributed training with DDP
    )

.. note::

    1. ``devices='auto'``: This will automatically detect and use all available GPUs. 
    Alternatively, specify a specific number of GPUs by setting devices=n, 
    where n is the desired number of GPUs (e.g., devices=2 for two GPUs).

    2. ``strategy='ddp'``: Use the Distributed Data Parallel (DDP) strategy 
    for training across multiple GPUs on a single node. DDP helps to parallelize 
    the model training by splitting the data and computing on different GPUs, improving performance.
    
    3. Run in ``.py`` script: For multi-GPU training, it is recommended to run 
    your code in a ``.py`` file rather than a Jupyter notebook (``.ipynb``).