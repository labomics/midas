Quick Start
===========

Configure and Train a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Set Up the Environment
------------------------------

Begin by importing the necessary modules:

.. code-block:: python
   
    from scmidas.model import MIDAS
    from scmidas.config import load_config
    import lightning as L
    from lightning.pytorch import loggers as pl_loggers

Step 2: Configure the Model
---------------------------
You can set up the MIDAS model using one of the following methods, depending on the format and organization of your input data.

Option 1: Configure from CSV Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data is stored as CSV files, you can configure the model as follows:

.. code-block:: python

    data = [
        {'mod1': '1.csv', 'mod2': '2.csv'},
        {'mod1': '3.csv', 'mod3': '4.csv'},
    ]
    transform = {'mod1': 'binarize'}
    # Masks for specific modalities
    mask = {'mod1': 'mask1.csv', 'mod3': 'mask3.csv'}
    # Set up the dimensions for each modalities
    dims_x = {'mod1':[200], 'mod2':[200], 'mod3':[100, 200, 300]}
    configs = load_config()
    # Configure the model with data, masks, and transformations
    datasets, dims_s, s_joint, combs = MIDAS.configure_data_from_csv(data, mask, transform)
    model = MIDAS.configure_data(configs, datasets, dims_x, dims_s, s_joint, combs)

.. note::
    1. This method is efficient as it avoids re-fetching data, making it well-suited for datasets that can fit into memory. For larger datasets that cannot be fully loaded, consider Option 2, which allows loading one sample at a time instead of the entire dataset.
    2. For modalities in `dims_x` with a length greater than 1, the data will be split into chunks based on these dimensions. For high-dimensional data, such as ATAC-seq, this can involve splitting by chromosomes.


Option 2: Configure from Each Modality's Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If each modality's data is stored in separate directories (with each sample saved as a CSV file):

.. code-block:: python

    data = [
        {'mod1': '1/', 'mod2': '2/'},
        {'mod1': '3/', 'mod3': '4/'},
    ]
    transform = {'mod1': 'binarize'}
    # Masks for specific modalities
    mask = {'mod1': 'mask1.csv', 'mod3': 'mask3.csv'}
    # Set up the dimensions for each modalities
    dims_x = {'mod1':[200], 'mod2':[200], 'mod3':[100, 200, 300]}
    configs = load_config()
    # Configure the model with data, masks, and transformations
    datasets, dims_s, s_joint, combs = MIDAS.configure_data_from_csv(data, mask, transform)
    model = MIDAS.configure_data(configs, datasets, dims_x, dims_s, s_joint, combs)

.. note::
    Option 1 and Option 2 can be combined for greater flexibility in handling your data.

Option 3: Configure from a Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data is organized in a directory with standard MIDAS input formats:

.. code-block:: python

    task = 'path/to/directory_containing_MIDAS_inputs'
    transform = {'mod': 'binarize'}  # Example of a transformation
    configs = load_config()
    model = MIDAS.configure_data_from_dir(configs, task, transform)


Step 3: Configure the Training Process
--------------------------------------

Set up a trainer to handle the training workflow. This includes configuring logging for real-time monitoring of the training process:

.. code-block:: python

    # Initialize TensorBoard logging
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='./logs/', 
        version='task_version'  # Replace with a descriptive version name
    )

    # Configure the trainer for single-device training
    trainer = L.Trainer(
        accelerator='auto',            # Automatically select accelerator (CPU/GPU)
        devices=1,                     # Use a single device
        precision=32,                  # Set precision to 32-bit
        strategy='auto',               # Automatically select training strategy
        num_nodes=1,                   # Use a single node
        max_epochs=2000,               # Maximum number of epochs
        logger=tb_logger,              # Attach the logger
        log_every_n_steps=5            # Log metrics every 5 steps
    )

Step 4: Train the Model
-----------------------

Once the model and trainer are configured, you can begin training:

.. code-block:: python

    trainer.fit(model=model)

Step 5: Infer with the Model
----------------------------

After training, you can use the `predict` method to generate and save predictions. Here's the syntax:

.. code-block:: python

    model.predict(output_dir,   
            joint_latent=True,
            mod_latent=True,
            impute=True,
            batch_correct=True,
            translate=True,
            input=True)

- pred_dir: The directory where prediction results will be saved.
- joint_latent: Whether to calculate and save joint latent representations (combined features from all modalities).
- mod_latent: Whether to calculate and save modality-specific latent representations (features for each individual modality).
- impute: Whether to perform data imputation, filling in missing or incomplete data.
- batch_correct: Whether to apply batch correction to the data to reduce batch effects.
- translate: Whether to perform modality translation (i.e., transforming data between different modalities).
- input : Whether to save the original input data. Note that if you’ve configured any transformations (e.g., `binarize`), the saved input data may differ from the original data in the file.

Step 6: Fetch Outputs
---------------------

To retrieve and load the predicted outputs, you can use the `load_predicted` function from the `scmidas.utils` module. Here's how to do it:

.. code-block:: python

    from scmidas.utils import load_predicted
    load_predicted('./predict/'+task, 
                    model.s_joint, 
                    model.combs, 
                    model.mods, 
                    joint_latent=True, 
                    mod_latend=True, 
                    impute=True, 
                    batch_correct=True, 
                    translate=True, 
                    input=True)


This function will load and return the predicted results, which can then be used for further analysis or visualization.

Visualize the Training Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you should pass a logger into the model:

.. code-block:: python

    # Initialize TensorBoard logging
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='./logs/', 
        version='task_version'  # Replace with a descriptive version name
    )

    # Configure the trainer for single-device training
    trainer = L.Trainer(
        logger=tb_logger,              # Attach the logger
    )

To monitor the training progress, run the following command in your terminal:

.. code-block:: python

    tensorboard --logdir './logs/lightning_logs'


Then, open the URL displayed in your terminal (e.g., `http://localhost:6006`) in a web browser to visualize the training metrics and results.

Train with Multi-GPU
~~~~~~~~~~~~~~~~~~~~

To enable training on multiple GPUs, create a `.py` file and modify the `strategy` parameter to `'ddp'` (Distributed Data Parallel):

.. code-block:: python
   :emphasize-lines: 5

    trainer = L.Trainer(
        accelerator='auto',            # Automatically select accelerator (CPU/GPU)
        devices='auto',                # Automatically use all available GPUs
        precision=32,                  # Set precision to 32-bit
        strategy='ddp',                # Enable distributed training with DDP
        num_nodes=1,                   # Use a single node
        max_epochs=2000,               # Maximum number of epochs
        logger=tb_logger,              # Attach the logger
        log_every_n_steps=5            # Log metrics every 5 steps
    )

.. note::
    1. Use a meaningful `version` name in `TensorBoardLogger` to differentiate between experiments.
    2. Set `devices='auto'` to utilize all available GPUs automatically. Alternatively, specify the exact number of GPUs by setting `devices=2` (for 2 GPUs).
    3. Use `'ddp'` for multi-GPU training on a single node.
