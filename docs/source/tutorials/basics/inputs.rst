Inputs of MIDAS
===============

This section provides instructions on preparing and loading input data for MIDAS. 
The tool requires two primary components: multi-modal data and masks. 
Each modality's data for every batch is stored separately. 
The mask for each batch and modality indicates the presence (1) or absence (0) of features.

Inputs Overview
~~~~~~~~~~~~~~~

Data
^^^^

MIDAS supports multi-modal data inputs, including both paired and unpaired data. 
By default, it supports the following modalities:

- ``RNA``: integer, RNA counts
- ``ADT``: integer, Protein counts
- ``ATAC``: integer, ATAC peaks (binarized during training by setting transform={'atac': 'binarize'} in MIDAS.configure_data_from_dir())

For custom modalities, refer to the `Advanced Development Instructions` in the `Tutorials/Advanced section` to configure additional data types.

Mask
^^^^

Masks indicate the presence ``1`` or absence ``0`` of features for each modality in every batch.

- **Mask Format**:

    Each batch and modality should have a corresponding CSV mask file with the following structure:

    - Shape: ``1 x m`` (``1`` row and ``m`` features).
    - Includes: A header and an index column.

- **Default Behavior**:

    If no mask file is provided, MIDAS assumes that all features are present.

Loading Inputs 
~~~~~~~~~~~~~~
To format your processed data, ensuring that the features are consistently aligned across batches, 
organize it by batch and modality in the following directory structure:

.. code-block:: bash
    
    ./dataset_path/
        batch_0/
            mask/
                mod1.csv
                mod2.csv
                ...
            data_input/ # This will either be in mat or vec format
        batch_1/
            ...
        feat/
            feat_dims.toml
            
The ``data_input`` directory will contain files in either **matrix** or **vector** format:

If in **matrix** format ``mat``, the files within the ``data_input`` directory can be in formats such as ``.mtx`` or ``.csv`` for each batch and modality.

If in **vector** format ``vec``, the files within the ``data_input`` directory should be ``.csv`` files for each cell and modality.

The ``feat_dims.toml`` file contains the feature dimensions for each modality, as shown below:

.. code-block:: ini  

    rna = [1000]
    adt = [100]
    atac = [100, 100, 100] # Truncated based on the number of peaks for each chromosome.


``.mtx`` matrix (default)
^^^^^^^^^^^^^^^^^^^^^^^^^

To input a sparse matrix stored in the ``.mtx`` format:

.. code-block:: bash
    
    ./dataset_path/batch_0/mat/
        mod1.mtx
        mod2.mtx

Use the following code to load the data:

.. code-block:: python

    model = MIDAS.configure_data_from_dir(configs, dataset_path, format='mtx')

``.csv`` matrix
^^^^^^^^^^^^^^^

To input data stored in the ``.csv`` format (with headers and column names):

.. code-block:: bash

    ./dataset_path/batch_0/mat/
        mod1.csv
        mod2.csv

Use the following code to load the data:

.. code-block:: python

    model = MIDAS.configure_data_from_dir(configs, dataset_path, format='csv')

``.csv`` vector
^^^^^^^^^^^^^^

To structure your data in a directory as follows:

.. code-block:: bash

    ./dataset_path/batch_0/vec/
        mod1/
            0000.csv  # Represents a vector of a cell (no header or index)
            0001.csv
            ...
        mod2/
            0000.csv
            0001.csv
            ...

Use the following code to load the data:

.. code-block:: python

    model = MIDAS.configure_data_from_dir(configs, dataset_path, format='vec')


On Choosing Input Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``.csv`` Matrix:  
  High memory usage with low I/O demands. Use this format to accelerate training when memory is abundant.

- ``.mtx`` Matrix:  
  Moderate memory usage with low I/O demands. However, it requires extra time for conversion to a dense format. 
  Choose this format when memory is moderate, as it provides a balance between memory usage and time consumption.

- ``.csv`` Vector:  
  Low memory usage with high I/O demands. Use this format when memory is constrained, 
  but be mindful that frequent I/O operations may increase processing time.