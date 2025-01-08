Inputs of MIDAS
===============

This section explains how to prepare and load input data for MIDAS. 
The tool requires two main components as input: multi-modal data and masks. 
Data for each modality is stored separately, 
while the mask indicates the presence or absence of features.

Inputs Overview
~~~~~~~~~~~~~~~~~~~

Data
^^^^

MIDAS supports multi-modal data inputs, 
including both paired and unpaired data. 
By default, it supports the following modalities:

- ``RNA``: RNA counts,  integer values.
- ``ADT``: Protein counts, integer values.
- ``ATAC``: ATAC peaks, integer values (binarized during training).

For custom modalities, refer to the `Advanced Development Instructions` in the `Tutorials/Advanced section` to configure additional data types.

Mask
^^^^

Masks indicate the presence (``1``) or absence (``0``) of features in each modality for every batch.

- **Mask Format**:
    Each batch and modality should have a corresponding CSV mask file. The CSV file must have the following structure:

    - Shape: ``1 x m`` (1 row and ``m`` columns, where ``m`` is the number of features).
    - Includes: A header and an index column.

- **Default Behavior**:
    If no mask file is provided, MIDAS assumes that all features are present.

To specify mask files, provide the paths in the format below:

.. code-block:: python

    mask_config = [
        {'rna': 'batch_1_rna_mask.csv', 'adt': 'batch_1_adt_mask.csv'},
        {'rna': 'batch_2_rna_mask.csv', 'adt': 'batch_2_adt_mask.csv'},
        {'rna': 'batch_3_rna_mask.csv', 'adt': 'batch_3_adt_mask.csv'}
    ]


Initial Setup
~~~~~~~~~~~~~~~

Begin by importing the necessary modules and loading default configurations:

.. code-block:: python  

    from scmidas.model import MIDAS
    from scmidas.config import load_config

    # Settings for the model, such as the layer dimensions.
    configs = load_config()

Approach 1: Loading Data from Single-level Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the steps below to prepare and load input data into MIDAS.

Step 1: Configuring Data
^^^^^^^^^^^^^^^^^^^^^^^^^

MIDAS supports two main formats for loading data. Choose the one that best suits your dataset's size and structure.

Option 1: CSV per Modality and Batch
""""""""""""""""""""""""""""""""""""

- **Description**:
    Data for each modality and batch is stored in separate CSV files. Each file represents a ``cell x feature`` matrix, where:
    
    - Rows: Cells
    - Columns: Features
    - The file includes a header and an index column.

- **Example Configuration**:
    .. code-block:: python  

        # Data for each modality and batch
        data_config = [
            {'rna': 'batch_1_rna.csv', 'adt': 'batch_1_adt.csv', 'atac': 'batch_1_atac.csv'},
            {'rna': 'batch_2_rna.csv', 'adt': 'batch_2_adt.csv', 'atac': 'batch_2_atac.csv'},
            {'rna': 'batch_3_rna.csv', 'adt': 'batch_3_adt.csv', 'atac': 'batch_3_atac.csv'}
        ]
- **Use Case**:
    This format is suitable when datasets fit into memory, as it avoids re-fetching data.


Option 2: CSV per Cell
""""""""""""""""""""""""""""""""""""

- **Description**:
    Data for each cell is stored in individual CSV files. Each file contains a ``1 x feature`` vector without a header or index column.

- **Example Configuration**:
    .. code-block:: python

        # Directory paths for each modality and batch
        data_config = [
            {'rna': 'batch_1_rna_dir/', 'adt': 'batch_1_adt_dir/', 'atac': 'batch_1_atac_dir/'},
            {'rna': 'batch_2_rna_dir/', 'adt': 'batch_2_adt_dir/', 'atac': 'batch_2_atac_dir/'},
            {'rna': 'batch_3_rna_dir/', 'adt': 'batch_3_adt_dir/', 'atac': 'batch_3_atac_dir/'}
        ]

- **Use Case**:
    This format is ideal for large datasets that cannot fit into memory, as it allows loading data one sample at a time.

.. tip::
    Both **Option 1** and **Option 2** can be combined for flexible data handling.

Step 2: Defining Data Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the dimensions for each modality. 
Example:

.. code-block:: python  

    # Dimensions per modality.
    # In this example, the ATAC data is split into chunks during training 
    # based on the specified dimensionality.
    dims_x = {
        'rna': [200],    # RNA data is represented as a cell x 200 matrix.
        'adt': [200],    # ADT data is represented as a cell x 100 matrix.
        'atac': [100, 200, 300, ..., 200]  # ATAC data is split into multiple chunks with varying dimensions
    }

.. note::

    For modalities with more than one dimension (e.g., ``ATAC``), 
    data will be split into chunks based on the specified dimensions. 
    This is useful for high-dimensional data like ATAC-seq, 
    where splitting occurs based on chromosomes.

Step 3: Specifying Transformation Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For certain modalities, 
you may want to apply transformations. 
In this example, we binarize the ATAC data and leave RNA and ADT data unchanged:

.. code-block:: python  

    transform = {'atac': 'binarize'}  # Binarize ATAC data, leave RNA and ADT unchanged

Step 4: Combining Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate the configurations, data paths, and transformations to set up MIDAS:

.. code-block:: python    

    # Configure MIDAS with the data
    datasets, dims_s, s_joint, combs = MIDAS.configure_data_from_csv(data_config, mask_config, transform)
    model = MIDAS.configure_data(configs, datasets, dims_x, dims_s, s_joint, combs)

Approach 2: Loading Data from Multi-level Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the previously mentioned method of loading data from single-level directories 
(where each path corresponds to an independent directory containing CSV files), 
MIDAS also supports loading data directly from a well-organized, multi-level directory structure. 
The required directory format is as follows:

.. code-block:: plaintext

    ./dataset_path/
        batch_0/
            mask/
                rna.csv
                adt.csv
            vec/
                rna/
                    0000.csv
                    0001.csv
                    ...
                adt/
                    0000.csv
                    0001.csv
                    ...
                atac/
                    0000.csv
                    0001.csv
                    ...
        batch_1/
            ...
        feat/
            feat_dims.toml

- ``mask``: Contains mask files for each modality.
- ``vec``: Contains cell-specific data files for each modality.
- ``feat/feat_dims.toml``: Specifies feature dimensions for each modality. Example:

.. code-block:: python

    rna = [200]
    adt = [100]
    atac = [100, 200, 300, ..., 200]

To load data from this structure, use the ``configure_data_from_dir()`` function:

.. code-block:: python

    # Load dataset using the directory structure
    model = MIDAS.configure_data_from_dir(configs, dataset_path, transform)
