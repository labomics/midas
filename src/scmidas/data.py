import math
import random
import numpy as np
import pandas as pd
import os
from typing import Iterator, Optional, TypeVar, Any, Dict

import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Sampler

from .nn import transform_registry
from .utils import load_csv

_T_co = TypeVar('_T_co', covariant=True)


class BasicModDataset(Dataset):
    """
    Base class for modular datasets.
    """

    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int
                Number of samples (default is 0 for base class).
        """
        return 0

    def __getitem__(self, idx: int) -> None:
        """
        Retrieve the data item at the specified index (not implemented in base class).

        Parameters:
            idx : int
                The index of the data item.

        Returns:
            None
        """
        return None


class VecDataset(BasicModDataset):
    """
    Dataset for vector-based data.

    Parameters:
        path : str
            Directory containing vector-based data files.
    """

    def __init__(self, path: str):
        super().__init__()
        self.root = path
        self.data_path = sorted(os.listdir(path))

    def __len__(self) -> int:
        """
        Return the number of files in the vector dataset.

        Returns:
            int
                Number of vector files in the dataset.
        """
        return len(self.data_path)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the vector data at the specified index.

        Parameters:
            idx : int
                The index of the vector file.

        Returns:
            np.ndarray
                The vector data as a NumPy array.
        """
        vector_data = np.array(
            load_csv(os.path.join(self.root, self.data_path[idx])), dtype=np.float32
        )[0]
        return vector_data


class MatDataset(BasicModDataset):
    """
    Dataset for matrix-based data.

    Parameters:
        csv_file : str
            Path to the CSV or compressed CSV file.
    """

    def __init__(self, csv_file: str):
        super().__init__()
        if csv_file.endswith('.csv'):
            # Load CSV into a NumPy array
            self.data_frame = np.array(load_csv(csv_file))[1:, 1:].astype(np.float32)
        elif csv_file.endswith('.csv.gz'):
            # Load compressed CSV using pandas
            self.data_frame = pd.read_csv(csv_file, index_col=0).values.astype(np.float32)
        else:
            raise ValueError(f'Unsupported file format: {csv_file}')

    def __len__(self) -> int:
        """
        Return the number of rows in the matrix dataset.

        Returns:
            int
                Number of rows in the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the matrix row at the specified index.

        Parameters:
            idx : int
                The index of the matrix row.

        Returns:
            np.ndarray
                The matrix row as a NumPy array.
        """
        return self.data_frame[idx]

modDataset_map = {'vec': VecDataset, 'mat': MatDataset}


class MultiModalDataset(Dataset):
    """
    A dataset class for handling multi-modal data with optional masking and transformations.

    Parameters:
        mod_dict : Dict[str, str]
            A dictionary mapping modality names to their respective file paths.
        mod_id_dict : Dict[str, int]
            A dictionary mapping modality names to their unique identifiers.
        file_type : Dict[str, str]
            A dictionary mapping modality names to their file types (e.g., 'vec', 'mat').
        mask_path : Optional[Dict[str, str]], optional
            A dictionary mapping modality names to their mask file paths, default is None.
        transform : Optional[Dict[str, str]], optional
            A dictionary specifying transformations to apply to each modality, default is None.

    Methods:
        __len__():
            Returns the size of the dataset.
        __getitem__(idx: int) -> Dict[str, Dict[str, Any]]:
            Retrieves the data at the given index across all modalities.
    """

    def __init__(
        self,
        mod_dict: Dict[str, str],
        mod_id_dict: Dict[str, int],
        file_type: Dict[str, str],
        mask_path: Optional[Dict[str, str]] = None,
        transform: Optional[Dict[str, str]] = None,
    ):
        self.mod_dict = mod_dict
        self.mod_id_dict = mod_id_dict
        self.data = {
            modality: modDataset_map[file_type[modality]](path)
            for modality, path in self.mod_dict.items()
        }
        self.mask = (
            {
                modality: np.array(load_csv(mask_path[modality])[1][1:]).astype(np.float32)
                for modality in mask_path
            }
            if mask_path
            else None
        )
        self.transform = transform or {}
        self.size = len(next(iter(self.data.values())))  # Determine dataset size from the first modality

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int
                The number of samples in the dataset.
        """
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves the data at the specified index across all modalities.

        Parameters:
            idx : int
                The index of the sample to retrieve.

        Returns:
            Dict[str, Dict[str, Any]]:
                A dictionary containing the following keys:
                - 'x': Modality data at the given index, with optional transformations applied.
                - 's': Modality IDs.
                - 'e': Masking information, if available.
        """
        items = {'x': {}, 's': {}, 'e': {}}

        # Retrieve data for each modality
        for modality, dataset in self.data.items():
            # Get raw data
            items['x'][modality] = dataset[idx]

            # Apply transformation if specified
            if modality in self.transform:
                transform_fn = transform_registry.get(self.transform[modality])
                items['x'][modality] = transform_fn(items['x'][modality])

            # Store modality ID
            items['s'][modality] = np.array([self.mod_id_dict[modality]], dtype=np.int64)

        # Add joint ID
        items['s']['joint'] = np.array([self.mod_id_dict['joint']], dtype=np.int64)

        # Add masking information if available
        if self.mask:
            for modality, mask_data in self.mask.items():
                items['e'][modality] = mask_data

        return items


class MultiBatchSampler(Sampler):
    """
    Custom sampler for multi-batch sampling across multiple datasets.

    Parameters:
        data_source : Any
            A dataset or a concatenated dataset (e.g., ConcatDataset) containing multiple sub-datasets.
        shuffle : bool, optional
            Whether to shuffle the samples within each dataset, default is True.
        batch_size : int, optional
            Number of samples per batch, default is 1.
        n_max : int, optional
            Maximum number of samples to draw from each dataset, default is 10000.
    """

    def __init__(
        self,
        data_source: Optional[Any] = None,
        shuffle: bool = True,
        batch_size: int = 1,
        n_max: int = 10000,
    ):
        super().__init__(data_source)
        if not hasattr(data_source, 'datasets') or not hasattr(data_source, 'cumulative_sizes'):
            raise ValueError('Data source must be a ConcatDataset or equivalent.')

        self.data = data_source
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_dataset = len(self.data.datasets)
        self.n_max = min(max(len(d) for d in self.data.datasets), n_max)
        self.Sampler = (
            torch.utils.data.RandomSampler if shuffle else torch.utils.data.SequentialSampler
        )

    def __len__(self) -> int:
        """
        Calculate the total number of samples across all sub-datasets.

        Returns:
            int
                The total number of samples.
        """
        return math.ceil(self.n_max / self.batch_size) * self.batch_size * self.n_dataset

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dataset indices in a multi-batch sampling manner.

        Returns:
            Iterator[int]
                An iterator over sampled indices.
        """
        # Number of iterations per dataset
        n_iter = math.ceil(self.n_max / self.batch_size)

        # Create individual samplers and iterators for each dataset
        sampler_indv = [
            self.Sampler(self.data.datasets[idx]) for idx in range(self.n_dataset)
        ]
        sampler_iter_indv = [iter(s) for s in sampler_indv]

        # Cumulative sizes for offset indexing
        push_index_val = [0] + self.data.cumulative_sizes[:-1]
        idx_dataset = list(range(self.n_dataset))

        indices = []
        for _ in range(n_iter):
            # Shuffle dataset order if required
            if self.shuffle:
                random.shuffle(idx_dataset)

            for i in idx_dataset:
                s = sampler_iter_indv[i]
                indices_indv = []
                for _ in range(self.batch_size):
                    try:
                        indices_indv.append(next(s) + push_index_val[i])
                    except StopIteration:
                        # Restart sampler iterator if exhausted
                        sampler_iter_indv[i] = iter(sampler_indv[i])
                        s = sampler_iter_indv[i]
                        indices_indv.append(next(s) + push_index_val[i])
                indices.extend(indices_indv)

        return iter(indices)


class MyDistributedSampler(DistributedSampler):
    """
    A custom distributed sampler for datasets split across multiple replicas.

    Parameters:
        dataset : Dataset
            The dataset to sample from.
        num_replicas : int, optional
            Number of replicas in the distributed setup, default is determined by `torch.distributed`.
        rank : int, optional
            The rank of the current process, default is determined by `torch.distributed`.
        shuffle : bool, optional
            Whether to shuffle the data, default is True.
        seed : int, optional
            Random seed for shuffling, default is 0.
        batch_size : int, optional
            Number of samples per batch, default is 256.
        n_max : int, optional
            Maximum number of samples per dataset, default is 10000.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 256,
        n_max: int = 10000,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]'
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.n_dataset = len(self.dataset.datasets)
        self.n_sample = [len(d) // num_replicas for d in self.dataset.datasets]
        self.batch_size = batch_size
        self.n_max = n_max

        # Cumulative dataset sizes for indexing
        self.push_index_val = [0] + self.dataset.cumulative_sizes
        self.all_indices = []
        self.all_length = []

        # Generate indices for each dataset
        for idx in range(self.n_dataset):
            indices = list(
                range(
                    self.rank + self.push_index_val[idx],
                    self.push_index_val[idx + 1],
                    self.num_replicas,
                )
            )
            self.all_indices.append(indices)
            self.all_length.append(len(indices))

    def __iter__(self) -> Iterator[_T_co]:
        """
        Iterate over the distributed dataset, ensuring balanced sampling across replicas.

        Returns:
            Iterator
                Iterator over indices for the current replica.
        """
        sampler_indv = []
        sampler_iter_indv = []
        n_sample_by_dataset = []

        # Prepare samplers for each dataset
        for idx in range(self.n_dataset):
            indices = self.all_indices[idx]
            if self.shuffle:
                random.shuffle(indices)
            indices = indices[: self.n_max]
            sampler_indv.append(indices)
            sampler_iter_indv.append(iter(indices))
            n_sample_by_dataset.append(len(indices))

        n_iter = math.ceil(max(n_sample_by_dataset) / self.batch_size) * self.n_dataset

        idx_dataset = list(range(self.n_dataset))
        indices = []

        # Main sampling loop
        for _ in range(n_iter):
            random.shuffle(idx_dataset)  # Shuffle dataset order
            for i in idx_dataset:
                s = sampler_iter_indv[i]
                order_indv = []
                for _ in range(self.batch_size):
                    try:
                        order_indv.append(next(s))
                    except StopIteration:
                        sampler_iter_indv[i] = iter(sampler_indv[i])
                        s = sampler_iter_indv[i]
                        order_indv.append(next(s))
                indices.extend(order_indv)

        return iter(indices)

    def __len__(self) -> int:
        """
        Calculate the number of samples in the sampler.

        Returns:
            int
                Number of samples across all datasets.
        """
        max_samples = min(max(self.all_length), self.n_max)
        return math.ceil(max_samples / self.batch_size) * self.n_dataset * self.batch_size

def download_file(url: str, dest_path: Path):
    """Helper function to download a file from a URL with progress display.
    
    Parameters:
        url : str
            URL for data.
        dest_path : str
            Path to save.
    """
    try:
        # Send HTTP GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the total size of the file from headers
        total_size = int(response.headers.get('Content-Length', 0))
        
        # Open the destination file in write-binary mode
        with open(dest_path, 'wb') as file:
            # Use tqdm to display download progress
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {dest_path.name}') as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))  # Update progress bar with the downloaded chunk size
        logging.info(f'Downloaded: {url} to {dest_path}')

    except requests.exceptions.RequestException as e:
        logging.error(f'Error downloading {url}: {e}')
        raise

def unzip_file(zip_path: Path, extract_to: Path):
    """Helper function to unzip a file.

    Parameters:
        zip_path : str
            Path of zip file.
        extract_to : str
            Path to save.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f'Unzipped: {zip_path} to {extract_to}')
    except zipfile.BadZipFile as e:
        logging.error(f'Error unzipping {zip_path}: {e}')
        raise

def download_data(name: str, des: str = './'):
    """
    Downloads the specified dataset and extracts it.

    Parameters:
        name : str
            Name of the dataset to download (e.g., 'teadog_mosaic_4k').
        des : str
            Destination path to save the dataset (default is the current directory).
    """
    # Set up the destination path
    des_path = Path(des) / 'dataset'
    des_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    urls_dict = {
        'teadog_mosaic_4k' : [('https://drive.usercontent.google.com/download?id=1MQtg5CHV3KDsmbRowiNnggKImYazBpOi&export=download&authuser=0&confirm=t&uuid=840e8dbf-6a9b-407f-89fe-cc5c82debc8a&at=APvzH3omA-S-4W1YkjAlCvyM6EuX:1733823042031', 
                 des_path / 'teadog_mosaic_4k.zip')],
        'wnn_mosaic_3batch' : [('https://drive.usercontent.google.com/download?id=11a62mlJ4tbqPMM7y6iF9XfMxeWMFqc-7&export=download&authuser=0&confirm=t&uuid=f6efdc19-ba0b-448a-bfa1-ab65a9784bee&at=APvzH3rBWhgaiST18uqbTjSu6uo4:1734661218069', des_path / 'wnn_mosaic_3batch.zip')],
        'wnn_full_3batch' : [('https://drive.usercontent.google.com/download?id=1W3ZkU8TWzlPcCuqlGvptfH_PnHjvWI4u&export=download&authuser=0&confirm=t&uuid=015fddd9-a789-4bc7-8fda-3f4ef202811a&at=APvzH3rhfWzjXrlKJedDEBGzhsXm:1734661020282', des_path / 'wnn_full_3batch.zip')],
        'wnn_full_8batch' : [('https://drive.usercontent.google.com/download?id=1kzlSd6iAM2UHifvlzu0OYbpq_MLPomrx&export=download&authuser=0&confirm=t&uuid=79c4ce32-18ca-4ba3-bbbd-e1c955ab1064&at=APvzH3q3nmmKLDSI1SNtF1CGNbnn:1734661120552', des_path / 'wnn_full_8batch.zip')],
    }

    if name in urls_dict:
        try:
            # Download and extract the TEADOG mosaic dataset
            urls = urls_dict[name]
            for url, file_path in urls:
                download_file(url, file_path)
                if file_path.suffix == '.zip':
                    unzip_file(file_path, des_path)
                    os.remove(file_path)
        except Exception as e:
            logging.error(f'An error occurred while downloading the dataset: {e}')
            raise

    else:
        logging.error(f'Dataset "{name}" is not recognized.')
        raise ValueError(f'Dataset "{name}" not supported.')
    