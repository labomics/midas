import math
from pathlib import Path
import random
import numpy as np
import pandas as pd
import os
from typing import Iterator, Optional, TypeVar, Any, Dict

from scipy.io import mmread
from scipy.sparse import csr_matrix

import zipfile
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
    Base class for modality data.
    """

    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int:
                Number of samples (default is 0 for base class).
        """
        return 0

    def __getitem__(self, idx: int):
        """
        Retrieve the data item at the specified index (not implemented in base class).

        Parameters:
            idx : int
                The index of the data item.
        """
        return None


class VECDataset(BasicModDataset):
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
            int:
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
            np.ndarray:
                The vector data as a NumPy array.
        """
        vector_data = np.array(
            load_csv(os.path.join(self.root, self.data_path[idx])), dtype=np.float32
        )[0]
        return vector_data


class MTXDataset(BasicModDataset):
    """
    Dataset for mtx-based data.

    Parameters:
        mtx_file : str
            Path to the mtx file.
    """
    def __init__(self, mtx_file: str):
        super().__init__()
        if mtx_file.endswith('.mtx'):
            self.data = csr_matrix(mmread(mtx_file))
        else:
            raise ValueError(f'Unsupported file format: {mtx_file}')
    def get_all(self) -> np.ndarray:
        """
        Return all data in the dataset as a NumPy array.

        Returns:
            np.ndarray:
                All data in the dataset as a NumPy array.
        """
        return self.data.toarray().astype(np.float32)
    def __len__(self) -> int:
        """
        Return the number of rows in the matrix dataset.

        Returns:
            int:
                Number of rows in the dataset.
        """
        return self.data.shape[0]
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve the matrix row at the specified index.

        Parameters:
            idx : int
                The index of the matrix row.

        Returns:
            np.ndarray:
                The matrix row as a NumPy array.
        """
        return self.data.getrow(idx).toarray()[0].astype(np.float32)
    

class CSVDataset(BasicModDataset):
    """
    Dataset for csv-based data.

    Parameters:
        csv_file : str
            Path to the CSV or compressed CSV file (csv.gz).
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
            int:
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
            np.ndarray:
                The matrix row as a NumPy array.
        """
        return self.data_frame[idx]


modDataset_map = {'vec': VECDataset, 'csv': CSVDataset, 'mtx': MTXDataset}


class MultiModalDataset(Dataset):
    """
    A dataset class for handling multi-modal data with optional masking and transformations.

    Parameters:
        mod_dict : Dict[str, str]
            A dictionary mapping modality names to their respective file paths.
        mod_id_dict : Dict[str, int]
            A dictionary mapping modality names to their unique identifiers.
        file_type : Dict[str, str]
            A dictionary mapping modality names to their file types (e.g., 'vec', 'csv', 'mtx').
        mask_path : Optional[Dict[str, str]]
            A dictionary mapping modality names to their mask file paths, default is None.
        transform : Optional[Dict[str, str]]
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
            int:
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
        data_source : Dataset
            Dataset.
        shuffle : bool
            Whether to shuffle the samples within each dataset, default is True.
        batch_size : int
            Number of samples per batch, default is 1.
        n_max : int
            Maximum number of samples to draw from each dataset, default is 10000.
    """

    def __init__(
        self,
        data_source: Dataset,
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
            int:
                The total number of samples.
        """
        return math.ceil(self.n_max / self.batch_size) * self.batch_size * self.n_dataset

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dataset indices in a multi-batch sampling manner.

        Returns:
            Iterator[int]:
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
        num_replicas : Optional[int]
            Number of replicas in the distributed setup, default is determined by `torch.distributed`.
        rank : Optional[int]
            The rank of the current process, default is determined by `torch.distributed`.
        shuffle : bool
            Whether to shuffle the data, default is True.
        seed : int
            Random seed for shuffling, default is 0.
        batch_size : int
            Number of samples per batch, default is 256.
        n_max : int
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
            Iterator:
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
            int:
                Number of samples across all datasets.
        """
        return sum(self.all_length)
        # max_samples = min(max(self.all_length), self.n_max)
        # return math.ceil(max_samples / self.batch_size) * self.n_dataset * self.batch_size

def download_file(url: str, dest_path: str):
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

def unzip_file(zip_path: str, extract_to: str):
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

def download_models(name: str, des: str = './'):
    """
    Downloads the specified model.

    Parameters:
        name : str
            Name of the model to download (e.g., 'wnn_mosaic_8batch_mtx').
        des : str
            Destination path to save the model (default is the current directory).
    """
    # Set up the destination path
    des_path = Path(des) / 'models'
    des_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    urls_dict = {
        'wnn_mosaic_8batch_mtx' : [('https://drive.usercontent.google.com/download?id=12gqdg12Nb3tXOx82OtxwKHt6bPwRuGFl&export=download&authuser=0&confirm=t&uuid=675566a6-d782-4586-83ea-d994bd277dee&at=ALoNOgnU0Twolb8wYDEQ72MYJBt5:1749205390939', des_path / 'wnn_mosaic_8batch_mtx.pt')],
        'wnn_full_8batch_mtx' : [('https://drive.usercontent.google.com/download?id=1nI3TVPkvF8uu8PGnxRKqSVrZCGLnz7ZC&export=download&authuser=0&confirm=t&uuid=3d85e299-401f-4591-af6e-e9b4741f0aed&at=ALoNOgl9EvZwibld2GcYsK7LVf1C:1749205432077', des_path / 'wnn_full_8batch_mtx.pt')],
        'teadog_mosaic_mtx' : [('https://drive.usercontent.google.com/download?id=1zU9E9OtQaZMGJKSy_4r0kRlzoSf7ojMn&export=download&authuser=0&confirm=t&uuid=afc6a0c2-d155-4e46-be9f-a1f2b8b21228&at=ALoNOgkzwD8Q5MCX0MFIC7ggnUHU:1749215287513', des_path / 'teadog_mosaic_mtx.pt')],
    }

    if name in urls_dict:
        try:
            # Download and extract the TEADOG mosaic dataset
            urls = urls_dict[name]
            for url, file_path in urls:
                logging.info(f'Downloading from {url}.')
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
        'teadog_mosaic_4k' : [('https://drive.usercontent.google.com/download?id=1MQtg5CHV3KDsmbRowiNnggKImYazBpOi&export=download&authuser=0&confirm=t&uuid=840e8dbf-6a9b-407f-89fe-cc5c82debc8a&at=APvzH3omA-S-4W1YkjAlCvyM6EuX:1733823042031', des_path / 'teadog_mosaic_4k.zip')],
        'wnn_mosaic_3batch' : [('https://drive.usercontent.google.com/download?id=11a62mlJ4tbqPMM7y6iF9XfMxeWMFqc-7&export=download&authuser=0&confirm=t&uuid=f6efdc19-ba0b-448a-bfa1-ab65a9784bee&at=APvzH3rBWhgaiST18uqbTjSu6uo4:1734661218069', des_path / 'wnn_mosaic_3batch.zip')],
        'wnn_full_3batch' : [('https://drive.usercontent.google.com/download?id=1W3ZkU8TWzlPcCuqlGvptfH_PnHjvWI4u&export=download&authuser=0&confirm=t&uuid=015fddd9-a789-4bc7-8fda-3f4ef202811a&at=APvzH3rhfWzjXrlKJedDEBGzhsXm:1734661020282', des_path / 'wnn_full_3batch.zip')],
        'wnn_full_8batch' : [('https://drive.usercontent.google.com/download?id=1kzlSd6iAM2UHifvlzu0OYbpq_MLPomrx&export=download&authuser=0&confirm=t&uuid=79c4ce32-18ca-4ba3-bbbd-e1c955ab1064&at=APvzH3q3nmmKLDSI1SNtF1CGNbnn:1734661120552', des_path / 'wnn_full_8batch.zip')],
        'wnn_mosaic_8batch_mtx' : [('https://drive.usercontent.google.com/download?id=1tQ-EpP8Mbw8qCChC_LbXfBLh8GsfBGSw&export=download&authuser=0&confirm=t&uuid=eed8c863-2b77-466e-a7f8-272408486f7a&at=ALoNOgla8Tuk8CBkZGWbtwWra7S6:1749205613084', des_path / 'wnn_mosaic_8batch_mtx.zip')],
        'wnn_full_8batch_mtx' : [('https://drive.usercontent.google.com/download?id=1fztQVy9EU91KSsyXiBbZlvTSb6JEuoq4&export=download&authuser=0&confirm=t&uuid=25c02410-37ff-4598-9a6a-40a9a3d1c992&at=ALoNOglSyxEsCjo4C0WvJx5fNX6F:1749205646785', des_path / 'wnn_full_8batch_mtx.zip')],
        'teadog_mosaic_mtx' : [('https://drive.usercontent.google.com/download?id=1vkejT5Zj_QyZPMkVfxSkE-ICivb8Q5du&export=download&authuser=0&confirm=t&uuid=cfd4bc42-9d4d-4f93-a92a-a2ecf5893f41&at=ALoNOgnZu2B-9fgOqH2XuUDyUOeY:1749205575410', des_path / 'teadog_mosaic_mtx.zip')],
    }

    if name in urls_dict:
        try:
            # Download and extract the TEADOG mosaic dataset
            urls = urls_dict[name]
            for url, file_path in urls:
                logging.info(f'Downloading from {url}.')
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
    