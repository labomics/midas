# Standard Library Imports
import os
import csv
import shutil
import itertools
import math
from glob import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Any, Optional
import re
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
import anndata as ad
import mudata as mu
import scipy.sparse as sp

# Third-Party Library Imports
import numpy as np
import torch
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

def load_csv(filename: str) -> list:
    """
    Load a CSV file and return its contents as a list of rows.

    Parameters:
        filename : str
            Path to the CSV file.

    Returns:
        list:
            A list of rows, where each row is a list of strings.
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def exp(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute a numerically stable exponential transformation.

    Handles negative and positive values to avoid numerical instability.

    Parameters:
        x : torch.Tensor
            Input tensor.
        eps : float, optional
            A small epsilon value to avoid division by zero, by default 1e-12.

    Returns:
        torch.Tensor:
            Transformed tensor with the exponential applied.
    """
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute a numerically stable logarithm transformation.

    Ensures numerical stability by adding a small epsilon.

    Parameters:
        x : torch.Tensor
            Input tensor.
        eps : float, optional
            A small epsilon value to avoid log(0), by default 1e-12.

    Returns:
        torch.Tensor:
            Transformed tensor with the logarithm applied.
    """
    return (x + eps).log()


def extract_params(config: dict, prefix: str) -> dict:
    """
    Extract parameters from a configuration dictionary with a specific prefix.

    Removes the specified prefix from the keys in the resulting dictionary.

    Parameters:
    
        config : dict
            Configuration dictionary containing various parameters.
        prefix : str
            Prefix to filter and remove from the keys.

    Returns:
        dict:
            A new dictionary containing the filtered parameters with the prefix removed.
    """
    extracted_params = {}
    
    # Iterate over all key-value pairs in the config dictionary
    for key, value in config.items():
        if key.startswith(prefix):  # Filter keys starting with the specified prefix
            new_key = key[len(prefix):]  # Remove the prefix from the key
            extracted_params[new_key] = value
    
    return extracted_params


def ref_sort(x: List[str], ref: List[str]) -> List[str]:
    """
    Sort the elements of `x` based on the order defined in `ref`.

    Parameters:
    
        x : list of str
            List of elements to be sorted.
        ref : list of str
            Reference list defining the sort order.

    Returns:
        List[str]:
            A sorted list of elements from `x` that appear in `ref`, 
            maintaining the order of `ref`.
    """
    return [v for v in ref if v in x]


def extract_values(x: Union[List[Any], Tuple[Any], Dict[Any, Any], Any]) -> List[Any]:
    """
    Recursively extract all values from a tuple, list, or dictionary.

    Parameters:
    
        x : list, tuple, dict, or any type
            The input structure containing nested values.

    Returns:
        List[Any]:
            A flattened list of all values extracted from the input.
    """
    values = []

    # Handle list and tuple
    if isinstance(x, (list, tuple)):
        for v in x:
            values.extend(extract_values(v))  # Recursive extraction

    # Handle dictionary
    elif isinstance(x, dict):
        for v in x.values():
            values.extend(extract_values(v))  # Recursive extraction

    # Handle individual value
    else:
        values.append(x)

    return values

def reverse_dict(original_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Reverse the keys and sub-keys of a nested dictionary.

    Parameters:
        
        original_dict : Dict[str, Dict[str, Any]]
            The original nested dictionary to be reversed.

    Returns:
        Dict[str, Dict[str, Any]]:
            A reconstructed dictionary where the keys and sub-keys are swapped.
    """
    reconstructed_dict = defaultdict(dict)
    for key, subdict in original_dict.items():
        for sub_key, value in subdict.items():
            reconstructed_dict[sub_key][key] = value
    return dict(reconstructed_dict)


def detach_tensors(x: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively detach all tensors in a dictionary.

    Parameters:
    
        x : Dict[str, Any]
            Dictionary containing tensors or nested dictionaries.

    Returns:
        Dict[str, Any]:
            A new dictionary with all tensors detached.
    """
    y = {}
    for kw, arg in x.items():
        if torch.is_tensor(arg):
            y[kw] = arg.detach()
        else:
            y[kw] = detach_tensors(arg)
    return y


def convert_tensor_to_list(data: Union[torch.Tensor, List[List[Any]]]) -> List[List[Any]]:
    """
    Convert a 2D tensor or list into a 2D list.

    Parameters:
    
        data : Union[torch.Tensor, List[List[Any]]]
            Input data to be converted.

    Returns:
        List[List[Any]]:
            Converted 2D list.
    """
    if torch.is_tensor(data):
        return data.cpu().detach().numpy().tolist()
    else:
        return [list(line) for line in data]


def save_list_to_csv(data: List[List[Any]], filename: str, delimiter: str = ','):
    """
    Save a 2D list to a CSV file.

    Parameters:
    
        data : List[List[Any]]
            Input data to be saved.
        filename : str
            Path to the CSV file.
        delimiter : str
            Delimiter to separate values in the CSV file, by default ','.
    """
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)


def save_list_to_mtx(data: torch.Tensor, filename: str):
    """
    Save a 2D list or tensor to a Matrix Market (MTX) file.

    Parameters:
    
        data : torch.Tensor
            Input data to be saved.
        filename : str
            Path to the MTX file.
    """
    sparse_mtx = sp.csr_matrix(data.numpy(), shape=data.shape)
    mmwrite(filename, sparse_mtx)


def save_tensor_to_mtx(data: torch.Tensor, filename: str):
    """
    Save a 2D tensor to a Matrix Market (MTX) file.

    Parameters:
    
        data : torch.Tensor
            Input tensor to be saved.
        filename : str
            Path to the MTX file.
    """
    save_list_to_mtx(data, filename)


def save_tensor_to_csv(data: torch.Tensor, filename: str, delimiter: str = ','):
    """
    Save a 2D tensor to a CSV file.

    Parameters:
    
        data : torch.Tensor
            Input tensor to be saved.
        filename : str
            Path to the CSV file.
        delimiter : str, optional
            Delimiter to separate values in the CSV file, by default ','.
    """
    data_list = convert_tensor_to_list(data)
    save_list_to_csv(data_list, filename, delimiter)


def get_name_fmt(file_num: int) -> str:
    """
    Generate a format string for filenames based on the total number of files.

    Parameters:
    
        file_num : int
            Total number of files to be named.

    Returns:
        str:
            Format string for filenames, e.g., '%03d' for three-digit naming.
    """
    digits = math.floor(math.log10(file_num)) + 1
    return f'%0{digits}d'


def convert_tensors_to_cuda(x: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Recursively convert all tensors in a dictionary to CUDA.

    Parameters:
    
        x : Dict[str, Any]
            Dictionary containing tensors or nested dictionaries.
        device : torch.device
            Device to move the tensors to (e.g., CUDA or CPU).

    Returns:
        Dict[str, Any]:
            A new dictionary with all tensors moved to the specified device.
    """
    y = {}
    for kw, arg in x.items():
        if torch.is_tensor(arg):
            y[kw] = arg.to(device)
        else:
            y[kw] = convert_tensors_to_cuda(arg, device)
    return y


def filter_keys(d: Dict[str, Any], substring: str) -> Dict[str, Any]:
    """
    Filter a dictionary to include only keys that contain a specific substring.

    Parameters:
    
        d : Dict[str, Any]
            The input dictionary to filter.
        substring : str
            The substring to look for in the keys.

    Returns:
        Dict[str, Any]:
            A new dictionary containing only the keys from the original 
            dictionary that include the specified substring.
    """
    return {k: v for k, v in d.items() if substring in k}


def get_filenames(directory: str, extension: str) -> List[str]:
    """
    Get sorted filenames with the given extension in the specified directory.

    Parameters:
        
        directory : str
            The directory to search for files.
        extension : str
            The file extension to filter by.

    Returns:
        List[str]:
            Sorted list of filenames with the specified extension.
    """
    filenames = glob(os.path.join(directory, f'*.{extension}'))
    filenames = [os.path.basename(filename) for filename in filenames]
    filenames.sort()
    return filenames

def load_mtx(filename: str) -> list:
    """
        load mtx file and convert to csr_matrix

        Parameters:
            filename : str
                Path to the mtx file.
    """
    return csr_matrix(mmread(filename)).toarray().tolist()


def get_s_joint_mods(combs: List[List[str]]) -> Tuple[List[Dict[str, int]], List[str]]:
    """
    Generate `s_joint` and `mods` from a list of modality combinations.

    Parameters:

        combs : List[List[str]]
            A list where each element is a list of strings representing combinations 
            of modalities for a specific batch.

    Returns:
        Tuple:
            - `s_joint`: A list of dictionaries, where each dictionary maps the modalities
            to their corresponding indices for each batch.
            - `mods`: A list of all unique modalities across the dataset.

    """
    s_joint = []
    mods = {}
    for b in combs:
        t = {}
        for m in b + ['joint']:
            if m in mods:
                mods[m] += 1
            else:
                mods[m] = 0
            t[m] = mods[m]
        s_joint.append(t)
    mods = list(mods.keys())
    mods.remove('joint')
    return s_joint, mods

def get_pred_dirs(
    pred_dir: str,
    combs: List[List[str]],
    joint_latent: bool,
    mod_latent: bool,
    impute: bool,
    batch_correct: bool,
    translate: bool,
    input: bool,
) -> Dict[int, Dict[str, Dict[str, str]]]:
    """
    Generate directory paths for predictions based on configurations.

    Parameters:
    
        pred_dir : str
            Base directory for predictions.
        combs : list of list of str
            Combinations of modalities for each batch.
        joint_latent : bool
            Include joint latent variables.
        mod_latent : bool
            Include modality-specific latent variables.
        impute : bool
            Include imputed data.
        batch_correct : bool
            Include batch-corrected data.
        translate : bool
            Include translated data.
        input : bool
            Include input data.

    Returns:
        Dict[int, Dict[str, Dict[str, str]]]:
            Dictionary of directories for each batch and variable.
    """
    dirs = {}
    s_joint, mods = get_s_joint_mods(combs)
    for batch_id in range(len(s_joint)):
        batch_dir = os.path.join(pred_dir, f'batch_{batch_id}')
        dirs[batch_id] = {}

        if joint_latent or mod_latent:
            dirs[batch_id]['z'] = {}
            if joint_latent:
                dirs[batch_id]['z']['joint'] = os.path.join(batch_dir, 'z', 'joint')
            if mod_latent:
                for mod in combs[batch_id]:
                    dirs[batch_id]['z'][mod] = os.path.join(batch_dir, 'z', mod)

        if impute:
            dirs[batch_id]['x_impt'] = {mod: os.path.join(batch_dir, 'x_impt', mod) for mod in mods}

        if batch_correct:
            dirs[batch_id]['x_bc'] = {mod: os.path.join(batch_dir, 'x_bc', mod) for mod in mods}

        if translate:
            dirs[batch_id]['x_trans'] = {}
            all_combinations = generate_all_combinations(mods)
            
            for input_mods, output_mods in all_combinations:
                f = True
                for i in input_mods:
                    if i not in combs[batch_id]:
                        f = False
                if f:
                    input_mods_sorted = sorted(input_mods)
                    for mod in output_mods:
                        key = '_'.join(input_mods_sorted) + '_to_' + mod
                        dirs[batch_id]['x_trans'][key] = os.path.join(batch_dir, 'x_trans', key)

        if input:
            dirs[batch_id]['x'] = {mod: os.path.join(batch_dir, 'x', mod) for mod in combs[batch_id]}

    return dirs


def rmdir(directory: str):
    """
    Remove a directory if it exists.

    Parameters:
    
        directory : str
            Path to the directory to remove.
    """
    if os.path.exists(directory):
        logging.warning(f'Removing directory "{directory}"')
        shutil.rmtree(directory)


def mkdir(directory: str, remove_old: bool = False):
    """
    Create a directory, optionally removing the old one.

    Parameters:
    
        directory : str
            Path to the directory.
        remove_old : bool, optional
            Whether to remove the old directory if it exists, by default False.
    """
    if remove_old:
        rmdir(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def mkdirs(directories: Union[str, List[str], Dict[str, Any]], remove_old: bool = False):
    """
    Recursively create directories.

    Parameters:
    
        directories :  Union[str, List[str], Dict[str, Any]]
            Path(s) to directories to create.
        remove_old : bool
            Whether to remove old directories if they exist, by default False.
    """
    if isinstance(directories, (list, tuple)):
        for d in directories:
            mkdirs(d, remove_old=remove_old)
    elif isinstance(directories, dict):
        for d in directories.values():
            mkdirs(d, remove_old=remove_old)
    else:
        mkdir(directories, remove_old=remove_old)
    

def reverse_trsf(name: str, data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply a reverse transformation to the given data.

    Parameters:
    
        name : str
            Name of the transformation to reverse (e.g., 'log1p').
        data : np.ndarray
            Data to transform.
        kwargs : dict
            Additional transformation parameters.

    Returns:
        np.ndarray:
            Transformed data.
    """
    # Extract parameters from kwargs
    params = {k.split('_')[-1]: v for k, v in kwargs.items()}

    # Perform the reverse transformation based on the name
    if name == 'log1p':
        return data.exp()
    else:
        return data


def generate_all_combinations(mods: List[str]) -> List[Tuple[Tuple[str, ...], List[str]]]:
    """
    Generate all possible input-output combinations for a given list of modalities.

    For N modalities, generate all combinations of size r (1 <= r < N) as input,
    and the remaining modalities as output.

    Parameters:
    
        mods : List[str]
            List of modality names.

    Returns:
        List[Tuple[Tuple[str, ...], List[str]]]:
            A list of tuples, where each tuple contains:
                - A tuple of input modalities.
                - A list of output modalities.
    """
    combinations = []
    for r in range(1, len(mods)):  # Generate combinations of size r
        for input_mods in itertools.combinations(mods, r):
            output_mods = list(set(mods) - set(input_mods))
            combinations.append((input_mods, output_mods))
    return combinations


def safe_append(pred:dict , batch_id:int, key_path:list, value:Any):
    """
    Append a value to a nested dictionary structure.

    Parameters:
    
        pred : dict
            The nested dictionary structure to append to.
        batch_id : int
            The batch ID to use as the key for the nested dictionary.
        key_path : list of str
            The path of keys to follow in the nested dictionary.
        value : Any
            The value to append to the nested dictionary.

    """
    current = pred.setdefault(batch_id, {})
    for key in key_path[:-1]:
        current = current.setdefault(key, {})
    current.setdefault(key_path[-1], []).append(value.cpu())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()

class BaseSink:
    """A sink receives minibatch outputs. It may keep them in memory or write to disk."""
    def write(self, batch_name: str, path: List[str], value: torch.Tensor | np.ndarray):
        raise NotImplementedError

    def write_meta(self, batch_name: str, path: List[str], value: Any):
        """For small non-tensor metadata (optional)."""
        raise NotImplementedError

    def finalize(self) -> Any:
        """Return final outputs (e.g., nested dict for MemorySink, or manifest for DiskSink)."""
        raise NotImplementedError


class MemorySink(BaseSink):
    def __init__(self):
        # nested dict: pred[batch][var][key] -> list[tensor]
        self.pred: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {}
        # metadata (e.g., masks) stored directly
        self.meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _ensure(self, batch_name: str, var: str, key: str):
        self.pred.setdefault(batch_name, {}).setdefault(var, {}).setdefault(key, [])

    def write(self, batch_name: str, path: List[str], value: torch.Tensor | np.ndarray):
        var, key = path[0], path[1]
        self._ensure(batch_name, var, key)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self.pred[batch_name][var][key].append(value.detach().cpu())

    def write_meta(self, batch_name: str, path: List[str], value: Any):
        var, key = path[0], path[1]
        self.meta.setdefault(batch_name, {}).setdefault(var, {})[key] = value

    def finalize(self) -> Dict[str, Any]:
        # Return raw lists (caller can post-process into z_c/z_u, etc.)
        out: Dict[str, Any] = {}
        for b, d in self.pred.items():
            out[b] = {}
            for var, d2 in d.items():
                out[b][var] = {}
                for k, lst in d2.items():
                    out[b][var][k] = torch.cat(lst, dim=0) if len(lst) else torch.empty((0,))
        # merge meta
        for b, d in self.meta.items():
            out.setdefault(b, {})
            for var, d2 in d.items():
                out[b].setdefault(var, {})
                out[b][var].update(d2)
        return out


@dataclass
class DiskSinkConfig:
    save_dir: str
    save_format: str = "npy"  # "npy" or "csv"
    fname_pattern: str = "{batch}/{var}/{key}/{i:06d}.{ext}"


class DiskSink(BaseSink):
    """
    Stream-to-disk sink (old-code style): each minibatch is saved immediately.
    Produces a manifest describing where things were written.
    """
    def __init__(self, cfg: DiskSinkConfig):
        self.cfg = cfg
        self.manifest: Dict[str, Dict[str, Dict[str, List[str]]]] = {}  # batch->var->key->files
        ensure_dir(cfg.save_dir)

    def _write_array(self, fpath: str, arr: np.ndarray):
        ensure_dir(os.path.dirname(fpath))
        if self.cfg.save_format == "npy":
            np.save(fpath, arr)
        elif self.cfg.save_format == "csv":
            # WARNING: large arrays to CSV can be slow and huge
            np.savetxt(fpath, arr, delimiter=",")
        else:
            raise ValueError(f"Unsupported save_format={self.cfg.save_format}")

    def write(self, batch_name: str, path: List[str], value: torch.Tensor | np.ndarray):
        var, key = path[0], path[1]
        self.manifest.setdefault(batch_name, {}).setdefault(var, {}).setdefault(key, [])
        i = len(self.manifest[batch_name][var][key])
        ext = "npy" if self.cfg.save_format == "npy" else "csv"
        rel = self.cfg.fname_pattern.format(batch=batch_name, var=var, key=key, i=i, ext=ext)
        fpath = os.path.join(self.cfg.save_dir, rel)

        arr = to_numpy(value)
        self._write_array(fpath, arr)
        self.manifest[batch_name][var][key].append(fpath)

    def write_meta(self, batch_name: str, path: List[str], value: Any):
        # meta is small; store in a side json-like structure
        var, key = path[0], path[1]
        self.manifest.setdefault(batch_name, {}).setdefault(var, {}).setdefault(key, [])
        # write meta once
        rel = f"{batch_name}/{var}/{key}/meta.npy"
        fpath = os.path.join(self.cfg.save_dir, rel)
        ensure_dir(os.path.dirname(fpath))
        np.save(fpath, np.array(value, dtype=object), allow_pickle=True)
        self.manifest[batch_name][var][key].append(fpath)

    def finalize(self) -> Dict[str, Any]:
        return {"saved_to": self.cfg.save_dir, "format": self.cfg.save_format, "manifest": self.manifest}


# -----------------------------
# Online stats for batch correction
# -----------------------------
class OnlineMeanByGroup:
    """
    Compute global mean and per-group means WITHOUT storing all samples.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.global_sum = torch.zeros(dim, dtype=torch.float64)
        self.global_n = 0
        self.group_sum: Dict[int, torch.Tensor] = {}
        self.group_n: Dict[int, int] = {}

    def update(self, x: torch.Tensor, g: torch.Tensor):
        """
        x: (N, D), g: (N,) int-like
        """
        x = x.detach().cpu().to(torch.float64)
        g = g.detach().cpu().to(torch.int64).flatten()

        self.global_sum += x.sum(dim=0)
        self.global_n += x.shape[0]

        for gid in torch.unique(g):
            gid_int = int(gid.item())
            mask = (g == gid)
            xs = x[mask]
            self.group_sum[gid_int] = self.group_sum.get(gid_int, torch.zeros(self.dim, dtype=torch.float64)) + xs.sum(dim=0)
            self.group_n[gid_int] = self.group_n.get(gid_int, 0) + xs.shape[0]

    def finalize_centroid(self) -> torch.Tensor:
        """
        Choose the group mean closest to global mean (L2), return that group's mean.
        """
        if self.global_n == 0:
            raise RuntimeError("No samples were seen.")
        global_mean = self.global_sum / self.global_n

        best_gid = None
        best_dist = None
        best_mean = None
        for gid, s in self.group_sum.items():
            n = self.group_n[gid]
            mu = s / n
            dist = torch.sum((mu - global_mean) ** 2).item()
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_gid = gid
                best_mean = mu
        return best_mean.to(torch.float32)
    

def _list_batches(save_dir: str) -> List[str]:
    batches = []
    for name in os.listdir(save_dir):
        p = os.path.join(save_dir, name)
        if os.path.isdir(p):
            batches.append(name)
    return sorted(batches)


def _list_vars(save_dir: str, batch: str) -> List[str]:
    p = os.path.join(save_dir, batch)
    if not os.path.isdir(p):
        return []
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


def _list_keys(save_dir: str, batch: str, var: str) -> List[str]:
    p = os.path.join(save_dir, batch, var)
    if not os.path.isdir(p):
        return []
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


_NUM_SUFFIX = re.compile(r"(\d+)(?=\.[^.]+$)")  # capture digits before extension


def _extract_index(fname: str) -> Optional[int]:
    m = _NUM_SUFFIX.search(fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _load_array_file(path: str, save_format: str) -> np.ndarray:
    if save_format == "npy":
        return np.load(path, allow_pickle=False)
    if save_format == "csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported save_format={save_format}")


def _load_meta_file(path: str) -> Any:
    # we saved meta as np.array(..., dtype=object) with allow_pickle=True
    return np.load(path, allow_pickle=True).item() if np.load(path, allow_pickle=True).shape == () else np.load(path, allow_pickle=True)


def _concat_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    # Ensure 2D
    normed = []
    for a in chunks:
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        normed.append(a)
    return np.concatenate(normed, axis=0)


def load_predicted(
    save_dir: str,
    *,
    save_format: str = "npy",   # must match DiskSinkConfig.save_format
    dim_c: Optional[int] = None,
    batch_names: Optional[List[str]] = None,
    var_names: Optional[List[str]] = None,
    split_z: bool = True,
    return_manifest: bool = False,
) -> Dict[str, Any]:
    """
    Load predictions saved by the streaming DiskSink.

    The function reads prediction results saved to disk during
    `predict(..., save_dir=...)` and reconstructs them into a
    prediction dictionary similar to the in-memory output format.
    
    Parameters:
        save_dir : str
            Root directory where predictions were saved by
            `predict(..., save_dir=...)`.

        save_format : {"npy", "csv"}
            File format used for saved prediction arrays.

        dim_c : int, optional
            Dimension of the content latent space (`z_c`).
            Required if `split_z=True` and latent variable `z` exists.

        batch_names : List[str], optional
            If provided, only the specified batches will be loaded
            (in the given order).

        joint_latent : bool, default=True
            Whether to include joint latent representations.
            If False, `z_*["joint"]` will be removed from the output.

        split_z : bool, default=True
            If True, split latent variable `z` into:

                - `z_c` : content latent representation
                - `z_u` : technical latent representation

            using `dim_c`. If False, keep the raw `z` arrays.

        return_manifest : bool, default=False
            Whether to include a manifest containing the file paths
            used to reconstruct the predictions.

    Returns:
        pred_b : Dict[str, Any]
            Prediction dictionary organized by batch. The structure
            matches the in-memory prediction output, for example:

                pred_b[batch]["z_c"][key]
                pred_b[batch]["z_u"][key]
                pred_b[batch]["x_impt"][modality]
                pred_b[batch]["x_bc"][modality]
                pred_b[batch]["x_trans"][translation_key]

            If metadata was saved, modality masks will be stored as:

                pred_b[batch]["mask"][modality]
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir not found: {save_dir}")

    batches = batch_names if batch_names is not None else _list_batches(save_dir)

    pred_b: Dict[str, Any] = {}
    manifest: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    for batch in batches:
        batch_dir = os.path.join(save_dir, batch)
        if not os.path.isdir(batch_dir):
            continue

        pred_b[batch] = {}
        manifest[batch] = {}

        for var in _list_vars(save_dir, batch) if var_names is None else var_names:
            pred_b[batch].setdefault(var, {})
            manifest[batch].setdefault(var, {})

            for key in _list_keys(save_dir, batch, var):
                key_dir = os.path.join(save_dir, batch, var, key)
                if not os.path.isdir(key_dir):
                    continue

                files = [f for f in os.listdir(key_dir) if os.path.isfile(os.path.join(key_dir, f))]
                meta_path = os.path.join(key_dir, "meta.npy")
                has_meta = os.path.isfile(meta_path)

                # Load meta if exists (commonly for mask)
                if has_meta:
                    pred_b[batch].setdefault(var, {})
                    pred_b[batch][var][key] = _load_meta_file(meta_path)
                    manifest[batch][var].setdefault(key, [])
                    manifest[batch][var][key].append(meta_path)
                    continue

                # Load chunk files (exclude meta)
                chunk_files = [f for f in files if f != "meta.npy"]
                # Sort by numeric suffix if present; else lexicographic
                chunk_files_sorted = sorted(
                    chunk_files,
                    key=lambda fn: (_extract_index(fn) is None, _extract_index(fn) or 0, fn),
                )

                paths = [os.path.join(key_dir, f) for f in chunk_files_sorted]
                chunks = [_load_array_file(p, save_format) for p in paths]
                arr = _concat_chunks(chunks)

                pred_b[batch][var][key] = arr
                manifest[batch][var][key] = paths

    # Move meta masks to a nicer place:
    # If var == 'mask' is present, keep as pred_b[batch]['mask'][m] = array
    # (Already true), but we also want to avoid later splitting logic touching it.
    # Nothing to do.

    # Split z into z_c / z_u if requested
    if split_z:
        if dim_c is None:
            # Only required if z exists
            any_z = any(("z" in pred_b[b]) for b in pred_b.keys())
            if any_z:
                raise ValueError("dim_c is required when split_z=True and z is present.")
        for batch, d in list(pred_b.items()):
            if "z" not in d:
                continue
            zdict = d["z"]
            d["z_c"] = {}
            d["z_u"] = {}
            for k, z in zdict.items():
                if not isinstance(z, np.ndarray):
                    # if someone saved meta under z (unlikely), skip
                    continue
                if z.ndim == 1:
                    z = z.reshape(-1, 1)
                d["z_c"][k] = z[:, :dim_c] if dim_c is not None else z
                d["z_u"][k] = z[:, dim_c:] if dim_c is not None else z
            # remove raw z
            d.pop("z", None)

    if return_manifest:
        return {"pred": pred_b, "manifest": manifest}

    return pred_b

def z_to_adata_or_mdata(pred, sparse_threshold=10000):
    """
    Convert prediction dictionary to AnnData (single modality) or MuData (multi-modality).

    If only one modality is present, an `AnnData` object will be returned.
    If multiple modalities are present, a `MuData` object will be constructed
    with one `AnnData` object per modality.

    Parameters:
        pred : Dict[str, Any]
            Prediction results generated by `predict()` or `load_predicted()`.

        sparse_threshold : int, default=10000
            If the number of features exceeds this threshold, the data matrix
            will be converted to a sparse CSR matrix to reduce memory usage.

    Returns:
        adata_or_mdata : Union[AnnData, MuData]
            - `AnnData` if a single modality is present.
            - `MuData` if multiple modalities are present.

    Notes:
        - The batch label is added to both:
            * `adata.obs["batch"]` for each modality
            * `mdata.obs["batch"]` at the top level (so `sc.pl.umap(mdata, color="batch")` works)

        - Latent embeddings are stored as:
            * `adata.obsm["z_c"]`, `adata.obsm["z_u"]` for single-modality data
            * `mdata.obsm["z_c"]`, `mdata.obsm["z_u"]` for multi-modality data

        - Modality masks are stored in:
            * `adata.uns["mask"]` for single-modality data
            * `adata.uns["mask_<modality>"]` or `adata.uns["mask"]` depending on the context
    """

    # ----------------------------------------------------
    # 1 Detect modalities
    # ----------------------------------------------------
    mods = set()
    for batch, data in pred.items():
        for key in ["x_impt", "x_bc", "x"]:
            if key in data:
                mods.update(data[key].keys())
    mods = sorted(list(mods))

    # ----------------------------------------------------
    # 2 Collect per modality matrices
    # ----------------------------------------------------
    mod_data = {m: [] for m in mods}
    obs_batch = []

    latent_c = []
    latent_u = []

    masks = {}

    for batch, data in pred.items():
        n_cells = None

        # latents
        if "z_c" in data and "joint" in data["z_c"]:
            latent_c.append(data["z_c"]["joint"])
            latent_u.append(data["z_u"]["joint"])
            n_cells = data["z_c"]["joint"].shape[0]

        # choose X with priority: imputed > batch-corrected > input
        for m in mods:
            if "x_impt" in data and m in data["x_impt"]:
                X = data["x_impt"][m]
            elif "x_bc" in data and m in data["x_bc"]:
                X = data["x_bc"][m]
            elif "x" in data and m in data["x"]:
                X = data["x"][m]
            else:
                continue

            mod_data[m].append(X)
            if n_cells is None:
                n_cells = X.shape[0]

        if n_cells is None:
            continue

        obs_batch.extend([batch] * n_cells)

        # masks (keep last seen per modality)
        if "mask" in data:
            for m, mask in data["mask"].items():
                masks[m] = mask

    # ----------------------------------------------------
    # 3 concatenate
    # ----------------------------------------------------
    for m in mods:
        mod_data[m] = np.concatenate(mod_data[m], axis=0) if len(mod_data[m]) else None

    latent_c = np.concatenate(latent_c, axis=0) if latent_c else None
    latent_u = np.concatenate(latent_u, axis=0) if latent_u else None

    # obs dataframe-like mapping
    obs = {"batch": np.array(obs_batch, dtype=object)}

    # ----------------------------------------------------
    # 4 Single modality → AnnData
    # ----------------------------------------------------
    if len(mods) == 1:
        m = mods[0]
        X = mod_data[m]
        if X is None:
            raise ValueError("No data found for the single modality.")

        if X.shape[1] > sparse_threshold:
            X = csr_matrix(X)

        adata = ad.AnnData(X=X, obs=obs)

        if latent_c is not None:
            adata.obsm["z_c"] = latent_c
        if latent_u is not None:
            adata.obsm["z_u"] = latent_u

        if m in masks:
            adata.uns["mask"] = masks[m]

        return adata

    # ----------------------------------------------------
    # 5 Multi modality → MuData
    # ----------------------------------------------------
    adatas = {}

    for m in mods:
        X = mod_data[m]
        if X is None:
            continue

        if X.shape[1] > sparse_threshold:
            X = csr_matrix(X)

        # IMPORTANT: each modality gets its own obs with batch
        adata_m = ad.AnnData(X=X, obs=obs)

        if m in masks:
            adata_m.uns["mask"] = masks[m]

        adatas[m] = adata_m

    if not adatas:
        raise ValueError("No modality matrices found to build MuData.")

    mdata = mu.MuData(adatas)

    # IMPORTANT: add batch to TOP-LEVEL obs so sc.pl.umap(mdata, color="batch") works
    # Use the obs from the first modality (same length/order)
    first_mod = next(iter(mdata.mod.keys()))
    mdata.obs["batch"] = mdata.mod[first_mod].obs["batch"].to_numpy()

    if latent_c is not None:
        mdata.obsm["z_c"] = latent_c
    if latent_u is not None:
        mdata.obsm["z_u"] = latent_u

    return mdata