# Standard Library Imports
import os
import csv
import shutil
import itertools
import math
from glob import glob
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Any

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
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

def load_predicted(
    pred_dir: str,
    combs: List[List[str]],
    joint_latent: bool = True,
    mod_latent: bool = False,
    impute: bool = False,
    batch_correct: bool = False,
    translate: bool = False,
    input: bool = False,
    group_by: str = 'modality',
    mtx: bool = True
) -> Union[Dict[int, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Load predicted variables from a specified directory.

    Parameters:
    
        pred_dir : str
            Path to the prediction directory.
        combs : list of list of str
            Combinations of modalities for each batch. Example: [['rna'], ['rna', 'adt']].
        joint_latent : bool, optional
            Whether to include joint latent variables, by default True.
        mod_latent : bool, optional
            Whether to include modality-specific latent variables, by default False.
        impute : bool, optional
            Whether to include imputed data, by default False.
        batch_correct : bool, optional
            Whether to include batch-corrected data, by default False.
        translate : bool, optional
            Whether to include translated data, by default False.
        input : bool, optional
            Whether to include input data, by default False.
        group_by : str, optional
            Grouping method for the data, either 'modality' or 'batch', by default 'modality'.

    Returns:
        Union[Dict[int, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
            Loaded predicted data grouped by the specified method.
    """
    logging.info('Loading predicted variables ...')
    dirs = get_pred_dirs(pred_dir, 
                         combs, 
                         joint_latent, 
                         mod_latent, 
                         impute, 
                         batch_correct, 
                         translate, 
                         input)
    data = {}
    # Load data from directories
    for batch_id, batch_dirs in dirs.items():
        data[batch_id] = {'s': {}}
        for variable, variable_dirs in batch_dirs.items():
            data[batch_id][variable] = {}
            for mod, dir_path in variable_dirs.items():
                logging.info(f'Loading batch {batch_id}: {variable}, {mod}')
                data[batch_id][variable][mod] = []
                if variable == 'z':
                    data[batch_id]['s'][mod] = []
                if mtx:
                    filenames = get_filenames(dir_path, 'mtx')
                else:
                    filenames = get_filenames(dir_path, 'csv')
                for filename in tqdm(filenames):
                    if mtx:
                        v = load_mtx(os.path.join(dir_path, filename))
                    else:
                        v = load_csv(os.path.join(dir_path, filename))
                    data[batch_id][variable][mod] += v
                    if variable == 'z':
                        data[batch_id]['s'][mod] += [batch_id] * len(v)

    logging.info('Converting to numpy ...')
    for batch_id, batch_data in data.items():
        for variable, variable_data in batch_data.items():
            for mod, values in variable_data.items():
                logging.info(f'Converting batch {batch_id}: {variable}, {mod}')
                if variable in ['z', 'x_trans', 'x_impt', 's', 'x', 'x_bc']:
                    data[batch_id][variable][mod] = values

    # Group data by modality if required
    if group_by == 'batch':
        for batch_id, batch_data in data.items():
            for variable, variable_data in batch_data.items():
                for mod, values in variable_data.items():
                    data[batch_id][variable][mod] = np.array(data[batch_id][variable][mod]).astype(np.float32)
        return data
    elif group_by == 'modality':
        data_m = {}
        for batch_id, batch_data in data.items():
            for variable, variable_data in batch_data.items():
                if variable not in data_m:
                    data_m[variable] = {}
                for mod, values in variable_data.items():
                    if mod not in data_m[variable]:
                        data_m[variable][mod] = {}
                    data_m[variable][mod][batch_id] = values

        # Concatenate data across batches
        for variable, variable_data in data_m.items():
            for mod, mod_data in variable_data.items():
                data_m[variable][mod] = np.concatenate(list(mod_data.values()), axis=0).astype(np.float32)

        return data_m

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
    current.setdefault(key_path[-1], []).append(value)