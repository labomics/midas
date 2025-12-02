import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union, List, Dict

import logging
logging.basicConfig(level=logging.INFO)


class DistributionRegistry:
    """
    A registry for managing and dynamically extending loss functions, 
    sampling functions, and activation functions.
    """

    def __init__(self):
        # Initialize the mappings for loss, sampling, and activation functions
        self.loss_map = {}
        self.sampling_map = {}
        self.activate_map = {}

        # Register default functions
        self.register(
            'POISSON',
            nn.PoissonNLLLoss(full=True, reduction='none'),
            self.poisson_sampling,
            self.null,
        )
        self.register(
            'BERNOULLI',
            nn.BCELoss(reduction='none'),
            self.bernoulli_sampling,
            nn.Sigmoid(),
        )
        self.register('CE', nn.CrossEntropyLoss(reduction='none'), self.null, self.null)

    def register(
        self, name: str, loss_fn: nn.Module, sampling_fn: Callable, activate_fn: Callable
    ):
        """
        Register a new set of loss, sampling, and activation functions.

        Parameters:
            name : str
                The name of the distribution (key for retrieval).
            loss_fn : nn.Module
                The loss function instance to register.
            sampling_fn : Callable
                The sampling function instance to register.
            activate_fn : Callable
                The activation function instance to register.

        Raises:
            ValueError:
                If the name is already registered in any of the maps.
        """
        if name in self.loss_map:
            logging.info(f'Loss function "{name}" is already registered. Override it.')
        self.loss_map[name] = loss_fn
        self.sampling_map[name] = sampling_fn
        self.activate_map[name] = activate_fn

    def get_activate(self, name: str) -> Callable:
        """
        Retrieve a registered activation function by name.

        Parameters:
            name : str
                The name of the activation function.

        Returns:
            Callable:
                The corresponding activation function instance.

        Raises:
            KeyError:
                If the activation function is not registered.
        """
        if name not in self.activate_map:
            raise KeyError(f'Activation function "{name}" is not registered.')
        return self.activate_map[name]

    def get_sampling(self, name: str) -> Callable:
        """
        Retrieve a registered sampling function by name.

        Parameters:
            name : str
                The name of the sampling function.

        Returns:
            Callable:
                The corresponding sampling function instance.

        Raises:
            KeyError:
                If the sampling function is not registered.
        """
        if name not in self.sampling_map:
            raise KeyError(f'Sampling function "{name}" is not registered.')
        return self.sampling_map[name]

    def get_loss(self, name: str) -> nn.Module:
        """
        Retrieve a registered loss function by name.

        Parameters:
            name : str
                The name of the loss function.

        Returns:
            nn.Module:
                The corresponding loss function instance.

        Raises:
            KeyError:
                If the loss function is not registered.
        """
        if name not in self.loss_map:
            raise KeyError(f'Loss function "{name}" is not registered.')
        return self.loss_map[name]

    def list_registered(self) -> List[str]:
        """
        List all registered distributions.

        Returns:
            List[str]:
                Names of all registered distributions.
        """
        return list(self.loss_map.keys())

    @staticmethod
    def bernoulli_sampling(data: torch.Tensor) -> torch.Tensor:
        """
        Perform Bernoulli sampling on the input tensor.

        Parameters:
            data : torch.Tensor
                Input probabilities for Bernoulli sampling.

        Returns:
            torch.Tensor:
                Sampled binary tensor.
        """
        return torch.bernoulli(data).int()

    @staticmethod
    def poisson_sampling(data: torch.Tensor) -> torch.Tensor:
        """
        Perform Poisson sampling on the input tensor.

        Parameters:
            data : torch.Tensor
                Input rates for Poisson sampling.

        Returns:
            torch.Tensor:
                Sampled tensor with Poisson-distributed values.
        """
        return torch.poisson(data).int()

    @staticmethod
    def null(data: torch.Tensor) -> torch.Tensor:
        """
        A placeholder function that returns the input tensor unchanged.

        Parameters:
            data : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor:
                The same tensor without any modification.
        """
        return data


distribution_registry = DistributionRegistry()


class TransformRegistry:
    """
    A registry for managing and dynamically extending transformation functions,
    with mandatory registration of inverse transformations.
    """

    def __init__(self):
        # Initialize mappings for transformations and their inverses
        self.transform_map = {}
        self.inverse_transform_map = {}

        # Register default transformations with their inverses
        self.register('binarize', self.binarize, self.null)
        self.register('log1p', self.log1p, self.exp)

    def register(self, name: str, fn: Callable, inverse_fn: Callable):
        """
        Register a new transformation function along with its inverse.

        Parameters:
            name : str
                The name of the transformation function (key for retrieval).
            fn : callable
                The transformation function.
            inverse_fn : callable
                The inverse of the transformation function.

        Raises:
            ValueError:
                If the transformation or its inverse is already registered.
        """
        if name in self.transform_map:
            logging.info(f'Transformation "{name}" is already registered. Override it.')

        self.transform_map[name] = fn
        self.inverse_transform_map[name] = inverse_fn

    def get(self, name: str) -> Callable:
        """
        Retrieve a registered transformation function by name.

        Parameters:
            name : str
                The name of the transformation function.

        Returns:
            Callable:
                The corresponding transformation function.

        Raises:
            KeyError:
                If the specified transformation function is not registered.
        """
        if name not in self.transform_map:
            raise KeyError(f'Transformation "{name}" is not registered.')
        return self.transform_map[name]

    def get_inverse(self, name: str) -> Callable:
        """
        Retrieve the inverse of a registered transformation function by name.

        Parameters:
            name : str
                The name of the transformation function.

        Returns:
            Callable:
                The corresponding inverse transformation function.

        Raises:
            KeyError:
                If the specified inverse transformation function is not registered.
        """
        if name not in self.inverse_transform_map:
            raise KeyError(f'Inverse transformation for "{name}" is not registered.')
        return self.inverse_transform_map[name]

    def list_registered(self) -> List[str]:
        """
        List all registered transformation functions.

        Returns:
            List[str]:
                Names of all registered transformation functions.
        """
        return list(self.transform_map.keys())

    @staticmethod
    def binarize(data: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> Union[np.ndarray, torch.Tensor]:
        """
        Binarize the data using the specified threshold.

        Parameters:
            data : np.ndarray or torch.Tensor
                Input data to binarize.
            threshold : float, optional
                Threshold for binarization, default is 0.5.

        Returns:
            Union[np.ndarray, torch.Tensor]:
                Binarized data.

        Raises:
            TypeError:
                If the input data is neither a numpy array nor a torch tensor.
        """
        if isinstance(data, np.ndarray):
            return np.where(data.astype(np.float32) > threshold, 1, 0).astype(np.float32)
        elif isinstance(data, torch.Tensor):
            return (data > threshold).float()
        else:
            raise TypeError('Data must be a numpy array or torch tensor.')

    @staticmethod
    def null(data: torch.Tensor) -> torch.Tensor:
        """
        A placeholder function that returns the input tensor unchanged.

        Parameters:
            data : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor:
                The same tensor without any modification.
        """
        return data

    @staticmethod
    def log1p(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply log1p transformation to the data.

        Parameters:
            data : np.ndarray or torch.Tensor
                Input data to transform.

        Returns:
            Union[np.ndarray, torch.Tensor]:
                Transformed data.

        Raises:
            TypeError:
                If the input data is neither a numpy array nor a torch tensor.
        """
        if isinstance(data, np.ndarray):
            return np.log1p(data)
        elif isinstance(data, torch.Tensor):
            return data.log1p()
        else:
            raise TypeError('Data must be a numpy array or torch tensor.')

    @staticmethod
    def exp(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply exponential transformation (inverse of log1p) to the data.

        Parameters:
            data : np.ndarray or torch.Tensor
                Input data to transform.

        Returns:
            Union[np.ndarray, torch.Tensor]:
                Transformed data.

        Raises:
            TypeError:
                If the input data is neither a numpy array nor a torch tensor.
        """
        if isinstance(data, np.ndarray):
            return np.expm1(data)
        elif isinstance(data, torch.Tensor):
            return torch.expm1(data)
        else:
            raise TypeError('Data must be a numpy array or torch tensor.')


# Initialize the registry
transform_registry = TransformRegistry()


class ActivationRegistry:
    """
    A registry for managing and dynamically extending activation functions.
    """

    def __init__(self):
        # Initialize the mapping for activation functions
        self.func_map: Dict[str, Callable] = {}

        # Register default activation functions
        self.register('tanh', nn.Tanh)
        self.register('relu', nn.ReLU)
        self.register('silu', nn.SiLU)
        self.register('mish', nn.Mish)
        self.register('sigmoid', nn.Sigmoid)
        self.register('softmax', lambda dim=1: nn.Softmax(dim=dim))
        self.register('log_softmax', lambda dim=1: nn.LogSoftmax(dim=dim))

    def register(self, name: str, func: Callable):
        """
        Register a new activation function.

        Parameters:
            name : str
                The name of the activation function (key for retrieval).
            func : Callable
                The activation function instance or a factory function.
        """
        if name in self.func_map:
            logging.info(f'Activation function "{name}" is already registered. Override it.')
        
        self.func_map[name] = func

    def get(self, name: str, **kwargs) -> Callable:
        """
        Retrieve a registered activation function by name.

        Parameters:
            name : str
                The name of the activation function.
            kwargs : dict, optional
                Additional parameters for the activation function (e.g., `dim` for Softmax).

        Returns:
            Callable:
                The corresponding activation function instance.

        Raises:
            KeyError:
                If the specified activation function is not registered.
            ValueError:
                If the activation function does not support dynamic parameters.
        """
        if name not in self.func_map:
            raise KeyError(f'Activation function "{name}" is not registered.')

        # If the function is parameterized (e.g., softmax), allow dynamic configuration
        func = self.func_map[name]
        if callable(func):
            return func(**kwargs) if kwargs else func()
        return func

    def list_registered(self) -> List[str]:
        """
        List all registered activation functions.

        Returns:
            List[str]:
                Names of all registered activation functions.
        """
        return list(self.func_map.keys())

activation_registry = ActivationRegistry()


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) module with customizable activation functions,
    normalization, and dropout layers.

    Parameters:
        features : list of int
            List of integers specifying the number of neurons in each layer.
        hid_trans : str, optional
            Activation function for hidden layers, default is 'mish'.
        out_trans : str or bool, optional
            Activation function for the output layer. If False, no activation is applied, default is False.
        norm : str or bool, optional
            Normalization type for all layers ('bn', 'ln', or False). Overrides `hid_norm` and `out_norm`.
        hid_norm : str or bool, optional
            Normalization type for hidden layers ('bn', 'ln', or False), default is False.
        drop : float or bool, optional
            Dropout rate for all layers. Overrides `hid_drop` and `out_drop`, default is False.
        hid_drop : float or bool, optional
            Dropout rate for hidden layers, default is False.

    Attributes:
        net : nn.Sequential
            Sequential container for the layers of the MLP.
    """

    def __init__(
        self,
        features: list,
        hid_trans: str = 'mish',
        out_trans: Union[str, bool] = False,
        norm: Union[str, bool] = False,
        hid_norm: Union[str, bool] = False,
        drop: Union[float, bool] = False,
        hid_drop: Union[float, bool] = False,
    ):
        super(MLP, self).__init__()
        assert len(features) > 1, 'MLP must have at least 2 layers (input and output)!'

        # Apply global normalization and dropout if specified
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False

        # Build the MLP layers
        layers = []
        for i in range(1, len(features)):
            layers.append(nn.Linear(features[i - 1], features[i]))
            if i < len(features) - 1:  # Hidden layers
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:  # Output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters:
            x : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor:
                Output tensor after passing through the MLP layers.
        """
        return self.net(x)


class Layer1D(nn.Module):
    """
    A single layer module that supports normalization, activation, and dropout.

    Parameters:
        dim : int, optional
            Dimension of the input tensor (required for normalization layers), default is False.
        norm : str or bool, optional
            Type of normalization to apply ('bn' for BatchNorm, 'ln' for LayerNorm, or False), default is False.
        trans : str or bool, optional
            Activation function name to apply. If False, no activation is applied, default is False.
        drop : float or bool, optional
            Dropout rate. If False, no dropout is applied, default is False.

    Attributes:
        net : nn.Sequential
            Sequential container for the components of the layer.
    """

    def __init__(
        self,
        dim: Union[int, bool] = False,
        norm: Union[str, bool] = False,
        trans: Union[str, bool] = False,
        drop: Union[float, bool] = False,
    ):
        super(Layer1D, self).__init__()
        layers = []

        # Add normalization layer
        if norm == 'bn':
            layers.append(nn.BatchNorm1d(dim))
        elif norm == 'ln':
            layers.append(nn.LayerNorm(dim))

        # Add activation function
        if trans:
            layers.append(activation_registry.get(trans))

        # Add dropout layer
        if drop:
            layers.append(nn.Dropout(drop))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters:
            x : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor:
                Output tensor after applying normalization, activation, and dropout.
        """
        return self.net(x)


