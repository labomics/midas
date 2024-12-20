import toml
import os
import logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path :str = None, config_name :str = "default"):
    """
    Load configurations from file to construct the model.

    Parameters:
        config_path : str
            Path of the configuration file. End with '.toml'. If not given, use the 'scmidas/model_config.toml'.
        config_name : str
            Item name from the configuration.
    """
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), 'model_config.toml')
        logging.info(f'The model is initialized with the configurations from "{config_path}" [{config_name}].')
    else:
        logging.info(f'The model is initialized with the configurations from "{config_path}" [{config_name}].')
    configs = toml.load(config_path).get(config_name, {})
    return configs