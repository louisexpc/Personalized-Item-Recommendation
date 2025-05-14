import yaml

class Config:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v)) 
            else:
                setattr(self, k, v)

def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML file.

    Returns:
        Config: Configuration dictionary-like object.
    """
    with open(config_path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    return Config(raw_dict)