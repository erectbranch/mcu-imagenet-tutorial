# python < 3.7
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ['yaml_to_config', 'SamplingConfig' 'CliConfig']

def yaml_to_config(config: Dict[Any, Any], config_class: type):
    return config_class(**config)


# sampling.yaml
@dataclass
class Sampling:
    random_seed: int
    subset_size: int
    num_subset: int
    save_path: str

@dataclass
class ValidSet:
    path: str
    image_size: int
    batch_size: int

@dataclass
class SamplingConfig:
    sampling: Sampling
    valid_set: ValidSet

    def __post_init__(self):
        self.sampling  = yaml_to_config(self.sampling, Sampling)
        self.valid_set = yaml_to_config(self.valid_set, ValidSet)            


# cli_command.yaml
@dataclass
class Stedgeai:
    path: str
    temp: str
    serial_port: str
    report_path: str

@dataclass
class Target:
    mcu: str
    model: str

@dataclass
class CliConfig:
    stedgeai: Stedgeai
    target: Target

    def __post_init__(self):
        self.stedgeai = yaml_to_config(self.stedgeai, Stedgeai)
        self.target   = yaml_to_config(self.target, Target)