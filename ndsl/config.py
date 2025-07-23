import dataclasses
from typing import Any

import f90nml
import yaml

from ndsl.namelist import namelist_to_flatish_dict


DEFAULT_INT = 0
DEFAULT_STR = ""
DEFAULT_FLOAT = 0.0
DEFAULT_BOOL = False


@dataclasses.dataclass
class Config:
    """
    Base class for configuration objects.
    Provides subclasses with base methods for loading from YAML and
    Fortran namelists files.
    """

    dt_atmos: int = DEFAULT_INT
    days: int = DEFAULT_INT
    hours: int = DEFAULT_INT
    minutes: int = DEFAULT_INT
    seconds: int = DEFAULT_INT
    npx: int = DEFAULT_INT
    npy: int = DEFAULT_INT
    npz: int = DEFAULT_INT

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_f90nml(cls, f90_namelist: f90nml.Namelist) -> "Config":
        namelist_dict = namelist_to_flatish_dict(f90_namelist.items())
        namelist_dict = {
            key: value
            for key, value in namelist_dict.items()
            if key in cls.__dataclass_fields__  # type: ignore
        }
        config = cls(**namelist_dict)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, yaml_config: str) -> "Config":
        # We're trying to make this generic, BUT we still assume that
        # dt_atmos, nx_tiles, and nz are defined

        config = cls()
        with open(yaml_config, "r") as f:
            raw_config = yaml.safe_load(f)
        flat_config: dict = {}
        for key, value in raw_config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey in config.__dataclass_fields__.keys():
                        if subkey in flat_config:
                            if subvalue != flat_config[subkey]:
                                raise ValueError(
                                    "Cannot flatten this config ",
                                    f"duplicate keys: {subkey}",
                                )
                        flat_config[subkey] = subvalue
            else:
                if key == "nx_tile":
                    flat_config["npx"] = value + 1
                    flat_config["npy"] = value + 1
                elif key == "nz":
                    flat_config["npz"] = value
                else:
                    if key in config.__dataclass_fields__.keys():
                        flat_config[key] = value
        for field in dataclasses.fields(config):
            if field.name in flat_config.keys():
                setattr(config, field.name, flat_config[field.name])
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Create a config object from a dictionary.
        """
        # This implementation assumes that the dictionary keys match the
        # dataclass field names.
        config = cls(**data)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config object to a dictionary.
        """
        return dataclasses.asdict(self)

    def validate(self):
        """
        Validate
        """
        # Convert these into proper throw statements
        assert self.npx == self.npy
        assert self.dt_atmos >= 0
        assert self.days >= 0
        assert self.hours >= 0
        assert self.minutes >= 0
        assert self.seconds >= 0
