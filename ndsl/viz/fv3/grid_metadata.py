import abc
import dataclasses


class GridMetadata(abc.ABC):
    @property
    @abc.abstractmethod
    def coord_vars(self) -> dict: ...


@dataclasses.dataclass
class GridMetadataFV3(GridMetadata):
    i: str = "i"
    j: str = "j"
    i_interface: str = "i_interface"
    j_interface: str = "j_interface"
    tile: str = "tile"
    lon: str = "lon"
    lonb: str = "lonb"
    lat: str = "lat"
    latb: str = "latb"

    @property
    def coord_vars(self):
        coord_vars = {
            self.lonb: [self.j_interface, self.i_interface, self.tile],
            self.latb: [self.j_interface, self.i_interface, self.tile],
            self.lon: [self.j, self.i, self.tile],
            self.lat: [self.j, self.i, self.tile],
        }
        return coord_vars


@dataclasses.dataclass
class GridMetadataScream(GridMetadata):
    ncol: str = "ncol"
    lon: str = "lon"
    lat: str = "lat"

    @property
    def coord_vars(self):
        coord_vars = {
            self.lon: [self.ncol],
            self.lat: [self.ncol],
        }
        return coord_vars
