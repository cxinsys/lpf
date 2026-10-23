from lpf.geometry.base import Geometry
from lpf.geometry.grid import GridGeometry
from lpf.geometry.domain import DomainGeometry
from lpf.geometry.mesh import MeshGeometry


# Geometries that model dictionaries can describe (see geometry_kwargs).
_SERIALIZABLE = (MeshGeometry, DomainGeometry)


def make_geometry(height, width, dx, domain=None, mesh=None):
    """The geometry for a model's constructor arguments: the grid, a domain
    on it (lpf.domains) or a mesh surface (lpf.surfaces)."""
    if domain is not None and mesh is not None:
        raise ValueError("domain and mesh cannot be used together.")
    if mesh is not None:
        return MeshGeometry(mesh)
    if domain is not None:
        return DomainGeometry(domain, height, width, dx)
    return GridGeometry(height, width, dx)


def geometry_kwargs(model_dicts, dpath="."):
    """Model constructor keyword arguments ({"domain": ...}, {"mesh": ...}
    or {} for the grid) that rebuild the geometry of saved models."""
    for cls in _SERIALIZABLE:
        kwargs = cls.kwargs_from_dicts(model_dicts, dpath)
        if kwargs is not None:
            return kwargs
    return {}
