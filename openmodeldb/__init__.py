"""
OpenModelDB - Browse and download AI upscaling models.

Usage:
    from openmodeldb import OpenModelDB

    db = OpenModelDB()
    models = db.find(scale=4)
    db.download(models[0])
"""

from openmodeldb.client import (
    DownloadError,
    FormatNotFoundError,
    Model,
    ModelNotFoundError,
    OpenModelDB,
    OpenModelDBError,
)

__all__ = [
    "OpenModelDB",
    "Model",
    "OpenModelDBError",
    "ModelNotFoundError",
    "FormatNotFoundError",
    "DownloadError",
]
__version__ = "1.1.1"
