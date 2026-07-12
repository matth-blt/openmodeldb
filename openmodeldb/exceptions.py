"""
Exception hierarchy for OpenModelDB.

Kept in its own module (rather than in client.py) so that other modules
(e.g. openmodeldb.downloader) can raise/catch these without importing
openmodeldb.client.
"""


class OpenModelDBError(Exception):
    """Base exception for OpenModelDB."""


class ModelNotFoundError(OpenModelDBError):
    """Raised when a model name/id cannot be resolved."""


class FormatNotFoundError(OpenModelDBError):
    """Raised when a requested format is not available for a model."""


class DownloadError(OpenModelDBError):
    """Raised when a download fails."""
