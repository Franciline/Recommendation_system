"""Compatibility shim for notebooks that still import `reco_systems`.

New code should import from `boardgames_recsys` instead.
"""

import boardgames_recsys

__path__ = boardgames_recsys.__path__
