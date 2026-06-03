from __future__ import annotations

import os
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

external_models_dir = os.environ.get("MODEL_DIR") or os.environ.get("HICACHE_MODEL_DIR")
if external_models_dir and os.path.isdir(external_models_dir):
    __path__.append(external_models_dir)
