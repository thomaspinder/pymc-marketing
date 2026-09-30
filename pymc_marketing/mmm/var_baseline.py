#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Vector autoregression (VAR) baseline support, built on Impulso.

`Impulso <https://github.com/QuantClimate/Impulso>`_ is an optional dependency of
PyMC-Marketing. Install it with ``pip install 'pymc-marketing[var]'``. This module
imports Impulso only when it is used, so PyMC-Marketing imports without it.
"""

from types import ModuleType


def _import_impulso() -> ModuleType:
    """Import Impulso, the optional dependency of the VAR baseline.

    Returns
    -------
    ModuleType
        The ``impulso`` module.

    Raises
    ------
    ImportError
        If Impulso is not installed.
    """
    try:
        import impulso
    except ImportError as exc:
        raise ImportError(
            "impulso is required for the VAR baseline. "
            "Install it with: pip install 'pymc-marketing[var]'"
        ) from exc
    return impulso
