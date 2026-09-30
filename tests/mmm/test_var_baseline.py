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
"""Tests for the VAR baseline effect and its optional Impulso dependency."""

import subprocess
import sys

import pytest


@pytest.mark.xfail(
    strict=True, reason="pymc_marketing.mmm.var_baseline is not implemented yet"
)
def test_import_impulso_missing_raises(monkeypatch):
    """Without Impulso, the error names the extra and the install command."""
    from pymc_marketing.mmm.var_baseline import _import_impulso

    monkeypatch.setitem(sys.modules, "impulso", None)

    with pytest.raises(ImportError, match=r"pip install 'pymc-marketing\[var\]'"):
        _import_impulso()


@pytest.mark.xfail(
    strict=True, reason="pymc_marketing.mmm.var_baseline is not implemented yet"
)
def test_pymc_marketing_imports_without_impulso():
    """Neither the package nor the effect's module needs Impulso at import time."""
    code = (
        "import sys; sys.modules['impulso'] = None; "
        "import pymc_marketing.mmm; import pymc_marketing.mmm.var_baseline"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
