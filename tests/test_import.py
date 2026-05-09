from __future__ import annotations

import topogpu


def test_import_has_version() -> None:
    assert topogpu.__version__ == "0.1.0"
