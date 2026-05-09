from __future__ import annotations

from topogpu.cli import main


def test_cli_verify_small_help_path(capsys) -> None:
    assert main(["cite"]) == 0
    assert "10.5281/zenodo.20100693" in capsys.readouterr().out
