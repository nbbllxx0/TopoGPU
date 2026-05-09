from __future__ import annotations

from topogpu.cli import main


def test_cli_version(capsys) -> None:
    assert main(["--version"]) == 0
    assert "0.1.0" in capsys.readouterr().out


def test_cli_list_cases(capsys) -> None:
    assert main(["list-cases"]) == 0
    assert "cantilever_3d" in capsys.readouterr().out


def test_cli_cite(capsys) -> None:
    assert main(["cite"]) == 0
    assert "TopoGPU" in capsys.readouterr().out

