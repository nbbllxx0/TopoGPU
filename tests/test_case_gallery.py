from __future__ import annotations

import topogpu as tg


def test_case_gallery_cantilever_fields() -> None:
    problem = tg.gallery.cantilever_3d(nel=(4, 3, 2), volfrac=0.25)
    problem.validate()
    assert problem.nel == (4, 3, 2)
    assert problem.volfrac == 0.25
    assert problem.metadata["support"] == "xmin"
