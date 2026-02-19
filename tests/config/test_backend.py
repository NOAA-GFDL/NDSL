import pytest

from ndsl import Backend


def test_backend_building():
    Backend("st:python:cpu:IJK")
    Backend("st:numpy:cpu:IJK")
    Backend("st:gt:cpu:IJK")
    Backend("st:gt:cpu:KJI")
    Backend("st:gt:gpu:KJI")
    Backend("st:dace:cpu:IJK")
    Backend("orch:dace:cpu:IJK")
    Backend("st:dace:cpu:KIJ")
    Backend("orch:dace:cpu:KIJ")
    Backend("st:dace:cpu:KJI")
    Backend("orch:dace:cpu:KJI")
    Backend("st:dace:gpu:KJI")
    Backend("orch:dace:gpu:KJI")

    with pytest.raises(ValueError):
        Backend("bad:name:good:number")


def test_backend_operators():
    backend_A = Backend("st:numpy:cpu:IJK")
    backend_B = Backend("st:numpy:cpu:IJK")

    assert backend_A == backend_B
    assert not (backend_A != backend_B)
