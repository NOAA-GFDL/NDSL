import pytest

from ndsl import Backend
from ndsl.config import BackendLoopOrder


def test_backend_building() -> None:
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

    unknown_backend = "bad:name:good:number"
    with pytest.raises(ValueError, match=f"Unknown {unknown_backend}, options are .*"):
        Backend(unknown_backend)


def test_backend_operators() -> None:
    backend_A = Backend("st:numpy:cpu:IJK")
    backend_B = Backend("st:numpy:cpu:IJK")

    assert backend_A == backend_B
    assert not (backend_A != backend_B)


def test_equivalent_backend() -> None:
    orchestrated_backend = Backend("orch:dace:cpu:IJK")
    stencil_backend = orchestrated_backend.equivalent_stencil_backend()
    assert stencil_backend.is_stencil()
    assert stencil_backend.equivalent_orchestration_backend() == orchestrated_backend

    cpu_backend = Backend("orch:dace:cpu:KJI")
    gpu_backend = cpu_backend.equivalent_gpu_backend()
    assert gpu_backend.is_gpu_backend()
    assert gpu_backend.equivalent_cpu_backend() == cpu_backend

    ijk_backend = Backend("st:dace:cpu:IJK")
    kji_backend = ijk_backend.equivalent_backend_with_loop_order(BackendLoopOrder.KJI)
    assert kji_backend.loop_order == BackendLoopOrder.KJI
    assert (
        kji_backend.equivalent_backend_with_loop_order(BackendLoopOrder.IJK)
        == ijk_backend
    )
