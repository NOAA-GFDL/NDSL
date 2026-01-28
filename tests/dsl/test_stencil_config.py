import pytest

from ndsl import CompilationConfig, DaceConfig, StencilConfig
from ndsl.config.backend import backend_python, Backend


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("rebuild", [True, False])
@pytest.mark.parametrize("format_source", [True, False])
@pytest.mark.parametrize("compare_to_numpy", [True, False])
@pytest.mark.parametrize("backend", [Backend.python(), Backend("st:gt:gpu:KJI")])
def test_same_config_equal(
    backend: Backend,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    compare_to_numpy: bool,
) -> None:
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
    )
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=False,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    same_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=False,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config == same_config


def test_different_backend_not_equal(
    backend: Backend = backend_python,
    rebuild: bool = True,
    validate_args: bool = True,
    format_source: bool = True,
    device_sync: bool = False,
    compare_to_numpy: bool = True,
) -> None:
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
    )
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            Backend("st:python:cpu:numpy"),
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config


def test_different_rebuild_not_equal(
    backend: Backend = backend_python,
    rebuild: bool = True,
    validate_args: bool = True,
    format_source: bool = True,
    device_sync: bool = False,
    compare_to_numpy: bool = True,
) -> None:
    dace_config = DaceConfig(
        communicator=None,
        backend=backend,
    )
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=not rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config


def test_different_device_sync_not_equal(
    rebuild: bool = True,
    validate_args: bool = True,
    format_source: bool = True,
    device_sync: bool = False,
    compare_to_numpy: bool = True,
) -> None:
    dace_config = DaceConfig(
        communicator=None,
        backend=Backend("st:gt:gpu:KJI"),
    )
    config = StencilConfig(
        compilation_config=CompilationConfig(
            Backend("st:gt:gpu:KJI"),
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            Backend("st:gt:gpu:KJI"),
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=not device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config


def test_different_validate_args_not_equal(
    backend: Backend = backend_python,
    rebuild: bool = True,
    validate_args: bool = True,
    format_source: bool = True,
    device_sync: bool = False,
    compare_to_numpy: bool = True,
) -> None:
    dace_config = DaceConfig(
        None,
        backend,
    )
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=not validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config


def test_different_format_source_not_equal(
    backend: Backend = backend_python,
    rebuild: bool = True,
    validate_args: bool = True,
    format_source: bool = True,
    device_sync: bool = False,
    compare_to_numpy: bool = True,
) -> None:
    dace_config = DaceConfig(communicator=None, backend=backend)
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=not format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config


@pytest.mark.parametrize("compare_to_numpy", [True, False])
def test_different_compare_to_numpy_not_equal(
    compare_to_numpy: bool,
    backend: Backend = backend_python,
    device_sync: bool = False,
    format_source: bool = True,
    rebuild: bool = True,
    validate_args: bool = False,
) -> None:
    dace_config = DaceConfig(communicator=None, backend=backend)
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=compare_to_numpy,
        dace_config=dace_config,
    )

    different_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=rebuild,
            validate_args=validate_args,
            format_source=format_source,
            device_sync=device_sync,
        ),
        compare_to_numpy=not compare_to_numpy,
        dace_config=dace_config,
    )
    assert config != different_config
