from ndsl.dsl.optimization_config import OptimizationConfig, OptimizationOption


def test_get_default_converts_boolean_optimization_options(
    tmp_path, monkeypatch
) -> None:
    config_file = tmp_path / "optimization.yaml"
    config_file.write_text("stree:\n  kernelize: true\nloop_vectorization: false\n")
    monkeypatch.setenv("NDSL_OPTIMIZATION_CONFIG", str(config_file))

    config = OptimizationConfig.get_default()

    assert config.stree.kernelize is OptimizationOption.APPLY
    assert config.loop_vectorization is OptimizationOption.DO_NOT_APPLY
