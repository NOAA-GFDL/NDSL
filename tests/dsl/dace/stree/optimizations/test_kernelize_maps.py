import pytest
from dace import nodes
from dace.sdfg.state import LoopRegion

from ndsl import Backend, NDSLRuntime, OptimizationConfig, orchestrate
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, interval
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import get_SDFG_and_purge
from tests.dsl.dace.stree.optimizations import Factories


def stencil_kernelize(in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
    with computation(PARALLEL), interval(...):
        value = in_field * 2
        tmp = value

    with computation(FORWARD), interval(0, -1):
        tmp = 0.5 * (tmp + tmp[0, 0, 1])

    with computation(PARALLEL), interval(...):
        out_field = tmp


def stencil_only_serial_noop(
    in_field: FloatField, out_field: FloatField
) -> None:  # type:ignore
    with computation(FORWARD), interval(...):
        tmp = in_field

    with computation(BACKWARD), interval(...):
        out_field = tmp


def stencil_only_parallel_noop(
    in_field: FloatField, out_field: FloatField
) -> None:  # type:ignore
    with computation(PARALLEL), interval(0, 2):
        out_field = in_field

    with computation(PARALLEL), interval(-2, None):
        out_field = in_field + 1


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        optimization_config = OptimizationConfig(
            stree=OptimizationConfig.Tree(
                enabled=True,
                merger=OptimizationConfig.Tree.Merger(enabled=True),
            )
        )
        super().__init__(stencil_factory, optimization_config)

        methods_to_orchestrate = [
            "kernelize_k",
            "only_serial_noop",
            "only_parallel_noop",
        ]
        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
                optimization_config=optimization_config,
            )

        self._stencil_kernelize_k = stencil_factory.from_dims_halo(
            func=stencil_kernelize,
            compute_dims=(I_DIM, J_DIM, K_DIM),
        )
        self._stencil_only_serial_noop = stencil_factory.from_dims_halo(
            func=stencil_only_serial_noop,
            compute_dims=(I_DIM, J_DIM, K_DIM),
        )
        self._stencil_only_parallel_noop = stencil_factory.from_dims_halo(
            func=stencil_only_parallel_noop,
            compute_dims=(I_DIM, J_DIM, K_DIM),
        )

    def kernelize_k(self, in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._stencil_kernelize_k(in_field, out_field)

    def only_serial_noop(self, in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._stencil_only_serial_noop(in_field, out_field)

    def only_parallel_noop(self, in_field: FloatField, out_field: FloatField) -> None:  # type: ignore
        self._stencil_only_parallel_noop(in_field, out_field)


class TestKernelizeMaps:
    @pytest.fixture(
        params=[
            "orch:dace:cpu:IJK",
            pytest.param("orch:dace:gpu:IJK", marks=pytest.mark.gpu),
        ]
    )
    def factories(self, request: pytest.FixtureRequest) -> Factories:
        domain = (3, 4, 5)
        return get_factories_single_tile(
            nx=domain[0],
            ny=domain[1],
            nz=domain[2],
            nhalo=0,
            backend=Backend(request.param),
        )

    def test_kernelize_k_gpu(self, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        code = OrchestratedCode(stencil_factory)

        in_field = quantity_factory.ones((I_DIM, J_DIM, K_DIM), "")
        out_field = quantity_factory.zeros((I_DIM, J_DIM, K_DIM), "")

        code.kernelize_k(in_field, out_field)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)

        if stencil_factory.backend.is_gpu_backend():
            # check for kernelization
            all_maps = [
                node
                for node, _ in precompiled_sdfg.sdfg.all_nodes_recursive()
                if isinstance(node, nodes.MapEntry)
            ]

            ij_maps = 0
            ijk_maps = 0
            for map_entry in all_maps:
                if map_entry.map.params == ["__i", "__j"]:
                    ij_maps += 1
                elif len(map_entry.map.params) == 3:
                    params = map_entry.map.params
                    k_param = params[2]
                    if (
                        params[0:2] == ["__i", "__j"]
                        and isinstance(k_param, str)
                        and k_param.startswith("__k")
                    ):
                        ijk_maps += 1

            # expect two IJK-maps and one IJ-map
            assert ij_maps == 1
            assert ijk_maps == 2
            assert len(all_maps) == 3

            all_loop_regions = [
                node
                for node, _ in precompiled_sdfg.sdfg.all_nodes_recursive()
                if isinstance(node, LoopRegion)
            ]
            # expect one k-loop is preserved
            assert len(all_loop_regions) == 1
            assert all_loop_regions[0].loop_variable.startswith("__k")
        else:
            # check that we keep IJ loops merged
            all_maps = [
                node
                for node, _ in precompiled_sdfg.sdfg.all_nodes_recursive()
                if isinstance(node, nodes.MapEntry)
            ]

            ij_maps = 0
            k_maps = 0
            for map_entry in all_maps:
                if map_entry.map.params == ["__i", "__j"]:
                    ij_maps += 1
                elif len(map_entry.map.params) == 1:
                    param = map_entry.map.params[0]
                    if isinstance(param, str) and param.startswith("__k"):
                        k_maps += 1

            # expect one IJ-map and two K-maps
            assert ij_maps == 1
            assert k_maps == 2
            assert len(all_maps) == 3

            all_loop_regions = [
                node
                for node, _ in precompiled_sdfg.sdfg.all_nodes_recursive()
                if isinstance(node, LoopRegion)
            ]
            # expect one k-loop is preserved
            assert len(all_loop_regions) == 1
            assert all_loop_regions[0].loop_variable.startswith("__k")
