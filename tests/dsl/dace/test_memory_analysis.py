import dace

from ndsl.dsl.dace.utils import memory_static_analysis, report_memory_static_analysis


def test_memory_static_analysis_includes_nested_lifetimes() -> None:
    sdfg = dace.SDFG("memory_test")
    state = sdfg.add_state()
    nested_sdfg = dace.SDFG("nested")
    nested_sdfg.add_array("scratch", [4], dace.float32, transient=True)
    nested_sdfg.add_array("persistent", [3], dace.float64, transient=False)
    nested_sdfg.add_scalar("scratch_scalar", dace.float32, transient=True)
    nested_sdfg.add_scalar("persistent_scalar", dace.float64, transient=False)
    nested_sdfg.add_array("empty_shape", [], dace.float32, transient=True)
    nested_sdfg.add_array("kilobyte_array", [256], dace.float64, transient=True)
    nested_sdfg.add_array("megabyte_array", [131072], dace.float64, transient=True)
    state.add_nested_sdfg(nested_sdfg, set(), set(), {})

    allocations = memory_static_analysis(sdfg)
    default_storage = allocations[dace.StorageType.Default]

    assert default_storage.transient_unreferenced_in_bytes == 1_050_648
    assert default_storage.non_transient_unreferenced_in_bytes == 32
    assert default_storage.unreferenced_in_bytes == 1_050_680
    assert default_storage.top_level_in_bytes == 0

    report = report_memory_static_analysis(sdfg, allocations, detail_report=True)
    assert "Transient: arrays 3, scalars 2" in report
    assert "Non-transient: arrays 1, scalars 1" in report
    assert "array | transient | unreferenced | not pooled | nested | 1D | 4 | 16.00 B | scratch" in report
    assert "array | non-transient | unreferenced | not pooled | nested | 1D | 3 | 24.00 B | persistent" in report
    assert "scalar | transient | unreferenced | not pooled | nested | 0D | - | 4.00 B | scratch_scalar" in report
    assert "scalar | non-transient | unreferenced | not pooled | nested | 0D | - | 8.00 B | persistent_scalar" in report
    assert "scalar | transient | unreferenced | not pooled | nested | 0D | - | 4.00 B | empty_shape" in report
    assert "2.00 KiB | kilobyte_array" in report
    assert "1.00 MiB | megabyte_array" in report
