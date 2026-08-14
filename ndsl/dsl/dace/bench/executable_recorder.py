import pickle
from pathlib import Path

from dace.sdfg import SDFG

from ndsl.dsl.dace.dace_executable import DaceExecutable

_BUNDLE_DIRECTORY_NAME = "DaceExecutableRecorder"
_GT4PY_OPT_SDFG_NAME = "gt4py_sdfg"
_OPT_SDFG_NAME = "opt_sdfg"


class DaceExecutableRecorder:
    """☢️ Many step recorder, carries a `_ready` flag when everything has been properly set"""

    def __init__(self) -> None:
        self._ready = False

        self._args = None
        self._args_hash = None
        self._skip_hash = None
        self._gt4py_sdfg = None
        self._opt_sdfg = None

    def _make_ready(self):
        if (
            self._args is None
            or self._args_hash is None
            or self._skip_hash is None
            or self._gt4py_sdfg is None
            or self._opt_sdfg is None
        ):
            return
        self._ready = True

    def set_exe_args(self, exe: DaceExecutable) -> None:
        self._args = exe.arguments
        self._args_hash = exe.arguments_hash
        self._skip_hash = exe._skip_hash
        self._opt_sdfg = exe.compiled_sdfg.sdfg
        self._make_ready()

    def set_gt4py_sdfg(self, unopt_sdfg: SDFG) -> None:
        self._gt4py_sdfg = unopt_sdfg
        self._make_ready()

    def save(self):
        if not self._ready:
            raise RuntimeError(f"DaceExecutableRecorder is not ready: {self}")

        assert self._opt_sdfg

        bundle_dir = self._opt_sdfg.build_folder + "/" + _BUNDLE_DIRECTORY_NAME
        Path(bundle_dir).mkdir(exist_ok=True, parents=True)

        with open(f"{bundle_dir}/de_args.pickle", "wb") as f:
            pickle.dump(self._args, f)

        with open(f"{bundle_dir}/de_args_hash.pickle", "wb") as f:
            pickle.dump(self._args_hash, f)

        with open(f"{bundle_dir}/de__skip_hash.pickle", "wb") as f:
            pickle.dump(self._skip_hash, f)

        assert self._gt4py_sdfg
        assert self._opt_sdfg
        self._gt4py_sdfg.save(
            f"{bundle_dir}/{_GT4PY_OPT_SDFG_NAME}.sdfgz", compress=True
        )
        self._opt_sdfg.save(f"{bundle_dir}/{_OPT_SDFG_NAME}.sdfgz", compress=True)

    def load(self, bundle_dir: str):
        with open(f"{bundle_dir}/de_args.pickle", "rb") as f:
            self._args = pickle.load(f)

        with open(f"{bundle_dir}/de_args_hash.pickle", "rb") as f:
            self._args_hash = pickle.load(f)

        with open(f"{bundle_dir}/de__skip_hash.pickle", "rb") as f:
            self._skip_hash = pickle.load(f)

        self._gt4py_sdfg = SDFG.from_file(f"{bundle_dir}/{_GT4PY_OPT_SDFG_NAME}.sdfgz")
        self._opt_sdfg = SDFG.from_file(f"{bundle_dir}/{_OPT_SDFG_NAME}.sdfgz")

        self._make_ready()
