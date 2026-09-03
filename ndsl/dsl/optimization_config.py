import enum
import os
from dataclasses import dataclass, field


class OptimizationHint(enum.Enum):
    """Hint for the configuration system that will drive the OptimizationOption.AUTO value"""

    SERIAL = enum.auto()
    PARALLEL = enum.auto()


class OptimizationOption(enum.Enum):
    """Options for configuration element. AUTO will rely on the best guess default"""

    AUTO = enum.auto()  # Best guess relying on the OptimizationHint
    APPLY = enum.auto()  # Pass will always be applied
    DO_NOT_APPLY = enum.auto()  # Pass will never be applied


@dataclass
class OptimizationConfig:
    """Configuration for running the full-program optimization"""

    @dataclass
    class Tree:
        """Optimization using the Schedule Tree IR"""

        @dataclass
        class Merger:
            enabled: bool = True
            """Enable cartesian axis merging."""

            overcompute: bool = (
                os.getenv("NDSL_STREE_OVERCOMPUTE_MERGE", "True").lower() == "true"
            )
            """When merging allow maps of different sizes to merge by inserting an `if` guard."""

            order: str = "default"
            """
            Allows to manually override the merging order (e.g. `KJI` will merge `K`, then `J`, then `I`).
            The default follows loop order of the backend given to `CartesianMerge`.
            """

        enabled: bool = os.getenv("NDSL_STREE_OPT", "False").lower() == "true"
        """Enable Schedule Tree transformations."""

        # TODO: Is it safe? Deactivate by default for now
        inline_K_loops_size_one: bool = False
        """"Remove serial for loops of size one in the K-axis."""

        kernelize: OptimizationOption = OptimizationOption.AUTO
        """Enable maximizing 3-axis kernelization by duplicating maps (GPU only)."""

        merger: Merger = field(default_factory=Merger)
        """Configuration object for cartesian axis merging."""

        refine_transients: bool = True
        """Reduce dimensionality of transient arrays based on their usage."""

    @dataclass
    class GPU:
        """Optimization dedicated for GPU"""

        common_gpu_xforms: OptimizationOption = OptimizationOption.DO_NOT_APPLY
        """DaCe common xforms bundled in `apply_gpu_transformations`"""

    stree: Tree = field(default_factory=Tree)
    """Schedule Tree optimization options"""

    gpu: GPU = field(default_factory=GPU)
    """GPU-only optimization options"""

    hint: OptimizationHint = OptimizationHint.PARALLEL
    """Hint for all optimizations passes"""

    name: str = "unset"
