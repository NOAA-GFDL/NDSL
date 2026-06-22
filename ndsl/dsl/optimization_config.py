import os
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    @dataclass
    class Tree:
        """Optimization using the Schedule Tree IR"""

        @dataclass
        class Merger:
            enabled: bool = False
            """Enable cartesian axis merging."""

            overcompute: bool = (
                os.getenv("NDSL_STREE_OVERCOMPUTE_MERGE", "True").lower() == "true"
            )
            """When merging allow maps of different sizes to merge by inserting an `if` guard."""

        enabled: bool = os.getenv("NDSL_STREE_OPT", "False").lower() == "true"
        """Enable Schedule Tree transformations."""

        # TODO: Is it safe? Deactivate by default for now
        inline_K_loops_size_one: bool = False
        """"Remove serial for loops of size one in the K-axis."""

        kernelize: bool = True
        """Enable maximizing 3-axis kernelization by duplicating maps (GPU only)."""

        merger: Merger = field(default_factory=Merger)
        """Configuration object for cartesian axis merging."""

        refine_transients: bool = True
        """Reduce dimensionality of transient arrays based on their usage."""

    @dataclass
    class GPU:
        """Optimization dedicated for GPU"""

        common_gpu_xforms: bool = False
        """DaCe common xforms bundled in `apply_gpu_transformations`"""

    stree: Tree = field(default_factory=Tree)
    gpu: GPU = field(default_factory=GPU)
