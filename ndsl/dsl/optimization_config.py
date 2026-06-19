import os
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    @dataclass
    class Tree:
        """Optimization using the Schedule Tree IR"""

        @dataclass
        class Merger:
            overcompute: bool = (
                os.getenv("NDSL_STREE_OVERCOMPUTE_MERGE", "True").lower() == "true"
            )
            """When merging allow map of different size to merge by inserting an if guard"""

        enabled: bool = os.getenv("NDSL_STREE_OPT", "False").lower() == "true"
        """Enable Schedule Tree transformations"""
        merger: Merger = field(default_factory=Merger)

    @dataclass
    class GPU:
        """Optimization dedicated for GPU"""

        common_gpu_xforms: bool = True
        """DaCe common xforms bundled in `apply_gpu_transformations`"""

    stree: Tree = field(default_factory=Tree)
    gpu: GPU = field(default_factory=GPU)
