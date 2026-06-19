import os
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    @dataclass
    class TreeConfig:
        @dataclass
        class MergerConfig:
            overcompute: bool = (
                os.getenv("NDSL_STREE_OVERCOMPUTE_MERGE", "True") == "True"
            )

        enabled: bool = os.getenv("NDSL_STREE_OPT", "False") == "True"
        merger: MergerConfig = field(default_factory=MergerConfig)

    stree: TreeConfig = field(default_factory=TreeConfig)
