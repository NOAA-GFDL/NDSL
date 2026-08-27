import re

from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def replace_variable_name(code_block: CodeBlock, old_var: str, new_var: str) -> None:
    """Replace a variable in a CodeBlock using regular expression (full word replacement)"""
    code_block.as_string = re.sub(rf"\b{old_var}\b", new_var, code_block.as_string)


def make_unique_container_name(name: str, root: tn.ScheduleTreeRoot) -> str:
    """Make a unique name from a given dace.Data registered within the root containers.

    Simply append `_X` until unicity is achieved.
    """

    # Dev TODO: caching the unique names and their counter would skip a few
    #           while loop iteration (and string compare) we just need a central
    #           bookeeper to do so
    if name not in root.containers:
        raise NameError(f"Unknown name {name} in root container")

    counter = 0
    candidate = f"{name}_{counter}"
    while candidate in root.containers:
        counter += 1
        candidate = f"{name}_{counter}"

    return candidate
