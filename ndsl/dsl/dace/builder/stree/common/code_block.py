import re

from dace.properties import CodeBlock


def replace_variable_name(code_block: CodeBlock, old_var: str, new_var: str) -> None:
    """Replace a variable in a CodeBlock using regular expression (full word replacement)"""
    code_block.as_string = re.sub(rf"\b{old_var}\b", new_var, code_block.as_string)
