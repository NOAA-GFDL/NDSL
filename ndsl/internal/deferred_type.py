import abc
from typing import Any, Type
from types import FrameType
import ast
import inspect


def get_lhs_name(frame: FrameType | None) -> str:
    """Inspect the back frame to retrieve the LHS of an assign
    operation, will fail if any of those aren't true"""
    if frame is None:
        raise RuntimeError("Code frame un-inspectable.")
    previous_frame = frame.f_back
    if previous_frame is None:
        raise RuntimeError("LHS retrieval failed: no back frame")
    code_context = inspect.getframeinfo(previous_frame).code_context
    if code_context is None:
        raise RuntimeError("LHS retrieval failed: code context cannot be read")
    module = ast.parse(code_context[0])
    if len(module.body) == 0 or not isinstance(
        module.body[0], ast.Assign | ast.AugAssign | ast.AnnAssign
    ):
        raise RuntimeError("LHS retrieval failed: AST body malformed")
    if isinstance(module.body[0], ast.Assign):
        if len(module.body[0].targets) != 1:
            raise RuntimeError(
                "Data dimension field declare: please assing only variable to the function"
            )
        target_node = module.body[0].targets[0]
    else:
        target_node = module.body[0].target
    if not isinstance(target_node, ast.Name):
        raise RuntimeError("LHS retrieval failed: AST LHS malformed")
    return target_node.id


class StencilTypeRegistrar:
    """Type registrar that knows how to give back true type.

    This acts as a registrar and also a dynamic resolver to be able to swap to the true
    type before
    """

    @classmethod
    @abc.abstractmethod
    def get(cls, name: str) -> Any:
        pass


class StencilDeferredType:
    """Placeholder stencil type for future (deferred) type resolve.

    This type should carry all the API of the true type given by `resolve`
    in order to be able to work with it as if it was resolved.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __descriptor__(self):
        # Ignore, use JIT
        return None

    @classmethod
    @abc.abstractmethod
    def resolve(cls) -> Type[StencilTypeRegistrar]: ...
