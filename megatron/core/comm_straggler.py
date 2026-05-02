import os
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist


_NVTX_ENV = "ENABLE_COMM_STRAGGLER_NVTX"


class _Context:
    iteration_id: Optional[int] = None
    microbatch_id: Optional[Any] = None
    _iteration_nvtx_active = False
    _microbatch_nvtx_active = False
    _module_nvtx_stack = []


def _enabled() -> bool:
    return os.getenv(_NVTX_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _rank_label() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"rank={dist.get_rank()}"
    return "rank=unknown"


def _push(message: str) -> bool:
    if not _enabled() or not torch.cuda.is_available():
        return False
    try:
        torch.cuda.nvtx.range_push(message)
    except Exception:
        return False
    return True


def _pop(active: bool) -> None:
    if not active:
        return
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def set_iteration(iteration_id: int) -> None:
    if _Context._iteration_nvtx_active:
        clear_iteration()
    _Context.iteration_id = int(iteration_id)
    _Context._iteration_nvtx_active = _push(
        f"comm_straggler.iteration {_rank_label()} iter={_Context.iteration_id}"
    )


def clear_iteration() -> None:
    clear_microbatch()
    _pop(_Context._iteration_nvtx_active)
    _Context._iteration_nvtx_active = False
    _Context.iteration_id = None


def set_microbatch(microbatch_id: Any) -> None:
    if _Context._microbatch_nvtx_active:
        clear_microbatch()
    _Context.microbatch_id = microbatch_id
    _Context._microbatch_nvtx_active = _push(
        "comm_straggler.microbatch "
        f"{_rank_label()} iter={_Context.iteration_id} microbatch={microbatch_id}"
    )


def clear_microbatch() -> None:
    _pop(_Context._microbatch_nvtx_active)
    _Context._microbatch_nvtx_active = False
    _Context.microbatch_id = None


@contextmanager
def module_scope(module_name: str, layer_id: Optional[Any] = None):
    message = (
        "comm_straggler.module "
        f"{_rank_label()} iter={_Context.iteration_id} "
        f"microbatch={_Context.microbatch_id} module={module_name}"
    )
    if layer_id is not None:
        message += f" layer={layer_id}"
    active = _push(message)
    _Context._module_nvtx_stack.append(active)
    try:
        yield
    finally:
        active = _Context._module_nvtx_stack.pop() if _Context._module_nvtx_stack else False
        _pop(active)


def mark_iteration_end() -> None:
    clear_iteration()


def finalize() -> None:
    clear_iteration()


def attach_model_hooks(model: Any) -> None:
    # Kept as a no-op compatibility shim for the old lightweight profiler branch.
    return None
