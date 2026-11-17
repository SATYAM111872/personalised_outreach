"""Optional LangSmith tracing. Off unless .env sets:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=zamp-outreach        # optional; default project otherwise
    LANGSMITH_HIDE_INPUTS=true             # optional: keep timings/structure, drop data
    LANGSMITH_HIDE_OUTPUTS=true            # optional

With tracing on, each prospect is one nested trace: lookup, each research round
(searches, page fetches, LLM calls), drafting and checks. Gemini calls made
through LangChain are traced automatically; `traceable` adds the plain-Python
steps. Traces hold prospect names, notes, page text and drafts, and are stored
on LangSmith's servers.

`Pool` replaces ThreadPoolExecutor so work done in worker threads (searches,
fetches, LLM timeouts) nests under its parent step instead of appearing as a
separate trace. Without the langsmith package both fall back to no-ops.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()


class _ContextPool(ThreadPoolExecutor):
    """ThreadPoolExecutor whose tasks run in a copy of the submitter's context, so
    observability.py's run id / stage reach worker threads (searches, fetches)
    even when langsmith is not installed."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree
    from langsmith.utils import ContextThreadPoolExecutor as Pool
except ImportError:  # tracing unavailable: same API, does nothing
    _traceable, Pool = None, _ContextPool

    def get_current_run_tree():
        return None


def traceable(**kw):
    """@traceable(name=..., run_type=...) or a no-op without langsmith."""
    if _traceable is None:
        return lambda f: f
    return _traceable(**kw)


def _short(v, limit=2000):
    """Inputs shown in a trace: long text trimmed, internal objects by name."""
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + f"... [{len(v)} chars]"
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _short(x, limit) for k, x in list(v.items())[:50]}
    if isinstance(v, (list, tuple, set, frozenset)):
        items = list(v)
        out = [_short(x, limit) for x in items[:30]]
        return out + [f"... {len(items) - 30} more"] if len(items) > 30 else out
    return f"<{type(v).__name__}>"


def trim_inputs(inputs):
    """process_inputs hook: drops the Emitter, trims page text and big lists."""
    return {k: _short(v) for k, v in inputs.items() if k not in ("em", "on_event")}


def trim_outputs(out):
    """process_outputs hook: same trimming for return values."""
    return {"output": _short(out)}


def step(name, run_type="chain"):
    """Decorator for a pipeline step, with trimmed inputs and outputs."""
    return traceable(name=name, run_type=run_type, process_inputs=trim_inputs, process_outputs=trim_outputs)


def tag_current(**metadata):
    """Add metadata (run id, company, final status) to the current trace step."""
    rt = get_current_run_tree()
    if rt is not None:
        rt.metadata.update({k: v for k, v in metadata.items() if v is not None})
