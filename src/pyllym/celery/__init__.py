"""Optional Celery integration for running pyllym operations on workers.

Requires the ``celery`` extra (``pip install pyllym[celery]``).

:func:`create_tasks` is a small factory that registers ready-made tasks
(``ask``, ``embed``, ``paint``, ``speak``, ``transcribe``, ``moderate``)
on your Celery app::

    from celery import Celery
    from pyllym.celery import create_tasks

    app = Celery("worker", broker="redis://localhost:6379/0")
    tasks = create_tasks(app)

    result = tasks.ask.delay("What's the capital of France?", model="gpt-5.4")
    result.get()  # -> the assistant Message as a dict

Celery workers are synchronous; each task runs the underlying coroutine on
a fresh event loop via :func:`run_async`, which also closes pyllym's shared
HTTP pools before the loop shuts down. Use :func:`run_async` directly when
writing your own tasks around richer pyllym features (tools, agents,
callbacks) that can't be expressed through broker-serializable arguments.
"""

from __future__ import annotations

try:
    import celery as _celery  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "pyllym.celery requires the 'celery' extra: pip install 'pyllym[celery]'"
    ) from exc

from .tasks import Tasks, create_tasks, run_async

__all__ = ["Tasks", "create_tasks", "run_async"]
