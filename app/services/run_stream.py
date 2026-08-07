"""Bounded transport buffer for synchronous work feeding an SSE response."""

from __future__ import annotations

import queue
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Event, Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class StreamItem:
    event: str
    payload: dict[str, Any]


_STREAM_CLOSED = object()


class StreamExecutorSaturatedError(RuntimeError):
    """Raised when all stream workers and pending slots are occupied."""


class BoundedStreamExecutor:
    """Thread pool with an explicit cap on running plus pending work."""

    def __init__(self, *, max_workers: int, max_pending: int) -> None:
        if max_workers < 1 or max_pending < 0:
            raise ValueError("stream workers must be positive and pending slots non-negative")
        self._capacity = BoundedSemaphore(max_workers + max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aker-stream",
        )

    def submit(self, work: Callable[[], Any]) -> Future[Any]:
        if not self._capacity.acquire(blocking=False):
            raise StreamExecutorSaturatedError("stream worker capacity is exhausted")
        try:
            future = self._executor.submit(work)
        except BaseException:
            self._capacity.release()
            raise
        future.add_done_callback(self._release_capacity)
        return future

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _release_capacity(self, _future: Future[Any]) -> None:
        self._capacity.release()


class RunStreamBuffer:
    """Bound memory while preserving control/final events over token deltas."""

    def __init__(self, max_size: int) -> None:
        if max_size < 2:
            raise ValueError("stream queue must hold at least two events")
        self._queue: queue.Queue[StreamItem | object] = queue.Queue(maxsize=max_size)
        self._publish_lock = Lock()
        self.cancelled = Event()
        self.dropped_tokens = 0
        self._closed = False

    @property
    def max_size(self) -> int:
        return self._queue.maxsize

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def publish(self, event: str, payload: dict[str, Any]) -> bool:
        if self.cancelled.is_set() or self._closed:
            return False
        item = StreamItem(event=event, payload=dict(payload))
        with self._publish_lock:
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                if event == "token":
                    self.dropped_tokens += 1
                    return False
                self._discard_oldest()
                self._queue.put_nowait(item)
                return True

    def publish_token(self, token: str) -> None:
        self.publish("token", {"delta": token})

    def close(self) -> None:
        with self._publish_lock:
            if self._closed:
                return
            self._closed = True
            if self._queue.full():
                self._discard_oldest()
            self._queue.put_nowait(_STREAM_CLOSED)

    def get(self, timeout: float) -> StreamItem | None:
        item = self._queue.get(timeout=timeout)
        if item is _STREAM_CLOSED:
            return None
        if not isinstance(item, StreamItem):
            raise AssertionError("unexpected stream buffer item")
        return item

    def request_cancellation(self) -> None:
        self.cancelled.set()

    def _discard_oldest(self) -> None:
        try:
            discarded = self._queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(discarded, StreamItem) and discarded.event == "token":
            self.dropped_tokens += 1


class RunCancellationRegistry:
    """Propagate API cancellation to an in-process synchronous run worker."""

    def __init__(self) -> None:
        self._events: dict[str, Event] = {}
        self._lock = Lock()

    def register(self, run_id: str, event: Event) -> None:
        with self._lock:
            self._events[run_id] = event

    def request(self, run_id: str) -> bool:
        with self._lock:
            event = self._events.get(run_id)
        if event is None:
            return False
        event.set()
        return True

    def unregister(self, run_id: str, event: Event) -> None:
        with self._lock:
            if self._events.get(run_id) is event:
                self._events.pop(run_id, None)


active_run_cancellations = RunCancellationRegistry()
