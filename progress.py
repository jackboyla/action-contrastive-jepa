"""Dependency-free console progress bars with ETA.

These helpers render progress as standalone lines (one per refresh) instead of
carriage-return animations. This keeps progress and ETA readable in captured,
non-interactive logs such as Modal/CI, where ``\\r``-based bars (tqdm, Lightning's
default) are unreliable, while still being perfectly readable in a terminal.
"""

from __future__ import annotations

import sys
import time


def format_duration(seconds) -> str:
    """Format a duration in seconds as ``M:SS`` or ``H:MM:SS`` (``?`` if unknown)."""
    if seconds is None:
        return "?"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if seconds != seconds or seconds < 0 or seconds == float("inf"):  # NaN/inf/neg
        return "?"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_bar(fraction, width: int = 24) -> str:
    """Render an ASCII progress bar like ``[#######-----------------]``."""
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        fraction = 0.0
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_rate(rate: float) -> str:
    """Format an iterations-per-second rate compactly."""
    if rate <= 0:
        return "?"
    if rate < 100:
        return f"{rate:.2f} it/s"
    return f"{rate:.0f} it/s"


class ProgressPrinter:
    """Throttled, log-friendly progress bar with ETA for plain Python loops.

    Every emitted update is a standalone line, so it is readable in
    non-interactive logs (Modal/CI) as well as terminals. Updates are throttled
    by ``min_interval_s`` so fast loops do not flood the logs; the final update
    is always emitted.

    Example::

        progress = ProgressPrinter(total=len(items), label="eval")
        for item in items:
            do_work(item)
            progress.update(1)
        progress.close()
    """

    def __init__(
        self,
        total,
        label: str = "progress",
        *,
        bar_width: int = 24,
        min_interval_s: float = 2.0,
        stream=None,
    ):
        self.total = int(total) if total else 0
        self.label = label
        self.bar_width = int(bar_width)
        self.min_interval_s = float(min_interval_s)
        self.stream = stream if stream is not None else sys.stdout
        self.count = 0
        self._start = time.perf_counter()
        self._last_emit_time = 0.0
        self._last_emit_count = -1

    def update(self, n: int = 1, *, suffix: str = "", force: bool = False) -> None:
        self.count += int(n)
        self._maybe_emit(suffix=suffix, force=force)

    def close(self, suffix: str = "") -> None:
        self._maybe_emit(suffix=suffix, force=True)

    def _maybe_emit(self, *, suffix: str, force: bool) -> None:
        is_last = bool(self.total) and self.count >= self.total
        now = time.perf_counter()
        if (
            not force
            and not is_last
            and now - self._last_emit_time < self.min_interval_s
        ):
            return
        if self.count == self._last_emit_count and not force:
            return
        self._last_emit_time = now
        self._last_emit_count = self.count
        self.stream.write(self._render(suffix) + "\n")
        self.stream.flush()

    def _render(self, suffix: str) -> str:
        elapsed = time.perf_counter() - self._start
        rate = self.count / elapsed if elapsed > 0 and self.count else 0.0
        if self.total:
            fraction = self.count / self.total
            remaining = (self.total - self.count) / rate if rate > 0 else None
            head = (
                f"[{self.label}] {render_bar(fraction, self.bar_width)} "
                f"{fraction * 100:5.1f}% {self.count}/{self.total}"
            )
            timing = f"{format_duration(elapsed)}<{format_duration(remaining)}"
        else:
            head = f"[{self.label}] {self.count}"
            timing = format_duration(elapsed)
        line = f"{head} | {format_rate(rate)} | {timing}"
        if suffix:
            line += f" | {suffix}"
        return line
