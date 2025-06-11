"""lsl_worker.py
================
Thread-based helper to acquire EEG samples from an **LSL** (Lab Streaming Layer)
stream and hand them to a bounded queue *and/or* a user callback.

The class is intended for real-time applications where keeping the most recent
samples is more important than preserving the very first ones: when the queue
is full, the oldest chunk is dropped.  All potentially silent failures are
captured and logged.

Usage example
-------------
```python
from lsl_worker import EEGStreamWorker
import time

with EEGStreamWorker(stream_name="MyEEG", pull_chunk_size=64) as worker:
    while True:
        data = worker.get_sample(timeout=0.5)  # ndarray or None
        if data is not None:
            print(data.shape)  # (64, n_channels)
        time.sleep(0.01)
```
"""
from __future__ import annotations

# ─────────────────────────────── Imports ────────────────────────────────
import logging
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams

# -----------------------------------------------------------------------------
# Configuration & logger setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("LSL_WORKER")

# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class EEGStreamWorker:
    """Continuously pulls chunks from an LSL inlet in a daemon thread.

    Parameters
    ----------
    stream_name : str | None, optional
        *If provided*, search by *name*; otherwise search by *type*.
    stream_type : str, default ``"EEG"``
        Fallback when *stream_name* is *None*.
    max_queue_samples : int, default ``250``
        Maximum number of chunks kept in memory.  The newest data always survive.
    pull_chunk_size : int, default ``20``
        ``StreamInlet.pull_chunk`` -> *max_samples*.
    callback : Callable[[np.ndarray, float], None] | None
        User callback executed for every chunk *before* it is placed on the
        queue.  Signature ``(chunk, perf_counter_ts)``.
    timeout : float, default ``1.0``
        Blocking timeout for ``pull_chunk``.
    """

    # ─────────────────────────────── ctor ────────────────────────────────
    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: str = "EEG",
        max_queue_samples: int = 250,  # Q = (Fs * L_buffer) / pull_chunk_size, where L_buffer is the buffer length in seconds 
        pull_chunk_size: int = 20, # N <= Fs * (L_max - t_proc), where L_max is the maximum latency and t_proc is the processing time
        callback: Optional[Callable[[np.ndarray, float], None]] = None,
        timeout: float = 1.0,
    ) -> None:
        
        # Store configuration parameters
        self._callback = callback
        self._timeout = timeout

        # ── Stream discovery ────────────────────────────────────────────
        logger.info("Resolving LSL stream (name=%s, type=%s)…", stream_name, stream_type)
        prop, val = ("name", stream_name) if stream_name else ("type", stream_type)
        streams = resolve_byprop(prop, val, timeout=5.0)

        if not streams:
            # If no stream was found, raise an error and list available streams
            available = [s.name() for s in resolve_streams()]
            logger.critical(
                "No LSL stream found with %s='%s'. Available streams: %s", prop, val, ", ".join(available) or "<none>",
            )
            raise RuntimeError(f"No LSL stream found with {prop}='{val}'")

        if len(streams) > 1:
            # Warn if more than one stream matched
            logger.warning("Multiple streams match; using '%s'", streams[0].name())

        # Create LSL inlet from the selected stream
        self._inlet = StreamInlet(streams[0], max_chunklen=pull_chunk_size)

        # Get stream metadata
        info = self._inlet.info()
        self.srate: float = info.nominal_srate()
        self.n_channels: int = info.channel_count()
        self._stream_info = info

        logger.info(
            "Connected to stream '%s' (%d ch, %.1f Hz)", info.name(), self.n_channels, self.srate
        )

        # ── Queue setup (always enabled, stdlib queue.Queue) ────────────
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue_samples)
        logger.debug("Queue created (max size: %d)", max_queue_samples)

        # Store additional parameters
        self._pull_chunk_size = pull_chunk_size
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─────────────────────────── life-cycle ──────────────────────────────
    def start(self) -> None:
        """Spawn the reader thread if it is not already running."""
        if self._thread and self._thread.is_alive():
            logger.debug("start() called but thread already running - ignoring")
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("EEGStreamWorker started (%d ch, %.1f Hz, thread ID=%s)", self.n_channels, self.srate, self._thread.ident)

    def stop(self) -> None:
        """Request the reader thread to exit and wait for it (graceful)."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        logger.info("EEGStreamWorker stopped")

    # Context-manager sugar (``with`` statement)
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ───────────────────────────── API ───────────────────────────────────
    def get_sample(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Blocking read for **one** chunk from the internal queue.

        When *timeout* is *None* the call blocks indefinitely. Returns *None*
        on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            logger.debug("Queue was empty during get_sample (timeout=%.3fs)", timeout or 0.0)
            return None

    def read_samples(self, max_samples: Optional[int] = None) -> List[np.ndarray]:
        """Non-blocking bulk read of up to *max_samples* chunks."""
        samples: List[np.ndarray] = []
        while (max_samples is None or len(samples) < max_samples) and not self._queue.empty():
            try:
                samples.append(self._queue.get_nowait())
            except queue.Empty:
                break
        logger.debug("Retrieved %d sample(s) from queue", len(samples))
        return samples

    def metadata(self) -> dict:
        """Return the LSL stream metadata XML parsed into a nested dict."""
        xml = self._stream_info.as_xml()
        root = ET.fromstring(xml)

        def _parse(node):
            children = list(node)
            if not children:
                return node.text
            d: dict[str, list] = defaultdict(list)
            for ch in children:
                d[ch.tag].append(_parse(ch))
            return {k: v[0] if len(v) == 1 else v for k, v in d.items()}

        logger.debug("Parsed metadata from stream info")
        return _parse(root)

    # ────────────────────────── internals ────────────────────────────────
    def _reader(self) -> None:
        """Background thread body - pulls data and dispatches them."""
        logger.debug("Reader thread started")
        while not self._stop_evt.is_set():
            try:
                # Attempt to pull a chunk from the LSL stream
                chunk, _ = self._inlet.pull_chunk(
                    timeout=self._timeout, max_samples=self._pull_chunk_size
                )
                logger.debug("Pulled chunk length=%s", len(chunk))
            except Exception:  # pylsl can raise various runtime errors
                logger.exception("StreamInlet.pull_chunk failed - retrying after 100 ms")
                time.sleep(0.1)
                continue

            # If no data was received, sleep briefly and retry
            if not chunk:
                time.sleep(0.001)
                continue

            # Timestamp the chunk and convert it to numpy array
            ts = time.perf_counter()
            chunk_array = np.asarray(chunk, dtype=np.float32)

            # ── User callback ────────────────────────────────────────────
            if self._callback is not None:
                try:
                    self._callback(chunk_array, ts)
                except Exception:  # noqa: BLE001 - never kill the thread
                    logger.exception("User callback raised - continuing")

            # ── Queue dispatch (drop-oldest policy) ─────────────────────
            try:
                self._queue.put_nowait(chunk_array)
            except queue.Full:
                try:
                    # Drop the oldest and put again (keeps newest data)
                    self._queue.get_nowait()
                    self._queue.put_nowait(chunk_array)
                    logger.warning(
                        "Queue full (max size: %d) – oldest sample discarded. "
                        "Data loss may occur if the processing pipeline is not keeping up.",
                        self._queue.maxsize,
                    )
                except queue.Empty:
                    # Extremely unlikely: queue was full but became empty.
                    pass

        logger.debug("Reader thread exited")


# -----------------------------------------------------------------------------
# Helper utilities - mostly for CLI/debugging
# -----------------------------------------------------------------------------

def list_available_streams() -> List[dict]:
    """Return minimal info for every visible LSL stream (best-effort)."""
    return [
        {
            "name": info.name(),
            "type": info.type(),
            "source_id": info.source_id(),
            "hostname": info.hostname(),
            "channel_count": info.channel_count(),
            "srate": info.nominal_srate(),
        }
        for info in resolve_streams()
    ]
