"""
serial_emg_interface.py
-----------------------
Full-duplex interface to an Arduino-based EMG front-end that streams raw
EMG samples (~1 kHz) over a USB/FTDI virtual COM port and accepts numeric
commands from Python.

Life-cycle (v2)
~~~~~~~~~~~~~~
* ``__init__`` **connects immediately**: opens the serial port and waits
  for the ``ARDUINO_READY`` handshake.
* ``start`` begins streaming by spawning the background reader thread.
* ``stop`` stops streaming and closes the port.  ``close`` is an alias.

This keeps connection latency out of ``start`` while giving you fine-
grained control over when the data-reader consumes CPU.

Example
-------
```python
emg = SerialEMGInterface(port="/dev/ttyACM0")  # connects & hand-shake
...
emg.start()              # begin reading samples in the background
...
emg.stop()               # terminate reader & close port
```
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional, Tuple

import serial

# ───────────────────────── Public re-exports ──────────────────────────
Sample = Tuple[int, float]  # Convenience alias: (value, timestamp)

# ───────────────────────── Logger configuration ───────────────────────
logger = logging.getLogger("SERIAL_EMG")

# ───────────────────────── Module constants ───────────────────────────
_READY_TOKEN = "ARDUINO_READY"  # Handshake token expected from firmware


class SerialEMGInterface:
    """Thread-based interface for ASCII EMG samples coming from an Arduino.

    Parameters
    ----------
    port : str
        Serial device path (e.g. ``'COM5'`` on Windows or ``'/dev/ttyACM0'`` on Linux).
    baudrate : int, default ``230_400``
        Baud rate used by the firmware.
    callback : Callable[[int, float], None] | None, default ``None``
        User function invoked for every sample *instead* of using the internal queue.
    queue_size : int, default ``5_000``
        Maximum number of samples kept if *callback* is ``None``.
    """

    # ───────────────────────────── ctor ──────────────────────────────
    def __init__(
        self,
        port: str,
        baudrate: int = 230_400,
        callback: Optional[Callable[[int, float], None]] = None,
        queue_size: int = 5_000,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._callback = callback
        self._queue: "queue.Queue[Sample]" = queue.Queue(maxsize=queue_size)
        self._stop_evt = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None

        # Connect immediately --------------------------------------------------
        logger.info("Opening serial port %s @ %d baud…", port, baudrate)
        try:
            self._ser = serial.Serial(port, baudrate, timeout=0.05)
        except serial.SerialException:
            logger.critical("Failed to open serial port %s @ %d baud", port, baudrate)
            raise

        # Handshake before returning ------------------------------------------
        self._wait_for_ready()
        logger.info(
            "SerialEMGInterface connected (queue=%d, callback=%s)",
            queue_size,
            "enabled" if callback else "disabled",
        )

    # ─────────────────────────── Properties ────────────────────────────
    @property
    def is_running(self) -> bool:
        """Return *True* while the background reader thread is alive."""
        return self._rx_thread is not None and self._rx_thread.is_alive()

    # ─────────────────────────── Life-cycle ────────────────────────────
    def start(self) -> None:
        """Spawn the reader thread and begin streaming EMG samples."""
        if self.is_running:
            logger.debug("start() called but interface already running – ignoring")
            return
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial port is not open")

        self._stop_evt.clear()
        self._rx_thread = threading.Thread(
            target=self._reader, name="SerialEMG-RX", daemon=True
        )
        self._rx_thread.start()
        logger.info("SerialEMGInterface streaming (thread ID=%s)", self._rx_thread.ident)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop streaming and close the serial connection."""
        if not (self.is_running or (self._ser and self._ser.is_open)):
            logger.debug("stop() called but interface already stopped – ignoring")
            return

        logger.info("Stopping SerialEMGInterface…")
        self._stop_evt.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=timeout)
            if self._rx_thread.is_alive():
                logger.warning("Reader thread did not exit within %.1f s", timeout)
            self._rx_thread = None

        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None
        logger.info("SerialEMGInterface stopped")

    # Context-manager sugar ----------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ─────────────────────────── Public API ────────────────────────────
    def send_command(self, code: int) -> None:
        """Send an integer command to the Arduino, appending ``'\n'``."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("SerialEMGInterface is not connected")

        msg = f"{code}\n".encode()
        try:
            self._ser.write(msg)
            self._ser.flush()
            logger.debug("Sent command: %d", code)
        except serial.SerialException:
            logger.exception("Failed to send command: %d", code)

    def get_sample(self, timeout: Optional[float] = None) -> Optional[Sample]:
        """Pop **one** sample from the internal queue (blocking)."""
        try:
            item = self._queue.get(timeout=timeout)
            logger.debug("get_sample → %s", item)
            return item
        except queue.Empty:
            logger.debug("get_sample timed-out (%.3fs)", timeout or 0.0)
            return None

    def read_samples(self, max_samples: Optional[int] = None) -> list[Sample]:
        """Retrieve up to *max_samples* queued items **without** blocking."""
        items: list[Sample] = []
        while (max_samples is None or len(items) < max_samples) and not self._queue.empty():
            items.append(self._queue.get_nowait())
        logger.debug("read_samples → %d item(s)", len(items))
        return items

    # ───────────────────────── Private helpers ──────────────────────────
    def _wait_for_ready(self) -> None:
        """Block until ``_READY_TOKEN`` is received or 30 s elapse."""
        assert self._ser is not None
        logger.info("Waiting for '%s' handshake…", _READY_TOKEN)
        start = time.time()
        while True:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line == _READY_TOKEN:
                logger.info("Arduino ready, handshake OK")
                return
            if time.time() - start > 30.0:
                logger.error("Handshake timeout (30 s)")
                raise TimeoutError(f"'{_READY_TOKEN}' not received within 30 s")

    def _reader(self) -> None:
        """Background loop that parses integer lines & dispatches them."""
        assert self._ser is not None
        logger.debug("Reader thread started")
        while not self._stop_evt.is_set():
            try:
                line = self._ser.readline()  # blocks up to 50 ms
                if not line:
                    continue  # timeout

                ts = time.perf_counter()

                # Convert ASCII line → int ---------------------------------
                try:
                    value = int(line.strip())
                except ValueError:
                    logger.debug("Discarding non-numeric line: %s", line[:20])
                    continue

                # Callback mode -------------------------------------------
                if self._callback is not None:
                    try:
                        self._callback(value, ts)
                    except Exception:
                        logger.exception("User callback raised an exception")
                    continue  # skip queue when callback present

                # Queue mode ---------------------------------------------
                if self._queue.full():
                    try:
                        self._queue.get_nowait()  # drop oldest
                        logger.debug("Queue full; dropped oldest sample")
                    except queue.Empty:
                        pass  # race condition
                self._queue.put_nowait((value, ts))

            except serial.SerialException:
                logger.exception("Serial port error — terminating reader thread")
                break

        logger.debug("Reader thread exited")