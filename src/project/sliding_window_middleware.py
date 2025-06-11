"""
sliding_window_middleware.py
===========================

Middleware *pull‑based* compatible con el patrón «submit / get» del backend
(APP NEURO). Convierte *chunks* `(chunk_size, n_channels)` en ventanas de
longitud fija con solapamiento opcional.

Diseño de alto nivel ──────────────────────────────────────────────────────

    LSLWorker.get_sample() ──► SlidingWindowMiddleware.submit()
                                       │
                                       ▼
             Backend llama ───► SlidingWindowMiddleware.get_windows()
                                       │            ▲
                                       ├──► GUI (op.)
                                       ▼
                             InferenceWorker.submit()

API mínima ───────────────────────────────────────────────────────────────

```python
wg = SlidingWindowMiddleware(win_len=256, hop_len=128,
                             n_channels=8, fs=256)
# en tu loop de bombeo EEG:
if wg.submit(sample_id, chunk):
    sample_id += len(chunk)
for win_id, win in wg.get_windows():
    inference.submit(win_id, win)
    gui_buffer.append(win)
```  

Implementación ────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Tuple

import numpy as np



class SlidingWindowMiddleware:
    """
    Convierte un flujo de *chunks* en ventanas con solapamiento opcional.
    """
    _SampleId = int
    _Window = np.ndarray

    def __init__(
        self,
        *,
        win_len: int,
        hop_len: int,
        n_channels: int,
        fs: float,
        max_buffer_s: int = 10,
    ) -> None:
        if win_len <= 0 or hop_len <= 0:
            raise ValueError("win_len y hop_len deben ser > 0")
        if hop_len > win_len:
            raise ValueError("hop_len no puede ser mayor que win_len")
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_channels = n_channels
        self.fs = fs
        # Capacidad del ring-buffer: histórico + ventana + hop para evitar recorte
        cap = int(max_buffer_s * fs) + win_len + hop_len
        self._buffer = np.empty((cap, n_channels), dtype=np.float32)
        self._cap = cap
        self._write_ptr = 0
        self._sample_id = 0    # total muestras vistas
        self._next_out = 0     # sample_id de próxima ventana
        self._lock = threading.Lock()
        self._ready: Deque[Tuple[int, np.ndarray]] = deque()

    def submit(self, sample_id: int, chunk: np.ndarray) -> bool:
        """
        Inserta un chunk. Devuelve False si hubo dropping interno.
        """
        if chunk.ndim != 2 or chunk.shape[1] != self.n_channels:
            raise ValueError(
                f"chunk inválido: expected (N, {self.n_channels}), got {chunk.shape}"
            )
        n = len(chunk)
        with self._lock:
            buf = self._buffer
            cap = self._cap
            # no recortamos chunk: escribimos en ring-buffer (mod cap)
            idx = (self._write_ptr + np.arange(n)) % cap
            buf[idx] = chunk
            self._write_ptr = (self._write_ptr + n) % cap
            self._sample_id += n
            # generar ventanas disponibles
            while self._sample_id - self._next_out >= self.win_len:
                ids = (self._next_out + np.arange(self.win_len)) % cap
                win = buf[ids].copy()
                self._ready.append((self._next_out, win))
                self._next_out += self.hop_len
            # ajustar next_out si backlog muy grande
            lag = self._sample_id - self._next_out
            if lag > cap - self.win_len:
                self._next_out = self._sample_id - (cap - self.win_len)
            return lag <= cap - self.win_len

    def get_windows(self) -> List[Tuple[_SampleId, _Window]]:
        """
        Devuelve todas las ventanas listas y las borra de la cola interna.
        """
        with self._lock:
            items = list(self._ready)
            self._ready.clear()
            return items

    def get_latest_samples(self, n_samples: int) -> np.ndarray:
        """
        Copia las últimas n_samples para la GUI.
        """
        if n_samples <= 0 or n_samples > self._cap:
            raise ValueError("n_samples inválido")
        with self._lock:
            idx = (self._write_ptr - n_samples + np.arange(n_samples)) % self._cap
            return self._buffer[idx].copy()

