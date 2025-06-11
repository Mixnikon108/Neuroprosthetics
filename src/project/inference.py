"""
Module: inference.py
Descripción
-----------
Worker multiproceso para inferencia en tiempo real de MI-EEG
usando un modelo Medusa (MIModelCSP) + baseline artificial.

Interfaz pública idéntica a la versión anterior:
    • InferenceWorker.submit(sample_id, window)
    • InferenceWorker.get_results() → [(sample_id, (pR, pL)), …]

Diferencias clave:
    • Ya **no** se usa TensorFlow/Keras.
    • El proceso hijo crea un `OnlineMIInferencer(model, baseline, fs)`.
    • El baseline y la frecuencia de muestreo **se pasan desde el
      proceso principal al instanciar InferenceWorker**.
"""

from __future__ import annotations
import logging
import multiprocessing as mp
from typing import Any, List, Tuple

import numpy as np
from medusa.bci.mi_paradigms import MIModel

from .utils import load_config

# ───────────────────────── Logger ─────────────────────────
logger = logging.getLogger("INFERENCE")

# ───────────────────────── Config ─────────────────────────
_cfg = load_config()
logger.info("Config loaded from config.yml")

# ───────────────────────── Inferencer ─────────────────────────
class OnlineMIInferencer:
    """
    Empaqueta un MIModel más el baseline:
        baseline  (n_ch × n_bsamp)  +  ventana (n_ch × n_wsamp)
    Devuelve (pR, 1-pR).
    """

    def __init__(self, model_path: str, baseline: np.ndarray, fs: float):
        # modelo
        logger.info("Loading MI model from: %s", model_path)
        self.model = MIModel.load(model_path)

        # baseline + parámetros
        self.baseline = baseline.astype(np.float32)
        self.fs = float(fs)
        self.n_ch, self.n_bsamp = self.baseline.shape
        self.signal_base = self.baseline.T                          # (samples × ch)
        self.times_base = np.arange(-self.n_bsamp, 0) / self.fs
        self.baseline_ms = int(self.n_bsamp / self.fs * 1000)
        logger.info("Inferencer ready — baseline %d ch × %d samp", self.n_ch, self.n_bsamp)

    # --------------------------------------------------
    def predict_window(self, window: np.ndarray) -> Tuple[float, float]:
        if window.shape[1] != self.n_ch:        # viene traspuesta
            raise ValueError("Channel count mismatch: "
                             f"{window.shape} vs expected {self.n_ch} channels")

        n_wsamp = window.shape[1]
        samp_ms = 1000 / self.fs
        win_ms = int((n_wsamp - 1) * samp_ms)           # última muestra incluida

        signal = np.vstack((self.signal_base, window))
        times = np.concatenate((self.times_base,
                                np.arange(n_wsamp) / self.fs))

        dec = self.model.predict(
            times=times,
            signal=signal,
            fs=self.fs,
            channel_set=self.model.channel_set,
            x_info={"onsets": [0.0], "mi_labels": None},
            w_epoch_t=(0, win_ms),
            w_baseline_t=(-self.baseline_ms, 0),
            baseline_mode="trial",
        )
        pR = float(dec["y_prob"][-1][1])
        return pR, 1.0 - pR


# ───────────────────────── Proceso hijo ─────────────────────────
def _worker(
    baseline: np.ndarray,
    fs: float,
    model_path: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
    stop_evt: mp.Event,
    ready_evt: mp.Event,
):
    """Loop de inferencia que corre en un proceso aparte."""
    try:
        infer = OnlineMIInferencer(model_path=model_path,
                                   baseline=baseline,
                                   fs=fs)
    except Exception:
        logger.exception("Worker init failed")
        ready_evt.set()
        return

    ready_evt.set()
    logger.debug("Worker ready (PID=%d)", mp.current_process().pid)

    while not stop_evt.is_set():
        try:
            sample_id, window = in_q.get(timeout=0.1)  # window = np.ndarray
        except mp.queues.Empty:
            continue
        except (EOFError, OSError):
            logger.error("Input queue closed — exiting worker")
            break

        try:
            probs = infer.predict_window(window)       # (pR, pL)
            out_q.put((sample_id, probs))
            logger.debug("Predicted sample_id=%s pR=%.3f", sample_id, probs[0])
        except Exception:
            logger.exception("Inference failed for sample_id=%s", sample_id)

    logger.debug("Worker exiting")


# ───────────────────────── Gestor alto nivel ─────────────────────────
class InferenceWorker:
    """
    Gestor del proceso de inferencia.

    Parameters
    ----------
    baseline : np.ndarray
        Grabación de reposo (n_ch × n_bsamp) en la **misma** referencia y orden
        de canales que el modelo.
    fs : float
        Frecuencia de muestreo de baseline y de las ventanas que se enviarán.
    """

    def __init__(self, baseline: np.ndarray, fs: float) -> None:
        maxsize = _cfg["queue"]["maxsize"]
        model_path = _cfg["model"]["path"]

        logger.info("Creating InferenceWorker (queue maxsize=%d)", maxsize)

        self.in_q: mp.Queue = mp.Queue(maxsize=maxsize)
        self.out_q: mp.Queue = mp.Queue()
        self.stop_evt, self.ready_evt = mp.Event(), mp.Event()

        # Proceso hijo
        self.proc = mp.Process(
            target=_worker,
            args=(
                baseline,
                fs,
                model_path,
                self.in_q,
                self.out_q,
                self.stop_evt,
                self.ready_evt,
            ),
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def start(self) -> None:
        logger.info("Starting inference process…")
        self.proc.start()
        self.ready_evt.wait()
        if not self.proc.is_alive():
            raise RuntimeError("Inference process terminated during startup")
        logger.info("Inference process running (PID=%d)", self.proc.pid)

    def stop(self, timeout: float = 2.0) -> None:
        logger.info("Stopping inference process…")
        self.stop_evt.set()
        self.proc.join(timeout=timeout)
        if self.proc.is_alive():
            logger.warning("Process did not terminate within %.1f s", timeout)
        else:
            logger.info("Inference process stopped")

    # ------------------------------------------------------------------
    # Comunicación
    # ------------------------------------------------------------------
    def submit(self, sample_id: int, window: np.ndarray) -> bool:
        """
        Envía una ventana (n_ch × n_wsamp) al worker.

        Returns
        -------
        bool
            True si la ventana se ha encolado; False si la cola está llena.
        """
        try:
            self.in_q.put_nowait((sample_id, window.astype(np.float32)))
            logger.debug("Submitted sample_id=%s shape=%s", sample_id, window.shape)
            return True
        except mp.queues.Full:
            logger.warning("Input queue full — sample_id=%s dropped", sample_id)
            return False

    def get_results(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Devuelve todos los resultados disponibles sin bloquear."""
        results: List[Tuple[int, Tuple[float, float]]] = []
        while not self.out_q.empty():
            results.append(self.out_q.get_nowait())
        if results:
            logger.debug("Fetched %d result(s)", len(results))
        return results

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
