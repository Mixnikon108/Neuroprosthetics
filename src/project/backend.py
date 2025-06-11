"""

> python -m backend --lsl Signal_generator --port COM6 

backend.py – Core backend orchestrator for the EEG ↔ EMG robotic‑control app.

Designed to run headless while the definitive Qt GUI is being developed.
It wires together four asynchronous actors:
    1. EEGStreamWorker    – pulls EEG from LSL in a background thread.
    2. InferenceWorker    – performs TensorFlow inference in a separate *process*.
    3. SerialEMGInterface – full‑duplex serial link with the Arduino (EMG → PC,
                            commands ← PC).
    4. Backend main loop  – the glue that moves data *without* blocking.

Key goals
──────────
• **Non‑blocking** – every queue get/put uses `nowait()` or ≤10 ms timeout.
• **Robust**      – unexpected exceptions are logged and do not crash the app.
• **Decoupled**   – exposes two queues (`q_to_gui`, `q_from_gui`) so the
  forthcoming Qt GUI can plug‑in with zero refactor.
• **Smart commands** – avoids sending redundant or excessively frequent
  commands to the Arduino (debounce + dedup + rate‑limit).

Standalone usage
────────────────
    $ python -m backend --lsl MyEEG --port COM3

The script streams EEG, performs dummy inference and pushes commands to the
Arduino. Press Ctrl‑C to exit cleanly.
"""


from __future__ import annotations

import argparse
import logging

from .utils import setup_color_logging
from .utils import load_config

import queue
import signal
import threading
import time
from types import FrameType
from typing import Any, Optional, Tuple, Callable, List
import sys
import numpy as np

from .arduino_interface import SerialEMGInterface
from .inference import InferenceWorker
from .lsl_worker import EEGStreamWorker
from .sliding_window_middleware import SlidingWindowMiddleware

from PySide6 import QtWidgets
from .gui import NeuroDashboard, ChannelSelectionDialog


# ═════════════════════════════════ LOGGING ════════════════════════════════
setup_color_logging(level=logging.INFO)
log = logging.getLogger("BACKEND")


# ═════════════════════════════ CONFIGURATION ═════════════════════════════════
_cfg               = load_config()
cfg_bk, cfg_lsl    = _cfg.get("backend",{}), _cfg.get("lsl",{})
cfg_sw, cfg_ard    = _cfg.get("sliding_window",{}), _cfg.get("arduino",{})


_POLL      = cfg_bk.get("poll_interval", .01)    # main‑loop sleep (s) when idle → 10 ms
_MAX_DROP_WARN   = cfg_bk.get("max_drop_warn", 100)    # warn if we drop >N EEG chunks consecutively
_MIN_CMD_INTERVAL    = cfg_bk.get("min_cmd_interval", .05) # ≥ 50 ms between commands (20 Hz) to Arduino
_STABILITY_REQUIRED   = cfg_bk.get("stability_required", 3) # need N identical preds before changing command

_LSL_CHUNK = cfg_lsl.get("pull_chunk_size", 64)
_LSL_QMAX  = cfg_lsl.get("max_queue_samples", 40)
_LSL_TOUT  = cfg_lsl.get("timeout", 2)

_WIN_LEN   = cfg_sw.get("win_len", 256)   # 1 s
_HOP_LEN   = cfg_sw.get("hop_len", 128)
_MAX_BUF   = cfg_sw.get("max_buffer_s", 10)

_BAUD_DEF  = cfg_ard.get("default_baudrate", 115_200)

_EMG_DECIM = 4             # 1 kHz → 250 Hz
_GUI_BUF_S = 5             # segundos que mantiene la GUI

# umbral de la prediccion para activar el comando
_THRESHOLD_CMD = 0.75


# ════════════════════════ DECORATORS / HELPERS ═══════════════════════════

def swallow_exceptions(fn):
    """Decorator: log *all* exceptions, never propagate further."""

    def wrapper(*args, **kwargs):  # type: ignore[return‑type]
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 – blanket catch on purpose
            log.exception("Unhandled exception in %s", fn.__qualname__)

    return wrapper

# ═════════════════════════════ BACKEND ═══════════════════════════════════
class Backend:
    """Orchestrates LSL → Inference → Arduino while exposing queues to a GUI."""

    def __init__(
            self, 
            *,
            lsl_name: str,
            serial_port: str,
            serial_baudrate: int,
            gui_queues: Optional[Tuple[queue.Queue[Any],queue.Queue[Any]]] = None
        ):
        
        # Queues for GUI ⇄ backend (plain thread‑safe queues)
        self.q_to_gui, self.q_from_gui = gui_queues or (queue.Queue(), queue.Queue())

        # ── Stats & command state ───────────────────────────
        self._sample_id        = 0
        self._dropped_streak    = 0
        self._pending_cmds: List[int]=[]
        self._last_cmd = None
        self._last_cmd_ts = 0.0
        self._emg_skip = 0

        # ── LSL ─────────────────────────────────────────────
        self._lsl = EEGStreamWorker(
            stream_name=lsl_name,
            pull_chunk_size=_LSL_CHUNK,
            max_queue_samples=_LSL_QMAX,
            timeout=_LSL_TOUT,
        )

        # ── Mini‑GUI de selección de canales ────────────────────────────
        self._selected_ch = self._ask_channel_selection(self._lsl.n_channels)
        n_sel = len(self._selected_ch)
        log.info("Canales seleccionados: %s", self._selected_ch)

        # ── ② CAPTURA DE BASELINE ANTES DE CREAR INFERENCE ────────────
        self._baseline = self._collect_baseline(seconds=5.0)   # ← aquí eliges la duración

        # ── Sliding-window ──────────────────────────────────
        self._sw = SlidingWindowMiddleware(
            win_len=_WIN_LEN, 
            hop_len=_HOP_LEN,
            n_channels=n_sel, 
            fs=self._lsl.srate,
            max_buffer_s=_MAX_BUF,
        )
        # EEG samples to send to GUI
        self._eeg_gui_samples = int(self._lsl.srate * _GUI_BUF_S)

        # ── Serial ──────────────────────────────────────────
        self._arduino   = SerialEMGInterface(
            port=serial_port,
            baudrate=serial_baudrate,
            callback=self._on_emg)
        
        
        # ── Inference ───────────────────────────────────────
        self._inference = InferenceWorker(
            baseline=self._baseline,
            fs=self._lsl.srate)


        # ── Thread control ──────────────────────────────────
        self._stop_evt     = threading.Event()
        self._main_thread  = threading.Thread(
            target=self._loop, name="backend‑main", daemon=True
        )

        
    # ────────────────────── NUEVO MÉTODO PRIVADO ───────────────────────
    def _collect_baseline(self, *, seconds: float) -> np.ndarray:
        """
        Arranca LSL, acumula `seconds` de datos de los canales seleccionados y
        devuelve un array (n_ch × n_samples).
        """
        log.info("Capturando baseline de %.1f s…", seconds)
        self._lsl.start()                              # arranca hilo LSL
        n_needed = int(seconds * self._lsl.srate)
        buf: list[np.ndarray] = []
        collected = 0

        while collected < n_needed:
            chunk = self._lsl.get_sample(timeout=1.0)  # bloquea hasta 1 s
            if chunk is None:
                continue                               # timeout, sigue esperando
            chunk = chunk[:, self._selected_ch]        # filtra canales
            buf.append(chunk)
            collected += len(chunk)

        baseline_arr = np.concatenate(buf, axis=0)[:n_needed]  # (samples × ch)
        log.info("Baseline capturado: %d muestras", n_needed)
        return baseline_arr.T.astype(np.float32)                # → (ch × samples)
    


    # ───────────────────────── GUI para elegir canales ──────────────────
    @staticmethod
    def _ask_channel_selection(n_channels: int) -> List[int]:
        """Muestra un diálogo modal para que el usuario seleccione canales."""
        # Asegurarnos de que hay una QApplication viva
        app = QtWidgets.QApplication.instance()
        created = False
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
            created = True

        dlg = ChannelSelectionDialog(n_channels)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            log.error("Channel selection cancelled – aborting start‑up")
            raise SystemExit(1)
        selected = dlg.selected_channels()
        if created:
            app.quit()
        if not selected:
            raise RuntimeError("At least one EEG channel must be selected")
        return selected

    # ─────────────────────────── LIFECYCLE ───────────────────────────────
    def start(self):
        log.info("Starting backend…")
        self._lsl.start()
        self._arduino.start()
        self._inference.start() 
        self._main_thread.start()

        self.q_to_gui.put({"event_type":"eeg_info",
                        "srate": self._lsl.srate,
                        "n_channels": len(self._selected_ch)})
        
        log.info("Backend running (thread ID=%s)", self._main_thread.ident)

    def stop(self):
        if self._stop_evt.is_set(): 
            return
        self._stop_evt.set(); 
        self._main_thread.join(timeout=2.0)
        self._lsl.stop(); 
        self._inference.stop(); 
        self._arduino.stop()
        log.info("Backend stopped")

    # ───────────────────────── GUI LOG HELPER ────────────────────────────
    def _log_gui(self, msg: str):
        try:
            self.q_to_gui.put_nowait({"event_type": "log", "msg": msg})
        except queue.Full:
            log.debug("GUI log queue full - dropped message: %s", msg)

    # ───────────────────────── EMG CALLBACK ──────────────────────────────
    def _on_emg(self, val:int, ts:float):
        self._emg_skip = (self._emg_skip+1) % _EMG_DECIM
        if self._emg_skip: return
        try: self.q_to_gui.put_nowait({"event_type":"emg","value":val})
        except queue.Full: pass

    # ───────────────────────── PUMPERS ───────────────────────────────────
    def _pump_gui_commands(self):
        while not self.q_from_gui.empty():
            ev = self.q_from_gui.get_nowait()
            if ev.get("event_type")=="arduino_cmd":
                self._arduino.send_command(ev.get("code",0))


    def _pump_eeg(self):
        """LSL → SlidingWindow → Inference (solo cuando haya ventana lista)."""
        chunk = self._lsl.get_sample(timeout=0.0)
        if chunk is None: 
            return  # no hay datos todavía
        
        # Filtrar canales seleccionados ---------------------------------
        chunk = chunk[:, self._selected_ch]
        
        # 1) Alimentar el middleware ------------------------------------
        log.debug(f"[pump_eeg] Llega chunk de {len(chunk)} muestras (sample_id={self._sample_id})")
        dropped = not self._sw.submit(self._sample_id, chunk)
        self._sample_id += len(chunk)

        # 2) Recuperar y_pred enviar ventanas listas -------------------------
        for win_id, win in self._sw.get_windows():
            log.debug(f"[pump_eeg] Ventana lista win_id={win_id}, shape={win.shape} → intentando enviar a inferencia")
            if not self._inference.submit(win_id, win):
                log.debug(f"[pump_eeg] Cola de inferencia llena → ventana win_id={win_id} descartada") 
                dropped=True
            else:
                log.debug(f"[pump_eeg] Ventana win_id={win_id} enviada correctamente a inferencia")

        # Enviar muestras más recientes a GUI
        latest = self._sw.get_latest_samples(self._eeg_gui_samples)
        try: 
            self.q_to_gui.put_nowait({"event_type":"eeg_latest",
                                    "samples": latest.tolist()})
        except queue.Full: 
            pass

        # 3) Estadísticas de pérdida ------------------------------------
        if dropped:
            self._dropped_streak += 1
            if self._dropped_streak >= _MAX_DROP_WARN:
                self._log_gui(f"!! {self._dropped_streak} ventanas perdidas")
                self._dropped_streak = 0
        else: self._dropped_streak = 0

    def _pump_predictions(self):
        now = time.perf_counter
        for sid, y_pred in self._inference.get_results():

            if y_pred[0] > _THRESHOLD_CMD:
                cmd = -1
            elif y_pred[1] > _THRESHOLD_CMD:
                cmd = 1
            else:
                cmd = 0
            
            self._pending_cmds.append(cmd)

            # DEBUG: mostrar predicción cruda y cmd
            log.debug("Predicción sid=%d → y_pred=%s → cmd=%d", sid, y_pred, cmd)

            signed_prob = (-1)**(1 - cmd) * (y_pred[cmd] - 0.5)
            
            try:
                self.q_to_gui.put_nowait({
                    "event_type":"inference_bar",
                    "value": signed_prob})
                
            except queue.Full:
                pass


            if len(self._pending_cmds) > _STABILITY_REQUIRED:
                self._pending_cmds.pop(0)

            

            if cmd == 0:
                continue # no hay comando

            stable = (
                len(self._pending_cmds) == _STABILITY_REQUIRED
                and len(set(self._pending_cmds)) == 1
            )

            if stable:
                if cmd != self._last_cmd:
                    if now() - self._last_cmd_ts >= _MIN_CMD_INTERVAL:
                        self._arduino.send_command(cmd)
                        self._last_cmd = cmd
                        self._last_cmd_ts = now()
                        log.info("→ Arduino cmd %d (sid=%d)", cmd, sid)
                        self._log_gui(f"✓ Predicción sid={sid} → cmd {cmd}")

                        try:
                            self.q_to_gui.put_nowait(
                                {"event_type": "pred", "cmd": cmd}
                            )
                        except queue.Full:
                            pass
                    else:
                        log.debug("Esperando intervalo mínimo para enviar cmd=%d", cmd)
                else:
                    log.debug("cmd=%d ya fue enviado antes, ignorado", cmd)
            else:
                log.debug("Comando inestable → ventana: %s", self._pending_cmds)

    
    # ───────────────────────── MAIN LOOP ─────────────────────────────────
    @swallow_exceptions
    def _loop(self):
        log.debug("Main loop started")
        while not self._stop_evt.is_set():
            self._pump_gui_commands()
            self._pump_eeg()
            self._pump_predictions()
            time.sleep(_POLL)

    # ───────────────────────── CONTEXT MANAGER ───────────────────────────
    def __enter__(self): 
        self.start()
        return self
    
    def __exit__(self,*_): 
        self.stop()

# ═════════════════════════════ CLI ENTRY ════════════════════════════════
def _main():
    parser = argparse.ArgumentParser(description="Run backend headless or with Qt GUI for testing.")
    parser.add_argument("--lsl", dest="lsl", help="LSL stream name (defaults to first EEG stream)")
    parser.add_argument("--port", required=True, help="Serial port for Arduino (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=_BAUD_DEF, help=f"Serial baudrate (default: {_BAUD_DEF})")
    parser.add_argument("--gui", action="store_true", help="Launch with Qt GUI (requires PySide6 and pyqtgraph)")
    args = parser.parse_args()

    app = None
    if args.gui:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

    backend = Backend(
        lsl_name=args.lsl,
        serial_port=args.port,
        serial_baudrate=args.baud
    )

    if args.gui:
        backend.start()
        win = NeuroDashboard(backend.q_to_gui, backend.q_from_gui)
        win.show()
        app.aboutToQuit.connect(backend.stop)
        sys.exit(app.exec())

    # Headless mode: catch SIGINT/SIGTERM and exit cleanly
    def _sig_handler(signum: int, frame: FrameType | None):
        log.info("Signal %s received, stopping…", signum)
        backend.stop()
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig_handler)
        except ValueError:
            # On some platforms only main thread may set signal handlers
            pass

    backend.start()

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _sig_handler(signal.SIGINT, None)


if __name__ == "__main__":
    _main()
