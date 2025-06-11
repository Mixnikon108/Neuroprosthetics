"""
gui.py – Dashboard Qt para APP-NEURO

* Rueda del ratón = cambia ganancia (no zoom de ejes)
* Panel fijo para activar/desactivar canales
"""
from __future__ import annotations
import queue, time
from collections import deque
from typing import Deque, List

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

FPS       = 60
BUFFER_S  = 5           # segundos de EEG visibles
OFFSET_UV = 500         # separación vertical entre canales
BAR_THICK = 20          # grosor de la barra de inferencia


class ChannelSelectionDialog(QtWidgets.QDialog):
    """Dialogo modal para elegir qué canales EEG usar."""

    def __init__(self, n_channels: int, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Selecciona los canales EEG a usar")
        self.resize(300, 400)
        vbox = QtWidgets.QVBoxLayout(self)

        self._checks: list[QtWidgets.QCheckBox] = []
        for ch in range(n_channels):
            cb = QtWidgets.QCheckBox(f"Ch {ch}")
            cb.setChecked(True)
            vbox.addWidget(cb)
            self._checks.append(cb)

        vbox.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

    # Helper -------------------------------------------------------------
    def selected_channels(self) -> List[int]:
        """Devuelve la lista de índices de canales marcados."""
        return [i for i, cb in enumerate(self._checks) if cb.isChecked()]

# ─── Custom ViewBox (desactiva zoom, gana con rueda) ─────────────────
class EEGViewBox(pg.ViewBox):
    def __init__(self, dashboard:"NeuroDashboard"):
        super().__init__(enableMenu=False)
        self.dash = dashboard
    def wheelEvent(self, ev):
        self.dash.adjust_gain(1.25 if ev.delta()>0 else 0.8)
        ev.accept()                         # evita zoom de ejes

# ─── Dashboard ───────────────────────────────────────────────────────
class NeuroDashboard(QtWidgets.QMainWindow):
    def __init__(self, q_from:"queue.Queue", q_to:"queue.Queue"):
        super().__init__()
        self.setWindowTitle("APP-NEURO • Dashboard")
        self.resize(1400, 850)

        self.q_from = q_from; self.q_to = q_to
        self.srate  = 256.0;     self.n_ch = 0
        self.gain   = 10.0

        self.emg_buf: Deque[int]      = deque(maxlen=1000)
        self.eeg_bufs: List[Deque[float]] = []
        self.eeg_curves: List[pg.PlotCurveItem] = []
        self.chan_checks: List[QtWidgets.QCheckBox] = []

        # ── Layout principal ─────────────────────────────────────
        central=QtWidgets.QWidget(); vbox=QtWidgets.QVBoxLayout(central)

        # EMG
        self.emg_plot=pg.PlotWidget(title="EMG (≈250 Hz)")
        self.emg_curve=self.emg_plot.plot(pen=pg.mkPen(width=1))
        self.emg_plot.setYRange(-1000,1000); vbox.addWidget(self.emg_plot,2)

        # EEG
        vb=EEGViewBox(self)
        self.eeg_plot=pg.PlotWidget(title="EEG",viewBox=vb)
        self.eeg_plot.setMouseEnabled(False,False)
        self.eeg_plot.showGrid(y=True,alpha=0.2); vbox.addWidget(self.eeg_plot,4)

        # ── Barra de inferencia ───────────────────────────────────────
        self.inf_value = 0.0
        self.inf_plot = pg.PlotWidget(title="Inferencia (-1 ↔ +1)")
        self.inf_plot.setMouseEnabled(False, False)
        self.inf_plot.setYRange(-0.5, 0.5)
        self.inf_plot.setXRange(-0.5, 0.5)
        self.inf_plot.hideAxis("left")
        self.inf_plot.getAxis("bottom")\
            .setTicks([[(-0.5,"Left"),(0,"Neutral"),(0.5,"Right")]])
        self.inf_plot.addLine(x=0, pen=pg.mkPen('#666'))  # línea central
        self.bar_curve = self.inf_plot.plot(
            pen=pg.mkPen('#007ACC', width=BAR_THICK,
                         capStyle=QtCore.Qt.FlatCap))
        vbox.addWidget(self.inf_plot, 1)
        # ──────────────────────────────────────────────────────────────

        # Pred + cmd manual
        hl=QtWidgets.QHBoxLayout()
        self.pred_lbl=QtWidgets.QLabel("Predicción: —")
        f=self.pred_lbl.font(); f.setPointSize(14); f.setBold(True)
        self.pred_lbl.setFont(f); hl.addWidget(self.pred_lbl)
        self.cmb=QtWidgets.QComboBox(); self.cmb.addItems([str(i) for i in range(10)])
        btn=QtWidgets.QPushButton("Enviar cmd")
        btn.clicked.connect(lambda: self._send_cmd(int(self.cmb.currentText())))
        hl.addStretch(1); hl.addWidget(QtWidgets.QLabel("Arduino cmd"))
        hl.addWidget(self.cmb); hl.addWidget(btn)
        vbox.addLayout(hl)

        # Log
        self.log=QtWidgets.QPlainTextEdit(readOnly=True,maximumBlockCount=800)
        vbox.addWidget(self.log,1); self.setCentralWidget(central)

        # ── Panel canales fijo ───────────────────────────────────
        dock=QtWidgets.QDockWidget("Canales EEG",self)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,dock)
        self.chan_widget=QtWidgets.QWidget()
        self.chan_layout=QtWidgets.QVBoxLayout(self.chan_widget)
        self.chan_layout.addStretch(1)
        dock.setWidget(self.chan_widget)

        # Timer
        self.timer=QtCore.QTimer(); self.timer.setInterval(1000//FPS)
        self.timer.timeout.connect(self._tick); self.timer.start()

    # ─── Ganancia con rueda ─────────────────────────────────────
    def adjust_gain(self,factor:float):
        self.gain=max(0.1,min(100.0,self.gain*factor))

    # ─── Tick loop ──────────────────────────────────────────────
    def _tick(self):
        self._drain_backend()
        self._draw_emg()
        self._draw_eeg()
        self._draw_inference()          # ← actualiza la barra

    # ─── Backend events ────────────────────────────────────────
    def _drain_backend(self):
        while not self.q_from.empty():
            ev=self.q_from.get_nowait(); et=ev.get("event_type")
            if et=="eeg_info":
                self.srate=ev["srate"]; self.n_ch=ev["n_channels"]; self._init_eeg()
            elif et=="eeg_latest":
                self._append_eeg(np.asarray(ev["samples"],float))
            elif et=="emg":
                self.emg_buf.append(ev["value"])
            elif et=="pred":
                self.pred_lbl.setText(f"Predicción estable → cmd {ev['cmd']}")
            elif et=="log":
                self.log.appendPlainText(ev["msg"])
            elif et=="inference_bar":
                # recibe signed_prob ∈ [-1,1]
                self.inf_value = max(-0.5, min(0.5, ev["value"]))
            elif et == "pred":
                self.pred_lbl.setText(f"Predicción estable → cmd {ev['cmd']}")

    # ─── Init EEG widgets/buffers ───────────────────────────────
    def _init_eeg(self):
        if self.eeg_bufs: return
        self.eeg_bufs=[deque(maxlen=int(self.srate*BUFFER_S))
                       for _ in range(self.n_ch)]
        colors=[pg.intColor(i,hues=self.n_ch) for i in range(self.n_ch)]
        for ch,col in enumerate(colors):
            curve=pg.PlotCurveItem(pen=pg.mkPen(col,width=1.2))
            self.eeg_plot.addItem(curve); self.eeg_curves.append(curve)
            cb=QtWidgets.QCheckBox(f"Ch {ch}"); cb.setChecked(True)
            cb.stateChanged.connect(lambda _,c=ch: self._toggle(c))
            self.chan_layout.insertWidget(ch,cb)
            self.chan_checks.append(cb)
        ticks=[(ch*OFFSET_UV,str(ch)) for ch in range(self.n_ch)]
        ax=self.eeg_plot.getAxis("left"); ax.setTicks([ticks])
        total=(self.n_ch-1)*OFFSET_UV
        self.eeg_plot.setYRange(-OFFSET_UV,total+OFFSET_UV)
        self.eeg_plot.showAxis("bottom"); self.eeg_plot.setXRange(-BUFFER_S,0)

    # ─── Buffer EEG ─────────────────────────────────────────────
    def _append_eeg(self,samples:np.ndarray):
        if not self.eeg_bufs: return
        for ch in range(self.n_ch):
            self.eeg_bufs[ch].extend(samples[:,ch])

    # ─── Draw EMG ───────────────────────────────────────────────
    def _draw_emg(self):
        if self.emg_buf:
            y=np.fromiter(self.emg_buf,float)
            self.emg_curve.setData(y)

    # ─── Draw EEG ───────────────────────────────────────────────
    def _draw_eeg(self):
        if not self.eeg_bufs: return
        N=len(self.eeg_bufs[0]); x=np.linspace(-BUFFER_S,0,N)
        for ch,curve in enumerate(self.eeg_curves):
            if not self.chan_checks[ch].isChecked():
                curve.hide(); continue
            y=np.array(self.eeg_bufs[ch],float)
            y=(y-np.median(y))*self.gain + ch*OFFSET_UV
            curve.setData(x,y); curve.show()

    # ─── Draw inference bar ─────────────────────────────────────
    def _draw_inference(self):
        # Línea horizontal desde 0 hasta el valor actual
        self.bar_curve.setData([0, self.inf_value], [0, 0])

    # ─── util ──────────────────────────────────────────────────
    def _toggle(self,ch:int):
        self.eeg_curves[ch].setVisible(self.chan_checks[ch].isChecked())

    def _send_cmd(self,code:int):
        try:
            self.q_to.put_nowait({"event_type":"arduino_cmd","code":code})
        except queue.Full:
            pass
        self.log.appendPlainText(f"[GUI] → cmd {code}")
