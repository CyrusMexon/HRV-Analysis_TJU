import sys
import numpy as np
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from hrvlib.data_handler import load_rr_file
from hrvlib.metrics.time_domain import HRVTimeDomainAnalysis
from hrvlib.metrics.freq_domain import HRVFreqDomainAnalysis

# from hrvlib.metrics.nonlinear import HRVNonlinearAnalysis   # (future extension)


class HRVWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kubios-like HRV MVP")
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.text = QtWidgets.QTextEdit(readOnly=True)

        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(self.canvas)
        layout.addWidget(self.text)
        self.setCentralWidget(w)

        open_act = self.menuBar().addMenu("&File").addAction("Open RR File")
        open_act.triggered.connect(self.open_file)

        # TODO: Add menu options for opening TXT and EDF files.
        # TODO: Add buttons for exporting results to CSV/PDF.

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open RR/ECG/PPG File",
            "",
            "All Supported (*.csv *.txt *.edf);;CSV (*.csv);;Text (*.txt);;EDF (*.edf)",
        )
        if not path:
            return

        # Load using DataHandler
        bundle = load_rr_file(path)
        rr = bundle.rri_ms if bundle.rri_ms else bundle.ppi_ms

        if not rr:
            self.text.setPlainText("No RR/PPI intervals found in file.")
            return

        # ---- Time-domain analysis ----
        td = HRVTimeDomainAnalysis(rr)
        s_sdnn = td.sdnn()
        s_rmssd = td.rmssd()
        s_pnn50 = td.pnn50()

        # ---- Frequency-domain analysis ----
        fd = HRVFreqDomainAnalysis(rr, fs=4.0)  # 4 Hz is common default
        vlf, lf, hf, lf_hf = fd.band_powers()

        # ---- Plot RR intervals ----
        ax = self.fig.clear() or self.fig.add_subplot(111)
        t = np.cumsum(rr) / 1000.0
        ax.plot(t, rr)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RR (ms)")
        self.canvas.draw()

        # ---- Display results ----
        txt = (
            f"File: {path}\n\n"
            f"SDNN: {s_sdnn:.2f} ms\n"
            f"RMSSD: {s_rmssd:.2f} ms\n"
            f"pNN50: {s_pnn50:.2f} %\n\n"
            f"VLF: {vlf:.3f}\nLF: {lf:.3f}\nHF: {hf:.3f}\nLF/HF: {lf_hf:.3f}"
        )
        self.text.setPlainText(txt)

        # TODO: Display nonlinear metrics (Poincaré, Sample Entropy).
        # TODO: Show preprocessing results before metric calculation.


def run():
    app = QtWidgets.QApplication(sys.argv)
    win = HRVWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
