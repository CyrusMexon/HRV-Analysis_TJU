import sys
import numpy as np
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from hrvlib.data_handler import load_rr_csv
from hrvlib.metrics.time_domain import sdnn, rmssd, pnn50
from hrvlib.metrics.freq_domain import band_powers

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

        open_act = self.menuBar().addMenu("&File").addAction("Open RR CSV")
        open_act.triggered.connect(self.open_file)

        # TODO: Add menu options for opening TXT and EDF files.
        # TODO: Add buttons for exporting results to CSV/PDF.

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open RR CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        rr = load_rr_csv(path)
        s_sdnn = sdnn(rr)
        s_rmssd = rmssd(rr)
        s_pnn50 = pnn50(rr)
        vlf, lf, hf, lf_hf = band_powers(rr)

        ax = self.fig.clear() or self.fig.add_subplot(111)
        t = np.cumsum(rr) / 1000.0
        ax.plot(t, rr)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RR (ms)")
        self.canvas.draw()

        txt = (
            f"SDNN: {s_sdnn:.2f} ms\nRMSSD: {s_rmssd:.2f} ms\npNN50: {s_pnn50:.2f}%\n"
            f"VLF: {vlf:.3f} | LF: {lf:.3f} | HF: {hf:.3f} | LF/HF: {lf_hf:.3f}"
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
