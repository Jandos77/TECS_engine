# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn.functional as F
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ElementSlotNetwork_LLM_training.ElementSlotNetwork_LLM_Dataset_training import ElementSlotNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================
# WINDOW 1 — DRAW
# =========================
class DrawWindow(QtWidgets.QWidget):
    def __init__(self, model, output_window):
        super().__init__()

        self.model = model
        self.output_window = output_window

        self.setWindowTitle("INPUT (draw here)")

        self.canvas = QtWidgets.QLabel()
        self.size = 280

        self.pixmap = QtGui.QPixmap(self.size, self.size)
        self.pixmap.fill(QtCore.Qt.white)
        self.canvas.setPixmap(self.pixmap)
        self.canvas.setFixedSize(self.size, self.size)

        self.last_point = None
        self.pen_width = 8

        btn_predict = QtWidgets.QPushButton("Predict")
        btn_clear = QtWidgets.QPushButton("Clear")

        btn_predict.clicked.connect(self.predict)
        btn_clear.clicked.connect(self.clear)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(btn_predict)
        h.addWidget(btn_clear)

        layout.addLayout(h)
        self.setLayout(layout)

        # mouse events
        self.canvas.mousePressEvent = self.mousePressEvent
        self.canvas.mouseMoveEvent = self.mouseMoveEvent
        self.canvas.mouseReleaseEvent = self.mouseReleaseEvent

    def mousePressEvent(self, e):
        self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if self.last_point is None:
            self.last_point = e.pos()

        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(QtCore.Qt.black, self.pen_width,
                         QtCore.Qt.SolidLine,
                         QtCore.Qt.RoundCap,
                         QtCore.Qt.RoundJoin)

        painter.setPen(pen)
        painter.drawLine(self.last_point, e.pos())
        self.last_point = e.pos()
        self.canvas.setPixmap(self.pixmap)

    def mouseReleaseEvent(self, e):
        self.last_point = None

    def clear(self):
        self.pixmap.fill(QtCore.Qt.white)
        self.canvas.setPixmap(self.pixmap)

    def get_numpy(self):
        img = self.pixmap.toImage()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(img.height(), img.width(), 4)
        return arr[..., 0]

    # =========================
    # PREPROCESS (IMPORTANT)
    # =========================
    def preprocess(self, img):
        img = 255 - img
        img = img.astype(np.float32) / 255.0
        img[img < 0.15] = 0.0

        coords = np.argwhere(img > 0)
        if len(coords):
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img[y0:y1+1, x0:x1+1]

        h, w = img.shape
        size = max(h, w)
        padded = np.zeros((size, size))
        padded[:h, :w] = img
        img = padded

        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((28, 28))

        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.1307) / 0.3081

        return img

    # =========================
    # PREDICT
    # =========================
    def predict(self):
        raw = self.get_numpy()
        img = self.preprocess(raw)

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = probs.argmax()

        # send to 2nd window
        self.output_window.update_output(img, pred, probs)


# =========================
# WINDOW 2 — OUTPUT
# =========================
class OutputWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OUTPUT (model result)")

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(280, 280)

        self.result_label = QtWidgets.QLabel("Prediction:")
        self.result_label.setStyleSheet("font-size: 20px")

        self.prob_label = QtWidgets.QLabel("")
        self.prob_label.setStyleSheet("font-size: 14px")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.prob_label)

        self.setLayout(layout)

    def update_output(self, img28, pred, probs):
        # show 28x28
        img = (img28 - img28.min()) / (img28.max() - img28.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

        qimg = QtGui.QImage(
            img.data, 28, 28, 28,
            QtGui.QImage.Format_Grayscale8
        )

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(280, 280)

        self.image_label.setPixmap(pix)

        self.result_label.setText(f"Prediction: {pred}")

        text = ""
        for i, p in enumerate(probs):
            text += f"{i}: {p:.2f}\n"

        self.prob_label.setText(text)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = ElementSlotNet()
    model_path = os.path.join(project_root, "ElementSlotNetwork_LLM_training", "best_element_slot.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    app = QtWidgets.QApplication(sys.argv)

    output_window = OutputWindow()
    draw_window = DrawWindow(model, output_window)

    draw_window.show()
    output_window.show()

    sys.exit(app.exec_())