# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ElementSlotNetwork_LLM_training.ElementSlotNetwork_LLM_Dataset_training import ElementSlotNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================
# DRAW WINDOW
# =========================
class DrawWindow(QtWidgets.QWidget):
    def __init__(self, predict_callback):
        super().__init__()
        self.setWindowTitle("DRAW DIGIT")

        self.canvas_size = 280
        self.image = QtGui.QImage(self.canvas_size, self.canvas_size, QtGui.QImage.Format_RGB32)
        self.image.fill(QtCore.Qt.white)

        self.last_point = None
        self.pen_width = 18

        self.predict_callback = predict_callback

        self.setFixedSize(self.canvas_size, self.canvas_size + 50)

        self.btn_predict = QtWidgets.QPushButton("Predict", self)
        self.btn_predict.move(10, self.canvas_size + 10)
        self.btn_predict.clicked.connect(self.run_predict)

        self.btn_clear = QtWidgets.QPushButton("Clear", self)
        self.btn_clear.move(120, self.canvas_size + 10)
        self.btn_clear.clicked.connect(self.clear)

    def paintEvent(self, event):
        canvas_painter = QtGui.QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_point is None:
            return

        painter = QtGui.QPainter(self.image)
        pen = QtGui.QPen(QtCore.Qt.black, self.pen_width,
                         QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.last_point, event.pos())

        self.last_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_point = None

    def clear(self):
        self.image.fill(QtCore.Qt.white)
        self.update()

    def run_predict(self):
        self.predict_callback(self.image)


# =========================
# RESULT WINDOW
# =========================
class ResultWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MODEL OUTPUT")

        self.label_img = QtWidgets.QLabel()
        self.label_img.setFixedSize(200, 200)

        self.label_text = QtWidgets.QLabel("Prediction: ")
        self.label_text.setStyleSheet("font-size: 24px;")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_img)
        layout.addWidget(self.label_text)

        self.setLayout(layout)
        self.setFixedSize(220, 260)

    def update_result(self, img28, pred):
        # show normalized image
        img = (img28 * 255).astype(np.uint8)
        qimg = QtGui.QImage(img.data, 28, 28, 28, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(200, 200)
        self.label_img.setPixmap(pix)

        self.label_text.setText(f"Prediction: {pred}")


# =========================
# MODEL WRAPPER
# =========================
class Predictor:
    def __init__(self):
        self.model = ElementSlotNet().to(device)
        model_path = os.path.join(project_root, "ElementSlotNetwork_LLM_training", "best_element_slot.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def preprocess(self, qimage):
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)

        gray = arr[..., 0].astype(np.float32)

        # invert
        gray = 255 - gray

        # normalize
        gray /= 255.0

        # crop (important!)
        coords = np.argwhere(gray > 0.2)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            digit = gray[y_min:y_max+1, x_min:x_max+1]

            h, w = digit.shape
            size = max(h, w)
            padded = np.zeros((size, size))
            padded[:h, :w] = digit
            gray = padded

        # resize
        gray = Image.fromarray((gray * 255).astype(np.uint8))
        gray = gray.resize((28, 28))

        gray = np.array(gray).astype(np.float32) / 255.0

        # MNIST normalize
        gray = (gray - 0.1307) / 0.3081

        return gray

    def predict(self, qimage):
        img = self.preprocess(qimage)

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.model(tensor)
            pred = logits.argmax(dim=1).item()

        return img, pred


# =========================
# MAIN APP
# =========================
class App:
    def __init__(self):
        self.predictor = Predictor()

        self.result_window = ResultWindow()
        self.draw_window = DrawWindow(self.on_predict)

        self.draw_window.show()
        self.result_window.show()

    def on_predict(self, qimage):
        img28, pred = self.predictor.predict(qimage)
        self.result_window.update_result(img28, pred)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = App()
    sys.exit(app.exec_())