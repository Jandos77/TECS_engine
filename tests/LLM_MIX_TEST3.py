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

from ElementSlotNetwork_LLM_training.ElementSlotNetwork_LLM_Dataset_training import ElementSlotNet  # import model from your training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Drawing GUI ======
class DrawCanvas(QtWidgets.QLabel):
    def __init__(self, parent=None, size=280):
        super().__init__(parent)
        self.size = size
        self.pixmap = QtGui.QPixmap(self.size, self.size)
        self.pixmap.fill(QtCore.Qt.white)
        self.setPixmap(self.pixmap)
        self.last_point = None
        self.pen_width = 18
        self.setFixedSize(self.size, self.size)

    def mousePressEvent(self, event):
        self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_point is None:
            self.last_point = event.pos()
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(QtCore.Qt.black, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.last_point, event.pos())
        self.last_point = event.pos()
        self.setPixmap(self.pixmap)

    def mouseReleaseEvent(self, event):
        self.last_point = None

    def clear(self):
        self.pixmap.fill(QtCore.Qt.white)
        self.setPixmap(self.pixmap)

    def get_image(self):
        return self.pixmap.toImage()

# ====== Main Interface ======
class DigitTester(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("Element-Slot MNIST Tester")
        self.model = model.to(device)
        self.model.eval()
        self.canvas = DrawCanvas()
        self.predict_button = QtWidgets.QPushButton("Predict")
        self.clear_button = QtWidgets.QPushButton("Clear Canvas")
        self.result_label = QtWidgets.QLabel("Prediction: ")
        self.routing_label = QtWidgets.QLabel("Routing Slots Heatmap")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.predict_button)
        hlayout.addWidget(self.clear_button)
        layout.addLayout(hlayout)
        layout.addWidget(self.result_label)
        layout.addWidget(self.routing_label)
        self.setLayout(layout)

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.canvas.clear)

    def predict(self):
        # Get image
        img = self.canvas.get_image()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(img.height(), img.width(), 4)
        
        # Extract R channel
        gray = arr[..., 0].astype(np.float32)
        
        # Invert colors (black background, white digit) and scale to 0-1
        gray = 255.0 - gray
        gray = gray / 255.0
        gray[gray < 0.15] = 0.0

        # Crop edges of the drawn digit and center it (like in MNIST)
        coords = np.argwhere(gray > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            digit = gray[y_min:y_max+1, x_min:x_max+1]

            h, w = digit.shape
            size = max(h, w)
            padded = np.zeros((size, size))
            # Center the cropped digit along the larger dimension
            padded_y = (size - h) // 2
            padded_x = (size - w) // 2
            padded[padded_y:padded_y+h, padded_x:padded_x+w] = digit
            gray = padded

        # Resize to 28x28
        gray = Image.fromarray((gray * 255).astype(np.uint8))
        gray = gray.resize((28, 28))

        # Normalize (like during MNIST model training)
        gray = np.array(gray).astype(np.float32) / 255.0
        gray = (gray - 0.1307) / 0.3081

        tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, routings = self.model(tensor)
            pred = logits.argmax(dim=1).item()
            self.result_label.setText(f"Prediction: {pred}")

            # Visualize routing slots of the last layer (last step, first batch element)
            last_routing = routings[-1][-1][0]  # [N, K]
            routing_img = (last_routing.cpu().numpy() * 255).astype(np.uint8)
            # Simple text heatmap visualization
            routing_text = "\n".join([" ".join(f"{int(v):3}" for v in row) for row in routing_img])
            self.routing_label.setText(f"Routing Slots (last layer):\n{routing_text}")

# ====== Run application ======
if __name__ == "__main__":
    model = ElementSlotNet()
    model_path = os.path.join(project_root, "ElementSlotNetwork_LLM_training", "best_element_slot.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    app = QtWidgets.QApplication(sys.argv)
    tester = DigitTester(model)
    tester.show()
    sys.exit(app.exec_())