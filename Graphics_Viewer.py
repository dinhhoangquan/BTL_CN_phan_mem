import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundRole(QPalette.Dark)

        scene = QGraphicsScene()
        self.setScene(scene)

        self._pixmap_item = QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)

    def load_pixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)
        self.fitToWindow()

    def fitToWindow(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitToWindow()
def main():
    app = QApplication(sys.argv)

    view = ImageViewer()
    view.resize(640, 480)
    view.show()

    pixmap = QPixmap("Save_png/1.png")
    view.load_pixmap(pixmap)

    ret = app.exec()
    sys.exit(ret)
if __name__ == "__main__":
    main()