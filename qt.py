from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys 

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(500, 540, 300, 300)
    win.setWindowTitle("New")
    win.show()
    sys.exit(app.exec_())

window()