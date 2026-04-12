import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication
from src.main_ui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
