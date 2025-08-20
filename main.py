import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow  # Assuming MainWindow is defined in main_window.py

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(experiment_name="image_classification_imagenet")
    win.show()
    sys.exit(app.exec())
