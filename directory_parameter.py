from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QSizePolicy
import pyqtgraph.parametertree.parameterTypes as pTypes

class DirectoryBrowseWidget(QWidget):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sigChanged = None  # No change signal needed for this custom widget
        self.setMinimumHeight(25)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(param.value()))
        self.lineEdit.textChanged.connect(self._on_text_changed)
        browse_dir_btn = QPushButton("Dir...")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setMinimumWidth(50)
        browse_dir_btn.clicked.connect(self._browse_directory)
        browse_file_btn = QPushButton("File...")
        browse_file_btn.setMaximumWidth(50)
        browse_file_btn.setMinimumWidth(50)
        browse_file_btn.clicked.connect(self._browse_file)
        layout.addWidget(self.lineEdit)
        layout.addWidget(browse_dir_btn)
        layout.addWidget(browse_file_btn)
    def _on_text_changed(self, text):
        self.param.setValue(text)
    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.param.name()}")
        if directory:
            self.lineEdit.setText(directory)
            self.param.setValue(directory)
    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(None, f"Select file for {self.param.name()}")
        if file_path:
            self.lineEdit.setText(file_path)
            self.param.setValue(file_path)
    def value(self):
        return self.lineEdit.text()
    def setValue(self, value):
        self.lineEdit.setText(str(value))
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.lineEdit.setFocus()
    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.show()

class DirectoryParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        widget = DirectoryBrowseWidget(self.param)
        widget.setVisible(True)
        return widget
    def valueChanged(self, param, data, info=None, force=False):
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.setValue(data)
            self.widget.show()
    def showEditor(self):
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
            return True
        return super().showEditor()
    def hideEditor(self):
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
        return True

class DirectoryParameter(pTypes.SimpleParameter):
    itemClass = DirectoryParameterItem
