#!/usr/bin/env python3
"""
Simple test script to view the styled parameter tree in ModelGardener.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from pyqtgraph.parametertree import ParameterTree, Parameter


def create_test_parameters():
    """Create test parameters similar to ModelGardener structure."""
    return Parameter.create(
        name='Configuration',
        type='group',
        children=[
            dict(name='Basic', type='group', children=[
                dict(name='Data', type='group', children=[
                    dict(name='train_dir', type='str', value='', tip='Training data directory'),
                    dict(name='val_dir', type='str', value='', tip='Validation data directory'),
                    dict(name='batch_size', type='int', value=32, limits=(1, 1024)),
                    dict(name='num_classes', type='int', value=1000, limits=(1, 10000)),
                    dict(name='shuffle', type='bool', value=True),
                ]),
                dict(name='Model', type='group', children=[
                    dict(name='backbone_type', type='list', limits=['resnet', 'efficientnet', 'mobilenet'], value='resnet'),
                    dict(name='model_id', type='int', value=50, limits=(1, 200)),
                    dict(name='dropout_rate', type='float', value=0.0, limits=(0.0, 0.9)),
                ]),
                dict(name='Training', type='group', children=[
                    dict(name='epochs', type='int', value=100, limits=(1, 1000)),
                    dict(name='learning_rate', type='float', value=0.001, limits=(1e-6, 1.0)),
                    dict(name='momentum', type='float', value=0.9, limits=(0.0, 1.0)),
                ])
            ]),
            dict(name='Advanced', type='group', children=[
                dict(name='Model Advanced', type='group', children=[
                    dict(name='depth_multiplier', type='float', value=1.0, limits=(0.1, 2.0)),
                    dict(name='use_sync_bn', type='bool', value=False),
                ]),
                dict(name='Training Advanced', type='group', children=[
                    dict(name='steps_per_loop', type='int', value=1000, limits=(1, 10000)),
                    dict(name='validation_interval', type='int', value=1000, limits=(1, 10000)),
                ])
            ])
        ]
    )


def apply_professional_styling(tree):
    """Apply the professional styling to parameter tree."""
    style_sheet = """
    QTreeWidget {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 12pt;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        selection-background-color: #0d6efd;
        selection-color: white;
        outline: none;
    }
    
    QTreeWidget::item {
        padding: 6px;
        border-bottom: 1px solid #e9ecef;
        color: #212529;
        font-weight: 500;
    }
    
    QTreeWidget::item:selected {
        background-color: #0d6efd;
        color: white;
        border: none;
    }
    
    QTreeWidget::item:hover {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    
    QTreeWidget::item:has-children {
        font-weight: 600;
        color: #495057;
        background-color: #f1f3f4;
    }
    
    QTreeWidget::item:has-children:selected {
        background-color: #1976d2;
        color: white;
        font-weight: 600;
    }
    
    /* Style for parameter value editors */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        font-size: 11pt;
        padding: 4px 8px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        background-color: white;
        selection-background-color: #0d6efd;
    }
    
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border-color: #86b7fe;
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25);
    }
    
    QCheckBox {
        font-size: 11pt;
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #ced4da;
        border-radius: 3px;
        background-color: white;
    }
    
    QCheckBox::indicator:checked {
        background-color: #0d6efd;
        border-color: #0d6efd;
    }
    
    QCheckBox::indicator:hover {
        border-color: #86b7fe;
    }
    """
    
    tree.setStyleSheet(style_sheet)
    tree.setAlternatingRowColors(False)
    tree.setRootIsDecorated(True)
    tree.setIndentation(20)
    tree.setHeaderHidden(False)
    
    # Set header styling
    header = tree.header()
    if header:
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #495057;
                color: white;
                padding: 8px;
                border: none;
                font-size: 12pt;
                font-weight: 600;
            }
        """)
        header.setDefaultSectionSize(200)
        header.setStretchLastSection(True)


def main():
    """Create and show the styled parameter tree test window."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("🎨 ModelGardener - Styled Parameter Tree Test")
    window.resize(800, 700)
    
    # Set window styling
    window.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
        }
        QWidget {
            font-family: 'Segoe UI', 'Arial', sans-serif;
        }
    """)
    
    # Create central widget and layout
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    window.setCentralWidget(central_widget)
    
    # Add title label
    title_label = QLabel("ModelGardener - Professional Parameter Tree Styling")
    title_label.setStyleSheet("""
        QLabel {
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 16pt;
            font-weight: 600;
            color: #495057;
            padding: 12px 0px;
            background-color: transparent;
        }
    """)
    layout.addWidget(title_label)
    
    # Create parameter tree
    tree = ParameterTree()
    params = create_test_parameters()
    tree.setParameters(params, showTop=False)
    
    # Apply professional styling
    apply_professional_styling(tree)
    
    layout.addWidget(tree)
    
    # Add info label
    info_label = QLabel("✨ Professional styling includes:\n• Larger, readable fonts\n• Modern color scheme\n• Better spacing and padding\n• Visual hierarchy for groups\n• Hover and focus effects")
    info_label.setStyleSheet("""
        QLabel {
            font-size: 10pt;
            color: #6c757d;
            padding: 8px;
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            margin-top: 8px;
        }
    """)
    layout.addWidget(info_label)
    
    # Show window
    window.show()
    
    print("✅ Professional parameter tree styling test window created!")
    print("   - Larger fonts for better readability")
    print("   - Modern color scheme")
    print("   - Professional appearance")
    print("   - Better visual hierarchy")
    
    return app, window


if __name__ == "__main__":
    try:
        app, window = main()
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
