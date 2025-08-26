import time
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PySide6.QtGui import QTextCursor, QFont


class ProgressLogWidget(QWidget):
    """Custom widget for displaying training logs with in-place progress bars."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.progress_line_index = -1  # Track which line has the current progress bar
        self.last_progress_info = None
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 9))  # Monospace font for better alignment
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        
    def append_log(self, text):
        """Append regular log text."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{ts}] {text}\n")
        self.text_edit.setTextCursor(cursor)
        self._scroll_to_bottom()
        
    def update_progress_bar(self, step_info, current_step, total_steps, metrics):
        """Update progress bar in place without creating new lines."""
        progress_text = self._create_progress_bar_text(step_info, current_step, total_steps, metrics)
        
        cursor = self.text_edit.textCursor()
        
        # If this is the first progress update or we reached 100%, add a new line
        if self.progress_line_index == -1 or current_step >= total_steps:
            # Add timestamp for new progress line
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.movePosition(QTextCursor.End)
            if current_step == 1:  # First step of epoch
                cursor.insertText(f"[{ts}] [TRAINING] {step_info}\n")
            cursor.insertText(f"[{ts}] [TRAINING] ")
            
            # Mark the position for future updates
            start_pos = cursor.position()
            cursor.insertText(progress_text + "\n")
            self.progress_line_index = start_pos
        else:
            # Update existing progress line
            cursor.setPosition(self.progress_line_index)
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(progress_text)
            
        # If we reached 100%, reset for next epoch
        if current_step >= total_steps:
            self.progress_line_index = -1
            
        self._scroll_to_bottom()
        
    def _create_progress_bar_text(self, step_info, current_step, total_steps, metrics):
        """Create colored progress bar text similar to Keras output."""
        # Progress calculation
        progress = min(current_step / max(total_steps, 1), 1.0)
        filled_chars = int(progress * 20)  # 20 character progress bar
        empty_chars = 20 - filled_chars
        
        # Create progress bar with colored blocks
        filled_bar = "━" * filled_chars
        empty_bar = "━" * empty_chars
        
        # Format step info
        step_text = f"{current_step}/{total_steps}"
        
        # Format timing (simplified)
        if current_step < total_steps:
            # Estimate remaining time (simplified)
            time_est = "estimating..."
        else:
            time_est = "completed"
            
        # Format metrics
        metrics_text = ""
        if metrics:
            metric_parts = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'loss' in key.lower():
                        metric_parts.append(f"{key}: {value:.4f}")
                    elif 'acc' in key.lower() or 'accuracy' in key.lower():
                        metric_parts.append(f"{key}: {value:.4f}")
                    else:
                        metric_parts.append(f"{key}: {value:.4f}")
            metrics_text = " - " + " - ".join(metric_parts) if metric_parts else ""
        
        # Combine all parts
        progress_text = f"{step_text} [{filled_bar}{empty_bar}] {time_est}{metrics_text}"
        
        return progress_text
        
    def _scroll_to_bottom(self):
        """Scroll to bottom of text edit."""
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear(self):
        """Clear all log content."""
        self.text_edit.clear()
        self.progress_line_index = -1
        
    def get_text_edit(self):
        """Get the underlying QTextEdit for compatibility."""
        return self.text_edit
