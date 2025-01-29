import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
from face_detector import FaceDetector
from database_manager import DatabaseManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        # Set fixed window size
        self.setFixedSize(800, 600)

        # Enable strong focus policy to capture key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Initialize components
        self.face_detector = FaceDetector()
        self.db_manager = DatabaseManager()
        
        # Create central widget and layout
        central_widget = QWidget()
        central_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        central_widget.setFocus()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create camera view with fixed size
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)  # Standard camera resolution
        layout.addWidget(self.camera_label)

        # Adjust button sizes
        button_style = """
        QPushButton {
            min-height: 40px;
            max-width: 200px;
            font-size: 14px;
        }
        """
        
        # Create buttons with styled size
        self.register_btn = QPushButton("Register New Face")
        self.register_btn.setStyleSheet(button_style)
        layout.addWidget(self.register_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.register_btn.clicked.connect(self.register_face)

        self.mark_attendance_btn = QPushButton("Mark Attendance")
        self.mark_attendance_btn.setStyleSheet(button_style)
        layout.addWidget(self.mark_attendance_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.mark_attendance_btn.clicked.connect(self.mark_attendance)

        # Add status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 14px; }")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.is_registering = False
        self.is_marking_attendance = False
        self.current_name = None

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Resize frame to fit the fixed label size
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

            if self.is_registering:
                self.face_detector.collect_face(frame)
            elif self.is_marking_attendance:
                self.face_detector.recognize_face(frame, self.db_manager)

    def register_face(self):
        if not self.is_registering:
            name, ok = QInputDialog.getText(self, 'Register New Face', 
                                          'Enter person name:')
            if ok and name:
                self.current_name = name
                self.is_registering = True
                self.register_btn.setText("Press SPACE to capture face")
                self.status_label.setText("Status: Position face and press SPACE")
                QMessageBox.information(self, "Instructions", 
                    "Position the face in the frame and press SPACE to capture.")
        else:
            self.is_registering = False
            self.current_name = None
            self.register_btn.setText("Register New Face")
            self.status_label.setText("Status: Ready")

    def mark_attendance(self):
        self.is_marking_attendance = not self.is_marking_attendance
        self.mark_attendance_btn.setText("Stop Marking" if self.is_marking_attendance else "Mark Attendance")

    def keyPressEvent(self, event):
        print("Key pressed:", event.key())  # Debug log
        if event.key() == Qt.Key.Key_Space and self.is_registering and self.current_name:
            self.status_label.setText("Status: Capturing face...")
            if hasattr(self.face_detector, 'current_frame'):
                success = self.face_detector.save_face(self.current_name)
                if success:
                    QMessageBox.information(self, "Success", 
                        f"Face registered successfully for {self.current_name}")
                    self.status_label.setText("Status: Face registered successfully!")
                    self.is_registering = False
                    self.current_name = None
                    self.register_btn.setText("Register New Face")
                else:
                    self.status_label.setText("Status: No face detected or could not encode face! Try again.")
                    QMessageBox.warning(self, "Error", 
                        "Could not detect or encode face properly. Please try again.")

    def closeEvent(self, event):
        self.capture.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
