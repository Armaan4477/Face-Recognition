import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
from face_detector import FaceDetector
from database_manager import DatabaseManager
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setFixedSize(800, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        if not self.check_models():
            QMessageBox.critical(self, "Missing Models", 
                "Dlib models not found. Please run setup_models.py first.")
            sys.exit()

        try:
            self.face_detector = FaceDetector()
            self.db_manager = DatabaseManager()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", 
                f"Failed to initialize face detector: {str(e)}")
            sys.exit()
        
        central_widget = QWidget()
        central_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        central_widget.setFocus()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        button_style = """
        QPushButton {
            min-height: 40px;
            max-width: 200px;
            font-size: 14px;
        }
        """
        
        self.register_btn = QPushButton("Register New Face")
        self.register_btn.setStyleSheet(button_style)
        layout.addWidget(self.register_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.register_btn.clicked.connect(self.register_face)

        self.mark_attendance_btn = QPushButton("Mark Attendance")
        self.mark_attendance_btn.setStyleSheet(button_style)
        layout.addWidget(self.mark_attendance_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.mark_attendance_btn.clicked.connect(self.mark_attendance)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 14px; }")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.is_registering = False
        self.is_marking_attendance = False
        self.current_name = None

    def check_models(self):
        """Check if required dlib models exist"""
        models_dir = "models"
        required_models = [
            "shape_predictor_68_face_landmarks.dat",
            "dlib_face_recognition_resnet_model_v1.dat"
        ]
        
        for model in required_models:
            if not os.path.exists(os.path.join(models_dir, model)):
                return False
        return True

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            
            if self.is_registering:
                self.face_detector.collect_face(frame)
            elif self.is_marking_attendance:
                self.face_detector.recognize_face(frame, self.db_manager)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

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
        if event.key() == Qt.Key.Key_Space and self.is_registering and self.current_name:
            self.status_label.setText("Status: Capturing face...")
            success = self.face_detector.save_face(self.current_name)
            if success:
                QMessageBox.information(self, "Success", 
                    f"Face registered successfully for {self.current_name}")
                self.status_label.setText("Status: Face registered successfully!")
                self.is_registering = False
                self.current_name = None
                self.register_btn.setText("Register New Face")
            else:
                self.status_label.setText("Status: No face detected! Try again.")
                QMessageBox.warning(self, "Error", 
                    "Could not detect face. Please try again.")

    def closeEvent(self, event):
        self.capture.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
