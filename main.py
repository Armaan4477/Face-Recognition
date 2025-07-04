import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QInputDialog, QMessageBox, 
                           QFileDialog, QHBoxLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
from face_detector import FaceDetector
from database_manager import DatabaseManager
from constant import ConfigManager
from loges import logger
import os

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setFixedSize(900, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.config_manager = ConfigManager()
        self.config_manager.config_ready.connect(self.on_config_ready)
        self.config_manager.config_updated.connect(self.on_config_updated)
        
        self.config_data = {}
        self.face_detector = None
        self.db_manager = None
        self.capture = None
        
        self.setup_ui()
        
        self.config_manager.start()

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        central_widget.setFocus()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        storage_layout = QHBoxLayout()
        self.storage_label = QLabel("Storage Location: Loading...")
        self.storage_label.setStyleSheet("QLabel { font-size: 12px; }")
        storage_layout.addWidget(self.storage_label)
        
        self.change_location_btn = QPushButton("Change Location")
        self.change_location_btn.setMaximumWidth(150)
        self.change_location_btn.clicked.connect(self.change_storage_location)
        self.change_location_btn.setEnabled(False)
        storage_layout.addWidget(self.change_location_btn)
        
        layout.addLayout(storage_layout)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("QLabel { border: 1px solid gray; }")
        layout.addWidget(self.camera_label, alignment=Qt.AlignmentFlag.AlignCenter)

        button_style = """
        QPushButton {
            min-height: 40px;
            max-width: 200px;
            font-size: 14px;
        }
        """
        
        self.register_btn = QPushButton("Register New Face")
        self.register_btn.setStyleSheet(button_style)
        self.register_btn.setEnabled(False)
        layout.addWidget(self.register_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.register_btn.clicked.connect(self.register_face)

        self.mark_attendance_btn = QPushButton("Mark Attendance")
        self.mark_attendance_btn.setStyleSheet(button_style)
        self.mark_attendance_btn.setEnabled(False)
        layout.addWidget(self.mark_attendance_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.mark_attendance_btn.clicked.connect(self.mark_attendance)

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("QLabel { color: blue; font-size: 14px; }")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.is_registering = False
        self.is_marking_attendance = False
        self.current_name = None

    def on_config_ready(self):
        self.config_data = self.config_manager.get_config()
        logger.info("Configuration loaded successfully")
        self.initialize_components()

    def on_config_updated(self, config_data):
        self.config_data = config_data
        logger.info("Configuration updated")
        self.update_storage_display()
        
        if self.face_detector and self.db_manager:
            logger.info("Reinitializing components with new storage path")
            self.initialize_components()

    def update_storage_display(self):
        if 'save_to_directory' in self.config_data:
            path = self.config_data['save_to_directory']
            display_path = path if len(path) <= 50 else f"...{path[-47:]}"
            self.storage_label.setText(f"Storage Location: {display_path}")
            logger.info(f"Storage location display updated: {path}")

    def initialize_components(self):
        if not self.check_models():
            logger.error("Required dlib models not found")
            QMessageBox.critical(self, "Missing Models", 
                "Dlib models not found. Please run setup_models.py first.")
            sys.exit()

        try:
            storage_path = self.config_data.get('save_to_directory', '')
            logger.info(f"Initializing components with storage path: {storage_path}")
            
            self.face_detector = FaceDetector(storage_path)
            self.db_manager = DatabaseManager(storage_path)
            
            self.update_storage_display()
            
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(30)
                logger.info("Camera initialized successfully")

            self.register_btn.setEnabled(True)
            self.mark_attendance_btn.setEnabled(True)
            self.change_location_btn.setEnabled(True)
            
            self.status_label.setText("Status: Ready")
            logger.info("Application initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            QMessageBox.critical(self, "Initialization Error", 
                f"Failed to initialize components: {str(e)}")
            sys.exit()

    def change_storage_location(self):
        current_path = self.config_data.get('save_to_directory', '')
        new_path = QFileDialog.getExistingDirectory(
            self, "Select Storage Directory", current_path
        )
        
        if new_path and new_path != current_path:
            logger.info(f"Changing storage location from {current_path} to {new_path}")
            self.config_data['save_to_directory'] = new_path
            self.config_manager.write_config(self.config_data)
            QMessageBox.information(self, "Success", 
                f"Storage location changed to: {new_path}")

    def check_models(self):
        """Check if required dlib models exist"""
        models_dir = get_resource_path("models")
        required_models = [
            "shape_predictor_68_face_landmarks.dat",
            "dlib_face_recognition_resnet_model_v1.dat"
        ]
        
        logger.info(f"Checking for models in: {models_dir}")
        for model in required_models:
            model_path = os.path.join(models_dir, model)
            if not os.path.exists(model_path):
                logger.error(f"Required model not found: {model_path}")
                return False
        logger.info("All required models found")
        return True

    def update_frame(self):
        if self.capture is None:
            return
            
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
                logger.info(f"Starting face registration for: {name}")
                self.current_name = name
                self.is_registering = True
                self.register_btn.setText("Press SPACE to capture face")
                self.status_label.setText("Status: Position face and press SPACE")
                QMessageBox.information(self, "Instructions", 
                    "Position the face in the frame and press SPACE to capture.")
        else:
            logger.info("Face registration cancelled")
            self.is_registering = False
            self.current_name = None
            self.register_btn.setText("Register New Face")
            self.status_label.setText("Status: Ready")

    def mark_attendance(self):
        self.is_marking_attendance = not self.is_marking_attendance
        self.mark_attendance_btn.setText("Stop Marking" if self.is_marking_attendance else "Mark Attendance")
        logger.info(f"Attendance marking {'started' if self.is_marking_attendance else 'stopped'}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.is_registering and self.current_name:
            self.status_label.setText("Status: Capturing face...")
            logger.info(f"Capturing face for: {self.current_name}")
            success = self.face_detector.save_face(self.current_name)
            if success:
                logger.info(f"Face registered successfully for: {self.current_name}")
                QMessageBox.information(self, "Success", 
                    f"Face registered successfully for {self.current_name}")
                self.status_label.setText("Status: Face registered successfully!")
                self.is_registering = False
                self.current_name = None
                self.register_btn.setText("Register New Face")
            else:
                logger.warning(f"Failed to capture face for: {self.current_name}")
                self.status_label.setText("Status: No face detected! Try again.")
                QMessageBox.warning(self, "Error", 
                    "Could not detect face. Please try again.")

    def closeEvent(self, event):
        logger.info("Application closing")
        if self.capture:
            self.capture.release()
        if hasattr(self, 'config_manager'):
            self.config_manager.quit()
            self.config_manager.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
