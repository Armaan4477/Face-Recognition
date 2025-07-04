import cv2
import dlib
import numpy as np
import os
import sys
from datetime import datetime
from loges import logger

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class FaceDetector:
    def __init__(self, storage_path=None):
        self.storage_path = storage_path or os.getcwd()
        self.faces_dir = os.path.join(self.storage_path, "faces")
        logger.info(f"Face detector initialized with storage path: {self.storage_path}")
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        
        self.detector = dlib.get_frontal_face_detector()
        
        # Use resource path for models
        predictor_path = get_resource_path("models/shape_predictor_68_face_landmarks.dat")
        recognition_model_path = get_resource_path("models/dlib_face_recognition_resnet_model_v1.dat")
        
        logger.info(f"Loading predictor from: {predictor_path}")
        logger.info(f"Loading recognition model from: {recognition_model_path}")
        
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(recognition_model_path)
        
        self.load_known_faces()
        self.current_frame = None
        self.current_face_coords = None

    def load_known_faces(self):
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            logger.info(f"Created faces directory: {self.faces_dir}")
            return

        face_count = 0
        for filename in os.listdir(self.faces_dir):
            if filename.endswith(".jpg"):
                path = os.path.join(self.faces_dir, filename)
                name = os.path.splitext(filename)[0]
                image = cv2.imread(path)
                encoding = self.get_face_encoding(image)
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    face_count += 1
                    logger.info(f"Loaded face for {name}")
                else:
                    logger.warning(f"Failed to load face encoding for {name}")
        
        logger.info(f"Loaded {face_count} known faces from {self.faces_dir}")

    def get_face_encoding(self, image):
        """Get face encoding using dlib"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = self.detector(gray)
            
            if len(faces) == 0:
                return None
                
            shape = self.predictor(gray, faces[0])
            
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
            
            return face_encoding
        except Exception as e:
            logger.error(f"Error getting face encoding: {str(e)}")
            return None

    def save_face(self, name):
        if self.current_frame is None or self.current_face_coords is None:
            logger.warning("No face detected to save")
            return False
            
        try:
            x, y, w, h = self.current_face_coords

            padding = 20
            face_image = self.current_frame[max(0, y-padding):y+h+padding, 
                                         max(0, x-padding):x+w+padding]
            
            os.makedirs(self.faces_dir, exist_ok=True)
            
            file_path = os.path.join(self.faces_dir, f"{name}.jpg")
            cv2.imwrite(file_path, face_image)
            logger.info(f"Saved face image to: {file_path}")
            
            face_encoding = self.get_face_encoding_from_coords(self.current_frame, self.current_face_coords)
            if face_encoding is not None:
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                logger.info(f"Face encoding saved for {name}")
                return True
            else:
                logger.error(f"Failed to generate face encoding for {name}")
            
            return False
        except Exception as e:
            logger.error(f"Error saving face: {str(e)}")
            return False

    def get_face_encoding_from_coords(self, image, coords):
        """Get face encoding using pre-detected face coordinates"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            x, y, w, h = coords
            face_rect = dlib.rectangle(x, y, x+w, y+h)
            
            shape = self.predictor(gray, face_rect)
            
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
            
            return face_encoding
        except Exception as e:
            logger.error(f"Error getting face encoding from coords: {str(e)}")
            return None

    def collect_face(self, frame):
        self.current_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if faces:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            self.current_face_coords = (x, y, w, h)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected - Press SPACE", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        else:
            self.current_face_coords = None

    def recognize_face(self, frame, db_manager):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                face_encoding = self.get_face_encoding_from_coords(frame, (x, y, w, h))
                
                if face_encoding is not None and len(self.known_face_encodings) > 0:
                    distances = []
                    for known_encoding in self.known_face_encodings:
                        distance = np.linalg.norm(known_encoding - face_encoding)
                        distances.append(distance)
                    
                    best_match_index = np.argmin(distances)
                    min_distance = distances[best_match_index]
                    
                    threshold = 0.6
                    
                    if min_distance < threshold:
                        name = self.known_face_names[best_match_index]
                        confidence = (1 - min_distance) * 100
                        
                        if db_manager.mark_attendance(name):
                            logger.info(f"Face recognized and attendance marked: {name} (confidence: {confidence:.1f}%)")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, f"{name} - Marked! ({confidence:.1f}%)", 
                                      (x + 6, y+h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        else:
                            logger.debug(f"Face recognized but attendance already marked: {name}")
                    else:
                        logger.debug(f"Unknown face detected (distance: {min_distance:.3f})")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
