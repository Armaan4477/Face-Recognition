import cv2
import dlib
import numpy as np
import os
from datetime import datetime

class FaceDetector:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        
        self.load_known_faces()
        self.current_frame = None
        self.current_face_coords = None

    def load_known_faces(self):
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            return

        for filename in os.listdir(faces_dir):
            if filename.endswith(".jpg"):
                path = os.path.join(faces_dir, filename)
                name = os.path.splitext(filename)[0]
                image = cv2.imread(path)
                encoding = self.get_face_encoding(image)
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)

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
            print(f"Error getting face encoding: {str(e)}")
            return None

    def save_face(self, name):
        if self.current_frame is None or self.current_face_coords is None:
            return False
            
        try:
            x, y, w, h = self.current_face_coords

            padding = 20
            face_image = self.current_frame[max(0, y-padding):y+h+padding, 
                                         max(0, x-padding):x+w+padding]
            
            os.makedirs("faces", exist_ok=True)
            
            file_path = os.path.join("faces", f"{name}.jpg")
            cv2.imwrite(file_path, face_image)
            
            face_encoding = self.get_face_encoding_from_coords(self.current_frame, self.current_face_coords)
            if face_encoding is not None:
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                return True
            
            return False
        except Exception as e:
            print(f"Error saving face: {str(e)}")
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
            print(f"Error getting face encoding from coords: {str(e)}")
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
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, f"{name} - Marked! ({confidence:.1f}%)", 
                                      (x + 6, y+h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
