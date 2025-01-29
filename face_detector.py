import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class FaceDetector:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.load_known_faces()
        self.current_frame = None

    def load_known_faces(self):
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            return

        for filename in os.listdir(faces_dir):
            if filename.endswith(".jpg"):
                path = os.path.join(faces_dir, filename)
                name = os.path.splitext(filename)[0]
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

    def save_face(self, name):
        if self.current_frame is None:
            return False
            
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model="hog")
            
            if not face_locations:
                return False
                
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                return False
                
            # Get the first face found
            top, right, bottom, left = face_locations[0]
            face_image = self.current_frame[top:bottom, left:right]
            
            # Ensure faces directory exists
            os.makedirs("faces", exist_ok=True)
            
            # Save face image with person's name
            file_path = os.path.join("faces", f"{name}.jpg")
            cv2.imwrite(file_path, face_image)
            
            # Add to known faces
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            return True
        except Exception as e:
            print(f"Error saving face: {str(e)}")
            return False

    def collect_face(self, frame):
        self.current_frame = frame.copy()  # Store a copy of the current frame
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        
        if face_locations:
            top, right, bottom, left = face_locations[0]
            # Draw green rectangle around detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected - Press SPACE", 
                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            # Removed waitKey block

    def recognize_face(self, frame, db_manager):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model="hog")
            self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

            for (top, right, bottom, left), face_encoding in zip(self.face_locations, self.face_encodings):
                if len(self.known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        if db_manager.mark_attendance(name):
                            # Draw rectangle around face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, f"{name} - Marked!", (left + 6, bottom - 6), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
