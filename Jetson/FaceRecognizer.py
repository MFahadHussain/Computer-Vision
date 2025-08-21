import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from datetime import datetime
import os
import time

class FaceRecognizer:
    def __init__(self, rtsp_url, model_name='buffalo_l'):
        self.rtsp_url = rtsp_url
        self.stream_handler = RTSPStreamHandler(rtsp_url)
        self.stream_handler.start()
        
        # Initialize InsightFace
        self.app = FaceAnalysis(name=model_name, root='~/.insightface/models')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Known faces database
        self.known_faces = []
        self.known_names = []
        self.known_embeddings = []
        
        # Cooldown settings
        self.last_recognition = {}
        self.cooldown_period = 5  # seconds
        
        # Load known faces
        self.load_known_faces()
        
        # Performance monitoring
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def load_known_faces(self):
        """Load known faces from database"""
        # Implementation would fetch from DB
        # For demo, we'll load from a directory
        known_faces_dir = "known_faces"
        if os.path.exists(known_faces_dir):
            for person_name in os.listdir(known_faces_dir):
                person_dir = os.path.join(known_faces_dir, person_name)
                if os.path.isdir(person_dir):
                    embeddings = []
                    for img_file in os.listdir(person_dir):
                        img_path = os.path.join(person_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            faces = self.app.get(img)
                            if faces:
                                embeddings.append(faces[0].embedding)
                    
                    if embeddings:
                        avg_embedding = np.mean(embeddings, axis=0)
                        self.known_names.append(person_name)
                        self.known_embeddings.append(avg_embedding)
    
    def recognize_frame(self, frame):
        """Recognize faces in frame"""
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Add FPS to frame
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add stream status
        status = "STREAM: ONLINE" if self.stream_handler.is_alive() else "STREAM: OFFLINE"
        color = (0, 255, 0) if self.stream_handler.is_alive() else (0, 0, 255)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        faces = self.app.get(frame)
        results = []
        
        for face in faces:
            # Get face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Get embedding
            embedding = face.embedding
            
            # Compare with known faces
            max_similarity = -1
            recognized_name = "Unknown"
            
            for i, known_embedding in enumerate(self.known_embeddings):
                similarity = np