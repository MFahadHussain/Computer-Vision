import cv2
import time
import threading
import queue
import numpy as np

class RTSPStreamHandler:
    def __init__(self, rtsp_url, buffer_size=10):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.last_frame = None
        self.last_frame_time = 0
        self.reconnect_interval = 5  # seconds
        self.max_reconnect_attempts = 3
        
    def start(self):
        """Start the RTSP stream thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the RTSP stream thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            
    def _process_stream(self):
        """Process RTSP stream in a separate thread"""
        reconnect_attempts = 0
        
        while self.running and reconnect_attempts < self.max_reconnect_attempts:
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise Exception(f"Failed to open RTSP stream: {self.rtsp_url}")
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                reconnect_attempts = 0  # Reset on successful connection
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Put frame in queue (non-blocking)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    # Update last frame
                    self.last_frame = frame.copy()
                    self.last_frame_time = time.time()
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"RTSP Stream Error: {e}")
                reconnect_attempts += 1
                if reconnect_attempts < self.max_reconnect_attempts:
                    print(f"Attempting to reconnect in {self.reconnect_interval} seconds...")
                    time.sleep(self.reconnect_interval)
                else:
                    print("Max reconnection attempts reached. Stopping stream.")
                    
            finally:
                if 'cap' in locals():
                    cap.release()
                    
    def get_frame(self):
        """Get the latest frame from the queue"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        elif self.last_frame is not None:
            # Return last known frame if queue is empty
            return self.last_frame
        else:
            # Return black frame if no frames available
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
    def is_alive(self):
        """Check if stream is active"""
        return self.running and (time.time() - self.last_frame_time < 5)