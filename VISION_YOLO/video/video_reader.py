"""Thread-safe video reader with enhanced error handling."""

import cv2
import numpy as np
from typing import Tuple, Optional
from threading import Thread, Lock
from queue import Queue, Empty
import time
import logging

logger = logging.getLogger(__name__)


class ThreadSafeVideoReader:
    """Thread-safe video reader with enhanced error handling."""
    
    def __init__(self, video_path: str, buffer_size: int = 32):
        """Initialize the video reader with error handling."""
        self.video_path = video_path
        self.cap = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.lock = Lock()
        
        # Video properties
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame_idx = 0
        
        try:
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.fps <= 0 or self.width <= 0 or self.height <= 0:
                raise ValueError(f"Invalid video properties: fps={self.fps}, size={self.width}x{self.height}")
                
        except Exception as e:
            logger.error(f"Error initializing video reader: {str(e)}")
            if self.cap:
                self.cap.release()
            raise
    
    def start(self) -> 'ThreadSafeVideoReader':
        """Start the background frame reading thread with error handling."""
        try:
            self.thread = Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            return self
        except Exception as e:
            logger.error(f"Error starting video reader thread: {str(e)}")
            self.stop()
            raise
    
    def _read_frames(self):
        """Background thread function to continuously read frames with error handling."""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self.stopped:
            try:
                if not self.frame_queue.full():
                    with self.lock:
                        ret, frame = self.cap.read()
                        
                    if not ret:
                        logger.info("End of video reached or error reading frame")
                        self.stopped = True
                        break
                        
                    self.frame_queue.put((self.current_frame_idx, frame))
                    self.current_frame_idx += 1
                    consecutive_errors = 0  # Reset error counter on successful read
                else:
                    time.sleep(0.001)  # Prevent busy waiting
            except Exception as e:
                logger.warning(f"Error reading frame: {str(e)}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({max_consecutive_errors}), stopping video reader")
                    self.stopped = True
                    break
                
                time.sleep(0.1)  # Brief pause before retrying
    
    def read(self) -> Tuple[bool, Optional[int], Optional[np.ndarray]]:
        """Read the next frame from the buffer with error handling."""
        try:
            frame_idx, frame = self.frame_queue.get(timeout=1.0)
            return True, frame_idx, frame
        except Empty:
            return False, None, None
        except Exception as e:
            logger.warning(f"Error getting frame from queue: {str(e)}")
            return False, None, None
    
    def read_first_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the first frame directly with error handling."""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position
            return ret, frame if ret else None
        except Exception as e:
            logger.error(f"Error reading first frame: {str(e)}")
            return False, None
    
    def stop(self):
        """Stop the background thread and release resources with error handling."""
        self.stopped = True
        if hasattr(self, 'thread'):
            try:
                self.thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping video reader thread: {str(e)}")
        
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing video capture: {str(e)}")
        
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()

