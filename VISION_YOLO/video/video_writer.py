"""Thread-safe video writer with enhanced error handling."""

import cv2
import numpy as np
from pathlib import Path
from threading import Thread, Lock
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class ThreadSafeVideoWriter:
    """Thread-safe video writer with enhanced error handling."""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int, 
                 buffer_size: int = 64):
        """Initialize the video writer with error handling."""
        self.output_path = output_path
        self.writer = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.lock = Lock()
        self.error_count = 0
        self.max_errors = 10
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not self.writer.isOpened():
                raise ValueError(f"Cannot create video writer: {output_path}")
                
        except Exception as e:
            logger.error(f"Error initializing video writer: {str(e)}")
            if self.writer:
                self.writer.release()
            raise
    
    def start(self) -> 'ThreadSafeVideoWriter':
        """Start the background frame writing thread with error handling."""
        try:
            self.thread = Thread(target=self._write_frames, daemon=True)
            self.thread.start()
            return self
        except Exception as e:
            logger.error(f"Error starting video writer thread: {str(e)}")
            self.stop()
            raise
    
    def _write_frames(self):
        """Background thread function to continuously write frames with error handling."""
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                with self.lock:
                    self.writer.write(frame)
                    self.error_count = 0  # Reset error counter on successful write
            except Empty:
                continue
            except Exception as e:
                logger.warning(f"Error writing frame: {str(e)}")
                self.error_count += 1
                
                if self.error_count >= self.max_errors:
                    logger.error(f"Too many consecutive write errors ({self.max_errors}), stopping video writer")
                    self.stopped = True
                    break
    
    def write(self, frame: np.ndarray):
        """Add a frame to the write buffer with error handling."""
        try:
            if frame is None or frame.size == 0:
                logger.warning("Attempted to write empty frame")
                return
                
            self.frame_queue.put(frame)
        except Exception as e:
            logger.warning(f"Error adding frame to write queue: {str(e)}")
    
    def stop(self):
        """Stop the background thread and release resources with error handling."""
        self.stopped = True
        if hasattr(self, 'thread'):
            try:
                self.thread.join(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error stopping video writer thread: {str(e)}")
        
        if self.writer:
            try:
                with self.lock:
                    self.writer.release()
            except Exception as e:
                logger.warning(f"Error releasing video writer: {str(e)}")
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()

