import time
import threading
import logging
from typing import Optional
from database import get_config_value

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BarrierController:
    """Barrier controller for GPIO relay control"""
    
    def __init__(self):
        """Initialize barrier controller"""
        self.barrier_type = get_config_value('barrier_type', 'gpio')
        self.activation_time = float(get_config_value('barrier_activation_time', 3.0))
        self.is_active = False
        self.lock = threading.Lock()
        
        # Initialize based on barrier type
        if self.barrier_type == 'gpio':
            self._init_gpio()
        elif self.barrier_type == 'serial':
            self._init_serial()
        elif self.barrier_type == 'network':
            self._init_network()
    
    def _init_gpio(self):
        """Initialize GPIO-based barrier control"""
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_pin = int(get_config_value('gpio_pin', 18))
            
            # Set GPIO mode and pin
            self.GPIO.setmode(GPIO.BCM)
            self.GPIO.setup(self.gpio_pin, GPIO.OUT)
            self.GPIO.output(self.gpio_pin, GPIO.LOW)  # Ensure relay is off
            
            logger.info(f"GPIO barrier control initialized on pin {self.gpio_pin}")
        except ImportError:
            logger.warning("RPi.GPIO not available, using mock GPIO")
            self.GPIO = None
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            self.GPIO = None
    
    def _init_serial(self):
        """Initialize serial-based barrier control"""
        try:
            import serial
            self.serial_port = get_config_value('serial_port', '/dev/ttyUSB0')
            self.baudrate = int(get_config_value('serial_baudrate', 9600))
            
            self.serial_conn = serial.Serial(
                self.serial_port, self.baudrate, timeout=1
            )
            
            logger.info(f"Serial barrier control initialized on {self.serial_port}")
        except ImportError:
            logger.warning("pyserial not available, using mock serial")
            self.serial_conn = None
        except Exception as e:
            logger.error(f"Serial initialization failed: {e}")
            self.serial_conn = None
    
    def _init_network(self):
        """Initialize network-based barrier control"""
        try:
            import requests
            self.requests = requests
            self.network_host = get_config_value('network_host', 'localhost')
            self.network_port = int(get_config_value('network_port', 80))
            
            logger.info(f"Network barrier control initialized for {self.network_host}:{self.network_port}")
        except ImportError:
            logger.warning("requests library not available, using mock network")
            self.requests = None
    
    def activate(self) -> bool:
        """
        Activate the barrier relay
        
        Returns:
            True if activation successful, False otherwise
        """
        with self.lock:
            if self.is_active:
                logger.warning("Barrier already active")
                return False
            
            try:
                if self.barrier_type == 'gpio':
                    result = self._activate_gpio()
                elif self.barrier_type == 'serial':
                    result = self._activate_serial()
                elif self.barrier_type == 'network':
                    result = self._activate_network()
                else:
                    logger.warning(f"Unknown barrier type: {self.barrier_type}")
                    result = False
                
                if result:
                    self.is_active = True
                    logger.info("Barrier activated successfully")
                    return True
                else:
                    logger.error("Barrier activation failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Barrier activation error: {e}")
                return False
    
    def deactivate(self):
        """Deactivate the barrier relay"""
        with self.lock:
            try:
                if self.barrier_type == 'gpio':
                    self._deactivate_gpio()
                elif self.barrier_type == 'serial':
                    self._deactivate_serial()
                elif self.barrier_type == 'network':
                    self._deactivate_network()
                
                self.is_active = False
                logger.info("Barrier deactivated")
                
            except Exception as e:
                logger.error(f"Barrier deactivation error: {e}")
    
    def _activate_gpio(self) -> bool:
        """Activate barrier via GPIO"""
        if self.GPIO:
            try:
                self.GPIO.output(self.gpio_pin, self.GPIO.HIGH)
                return True
            except Exception as e:
                logger.error(f"GPIO activation failed: {e}")
                return False
        else:
            logger.info("Mock: GPIO barrier activated")
            return True
    
    def _deactivate_gpio(self):
        """Deactivate barrier via GPIO"""
        if self.GPIO:
            try:
                self.GPIO.output(self.gpio_pin, self.GPIO.LOW)
            except Exception as e:
                logger.error(f"GPIO deactivation failed: {e}")
        else:
            logger.info("Mock: GPIO barrier deactivated")
    
    def _activate_serial(self) -> bool:
        """Activate barrier via serial command"""
        if self.serial_conn:
            try:
                # Example command - actual command depends on relay module
                self.serial_conn.write(b'ACTIVATE\r\n')
                response = self.serial_conn.readline().decode().strip()
                logger.info(f"Serial barrier activated: {response}")
                return True
            except Exception as e:
                logger.error(f"Serial activation failed: {e}")
                return False
        else:
            logger.info("Mock: Serial barrier activated")
            return True
    
    def _deactivate_serial(self):
        """Deactivate barrier via serial command"""
        if self.serial_conn:
            try:
                # Example command - actual command depends on relay module
                self.serial_conn.write(b'DEACTIVATE\r\n')
                response = self.serial_conn.readline().decode().strip()
                logger.info(f"Serial barrier deactivated: {response}")
            except Exception as e:
                logger.error(f"Serial deactivation failed: {e}")
        else:
            logger.info("Mock: Serial barrier deactivated")
    
    def _activate_network(self) -> bool:
        """Activate barrier via HTTP request"""
        if self.requests:
            try:
                url = f"http://{self.network_host}:{self.network_port}/activate"
                response = self.requests.post(url, timeout=5)
                logger.info(f"Network barrier activated: {response.status_code}")
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Network activation failed: {e}")
                return False
        else:
            logger.info("Mock: Network barrier activated")
            return True
    
    def _deactivate_network(self):
        """Deactivate barrier via HTTP request"""
        if self.requests:
            try:
                url = f"http://{self.network_host}:{self.network_port}/deactivate"
                response = self.requests.post(url, timeout=5)
                logger.info(f"Network barrier deactivated: {response.status_code}")
            except Exception as e:
                logger.error(f"Network deactivation failed: {e}")
        else:
            logger.info("Mock: Network barrier deactivated")
    
    def trigger_barrier(self) -> bool:
        """
        Trigger barrier activation with automatic deactivation
        
        Returns:
            True if activation successful, False otherwise
        """
        if not self.activate():
            return False
        
        # Wait for activation time
        time.sleep(self.activation_time)
        
        # Deactivate
        self.deactivate()
        return True
    
    def emergency_stop(self):
        """Emergency stop of barrier operation"""
        with self.lock:
            try:
                self.deactivate()
                self.is_active = False
                logger.info("Emergency stop activated")
            except Exception as e:
                logger.error(f"Emergency stop error: {e}")

# Global barrier controller instance
barrier_controller = BarrierController()

# Convenience functions
def activate_barrier() -> bool:
    """Activate barrier"""
    return barrier_controller.activate()

def deactivate_barrier():
    """Deactivate barrier"""
    barrier_controller.deactivate()

def trigger_barrier() -> bool:
    """Trigger barrier with automatic deactivation"""
    return barrier_controller.trigger_barrier()

def emergency_stop():
    """Emergency stop barrier"""
    barrier_controller.emergency_stop()

# Backward compatibility with existing code
def open_barrier(duration=3):
    """Open barrier for specified duration (backward compatibility)"""
    original_activation_time = barrier_controller.activation_time
    barrier_controller.activation_time = duration
    result = trigger_barrier()
    barrier_controller.activation_time = original_activation_time
    return result