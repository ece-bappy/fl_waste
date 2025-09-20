"""
Python interface for Arduino-based medical waste sorting robot.
Communicates with Arduino controller via serial interface.
"""

import serial
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

class WasteCategory(Enum):
    """Medical waste categories."""
    SHARPS = "SHARPS"
    INFECTIOUS = "INFECTIOUS"
    PHARMACEUTICAL = "PHARMACEUTICAL"
    PATHOLOGICAL = "PATHOLOGICAL"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"

class RobotState(Enum):
    """Robot operational states."""
    IDLE = "IDLE"
    ROTATING = "ROTATING"
    PICKING_UP = "PICKING_UP"
    DROPPING = "DROPPING"
    HOMING = "HOMING"
    ERROR = "ERROR"

class SortingController:
    """Interface to Arduino-based sorting robot."""
    
    def __init__(self, 
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 1.0):
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False
        self.current_state = RobotState.IDLE
        self.response_queue = queue.Queue()
        self.listener_thread = None
        self.stop_listener = False
        
        self._connect()
    
    def _connect(self):
        """Establish serial connection with Arduino."""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Start response listener thread
            self._start_listener()
            
            # Test connection
            if self._test_connection():
                self.is_connected = True
                logger.info(f"Connected to Arduino on {self.port}")
            else:
                self.is_connected = False
                logger.error("Failed to establish connection with Arduino")
                
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.is_connected = False
    
    def _start_listener(self):
        """Start background thread to listen for Arduino responses."""
        self.stop_listener = False
        self.listener_thread = threading.Thread(target=self._listen_for_responses)
        self.listener_thread.daemon = True
        self.listener_thread.start()
    
    def _listen_for_responses(self):
        """Listen for responses from Arduino."""
        while not self.stop_listener and self.serial_connection:
            try:
                if self.serial_connection.in_waiting > 0:
                    response = self.serial_connection.readline().decode('utf-8').strip()
                    if response:
                        self._process_response(response)
                        
            except Exception as e:
                logger.error(f"Error listening for responses: {e}")
                break
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def _process_response(self, response: str):
        """Process response from Arduino."""
        try:
            if response.startswith("INFO:"):
                logger.info(f"Arduino: {response[5:]}")
                
            elif response.startswith("ERROR:"):
                logger.error(f"Arduino: {response[6:]}")
                
            elif response.startswith("WARNING:"):
                logger.warning(f"Arduino: {response[8:]}")
                
            elif response.startswith("STATUS:"):
                self._parse_status_response(response[7:])
                
            elif response.startswith("TEST:"):
                logger.info(f"Arduino Test: {response[5:]}")
                
            # Add to response queue for command responses
            self.response_queue.put(response)
            
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
    
    def _parse_status_response(self, status_data: str):
        """Parse status response from Arduino."""
        try:
            status_parts = status_data.split(',')
            status_dict = {}
            
            for part in status_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    status_dict[key] = value
            
            # Update current state
            if 'State' in status_dict:
                try:
                    self.current_state = RobotState(status_dict['State'])
                except ValueError:
                    logger.warning(f"Unknown robot state: {status_dict['State']}")
            
            logger.debug(f"Robot status: {status_dict}")
            
        except Exception as e:
            logger.error(f"Failed to parse status response: {e}")
    
    def _test_connection(self) -> bool:
        """Test connection with Arduino."""
        try:
            # Send status command
            self._send_command("STATUS")
            
            # Wait for response
            response = self._wait_for_response(timeout=5.0)
            
            return response is not None and "State=" in response
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _send_command(self, command: str):
        """Send command to Arduino."""
        if not self.is_connected or not self.serial_connection:
            raise ConnectionError("Not connected to Arduino")
        
        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_connection.write(command_bytes)
            logger.debug(f"Sent command: {command}")
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            raise
    
    def _wait_for_response(self, timeout: float = 5.0) -> Optional[str]:
        """Wait for response from Arduino."""
        try:
            response = self.response_queue.get(timeout=timeout)
            return response
            
        except queue.Empty:
            logger.warning("Timeout waiting for Arduino response")
            return None
    
    def sort_waste(self, category: WasteCategory) -> bool:
        """Sort waste item to specified category."""
        if not self.is_connected:
            logger.error("Not connected to Arduino")
            return False
        
        if self.current_state != RobotState.IDLE:
            logger.warning(f"Robot not idle, current state: {self.current_state}")
            return False
        
        try:
            # Send sort command
            command = f"SORT:{category.value}"
            self._send_command(command)
            
            # Wait for acknowledgment
            response = self._wait_for_response(timeout=10.0)
            
            if response and "INFO:Sorting to" in response:
                logger.info(f"Started sorting to {category.value}")
                return True
            else:
                logger.error(f"Failed to start sorting: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to sort waste: {e}")
            return False
    
    def home_robot(self) -> bool:
        """Home the robot to initial position."""
        if not self.is_connected:
            logger.error("Not connected to Arduino")
            return False
        
        try:
            self._send_command("HOME")
            response = self._wait_for_response(timeout=10.0)
            
            if response and "INFO:Homing system" in response:
                logger.info("Robot homing initiated")
                return True
            else:
                logger.error(f"Failed to home robot: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to home robot: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current robot status."""
        if not self.is_connected:
            return {
                'connected': False,
                'state': 'DISCONNECTED',
                'error': 'Not connected to Arduino'
            }
        
        try:
            self._send_command("STATUS")
            response = self._wait_for_response(timeout=5.0)
            
            if response and response.startswith("STATUS:"):
                # Parse status response
                status_data = response[7:]
                status_parts = status_data.split(',')
                status_dict = {'connected': True}
                
                for part in status_parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        status_dict[key.lower()] = value
                
                return status_dict
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
        
        return {
            'connected': True,
            'state': self.current_state.value,
            'error': 'Failed to get detailed status'
        }
    
    def run_self_test(self) -> bool:
        """Run self-test on the robot."""
        if not self.is_connected:
            logger.error("Not connected to Arduino")
            return False
        
        try:
            self._send_command("TEST")
            
            # Wait for test completion
            test_responses = []
            start_time = time.time()
            
            while time.time() - start_time < 30:  # 30 second timeout
                response = self._wait_for_response(timeout=1.0)
                if response:
                    test_responses.append(response)
                    if "INFO:Self test completed" in response:
                        break
            
            # Check for test failures
            failed_tests = [resp for resp in test_responses if "FAIL" in resp]
            
            if failed_tests:
                logger.error(f"Self-test failed: {failed_tests}")
                return False
            else:
                logger.info("Self-test completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Self-test failed: {e}")
            return False
    
    def wait_for_idle(self, timeout: float = 60.0) -> bool:
        """Wait for robot to return to idle state."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.current_state == RobotState.IDLE:
                return True
            time.sleep(0.5)
        
        logger.warning(f"Timeout waiting for robot to become idle")
        return False
    
    def is_busy(self) -> bool:
        """Check if robot is currently busy."""
        return self.current_state != RobotState.IDLE
    
    def disconnect(self):
        """Disconnect from Arduino."""
        try:
            self.stop_listener = True
            
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1.0)
            
            if self.serial_connection:
                self.serial_connection.close()
            
            self.is_connected = False
            logger.info("Disconnected from Arduino")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()
