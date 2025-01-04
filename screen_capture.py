import cv2
import numpy as np
import pyautogui
import time
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from collections import deque
import win32gui
import win32ui
import win32con
from ctypes import windll

class WindowCapture:
    def __init__(self, window_title):
        """
        Initialize window capture
        Args:
            window_title: Title of the window to capture (can be partial match)
        """
        self.window_title = window_title
        self.hwnd = None
        self.update_window_handle()

    def update_window_handle(self):
        """Find and update the window handle"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in title.lower():
                    extra.append(hwnd)
            return True

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        
        if not hwnds:
            raise ValueError(f"Could not find window with title containing '{self.window_title}'")
        
        self.hwnd = hwnds[0]  # Use the first matching window
        print(f"Found window: {win32gui.GetWindowText(self.hwnd)}")
        
    def get_window_position(self) -> Tuple[int, int, int, int]:
        """Get the position and size of the window"""
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            x = rect[0]
            y = rect[1]
            width = rect[2] - x
            height = rect[3] - y
            return x, y, width, height
        except win32gui.error:
            self.update_window_handle()
            return self.get_window_position()

    def capture(self) -> Optional[np.ndarray]:
        """Capture the window content"""
        try:
            x, y, width, height = self.get_window_position()
            
            # Get window DC
            wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            
            # Create bitmap object
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
            cDC.SelectObject(dataBitMap)
            
            # Copy window content
            result = windll.user32.PrintWindow(self.hwnd, cDC.GetSafeHdc(), 2)
            if result == 0:
                raise Exception("PrintWindow failed")
            
            # Convert to numpy array
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)
            
            # Clean up
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
            
        except Exception as e:
            print(f"Error capturing window: {e}")
            self.update_window_handle()
            return None

class ScreenBuffer:
    def __init__(self, max_size=5):
        """
        Buffer to store recent screenshots
        Args:
            max_size: Number of screenshots to keep in buffer
        """
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add(self, screenshot: np.ndarray):
        """Add a new screenshot to the buffer"""
        self.buffer.append(screenshot)
        self.timestamps.append(time.time())
    
    def get_recent(self, within_seconds: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """Get screenshots captured within the last n seconds"""
        current_time = time.time()
        recent = []
        for screenshot, timestamp in zip(self.buffer, self.timestamps):
            if current_time - timestamp <= within_seconds:
                recent.append((screenshot, timestamp))
        return recent

class GameScreenshotTaker:
    def __init__(
        self,
        window_title: str,
        template_paths: List[str],
        save_dir: str = "screenshots",
        threshold: float = 0.8,
        capture_delay: float = 0.1,
        dialogue_width: int = 600,
        dialogue_height: int = 100,
        vertical_offset: int = -10,
        horizontal_offset: int = 0,    # Added horizontal offset parameter
        buffer_size: int = 5,
        buffer_time_window: float = 0.5
    ):
        """
        Initialize the screenshot taker with template images
        Uses color-based template matching with screenshot buffering
        """
        self.window_capture = WindowCapture(window_title)
        self.templates = {}
        self.template_dimensions = {}
        
        # Load all templates
        for path in template_paths:
            template_name = os.path.splitext(os.path.basename(path))[0]
            template = cv2.imread(path)
            if template is None:
                raise ValueError(f"Could not load template: {path}")
            
            self.templates[template_name] = template
            self.template_dimensions[template_name] = template.shape
            
        self.threshold = threshold
        self.capture_delay = capture_delay
        self.dialogue_width = dialogue_width
        self.dialogue_height = dialogue_height
        self.vertical_offset = vertical_offset
        self.horizontal_offset = horizontal_offset  # Store horizontal offset
        self.buffer_time_window = buffer_time_window
        
        # Initialize screenshot buffer
        self.screen_buffer = ScreenBuffer(max_size=buffer_size)
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def take_screenshot(self) -> Optional[np.ndarray]:
        """Take a screenshot of the window and add it to the buffer"""
        screenshot = self.window_capture.capture()
        if screenshot is not None:
            self.screen_buffer.add(screenshot)
        return screenshot
    
    def get_dialogue_region(self, screenshot: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Calculate and extract the dialogue box region"""
        y1 = max(0, y + self.vertical_offset)
        y2 = min(screenshot.shape[0], y1 + self.dialogue_height)
        x1 = max(0, x + self.horizontal_offset)  # Apply horizontal offset
        x2 = min(screenshot.shape[1], x1 + self.dialogue_width)
        
        region = screenshot[y1:y2, x1:x2].copy()
        return region, (x1, y1, x2, y2)
    
    def find_best_match(self, screenshot: np.ndarray, template_name: str, template: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """Find the best match for a template in the screenshot"""
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_loc, max_val  # Always return location and confidence
    
    def find_best_region(self, x: int, y: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Find the best dialogue region from recent screenshots
        Returns the region and its capture timestamp
        """
        recent_screenshots = self.screen_buffer.get_recent(self.buffer_time_window)
        best_region = None
        best_timestamp = None
        highest_text_content = -1
        
        for screenshot, timestamp in recent_screenshots:
            region, _ = self.get_dialogue_region(screenshot, x, y)
            
            # Convert to grayscale for text content estimation
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)
            text_content = np.sum(binary == 0)  # Count dark pixels
            
            if text_content > highest_text_content:
                highest_text_content = text_content
                best_region = region
                best_timestamp = timestamp
        
        return best_region, best_timestamp
    
    def save_screenshot(self, region: np.ndarray, template_name: str) -> str:
        """Save the dialogue region"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dialogue_{template_name}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, region)
        return filepath
    
    def monitor(self, interval: float = 0.2):
        """Continuously monitor the window for any of the templates"""
        print(f"Starting color-based monitor with screenshot buffer... Press Ctrl+C to stop")
        print(f"Monitoring window: {self.window_capture.window_title}")
        print(f"Dialogue box size: {self.dialogue_width}x{self.dialogue_height}")
        print(f"Buffer window: {self.buffer_time_window} seconds")
        
        last_capture_time = 0
        min_capture_interval = 1.0
        
        try:
            while True:
                current_time = time.time()
                
                if current_time - last_capture_time >= min_capture_interval:
                    screenshot = self.take_screenshot()
                    
                    if screenshot is not None:
                        for template_name, template in self.templates.items():
                            location, confidence = self.find_best_match(screenshot, template_name, template)
                            
                            if location is not None and confidence >= 0.95:  # Strict confidence threshold
                                x, y = location
                                print(f"Match found for '{template_name}' at ({x}, {y}) with confidence {confidence:.3f}")
                                
                                if self.capture_delay > 0:
                                    print(f"Waiting {self.capture_delay} seconds...")
                                    time.sleep(self.capture_delay)
                                    self.take_screenshot()  # Add one more to buffer after delay
                                
                                # Find best region from recent screenshots
                                best_region, timestamp = self.find_best_region(x, y)
                                if best_region is not None:
                                    filename = self.save_screenshot(best_region, template_name)
                                    print(f"Saved best region from {time.time() - timestamp:.3f}s ago as: {filename}")
                                    last_capture_time = current_time
                            elif location is not None:
                                # Match found but confidence too low
                                x, y = location

                                # print for debugging purposes only
                                # print(f"Match found but confidence too low: {confidence:.3f} < 0.95 for '{template_name}' at ({x}, {y})")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

# Example usage
if __name__ == "__main__":

    # generate template paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_dir, "templates")

    screenshot_taker = GameScreenshotTaker(
        window_title="Shin chan Shiro and the Coal Town",  # Replace with your game's window title
        template_paths= [os.path.join(templates_path, f) for f in os.listdir(templates_path)],
        # template_paths=[r"C:\Users\andre\Desktop\AutoScreen_v0\misae_name.png"],
        save_dir= os.path.join(current_dir, "screenshots"),
        dialogue_width=740,
        dialogue_height=260,
        vertical_offset=-25,
        horizontal_offset=-95,
        threshold=0.7,
        capture_delay=0.20,
        buffer_size=10,
        buffer_time_window=0.4
    )
    
    screenshot_taker.monitor(interval=0.2)

# Params tuning
# Lower capture_delay = Faster capture but might miss some text
# Smaller buffer_time_window = Faster processing but less chance to catch the best frame
# Lower min_capture_interval = More frequent captures but might get duplicates
# Lower interval = More frequent checks but higher CPU usage