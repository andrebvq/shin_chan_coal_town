# Python 3.6+ required

import os
import cv2
import time
import win32ui
import win32gui
import numpy as np
from ctypes import windll
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Dict


class GPUAccelerator:
    def __init__(self):
        self.has_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.has_gpu:
            self.gpu_matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC3, cv2.TM_CCOEFF_NORMED)
            print("GPU acceleration enabled")
        else:
            print("GPU acceleration not available, using CPU")
    
    def to_gpu(self, img):
        return cv2.cuda.GpuMat(img) if self.has_gpu else img
    
    def from_gpu(self, gpu_mat):
        return gpu_mat.download() if self.has_gpu else gpu_mat
    
    def match_template(self, img, template):
        if self.has_gpu:
            gpu_img = self.to_gpu(img)
            gpu_template = self.to_gpu(template)
            result = self.gpu_matcher.match(gpu_img, gpu_template)
            return self.from_gpu(result)
        return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    
    def threshold(self, img, thresh, maxval, type):
        if self.has_gpu:
            gpu_img = self.to_gpu(img)
            _, result = cv2.cuda.threshold(gpu_img, thresh, maxval, type)
            return self.from_gpu(result)
        return cv2.threshold(img, thresh, maxval, type)[1]

class WindowCapture:
    def __init__(self, window_title):
        self.window_title = window_title
        self.hwnd = None
        self.update_window_handle()

    def update_window_handle(self):
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
        
        self.hwnd = hwnds[0]
        print(f"Found window: {win32gui.GetWindowText(self.hwnd)}")
    
    def capture(self) -> Optional[np.ndarray]:
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y = rect[0], rect[1]
            width, height = rect[2] - x, rect[3] - y
            
            wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
            cDC.SelectObject(dataBitMap)
            
            if windll.user32.PrintWindow(self.hwnd, cDC.GetSafeHdc(), 2) == 0:
                raise Exception("PrintWindow failed")
            
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)
            
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        except Exception as e:
            print(f"Error capturing window: {e}")
            self.update_window_handle()
            return None

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
        horizontal_offset: int = 0,
        buffer_size: int = 5,
        buffer_time_window: float = 0.5,
        debug: bool = False
    ):
        self.window_capture = WindowCapture(window_title)
        self.gpu = GPUAccelerator()
        self.templates = {}
        self.gray_templates = {}
        
        for path in template_paths:
            template_name = os.path.splitext(os.path.basename(path))[0]
            template = cv2.imread(path)
            if template is None:
                raise ValueError(f"Could not load template: {path}")
            
            self.templates[template_name] = template
            if template_name == "dialogue_box":
                self.gray_templates[template_name] = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.dialogue_dims = (dialogue_width, dialogue_height)
        self.offsets = (vertical_offset, horizontal_offset)
        self.params = {
            'threshold': threshold,
            'capture_delay': capture_delay,
            'buffer_time': buffer_time_window
        }
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.debug = debug
        self.debug_window_name = "Template Matching Debug"

    def find_best_match(self, screenshot: np.ndarray, template_name: str, template: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        if template_name == "dialogue_box":
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            bin_screenshot = self.gpu.threshold(gray_screenshot, 200, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(bin_screenshot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_match = {'confidence': 0.0, 'location': None}
            template_area = template.shape[0] * template.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < template_area * 0.5:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                if not (2.0 < w/h < 4.0) or area/(w*h) < 0.7:
                    continue
                
                confidence = self._calculate_box_confidence(area)
                if confidence > best_match['confidence']:
                    best_match.update({
                        'confidence': min(confidence, 0.95),
                        'location': (x, y)
                    })
            
            return best_match['location'], best_match['confidence']
        else:
            result = self.gpu.match_template(screenshot, template)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            confidence = max(max_val, 0.95) if max_val >= 0.95 else max_val
            return max_loc, confidence

    def _calculate_box_confidence(self, area: float) -> float:
        MIN_AREA, MAX_AREA = 130000, 160000
        if MIN_AREA <= area <= MAX_AREA:
            return 0.90 + (0.05 * (area - MIN_AREA) / (MAX_AREA - MIN_AREA))
        return 0.90 * (min(area, MAX_AREA) / max(area, MIN_AREA))
    
    def get_dialogue_region(self, screenshot: np.ndarray, x: int, y: int) -> np.ndarray:
        height, width = screenshot.shape[:2]
        y1 = max(0, y + self.offsets[0])
        y2 = min(height, y1 + self.dialogue_dims[1])
        x1 = max(0, x + self.offsets[1])
        x2 = min(width, x1 + self.dialogue_dims[0])
        return screenshot[y1:y2, x1:x2].copy()
    
    def save_screenshot(self, img: np.ndarray, template_name: str) -> str:
        filename = f"{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, img)
        return filepath

    def draw_debug_overlay(self, screenshot: np.ndarray, matches: List[Dict]) -> np.ndarray:
        debug_img = screenshot.copy()
        
        for idx, match in enumerate(matches):
            if match['location'] and match['confidence'] > 0:
                x, y = match['location']
                template = self.templates[match['template_name']]
                w, h = template.shape[1], template.shape[0]
                
                # Color based on confidence (Red->Yellow->Green)
                confidence = match['confidence']
                color = (
                    0,
                    int(255 * confidence),
                    int(255 * (1 - confidence))
                )
                
                # Draw rectangle around match
                cv2.rectangle(
                    debug_img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )
                
                # Draw text with template name and confidence
                text = f"{match['template_name']}: {confidence:.2f}"
                cv2.putText(
                    debug_img,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        return debug_img

    def monitor(self, interval: float = 0.2):
        last_capture = 0
        min_interval = 1.0
        
        try:
            while True:
                current_time = time.time()
                if current_time - last_capture >= min_interval:
                    screenshot = self.window_capture.capture()
                    
                    if screenshot is not None:
                        matches = []
                        
                        for name, template in self.templates.items():
                            location, confidence = self.find_best_match(screenshot, name, template)
                            matches.append({
                                'template_name': name,
                                'location': location,
                                'confidence': confidence
                            })
                        
                        # Sort matches by confidence
                        matches.sort(key=lambda x: x['confidence'], reverse=True)
                        best_match = matches[0]
                        
                        if self.debug:
                            debug_img = self.draw_debug_overlay(screenshot, matches)
                            cv2.imshow(self.debug_window_name, debug_img)
                            key = cv2.waitKey(1)
                            if key == ord('q'):
                                break
                        
                        if best_match['confidence'] >= 0.65:
                            name = best_match['template_name']
                            x, y = best_match['location']
                            print(f"Match: {name} at ({x}, {y}) conf={best_match['confidence']:.3f}")
                            
                            if self.params['capture_delay'] > 0:
                                time.sleep(self.params['capture_delay'])
                            
                            region = self.get_dialogue_region(screenshot, x, y)
                            filepath = self.save_screenshot(region, name)
                            print(f"Saved: {filepath}")
                            last_capture = current_time
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopped monitoring")
        finally:
            if self.debug:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(current_dir, "templates")

    screenshot_taker = GameScreenshotTaker(
        window_title="Shin chan Shiro and the Coal Town",
        template_paths=[os.path.join(templates_path, f) for f in os.listdir(templates_path)],
        save_dir=os.path.join(current_dir, "screenshots"),
        dialogue_width=740,
        dialogue_height=260,
        vertical_offset=-25,
        horizontal_offset=-95,
        threshold=0.7,
        capture_delay=0.20,
        buffer_size=10,
        buffer_time_window=0.4,
        debug=True  # Enable debug visualization
    )
    
    screenshot_taker.monitor(interval=0.2)
